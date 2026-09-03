from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chipmem.dataset import load_dataset
from chipmem.memory import ChipMEM
from chipmem.statistical.hooks import ToolHooks
from chipmem.statistical.model import AgentModel
from chipmem.statistical.online import StatisticalMemory
from chipmem.store import _atomic_bytes, _mkdir_chain_durable

MODE_CONTRACT = {
    "memory_off": {"procedural": False, "statistical": False},
    "procedural_only": {"procedural": True, "statistical": False},
    "statistical_only": {"procedural": False, "statistical": True},
    "chipmem": {"procedural": True, "statistical": True},
}

_REQUIRED = {
    "mode",
    "domain",
    "task_directory",
    "task_order",
    "task_artifact",
    "state_directory",
    "output_directory",
    "top_k",
    "threshold",
    "embedding_dimension",
    "embedding_callable",
    "agent_callable",
    "harness_callable",
    "distill_callable",
}
_OPTIONAL = {
    "agent_endpoint_env",
    "features_callable",
    "max_nudges",
    "nudge_threshold",
    "planner_callable",
    "procedural_policy",
    "procedural_seed_directory",
    "session_timeout_seconds",
    "statistical_seed_directory",
    "step_limit",
}


def _json_object(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must contain valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return payload


def _jsonl_objects(path: Path, label: str) -> list[dict]:
    rows = []
    try:
        lines = path.read_bytes().decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must contain valid UTF-8 JSONL") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{label} contains invalid JSON at line {line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise TypeError(f"{label} line {line_number} must be a JSON object")
        rows.append(row)
    return rows


def _positive_number(value, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{label} must be a finite positive number")
    return float(value)


def _validate_count_map(value, label: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    for key, pair in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} keys must be non-empty strings")
        if not isinstance(pair, list) or len(pair) != 2:
            raise TypeError(f"{label} values must be two-element lists")
        for count in pair:
            if (
                not isinstance(count, (int, float))
                or isinstance(count, bool)
                or not math.isfinite(float(count))
                or float(count) < 0
            ):
                raise ValueError(f"{label} counts must be finite and nonnegative")


def _validate_model_payload(payload: dict) -> AgentModel:
    _validate_count_map(payload.get("counts", {}), "retry counts")
    _positive_number(payload.get("prior_alpha", 1.0), "retry prior_alpha")
    _positive_number(payload.get("prior_beta", 1.0), "retry prior_beta")
    _positive_number(
        payload.get("parent_strength", 8.0),
        "retry parent_strength",
    )
    recovery = payload.get("recovery_advisor", {})
    if not isinstance(recovery, dict):
        raise TypeError("recovery_advisor must be a JSON object")
    _validate_count_map(recovery.get("counts", {}), "recovery counts")
    _positive_number(recovery.get("prior_alpha", 1.0), "recovery prior_alpha")
    _positive_number(recovery.get("prior_beta", 1.0), "recovery prior_beta")
    _positive_number(
        recovery.get("parent_strength", 8.0),
        "recovery parent_strength",
    )
    examples = recovery.get("examples", {})
    if not isinstance(examples, dict):
        raise TypeError("recovery examples must be a JSON object")
    for key, excerpts in examples.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(excerpts, list)
            or any(not isinstance(excerpt, str) for excerpt in excerpts)
        ):
            raise TypeError("recovery examples must map strings to string lists")
    if not isinstance(payload.get("metadata", {}), dict):
        raise TypeError("statistical model metadata must be a JSON object")
    try:
        return AgentModel.from_dict(payload)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("statistical model structure is invalid") from exc


def _nonnegative_integer(value, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _nonempty_string(value, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _binary_label(value, label: str) -> int:
    if type(value) is not int or value not in {0, 1}:
        raise ValueError(f"{label} must be exactly 0 or 1")
    return value


def _validate_retry_rows(rows: list[dict]) -> None:
    required = {
        "session",
        "step",
        "ct",
        "rc",
        "pr",
        "pec",
        "te",
        "success",
        "error_class",
    }
    seen = set()
    for line_number, row in enumerate(rows, 1):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(
                f"nudge experience line {line_number} is missing fields: {missing}"
            )
        session = _nonempty_string(row["session"], "nudge session")
        step = _nonnegative_integer(row["step"], "nudge step")
        _nonempty_string(row["ct"], "nudge command type")
        _nonnegative_integer(row["rc"], "nudge retry count")
        if row["pr"] not in {"none", "fail", "success"}:
            raise ValueError("nudge previous result is invalid")
        _nonempty_string(row["pec"], "nudge previous error class")
        _nonnegative_integer(row["te"], "nudge turns elapsed")
        _binary_label(row["success"], "nudge success")
        _nonempty_string(row["error_class"], "nudge error class")
        call_hash = row.get("call_sha256")
        if call_hash is not None and (
            not isinstance(call_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", call_hash)
        ):
            raise ValueError("nudge call_sha256 is invalid")
        key = (session, step)
        if key in seen:
            raise ValueError("duplicate nudge experience session-step")
        seen.add(key)


def _validate_recovery_rows(rows: list[dict]) -> None:
    required = {
        "record_type",
        "session",
        "step",
        "failure_idx",
        "error",
        "stage",
        "strategy",
        "recovered",
    }
    seen = set()
    for line_number, row in enumerate(rows, 1):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(
                f"recovery experience line {line_number} is missing fields: {missing}"
            )
        if row["record_type"] != "recovery":
            raise ValueError("recovery record_type must be recovery")
        session = _nonempty_string(row["session"], "recovery session")
        _nonnegative_integer(row["step"], "recovery step")
        failure_idx = _nonnegative_integer(
            row["failure_idx"],
            "recovery failure_idx",
        )
        _nonempty_string(row["error"], "recovery error")
        _nonempty_string(row["stage"], "recovery stage")
        strategy = _nonempty_string(row["strategy"], "recovery strategy")
        _binary_label(row["recovered"], "recovery recovered")
        if row.get("plan_excerpt") is not None and not isinstance(
            row["plan_excerpt"], str
        ):
            raise TypeError("recovery plan_excerpt must be null or a string")
        if row.get("context_window") is not None and not isinstance(
            row["context_window"], list
        ):
            raise TypeError("recovery context_window must be a list")
        key = (session, failure_idx, strategy)
        if key in seen:
            raise ValueError("duplicate recovery experience key")
        seen.add(key)


def _validate_procedural_seed(
    seed_root: Path,
    domain: str,
    embedding_dimension: int,
) -> tuple[Path, list[Path]]:
    source = seed_root / "agents" / domain / "memory"
    if not source.is_dir() or source.is_symlink():
        raise FileNotFoundError(f"procedural seed is missing: {source}")
    skill_dirs = [
        path
        for path in sorted(source.iterdir())
        if path.is_dir() and re.fullmatch(r"skill_[0-9]{4,}", path.name)
    ]
    if not skill_dirs:
        raise ValueError("procedural seed contains no verified skills")
    canonical = {"SKILL.md", "skill.json", "embeddings.json"}
    for skill in skill_dirs:
        if skill.is_symlink() or {path.name for path in skill.iterdir()} != canonical:
            raise ValueError("procedural seed skill has noncanonical files")
        if any(
            not (skill / name).is_file() or (skill / name).is_symlink()
            for name in canonical
        ):
            raise ValueError("procedural seed contains an invalid skill file")
        try:
            skill_text = (skill / "SKILL.md").read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("procedural SKILL.md must be valid UTF-8") from exc
        if not skill_text.strip():
            raise ValueError("procedural SKILL.md must be non-empty")
        metadata = _json_object(skill / "skill.json", "procedural skill metadata")
        source_hash = metadata.get("source_task_sha256")
        if metadata.get("agent_type") != domain:
            raise ValueError("procedural skill domain does not match configuration")
        if metadata.get("verdict") != "pass":
            raise ValueError("procedural skill lacks verified PASS provenance")
        source_task = metadata.get("source_task")
        if not isinstance(source_task, str) or not source_task.strip():
            raise ValueError("procedural skill source_task must be non-empty")
        if not isinstance(source_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", source_hash
        ):
            raise ValueError("procedural skill source hash is invalid")
        embedding = _json_object(
            skill / "embeddings.json",
            "procedural skill embedding",
        ).get("vector")
        if not isinstance(embedding, list):
            raise TypeError("procedural skill embedding vector must be a list")
        try:
            vector = [float(value) for value in embedding]
        except (TypeError, ValueError) as exc:
            raise ValueError("procedural skill embedding is not numeric") from exc
        if len(vector) != embedding_dimension or any(
            not math.isfinite(value) for value in vector
        ):
            raise ValueError("procedural skill embedding dimension or values are invalid")
    return source, skill_dirs


def _validate_statistical_seed(seed_root: Path, domain: str) -> Path:
    source = seed_root / "agents" / domain / "memory"
    if not source.is_dir() or source.is_symlink():
        raise FileNotFoundError(f"statistical seed is missing: {source}")
    paths = {
        name: source / name
        for name in (
            "model.json",
            "nudge_experience.jsonl",
            "recovery_experience.jsonl",
        )
    }
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"statistical seed file is missing: {name}")
    model = _validate_model_payload(
        _json_object(paths["model.json"], "statistical model")
    )
    model.to_dict()
    nudge_rows = _jsonl_objects(
        paths["nudge_experience.jsonl"],
        "nudge experience",
    )
    recovery_rows = _jsonl_objects(
        paths["recovery_experience.jsonl"],
        "recovery experience",
    )
    _validate_retry_rows(nudge_rows)
    _validate_recovery_rows(recovery_rows)
    return source


def _copy_statistical_seed(
    seed_root: Path,
    destination_root: Path,
    domain: str,
) -> None:
    source = _validate_statistical_seed(seed_root, domain)
    destination = destination_root / "agents" / domain / "memory"
    if destination.exists():
        raise FileExistsError(f"statistical destination already exists: {destination}")
    canonical = (
        "model.json",
        "nudge_experience.jsonl",
        "recovery_experience.jsonl",
    )
    _mkdir_chain_durable(destination)
    for name in canonical:
        source_file = source / name
        if not source_file.is_file() or source_file.is_symlink():
            raise FileNotFoundError(f"statistical seed file is missing: {name}")
        payload = source_file.read_bytes()
        target = destination / name
        _atomic_bytes(target, payload)
        if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(payload).digest():
            raise RuntimeError(f"statistical seed copy mismatch: {name}")


def _copy_procedural_seed(
    seed_root: Path,
    destination_root: Path,
    domain: str,
    embedding_dimension: int,
) -> None:
    _source, skill_dirs = _validate_procedural_seed(
        seed_root,
        domain,
        embedding_dimension,
    )
    destination = destination_root / "agents" / domain / "memory"
    if destination.exists():
        raise FileExistsError(f"procedural destination already exists: {destination}")
    _mkdir_chain_durable(destination)
    canonical = {"SKILL.md", "skill.json", "embeddings.json"}
    for source_skill in skill_dirs:
        if source_skill.is_symlink():
            raise ValueError("procedural seed skills must not be symlinks")
        entries = {path.name for path in source_skill.iterdir()}
        if entries != canonical:
            raise ValueError("procedural seed skill has noncanonical files")
        target_skill = destination / source_skill.name
        _mkdir_chain_durable(target_skill)
        for name in sorted(canonical):
            source_file = source_skill / name
            if not source_file.is_file() or source_file.is_symlink():
                raise ValueError("procedural seed contains an invalid skill file")
            payload = source_file.read_bytes()
            target = target_skill / name
            _atomic_bytes(target, payload)
            if target.read_bytes() != payload:
                raise RuntimeError(f"procedural seed copy mismatch: {name}")


def _load_callable(specification: str) -> Callable:
    if not isinstance(specification, str) or specification.count(":") != 1:
        raise ValueError("callable specifications must use module:function")
    module_name, function_name = specification.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise TypeError(f"configured callable is not callable: {specification}")
    return function


def _path(value: str, base: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _is_runtime_lock(name: str) -> bool:
    return bool(
        name == ".store.lock"
        or re.fullmatch(r"\.source-[0-9a-f]{64}\.lock", name)
        or re.fullmatch(r"[0-9a-f]{64}_[0-9a-f]{64}\.lock", name)
    )


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.exists():
        return digest.hexdigest()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if (
            not path.is_symlink()
            and path.is_file()
            and _is_runtime_lock(path.name)
        ):
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        if path.is_symlink():
            entry_type = b"L"
            payload = os.readlink(path).encode("utf-8")
        elif path.is_dir():
            entry_type = b"D"
            payload = b""
        elif path.is_file():
            entry_type = b"F"
            payload = path.read_bytes()
        else:
            raise ValueError(f"unsupported memory-state entry: {path}")
        digest.update(entry_type)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_durable(path: Path) -> None:
    path.mkdir(parents=False, exist_ok=False)
    _fsync_directory(path.parent)


def _write_json(path: Path, payload: dict) -> None:
    try:
        content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    except (TypeError, ValueError) as exc:
        raise TypeError("payload must contain only JSON-serializable values") from exc
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _append_jsonl(path: Path, payload: dict) -> None:
    created = not path.exists()
    content = json.dumps(payload, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    if created:
        _fsync_directory(path.parent)


def _validated_verdict(raw_verdict: object) -> str:
    if type(raw_verdict) is not str:
        raise TypeError("harness verdict must be a string")
    verdict = raw_verdict.lower()
    if verdict not in {"pass", "fail", "invalid"}:
        raise ValueError("harness must return pass, fail, or invalid")
    return verdict


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("experiment configuration must be a JSON object")
    if overrides:
        payload.update(overrides)
    missing = sorted(_REQUIRED - set(payload))
    unknown = sorted(set(payload) - _REQUIRED - _OPTIONAL)
    if missing:
        raise ValueError(f"missing config fields: {missing}")
    if unknown:
        raise ValueError(f"unknown config fields: {unknown}")
    domain = payload["domain"]
    if not isinstance(domain, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*",
        domain,
    ):
        raise ValueError("domain must be a simple non-empty name")
    for field in (
        "task_directory",
        "task_artifact",
        "state_directory",
        "output_directory",
    ):
        if not isinstance(payload[field], str) or not payload[field]:
            raise ValueError(f"{field} must be a non-empty string")
    if type(payload["top_k"]) is not int or payload["top_k"] <= 0:
        raise ValueError("top_k must be a positive integer")
    retrieval_threshold = payload["threshold"]
    if (
        not isinstance(retrieval_threshold, (int, float))
        or isinstance(retrieval_threshold, bool)
        or not math.isfinite(float(retrieval_threshold))
    ):
        raise ValueError("threshold must be a finite number")
    if (
        type(payload["embedding_dimension"]) is not int
        or payload["embedding_dimension"] <= 0
    ):
        raise ValueError("embedding_dimension must be a positive integer")
    for field in (
        "embedding_callable",
        "agent_callable",
        "harness_callable",
        "distill_callable",
    ):
        if not isinstance(payload[field], str) or payload[field].count(":") != 1:
            raise ValueError(f"{field} must use module:function")
    if payload["mode"] not in MODE_CONTRACT:
        raise ValueError(
            "mode must be memory_off, procedural_only, statistical_only, or chipmem"
        )
    mode_contract = MODE_CONTRACT[payload["mode"]]
    default_policy = "evolving" if mode_contract["procedural"] else "disabled"
    requested_policy = payload.get("procedural_policy")
    procedural_policy = (
        default_policy if requested_policy is None else requested_policy
    )
    if mode_contract["procedural"]:
        if procedural_policy not in {"evolving", "read_only"}:
            raise ValueError("procedural_policy must be evolving or read_only")
        if procedural_policy == "read_only" and not payload.get(
            "procedural_seed_directory"
        ):
            raise ValueError("read_only procedural policy requires a seed directory")
    elif procedural_policy != "disabled" or payload.get("procedural_seed_directory"):
        raise ValueError("procedural state must be disabled for this mode")
    order = payload["task_order"]
    if not isinstance(order, list) or any(
        not isinstance(task, str) or not task or "/" in task or ".." in task
        for task in order
    ):
        raise ValueError("task_order must contain simple task identifiers")
    if len(order) != len(set(order)):
        raise ValueError("task_order contains duplicates")
    endpoint_env = payload.get("agent_endpoint_env")
    if endpoint_env is not None and (
        not isinstance(endpoint_env, str) or not endpoint_env
    ):
        raise ValueError("agent_endpoint_env must be null or a non-empty string")
    for field in ("procedural_seed_directory", "statistical_seed_directory"):
        value = payload.get(field)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{field} must be null or a non-empty string")
    features_callable = payload.get(
        "features_callable",
        "chipmem.statistical.features:OpenFlowFeatures",
    )
    if not isinstance(features_callable, str) or features_callable.count(":") != 1:
        raise ValueError("features_callable must use module:class")
    planner_callable = payload.get("planner_callable")
    if planner_callable is not None and (
        not isinstance(planner_callable, str) or planner_callable.count(":") != 1
    ):
        raise ValueError("planner_callable must be null or use module:function")
    for field, default in (
        ("step_limit", 32),
        ("session_timeout_seconds", 600),
    ):
        value = payload.get(field, default)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    max_nudges = payload.get("max_nudges")
    if max_nudges is not None and (
        type(max_nudges) is not int or max_nudges < 0
    ):
        raise ValueError("max_nudges must be null or a non-negative integer")
    threshold = payload.get("nudge_threshold", 0.20)
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(float(threshold))
        or not 0 <= threshold <= 1
    ):
        raise ValueError("nudge_threshold must be between 0 and 1")
    payload.setdefault("agent_endpoint_env", None)
    payload.setdefault(
        "features_callable",
        "chipmem.statistical.features:OpenFlowFeatures",
    )
    payload.setdefault("max_nudges", None)
    payload.setdefault("nudge_threshold", 0.20)
    payload.setdefault("planner_callable", None)
    payload["procedural_policy"] = procedural_policy
    payload.setdefault("procedural_seed_directory", None)
    payload.setdefault("session_timeout_seconds", 600)
    payload.setdefault("statistical_seed_directory", None)
    payload.setdefault("step_limit", 32)
    return payload


def _configuration_context(
    config_path: str | Path,
    overrides: dict | None = None,
) -> tuple[Path, dict, Path, Path, Path, list]:
    config_path = Path(config_path).resolve()
    config = load_config(config_path, overrides)
    base = config_path.parent
    task_root = _path(config["task_directory"], base)
    state_root = _path(config["state_directory"], base)
    output_root = _path(config["output_directory"], base)
    if output_root.exists():
        raise FileExistsError(f"output directory already exists: {output_root}")
    if not task_root.is_dir():
        raise FileNotFoundError(f"task directory does not exist: {task_root}")
    roots = {
        "task and state": (task_root, state_root),
        "task and output": (task_root, output_root),
        "state and output": (state_root, output_root),
    }
    for label, (left, right) in roots.items():
        if _paths_overlap(left, right):
            raise ValueError(f"configured {label} directories overlap")
    for field in ("procedural_seed_directory", "statistical_seed_directory"):
        value = config[field]
        if value is None:
            continue
        seed_root = _path(value, base)
        for label, writable in (
            ("state", state_root),
            ("output", output_root),
        ):
            if _paths_overlap(seed_root, writable):
                raise ValueError(f"configured {field} and {label} directories overlap")
    dataset = load_dataset(
        task_root,
        config["task_order"],
        config["task_artifact"],
    )
    return config_path, config, task_root, state_root, output_root, dataset


def preflight_experiment(
    config_path: str | Path,
    overrides: dict | None = None,
) -> dict:
    (
        config_path,
        config,
        task_root,
        state_root,
        output_root,
        dataset,
    ) = _configuration_context(config_path, overrides)
    for name in (
        "embedding_callable",
        "agent_callable",
        "harness_callable",
        "distill_callable",
    ):
        _load_callable(config[name])
    contract = MODE_CONTRACT[config["mode"]]
    if contract["statistical"]:
        _load_callable(config["features_callable"])
        if config["planner_callable"]:
            _load_callable(config["planner_callable"])
    endpoint_env = config["agent_endpoint_env"]
    if endpoint_env and endpoint_env not in os.environ:
        raise ValueError(
            f"configured endpoint environment variable is unset: {endpoint_env}"
        )
    if config["procedural_seed_directory"]:
        seed = _path(config["procedural_seed_directory"], config_path.parent)
        procedural_root = state_root / "procedural"
        if _paths_overlap(seed, procedural_root):
            raise ValueError("procedural seed and destination overlap")
        _validate_procedural_seed(
            seed,
            config["domain"],
            int(config["embedding_dimension"]),
        )
        destination = (
            procedural_root
            / "agents"
            / config["domain"]
            / "memory"
        )
        if destination.exists():
            raise FileExistsError(
                f"procedural destination already exists: {destination}"
            )
    if config["statistical_seed_directory"]:
        seed = _path(config["statistical_seed_directory"], config_path.parent)
        statistical_root = state_root / "statistical"
        if _paths_overlap(seed, statistical_root):
            raise ValueError("statistical seed and destination overlap")
        _validate_statistical_seed(seed, config["domain"])
        destination = (
            statistical_root
            / "agents"
            / config["domain"]
            / "memory"
        )
        if destination.exists():
            raise FileExistsError(
                f"statistical destination already exists: {destination}"
            )
    return {
        "config_path": str(config_path),
        "mode": config["mode"],
        "output_directory": str(output_root),
        "procedural_enabled": contract["procedural"],
        "state_directory": str(state_root),
        "statistical_enabled": contract["statistical"],
        "status": "valid",
        "task_count": len(dataset),
        "task_directory": str(task_root),
        "tasks": [task.task_id for task in dataset],
    }


def run_experiment(
    config_path: str | Path,
    overrides: dict | None = None,
) -> list[dict]:
    preflight_experiment(config_path, overrides)
    (
        config_path,
        config,
        _task_root,
        state_root,
        output_root,
        dataset,
    ) = _configuration_context(config_path, overrides)
    base = config_path.parent
    _mkdir_chain_durable(output_root.parent)
    _mkdir_durable(output_root)
    sessions_root = output_root / "sessions"
    _mkdir_durable(sessions_root)
    _write_json(output_root / "config.json", config)

    embed = _load_callable(config["embedding_callable"])
    agent = _load_callable(config["agent_callable"])
    harness = _load_callable(config["harness_callable"])
    distill = _load_callable(config["distill_callable"])
    dimension = int(config["embedding_dimension"])
    identity = {
        "dimension": dimension,
        "model": config["embedding_callable"],
        "provider": "configured-python-callable",
    }
    endpoint_env = config.get("agent_endpoint_env")
    if endpoint_env and endpoint_env not in os.environ:
        raise ValueError(
            f"configured endpoint environment variable is unset: {endpoint_env}"
        )
    execution = {
        "endpoint": os.environ.get(endpoint_env) if endpoint_env else None,
        "session_timeout_seconds": config.get("session_timeout_seconds", 600),
        "step_limit": config.get("step_limit", 32),
    }
    mode_contract = MODE_CONTRACT[config["mode"]]
    procedural_enabled = mode_contract["procedural"]
    statistical_enabled = mode_contract["statistical"]
    procedural_policy = config.get(
        "procedural_policy",
        "evolving" if procedural_enabled else "disabled",
    )
    procedural_root = state_root / "procedural"
    statistical_root = state_root / "statistical"

    memory = None
    if procedural_enabled:
        if config.get("procedural_seed_directory"):
            seed_root = _path(config["procedural_seed_directory"], base)
            if _paths_overlap(seed_root, procedural_root):
                raise ValueError("procedural seed and destination overlap")
            _copy_procedural_seed(
                seed_root,
                procedural_root,
                config["domain"],
                int(config["embedding_dimension"]),
            )
        memory = ChipMEM(
            procedural_root,
            config["domain"],
            top_k=int(config["top_k"]),
            threshold=float(config["threshold"]),
            allow_empty=True,
        )
        if procedural_policy == "read_only":
            memory.cache_directory = (
                state_root / "runtime_query_cache" / config["domain"]
            )

    statistical_memory = None
    if statistical_enabled:
        if config.get("statistical_seed_directory"):
            seed_root = _path(config["statistical_seed_directory"], base)
            if _paths_overlap(seed_root, statistical_root):
                raise ValueError("statistical seed and destination overlap")
            _copy_statistical_seed(
                seed_root,
                statistical_root,
                config["domain"],
            )
        features_factory = _load_callable(
            config.get(
                "features_callable",
                "chipmem.statistical.features:OpenFlowFeatures",
            )
        )
        features = features_factory()
        planner = (
            _load_callable(config["planner_callable"])
            if config.get("planner_callable")
            else None
        )
        statistical_memory = StatisticalMemory(
            statistical_root,
            config["domain"],
            features,
            threshold=float(config.get("nudge_threshold", 0.20)),
            planner=planner,
            max_nudges=config.get("max_nudges"),
        )

    rows: list[dict] = []
    for index, dataset_task in enumerate(dataset, start=1):
        task_id = dataset_task.task_id
        task_directory = dataset_task.directory
        document = dataset_task.document
        task_sha256 = dataset_task.sha256
        session = sessions_root / f"{index:04d}_{task_id}"
        _mkdir_durable(session)
        _atomic_bytes(
            session / "task_sha256.txt",
            (task_sha256 + "\n").encode("utf-8"),
        )

        before_ids = memory.store.list_skills() if memory else []
        before_hash = _tree_hash(state_root)
        procedural_before_hash = _tree_hash(procedural_root)
        statistical_before_hash = _tree_hash(statistical_root)
        statistical_updates_before = (
            statistical_memory.update_count if statistical_memory else 0
        )
        nudges_before = statistical_memory.nudge_count if statistical_memory else 0
        task_embedding = None
        selected = []
        context = ""
        if memory:
            task_embedding = memory.embed_task(
                document,
                lambda text: embed(text, dimension),
                identity=identity,
            )
            prepared = memory.prepare(task_embedding)
            selected = prepared.skills
            context = prepared.injection
            if selected:
                memory.materialize(prepared, session / "retrieved_skills")

        session_id = f"{config['mode']}:{index}:{task_id}"
        hooks = ToolHooks(
            statistical_memory,
            session_id,
            required=statistical_enabled,
        )
        if statistical_enabled:
            result = agent(
                document,
                context,
                task_directory,
                session,
                execution,
                hooks,
            )
        else:
            result = agent(
                document,
                context,
                task_directory,
                session,
                execution,
            )
        hooks.finalize()
        if not isinstance(result, dict):
            raise TypeError("agent callable must return a dictionary")
        transcript = result.get("transcript")
        if not isinstance(transcript, list):
            raise TypeError("agent result must contain a transcript list")
        _write_json(session / "agent_result.json", result)
        _write_json(session / "transcript.json", {"transcript": transcript})
        verified_verdict = None
        if memory:
            verified_verdict = memory.run_harness(
                task_embedding,
                harness,
                document,
                result,
                task_directory,
                session,
            )
            verdict = verified_verdict.label
        else:
            verdict = _validated_verdict(
                harness(document, result, task_directory, session)
            )
        gate_row = {
            "event": "gate",
            "index": index,
            "mode": config["mode"],
            "retrieval_scores": [item.score for item in selected],
            "retrieved_skills": [item.skill_id for item in selected],
            "skill_count_before": len(before_ids),
            "task_id": task_id,
            "task_sha256": task_sha256,
            "verdict": verdict,
        }
        _write_json(session / "gate_result.json", gate_row)
        _append_jsonl(output_root / "session_log.jsonl", gate_row)

        created = None
        if memory and verdict == "pass" and procedural_policy == "evolving":
            created = memory.learn(
                transcript,
                source_task=task_id,
                verdict=verified_verdict,
                distill=distill,
                task_embedding=task_embedding,
            )
        after_ids = memory.store.list_skills() if memory else []
        statistical_updates = (
            statistical_memory.update_count - statistical_updates_before
            if statistical_memory
            else 0
        )
        nudges = (
            statistical_memory.nudge_count - nudges_before
            if statistical_memory
            else 0
        )
        row = {
            "created_skill": created,
            "event": "finalized",
            "index": index,
            "memory_state_after_sha256": _tree_hash(state_root),
            "memory_state_before_sha256": before_hash,
            "mode": config["mode"],
            "nudges": nudges,
            "procedural_enabled": procedural_enabled,
            "procedural_policy": procedural_policy,
            "procedural_state_after_sha256": _tree_hash(procedural_root),
            "procedural_state_before_sha256": procedural_before_hash,
            "retrieval_scores": [item.score for item in selected],
            "retrieved_skills": [item.skill_id for item in selected],
            "skill_count_after": len(after_ids),
            "skill_count_before": len(before_ids),
            "statistical_enabled": statistical_enabled,
            "statistical_state_after_sha256": _tree_hash(statistical_root),
            "statistical_state_before_sha256": statistical_before_hash,
            "statistical_updates": statistical_updates,
            "task_id": task_id,
            "task_sha256": task_sha256,
            "verdict": verdict,
        }
        _append_jsonl(output_root / "session_log.jsonl", row)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a ChipMEM experiment")
    parser.add_argument("--config", required=True, help="Path to a JSON config")
    args = parser.parse_args()
    rows = run_experiment(args.config)
    summary = {
        "fail": sum(row["verdict"] == "fail" for row in rows),
        "invalid": sum(row["verdict"] == "invalid" for row in rows),
        "pass": sum(row["verdict"] == "pass" for row in rows),
        "tasks": len(rows),
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
