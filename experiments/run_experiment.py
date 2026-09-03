from __future__ import annotations

import argparse
import hashlib
import importlib
import json
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


def _copy_statistical_seed(
    seed_root: Path,
    destination_root: Path,
    domain: str,
) -> None:
    source = seed_root / "agents" / domain / "memory"
    destination = destination_root / "agents" / domain / "memory"
    if not source.is_dir():
        raise FileNotFoundError(f"statistical seed is missing: {source}")
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
) -> None:
    source = seed_root / "agents" / domain / "memory"
    destination = destination_root / "agents" / domain / "memory"
    if not source.is_dir():
        raise FileNotFoundError(f"procedural seed is missing: {source}")
    if destination.exists():
        raise FileExistsError(f"procedural destination already exists: {destination}")
    skill_dirs = [
        path
        for path in sorted(source.iterdir())
        if path.is_dir() and re.fullmatch(r"skill_[0-9]{4,}", path.name)
    ]
    if not skill_dirs:
        raise ValueError("procedural seed contains no verified skills")
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


def _validated_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted(_REQUIRED - set(payload))
    unknown = sorted(set(payload) - _REQUIRED - _OPTIONAL)
    if missing:
        raise ValueError(f"missing config fields: {missing}")
    if unknown:
        raise ValueError(f"unknown config fields: {unknown}")
    if payload["mode"] not in MODE_CONTRACT:
        raise ValueError(
            "mode must be memory_off, procedural_only, statistical_only, or chipmem"
        )
    mode_contract = MODE_CONTRACT[payload["mode"]]
    default_policy = "evolving" if mode_contract["procedural"] else "disabled"
    procedural_policy = payload.get("procedural_policy", default_policy)
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
    for field, default in (
        ("step_limit", 32),
        ("session_timeout_seconds", 600),
    ):
        value = payload.get(field, default)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    max_nudges = payload.get("max_nudges")
    if max_nudges is not None and (
        not isinstance(max_nudges, int) or max_nudges < 0
    ):
        raise ValueError("max_nudges must be null or a non-negative integer")
    threshold = payload.get("nudge_threshold", 0.20)
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("nudge_threshold must be between 0 and 1")
    return payload


def run_experiment(config_path: str | Path) -> list[dict]:
    config_path = Path(config_path).resolve()
    config = _validated_config(config_path)
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
    dataset = load_dataset(
        task_root,
        config["task_order"],
        config["task_artifact"],
    )
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
            _copy_procedural_seed(seed_root, procedural_root, config["domain"])
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
