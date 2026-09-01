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
from chipmem.store import _atomic_bytes

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
    "session_timeout_seconds",
    "step_limit",
}


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
    if payload["mode"] not in {"memory_off", "chipmem"}:
        raise ValueError("mode must be memory_off or chipmem")
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
    memory = None
    if config["mode"] == "chipmem":
        memory = ChipMEM(
            state_root,
            config["domain"],
            top_k=int(config["top_k"]),
            threshold=float(config["threshold"]),
            allow_empty=True,
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

        result = agent(document, context, task_directory, session, execution)
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
        if memory and verdict == "pass":
            created = memory.learn(
                transcript,
                source_task=task_id,
                verdict=verified_verdict,
                distill=distill,
                task_embedding=task_embedding,
            )
        after_ids = memory.store.list_skills() if memory else []
        row = {
            "created_skill": created,
            "event": "finalized",
            "index": index,
            "memory_state_after_sha256": _tree_hash(state_root),
            "memory_state_before_sha256": before_hash,
            "mode": config["mode"],
            "retrieval_scores": [item.score for item in selected],
            "retrieved_skills": [item.skill_id for item in selected],
            "skill_count_after": len(after_ids),
            "skill_count_before": len(before_ids),
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
