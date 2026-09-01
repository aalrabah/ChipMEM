from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from chipmem.embedding import TaskEmbedding

_DOMAIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SKILL_ID = re.compile(r"^skill_[0-9]{4,}$")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_chain_durable(path: Path) -> None:
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        if current.parent == current:
            raise FileNotFoundError(f"no existing ancestor for {path}")
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir()
        _fsync_directory(directory.parent)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _atomic_json(path: Path, payload: dict) -> None:
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _atomic_bytes(path, content)


def _validated_vector(vector: list[float]) -> list[float]:
    if not isinstance(vector, list) or not vector:
        raise ValueError("skill embedding must be a non-empty list")
    values = [float(value) for value in vector]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("skill embedding must contain only finite values")
    return values


class SkillStore:
    """Flat domain-private store of immutable verified procedural skills."""

    def __init__(
        self,
        state_root: str | Path,
        domain: str,
        *,
        _mint_authority: object | None = None,
    ):
        if not isinstance(domain, str) or not _DOMAIN.fullmatch(domain):
            raise ValueError("domain must be a simple non-empty name")
        self.state_root = Path(state_root).resolve()
        self.domain = domain
        self.memory_dir = self.state_root / "agents" / domain / "memory"
        _mkdir_chain_durable(self.memory_dir)
        self.lock_path = self.memory_dir / ".store.lock"
        self.__minted_task_embeddings: dict[int, TaskEmbedding] = {}
        self.__mint_authority = (
            _mint_authority if _mint_authority is not None else object()
        )

    def _register_task_embedding(
        self,
        task_embedding: TaskEmbedding,
        *,
        authority: object | None = None,
    ) -> None:
        if authority is not self.__mint_authority:
            raise TypeError("invalid task-embedding mint authority")
        if type(task_embedding) is not TaskEmbedding:
            raise TypeError("task embedding must have the exact TaskEmbedding type")
        task_embedding.verify()
        self.__minted_task_embeddings[id(task_embedding)] = task_embedding

    def _require_minted_task_embedding(
        self, task_embedding: TaskEmbedding
    ) -> None:
        if type(task_embedding) is not TaskEmbedding:
            raise TypeError("task embedding must have the exact TaskEmbedding type")
        registered = self.__minted_task_embeddings.get(id(task_embedding))
        if registered is not task_embedding:
            raise TypeError("task embedding was not minted for this ChipMEM store")
        task_embedding.verify()

    def _require_learning_authority(self, authority: object | None) -> None:
        if authority is not self.__mint_authority:
            raise TypeError("invalid verified-learning authority")

    def list_skills(self) -> list[str]:
        return [
            path.name
            for path in sorted(self.memory_dir.glob("skill_*"))
            if path.is_dir()
            and (path / "SKILL.md").is_file()
            and (path / "skill.json").is_file()
            and (path / "embeddings.json").is_file()
        ]

    def _all_skill_ids(self) -> list[str]:
        return [
            path.name
            for path in sorted(self.memory_dir.glob("skill_*"))
            if path.is_dir() and _SKILL_ID.fullmatch(path.name)
        ]

    def skill_dir(self, skill_id: str) -> Path:
        if not isinstance(skill_id, str) or not _SKILL_ID.fullmatch(skill_id):
            raise ValueError("invalid skill id")
        path = (self.memory_dir / skill_id).resolve()
        path.relative_to(self.memory_dir)
        return path

    def read_text(self, skill_id: str) -> str:
        return (self.skill_dir(skill_id) / "SKILL.md").read_bytes().decode("utf-8")

    def read_metadata(self, skill_id: str) -> dict:
        return json.loads(
            (self.skill_dir(skill_id) / "skill.json").read_text(encoding="utf-8")
        )

    def read_embedding(self, skill_id: str) -> list[float]:
        payload = json.loads(
            (self.skill_dir(skill_id) / "embeddings.json").read_text(
                encoding="utf-8"
            )
        )
        return _validated_vector(payload.get("vector"))

    def _next_skill_id_locked(self) -> str:
        numbers = [int(name.split("_", 1)[1]) for name in self._all_skill_ids()]
        return f"skill_{max(numbers, default=0) + 1:04d}"

    def create_skill(
        self,
        text: str,
        vector: list[float],
        *,
        source_task: str,
        source_task_sha256: str | None = None,
        verdict: str,
        skill_id: str | None = None,
    ) -> str:
        raise TypeError("skills can only be created through verified ChipMEM.learn")

    def _create_verified_skill(
        self,
        text: str,
        vector: list[float],
        *,
        source_task: str,
        source_task_sha256: str,
        verdict: str,
        authority: object | None,
        skill_id: str | None = None,
    ) -> str:
        self._require_learning_authority(authority)
        if verdict != "pass":
            raise ValueError("procedural skills require a verified PASS")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("skill text must be non-empty")
        if not isinstance(source_task, str) or not source_task.strip():
            raise ValueError("source_task must be non-empty")
        source_hash = source_task_sha256 or hashlib.sha256(
            source_task.encode("utf-8")
        ).hexdigest()
        if len(source_hash) != 64 or any(
            character not in "0123456789abcdef" for character in source_hash
        ):
            raise ValueError("source_task_sha256 must be a lowercase SHA-256")
        values = _validated_vector(vector)

        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            for existing_id in self.list_skills():
                metadata = self.read_metadata(existing_id)
                if (
                    metadata.get("verdict") == "pass"
                    and metadata.get("source_task_sha256") == source_hash
                ):
                    raise FileExistsError(
                        f"verified skill already exists for source task: {source_task}"
                    )
            sid = skill_id or self._next_skill_id_locked()
            directory = self.skill_dir(sid)
            _mkdir_chain_durable(directory)
            _atomic_bytes(directory / "SKILL.md", text.encode("utf-8"))
            _atomic_json(
                directory / "skill.json",
                {
                    "agent_type": self.domain,
                    "created": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "source_task": source_task,
                    "source_task_sha256": source_hash,
                    "verdict": verdict,
                },
            )
            _atomic_json(directory / "embeddings.json", {"vector": values})
            return sid
