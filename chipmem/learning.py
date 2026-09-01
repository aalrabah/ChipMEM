from __future__ import annotations

import fcntl
from collections.abc import Callable

from chipmem.embedding import TaskEmbedding
from chipmem.store import SkillStore


def learn_from_verified_run(
    store: SkillStore,
    transcript: list[dict],
    *,
    source_task: str,
    verdict: str,
    distill: Callable[[list[dict], str], str],
    task_embedding: TaskEmbedding,
    authority: object | None = None,
) -> str | None:
    """Append at most one immutable skill after a deterministic PASS."""
    if verdict not in {"pass", "fail", "invalid"}:
        raise ValueError("verdict must be pass, fail, or invalid")
    if verdict != "pass":
        return None
    store._require_learning_authority(authority)
    store._require_minted_task_embedding(task_embedding)

    source_lock = store.memory_dir / f".source-{task_embedding.task_sha256}.lock"
    with source_lock.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for skill_id in store.list_skills():
            metadata = store.read_metadata(skill_id)
            if (
                metadata.get("verdict") == "pass"
                and metadata.get("source_task_sha256")
                == task_embedding.task_sha256
            ):
                return None

        text = distill(transcript, verdict)
        if not isinstance(text, str) or not text.strip():
            return None
        try:
            return store._create_verified_skill(
                text,
                list(task_embedding.vector),
                source_task=source_task,
                source_task_sha256=task_embedding.task_sha256,
                verdict=verdict,
                authority=authority,
            )
        except FileExistsError:
            for skill_id in store.list_skills():
                metadata = store.read_metadata(skill_id)
                if (
                    metadata.get("verdict") == "pass"
                    and metadata.get("source_task_sha256")
                    == task_embedding.task_sha256
                ):
                    return None
            raise
