from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from chipmem.store import SkillStore


@dataclass(frozen=True)
class RetrievedSkill:
    skill_id: str
    path: Path
    score: float
    text: str
    metadata: dict


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        raise ValueError("embeddings must be non-empty")
    if len(a) != len(b):
        raise ValueError("embedding dimension mismatch")
    left = [float(value) for value in a]
    right = [float(value) for value in b]
    if any(not math.isfinite(value) for value in left + right):
        raise ValueError("embeddings must contain only finite values")
    dot = sum(x * y for x, y in zip(left, right))
    left_norm = math.sqrt(sum(x * x for x in left))
    right_norm = math.sqrt(sum(y * y for y in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def retrieve_top_k(
    store: SkillStore,
    query_vector: list[float],
    *,
    top_k: int = 2,
    threshold: float = 0.6,
) -> list[RetrievedSkill]:
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not math.isfinite(float(threshold)) or not -1.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be between -1 and 1")

    scored: list[RetrievedSkill] = []
    for skill_id in store.list_skills():
        metadata = store.read_metadata(skill_id)
        if metadata.get("verdict") != "pass":
            raise ValueError(f"skill {skill_id} lacks verified PASS provenance")
        score = cosine_similarity(query_vector, store.read_embedding(skill_id))
        if score >= threshold:
            scored.append(
                RetrievedSkill(
                    skill_id=skill_id,
                    path=store.skill_dir(skill_id),
                    score=score,
                    text=store.read_text(skill_id),
                    metadata=metadata,
                )
            )
    scored.sort(key=lambda item: (-item.score, item.skill_id))
    return scored[:top_k]
