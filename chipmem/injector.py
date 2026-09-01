from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from chipmem.retrieval import RetrievedSkill
from chipmem.store import _atomic_bytes, _mkdir_chain_durable

_SKILL_ID = re.compile(r"^skill_[0-9]{4,}$")


def render_skills(skills: Sequence[RetrievedSkill]) -> str:
    """Render complete selected skill files without summarization."""
    return "\n\n".join(
        f"## CHIPMEM SKILL: {item.skill_id}\n{item.text}" for item in skills
    )


def materialize_skills(
    skills: Sequence[RetrievedSkill], target: str | Path
) -> list[Path]:
    raise TypeError("use ChipMEM.materialize with PreparedMemory from prepare")


def _materialize_skills(
    skills: Sequence[RetrievedSkill], target: str | Path
) -> list[Path]:
    destination = Path(target)
    if destination.exists():
        raise FileExistsError(destination)
    _mkdir_chain_durable(destination)
    copied: list[Path] = []
    for item in skills:
        if not _SKILL_ID.fullmatch(item.skill_id):
            raise ValueError("invalid skill id for materialization")
        output = (destination / item.skill_id).resolve()
        output.relative_to(destination.resolve())
        source = Path(item.path).resolve()
        if Path(item.path).is_symlink():
            raise ValueError("stored skills must not contain symlinks")
        _mkdir_chain_durable(output)
        for source_item in sorted(
            source.rglob("*"), key=lambda path: path.relative_to(source).as_posix()
        ):
            if source_item.is_symlink():
                raise ValueError("stored skills must not contain symlinks")
            relative = source_item.relative_to(source)
            target_item = output / relative
            if source_item.is_dir():
                _mkdir_chain_durable(target_item)
            elif source_item.is_file():
                _atomic_bytes(target_item, source_item.read_bytes())
            else:
                raise ValueError(f"unsupported stored skill entry: {source_item}")
        copied.append(output)
    return copied
