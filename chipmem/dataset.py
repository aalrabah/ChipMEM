from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from chipmem.task_document import task_file_document

_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True, slots=True)
class DatasetTask:
    task_id: str
    directory: Path
    artifact: Path
    document: str
    sha256: str


def load_dataset(
    dataset_root: str | Path,
    task_order: list[str],
    task_artifact: str,
) -> list[DatasetTask]:
    """Load ordered exact task artifacts from a benchmark dataset directory."""
    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {root}")
    if not isinstance(task_order, list) or any(
        not isinstance(task_id, str) or not _TASK_ID.fullmatch(task_id)
        for task_id in task_order
    ):
        raise ValueError("task_order must contain simple task identifiers")
    if len(task_order) != len(set(task_order)):
        raise ValueError("task_order contains duplicates")
    artifact_relative = Path(task_artifact)
    if (
        not isinstance(task_artifact, str)
        or not task_artifact
        or artifact_relative.is_absolute()
        or ".." in artifact_relative.parts
    ):
        raise ValueError("task_artifact must be a safe relative path")

    tasks: list[DatasetTask] = []
    for task_id in task_order:
        directory = (root / task_id).resolve()
        directory.relative_to(root)
        artifact = (directory / artifact_relative).resolve()
        artifact.relative_to(directory)
        document = task_file_document(artifact)
        tasks.append(
            DatasetTask(
                task_id=task_id,
                directory=directory,
                artifact=artifact,
                document=document,
                sha256=hashlib.sha256(document.encode("utf-8")).hexdigest(),
            )
        )
    return tasks
