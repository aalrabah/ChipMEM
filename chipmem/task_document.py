from __future__ import annotations

from pathlib import Path


def rtl_task_document(design_directory: str | Path, top: str) -> str:
    """Return the exact top-level RTL source used as the retrieval document."""
    if (
        not isinstance(top, str)
        or not top
        or top in {".", ".."}
        or "/" in top
        or "\\" in top
    ):
        raise ValueError("top must be a safe file stem")
    root = Path(design_directory).resolve()
    candidates = [(root / f"{top}.v").resolve(), (root / f"{top}.sv").resolve()]
    for candidate in candidates:
        candidate.relative_to(root)
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"exact top RTL is missing or ambiguous for {top}: {root}"
        )
    return existing[0].read_bytes().decode("utf-8")


def task_file_document(path: str | Path) -> str:
    """Return exact public task-file text without prompt or tool additions."""
    task = Path(path)
    if not task.is_file():
        raise FileNotFoundError(f"benchmark task file missing: {task}")
    return task.read_bytes().decode("utf-8")
