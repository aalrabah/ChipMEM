import hashlib

import pytest

from chipmem.dataset import load_dataset


def test_dataset_loader_returns_ordered_exact_task_artifacts(tmp_path):
    root = tmp_path / "dataset"
    for task_id, content in [("task_b", "second\n"), ("task_a", "first\n")]:
        task = root / task_id
        task.mkdir(parents=True)
        (task / "TASK.md").write_text(content)

    loaded = load_dataset(root, ["task_a", "task_b"], "TASK.md")

    assert [task.task_id for task in loaded] == ["task_a", "task_b"]
    assert [task.document for task in loaded] == ["first\n", "second\n"]
    assert loaded[0].sha256 == hashlib.sha256(b"first\n").hexdigest()


def test_dataset_loader_rejects_task_path_escape(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()

    with pytest.raises(ValueError, match="simple task identifiers"):
        load_dataset(root, ["../private"], "TASK.md")


def test_dataset_loader_rejects_duplicate_task_ids(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()

    with pytest.raises(ValueError, match="duplicates"):
        load_dataset(root, ["task_a", "task_a"], "TASK.md")


def test_shipped_synthetic_dataset_exists():
    loaded = load_dataset("dataset/synthetic", ["task_a", "task_b"], "TASK.md")

    assert [task.task_id for task in loaded] == ["task_a", "task_b"]
    assert loaded[0].document.startswith("PASS")
    assert loaded[1].document.startswith("FAIL")
