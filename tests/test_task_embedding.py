import hashlib

import pytest

from chipmem.embedding import TaskEmbedding
from chipmem.memory import ChipMEM
from chipmem.task_document import rtl_task_document, task_file_document


def test_task_embedding_uses_exact_task_and_is_cached(tmp_path):
    memory = ChipMEM(tmp_path / "state", "rtl", allow_empty=True)
    document = "module top; endmodule\n"
    calls = []
    identity = {"model": "test", "dimension": 2, "provider": "local"}

    def embed(text):
        calls.append(text)
        return [1.0, 0.0]

    first = memory.embed_task(document, embed, identity=identity)
    second = memory.embed_task(
        document,
        lambda _: (_ for _ in ()).throw(AssertionError("task embedded twice")),
        identity=identity,
    )

    assert first.vector == second.vector == (1.0, 0.0)
    assert first.task_sha256 == hashlib.sha256(document.encode()).hexdigest()
    assert calls == [document]


def test_private_factory_cannot_self_register(tmp_path):
    memory = ChipMEM(tmp_path / "state", "rtl", allow_empty=True)
    forged = TaskEmbedding._create(
        "e" * 64,
        {"model": "forged", "dimension": 2, "provider": "local"},
        [1.0, 0.0],
    )

    with pytest.raises(TypeError, match="authority"):
        memory.store._register_task_embedding(forged)

    with pytest.raises(TypeError, match="not minted"):
        memory.learn(
            [],
            source_task="forged",
            verdict="pass",
            distill=lambda *_: "skill",
            task_embedding=forged,
        )


def test_task_embedding_private_factory_is_not_authorized_for_learning(tmp_path):
    memory = ChipMEM(tmp_path / "state", "rtl", allow_empty=True)
    forged = TaskEmbedding._create(
        "f" * 64,
        {"model": "forged", "dimension": 2, "provider": "local"},
        [1.0, 0.0],
    )

    with pytest.raises(TypeError, match="not minted"):
        memory.learn(
            [],
            source_task="forged",
            verdict="pass",
            distill=lambda *_: "skill",
            task_embedding=forged,
        )

    assert memory.store.list_skills() == []


def test_task_embedding_is_bound_to_its_memory_store(tmp_path):
    first = ChipMEM(tmp_path / "first", "rtl", allow_empty=True)
    second = ChipMEM(tmp_path / "second", "rtl", allow_empty=True)
    handle = first.embed_task(
        "first artifact",
        lambda _: [1.0, 0.0],
        identity={"model": "test", "dimension": 2, "provider": "local"},
    )

    with pytest.raises(TypeError, match="not minted"):
        second.learn(
            [],
            source_task="cross-store",
            verdict="pass",
            distill=lambda *_: "skill",
            task_embedding=handle,
        )

    assert second.store.list_skills() == []


def test_task_embedding_rejects_subclass_bypass(tmp_path):
    with pytest.raises(TypeError, match="cannot be subclassed"):

        class ForgedEmbedding(TaskEmbedding):
            def verify(self):
                return None


def test_task_embedding_rejects_dataclass_replacement(tmp_path):
    from dataclasses import replace

    memory = ChipMEM(tmp_path / "state", "rtl", allow_empty=True)
    handle = memory.embed_task(
        "bound artifact",
        lambda _: [1.0, 0.0],
        identity={"model": "test", "dimension": 2, "provider": "local"},
    )

    with pytest.raises(TypeError, match="minted"):
        replace(handle, vector=(0.0, 1.0))


def test_task_embedding_rejects_post_creation_mutation(tmp_path):
    memory = ChipMEM(tmp_path / "state", "rtl", allow_empty=True)
    handle = memory.embed_task(
        "bound artifact",
        lambda _: [1.0, 0.0],
        identity={"model": "test", "dimension": 2, "provider": "local"},
    )
    object.__setattr__(handle, "vector", (0.0, 1.0))

    with pytest.raises(TypeError, match="integrity"):
        memory.learn(
            [],
            source_task="task",
            verdict="pass",
            distill=lambda *_: "skill",
            task_embedding=handle,
        )

    assert memory.store.list_skills() == []


def test_task_embedding_rejects_wrong_dimension(tmp_path):
    memory = ChipMEM(tmp_path / "state", "rtl", allow_empty=True)
    identity = {"model": "test", "dimension": 2, "provider": "local"}
    with pytest.raises(ValueError, match="dimension"):
        memory.embed_task("task", lambda _: [1.0], identity=identity)


def test_task_file_document_preserves_crlf_bytes(tmp_path):
    path = tmp_path / "TASK.md"
    raw = b"line one\r\nline two\r\n"
    path.write_bytes(raw)

    document = task_file_document(path)

    assert document.encode("utf-8") == raw


def test_rtl_task_document_rejects_top_path_escape(tmp_path):
    design = tmp_path / "design"
    design.mkdir()
    (tmp_path / "secret.v").write_text("PRIVATE RTL\n")

    with pytest.raises(ValueError, match="safe file stem"):
        rtl_task_document(design, "../secret")


def test_rtl_task_document_reads_only_exact_top_file(tmp_path):
    design = tmp_path / "design"
    design.mkdir()
    (design / "top.v").write_text("module top; endmodule\n")
    (design / "helper.v").write_text("module helper; endmodule\n")
    assert rtl_task_document(design, "top") == "module top; endmodule\n"


def test_task_file_document_preserves_exact_text(tmp_path):
    path = tmp_path / "TASK.md"
    path.write_text("  exact public task\n\n")
    assert task_file_document(path) == "  exact public task\n\n"
