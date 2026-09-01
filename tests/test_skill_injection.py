import pytest

from chipmem.injector import render_skills
from chipmem.memory import ChipMEM
from chipmem.retrieval import retrieve_top_k


def seed(memory, text):
    handle = memory.embed_task(
        f"artifact-{len(memory.store.list_skills())}",
        lambda _: [1.0, 0.0],
        identity={"model": "test", "dimension": 2, "provider": "local"},
    )
    created = memory.learn(
        [],
        source_task=f"task-{len(memory.store.list_skills())}",
        verdict=memory.run_harness(handle, lambda: "pass"),
        distill=lambda *_: text,
        task_embedding=handle,
    )
    assert created is not None
    return handle


def test_injection_preserves_complete_skill_text(tmp_path):
    memory = ChipMEM(tmp_path / "state", "rtl", allow_empty=True)
    exact = "# Skill\n\n## DO\n- Preserve every byte.\n\n## AVOID\n- Summarization.\n"
    handle = seed(memory, exact)
    prepared = memory.prepare(handle)

    rendered = prepared.injection
    target = tmp_path / "materialized"
    copied = memory.materialize(prepared, target)

    assert exact in rendered
    assert len(copied) == 1
    assert (target / "skill_0001" / "SKILL.md").read_text() == exact


def test_injection_preserves_crlf_skill_payload_exactly(tmp_path):
    memory = ChipMEM(tmp_path / "state", "rtl", allow_empty=True)
    store = memory.store
    exact = "# Skill\r\n\r\nline 1\r\nline 2  \r\n"
    seed(memory, exact)

    selected = retrieve_top_k(store, [1.0, 0.0], top_k=2, threshold=0)
    rendered = render_skills(selected)

    assert exact in rendered
    assert selected[0].text.encode("utf-8") == exact.encode("utf-8")


def test_materialization_never_overwrites_existing_target(tmp_path):
    memory = ChipMEM(tmp_path / "state", "rtl", allow_empty=True)
    handle = seed(memory, "skill")
    prepared = memory.prepare(handle)
    target = tmp_path / "materialized"
    memory.materialize(prepared, target)
    with pytest.raises(FileExistsError):
        memory.materialize(prepared, target)
