import pytest

from chipmem.memory import ChipMEM
from chipmem.retrieval import cosine_similarity, retrieve_top_k


def seed(memory, skill_id, text, vector, source):
    handle = memory.embed_task(
        f"artifact-{source}",
        lambda _: vector,
        identity={
            "model": f"test-{source}",
            "dimension": len(vector),
            "provider": "local",
        },
    )
    created = memory.learn(
        [],
        source_task=source,
        verdict=memory.run_harness(handle, lambda: "pass"),
        distill=lambda *_: text,
        task_embedding=handle,
    )
    assert created == skill_id
    return created


def test_retrieval_returns_top_k_above_threshold(tmp_path):
    memory = ChipMEM(tmp_path / "state", "rtl", allow_empty=True)
    store = memory.store
    seed(memory, "skill_0001", "best", [1.0, 0.0], "a")
    seed(memory, "skill_0002", "second", [0.8, 0.2], "b")
    seed(memory, "skill_0003", "irrelevant", [0.0, 1.0], "c")

    found = retrieve_top_k(store, [1.0, 0.0], top_k=2, threshold=0.5)

    assert [item.skill_id for item in found] == ["skill_0001", "skill_0002"]
    assert all(item.score >= 0.5 for item in found)


def test_retrieval_is_domain_private(tmp_path):
    rtl_memory = ChipMEM(tmp_path / "state", "rtl", allow_empty=True)
    sim_memory = ChipMEM(tmp_path / "state", "simulation", allow_empty=True)
    seed(rtl_memory, "skill_0001", "rtl", [1.0, 0.0], "a")
    seed(sim_memory, "skill_0001", "simulation", [1.0, 0.0], "b")
    assert [
        x.text
        for x in retrieve_top_k(
            rtl_memory.store, [1.0, 0.0], top_k=2, threshold=0
        )
    ] == ["rtl"]


def test_cosine_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="dimension"):
        cosine_similarity([1.0], [1.0, 0.0])
