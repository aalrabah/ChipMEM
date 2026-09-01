import json

import pytest

from chipmem.memory import ChipMEM
from chipmem.store import SkillStore


def test_pass_creates_one_skill_and_fail_invalid_create_none(tmp_path):
    memory = ChipMEM(tmp_path / "state", "rtl", allow_empty=True)
    store = memory.store
    task_embedding = memory.embed_task(
        "task-pass artifact",
        lambda _: [1.0, 0.0],
        identity={"model": "test", "dimension": 2, "provider": "local"},
    )
    calls = []

    def distill(transcript, verdict):
        calls.append((transcript, verdict))
        return "## DO\n- Reuse the verified transformation.\n\n## AVOID\n- Skip verification.\n"

    created = memory.learn(
        [{"tool": "synth", "output": "ok"}],
        source_task="task-pass",
        verdict=memory.run_harness(task_embedding, lambda: "pass"),
        distill=distill,
        task_embedding=task_embedding,
    )
    failed = memory.learn(
        [{"tool": "synth", "output": "failed"}],
        source_task="task-fail",
        verdict=memory.run_harness(task_embedding, lambda: "fail"),
        distill=lambda *_: (_ for _ in ()).throw(AssertionError("FAIL called distiller")),
        task_embedding=task_embedding,
    )
    invalid = memory.learn(
        [],
        source_task="task-invalid",
        verdict=memory.run_harness(task_embedding, lambda: "invalid"),
        distill=lambda *_: (_ for _ in ()).throw(AssertionError("INVALID called distiller")),
        task_embedding=task_embedding,
    )

    assert created == "skill_0001"
    assert failed is None
    assert invalid is None
    assert len(calls) == 1
    assert store.list_skills() == ["skill_0001"]
    metadata = json.loads((store.skill_dir(created) / "skill.json").read_text())
    assert metadata["source_task"] == "task-pass"
    assert metadata["verdict"] == "pass"


def test_repeated_pass_for_same_task_does_not_distill_or_mutate(tmp_path):
    memory = ChipMEM(tmp_path / "state", "rtl", allow_empty=True)
    store = memory.store
    task_embedding = memory.embed_task(
        "same task artifact",
        lambda _: [1.0, 0.0],
        identity={"model": "test", "dimension": 2, "provider": "local"},
    )
    created = memory.learn(
        [{"tool": "synth", "output": "ok"}],
        source_task="same-task",
        verdict=memory.run_harness(task_embedding, lambda: "pass"),
        distill=lambda *_: "## DO\n- Keep this exact skill.\n",
        task_embedding=task_embedding,
    )
    before = {p.name: p.read_bytes() for p in store.skill_dir(created).iterdir()}

    repeated = memory.learn(
        [{"tool": "synth", "output": "ok again"}],
        source_task="same-task",
        verdict=memory.run_harness(task_embedding, lambda: "pass"),
        distill=lambda *_: (_ for _ in ()).throw(AssertionError("duplicate PASS redistilled")),
        task_embedding=task_embedding,
    )
    after = {p.name: p.read_bytes() for p in store.skill_dir(created).iterdir()}

    assert repeated is None
    assert before == after
    assert store.list_skills() == [created]


def test_store_rejects_non_pass_skill(tmp_path):
    store = SkillStore(tmp_path / "state", "rtl")
    with pytest.raises(TypeError, match="ChipMEM.learn"):
        store.create_skill(
            "unverified", [1.0, 0.0], source_task="bad", verdict="fail"
        )
