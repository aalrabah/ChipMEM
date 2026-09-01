import json
import threading
from pathlib import Path

import pytest

from chipmem.injector import materialize_skills
from chipmem.memory import ChipMEM, PreparedMemory
from chipmem.retrieval import RetrievedSkill
from chipmem.store import SkillStore
from chipmem.task_document import task_file_document
from experiments.run_experiment import (
    _append_jsonl,
    _mkdir_durable,
    _tree_hash,
    _validated_verdict,
    _write_json,
    run_experiment,
)


def test_durable_directory_chain_and_first_log_creation(tmp_path, monkeypatch):
    import os

    opened_directories = {}
    fsynced_directories = []
    original_open = os.open
    original_close = os.close
    original_fsync = os.fsync

    def tracking_open(path, flags, *args, **kwargs):
        descriptor = original_open(path, flags, *args, **kwargs)
        if os.path.isdir(path):
            opened_directories[descriptor] = str(path)
        return descriptor

    def tracking_fsync(descriptor):
        if descriptor in opened_directories:
            fsynced_directories.append(opened_directories[descriptor])
        original_fsync(descriptor)

    def tracking_close(descriptor):
        original_close(descriptor)
        opened_directories.pop(descriptor, None)

    monkeypatch.setattr(os, "open", tracking_open)
    monkeypatch.setattr(os, "fsync", tracking_fsync)
    monkeypatch.setattr(os, "close", tracking_close)

    output = tmp_path / "output"
    sessions = output / "sessions"
    session = sessions / "0001_task"
    _mkdir_durable(output)
    _mkdir_durable(sessions)
    _mkdir_durable(session)
    _append_jsonl(output / "session_log.jsonl", {"event": "gate"})

    expected = {str(tmp_path), str(output), str(sessions)}
    assert expected.issubset(set(fsynced_directories))


def test_tree_hash_includes_unrecognized_lock_file(tmp_path):
    empty = tmp_path / "empty"
    modified = tmp_path / "modified"
    empty.mkdir()
    modified.mkdir()
    (modified / "unexpected.lock").write_text("semantic payload")

    assert _tree_hash(empty) != _tree_hash(modified)


def test_tree_hash_includes_lock_named_symlink(tmp_path):
    empty = tmp_path / "empty"
    linked = tmp_path / "linked"
    target = tmp_path / "target"
    empty.mkdir()
    linked.mkdir()
    target.write_text("target")
    (linked / "hidden.lock").symlink_to(target)

    assert _tree_hash(empty) != _tree_hash(linked)


def test_tree_hash_frames_paths_and_contents(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a").write_bytes(b"bc")
    (second / "ab").write_bytes(b"c")

    assert _tree_hash(first) != _tree_hash(second)


def test_materialization_rejects_forged_traversal_skill(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text("private")
    forged = RetrievedSkill(
        skill_id="../escaped",
        path=source,
        score=1.0,
        text="private",
        metadata={"verdict": "pass"},
    )
    destination = tmp_path / "destination"

    with pytest.raises(TypeError, match="ChipMEM.materialize"):
        materialize_skills([forged], destination)

    assert not (tmp_path / "escaped").exists()


def test_materialization_rejects_unexpected_stored_file(tmp_path):
    memory = ChipMEM(tmp_path / "state", "rtl", threshold=0.0, allow_empty=True)
    handle = memory.embed_task(
        "legitimate task",
        lambda _: [1.0, 0.0],
        identity={"model": "test", "dimension": 2, "provider": "local"},
    )
    memory.learn(
        [],
        source_task="task",
        verdict=memory.run_harness(handle, lambda: "pass"),
        distill=lambda *_: "skill",
        task_embedding=handle,
    )
    prepared = memory.prepare(handle)
    skill_dir = memory.store.skill_dir("skill_0001")
    (skill_dir / "unexpected-private.bin").write_bytes(b"PRIVATE SENTINEL")

    with pytest.raises(ValueError, match="canonical files"):
        memory.materialize(prepared, tmp_path / "materialized")

    assert not (tmp_path / "materialized" / "skill_0001" / "unexpected-private.bin").exists()


def test_materialization_rejects_forged_source_skill(tmp_path):
    memory = ChipMEM(tmp_path / "state", "rtl", threshold=0.0, allow_empty=True)
    handle = memory.embed_task(
        "legitimate task",
        lambda _: [1.0, 0.0],
        identity={"model": "test", "dimension": 2, "provider": "local"},
    )
    memory.learn(
        [],
        source_task="legitimate",
        verdict=memory.run_harness(handle, lambda: "pass"),
        distill=lambda *_: "legitimate skill",
        task_embedding=handle,
    )
    prepared = memory.prepare(handle)
    outside = tmp_path / "outside" / "skill_9999"
    outside.mkdir(parents=True)
    (outside / "SKILL.md").write_text("PRIVATE SENTINEL")
    forged = RetrievedSkill(
        skill_id="skill_9999",
        path=outside,
        score=1.0,
        text="PRIVATE SENTINEL",
        metadata={"verdict": "pass"},
    )
    tampered = PreparedMemory(
        skills=tuple(prepared.skills) + (forged,),
        injection=prepared.injection,
    )

    with pytest.raises(ValueError, match="stored skill"):
        memory.materialize(tampered, tmp_path / "materialized")

    assert not (tmp_path / "materialized" / "skill_9999").exists()


def test_verified_verdict_rejects_label_mutation(tmp_path):
    memory = ChipMEM(tmp_path / "state", "rtl", allow_empty=True)
    handle = memory.embed_task(
        "evaluated fail task",
        lambda _: [1.0, 0.0],
        identity={"model": "test", "dimension": 2, "provider": "local"},
    )
    verdict = memory.run_harness(handle, lambda: "fail")
    object.__setattr__(verdict, "label", "pass")

    with pytest.raises(TypeError, match="integrity"):
        memory.learn(
            [],
            source_task="task",
            verdict=verdict,
            distill=lambda *_: "skill",
            task_embedding=handle,
        )

    assert memory.store.list_skills() == []


def test_verified_verdict_is_task_bound_and_single_use(tmp_path):
    memory = ChipMEM(tmp_path / "state", "rtl", allow_empty=True)
    first_handle = memory.embed_task(
        "first task",
        lambda _: [1.0, 0.0],
        identity={"model": "test", "dimension": 2, "provider": "local"},
    )
    second_handle = memory.embed_task(
        "second task",
        lambda _: [0.0, 1.0],
        identity={"model": "test", "dimension": 2, "provider": "local"},
    )
    verdict = memory.run_harness(first_handle, lambda: "pass")

    memory.learn(
        [],
        source_task="first",
        verdict=verdict,
        distill=lambda *_: "first skill",
        task_embedding=first_handle,
    )

    with pytest.raises(TypeError, match="consumed|task"):
        memory.learn(
            [],
            source_task="second",
            verdict=verdict,
            distill=lambda *_: "replayed skill",
            task_embedding=second_handle,
        )

    assert memory.store.list_skills() == ["skill_0001"]


def test_verified_verdict_is_bound_to_exact_embedding_object(tmp_path):
    memory = ChipMEM(tmp_path / "state", "rtl", allow_empty=True)
    first = memory.embed_task(
        "same artifact",
        lambda _: [1.0, 0.0],
        identity={"model": "test", "dimension": 2, "provider": "local"},
    )
    second = memory.embed_task(
        "same artifact",
        lambda _: [1.0, 0.0],
        identity={"model": "test", "dimension": 2, "provider": "local"},
    )
    verdict = memory.run_harness(first, lambda: "pass")

    with pytest.raises(TypeError, match="different task embedding"):
        memory.learn(
            [],
            source_task="task",
            verdict=verdict,
            distill=lambda *_: "skill",
            task_embedding=second,
        )

    assert memory.store.list_skills() == []


def test_public_learn_rejects_plain_pass_string(tmp_path):
    memory = ChipMEM(tmp_path / "state", "rtl", allow_empty=True)
    handle = memory.embed_task(
        "never evaluated",
        lambda _: [1.0, 0.0],
        identity={"model": "test", "dimension": 2, "provider": "local"},
    )

    with pytest.raises(TypeError, match="run_harness"):
        memory.learn(
            [],
            source_task="never-evaluated",
            verdict="pass",
            distill=lambda *_: "UNGATED SKILL",
            task_embedding=handle,
        )

    assert memory.store.list_skills() == []


def test_public_store_cannot_create_claimed_pass_skill(tmp_path):
    store = SkillStore(tmp_path / "state", "rtl")

    with pytest.raises(TypeError, match="ChipMEM.learn"):
        store.create_skill(
            "ungated",
            [1.0, 0.0],
            source_task="never-evaluated",
            verdict="pass",
        )

    assert store.list_skills() == []


def test_harness_verdict_must_be_an_actual_string():
    class PassLike:
        def __str__(self):
            return "pass"

    with pytest.raises(TypeError, match="string"):
        _validated_verdict(PassLike())


def test_learning_requires_task_embedding_handle(tmp_path):
    memory = ChipMEM(tmp_path / "state", "rtl", allow_empty=True)

    with pytest.raises(TypeError, match="TaskEmbedding"):
        memory.learn(
            [],
            source_task="task",
            verdict="pass",
            distill=lambda *_: "skill",
            task_embedding=[0.0, 1.0],
        )

    assert memory.store.list_skills() == []


def test_distinct_invalid_utf8_tasks_are_rejected_not_collapsed(tmp_path):
    first = tmp_path / "first.task"
    second = tmp_path / "second.task"
    first.write_bytes(b"\xff")
    second.write_bytes(b"\xfe")

    with pytest.raises(UnicodeDecodeError):
        task_file_document(first)
    with pytest.raises(UnicodeDecodeError):
        task_file_document(second)


def test_json_writer_rejects_non_json_values_without_temp_file(tmp_path):
    target = tmp_path / "agent_result.json"

    with pytest.raises(TypeError, match="JSON"):
        _write_json(target, {"artifact": Path("artifact.bin")})

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_json_writer_fsyncs_parent_directory(tmp_path, monkeypatch):
    import os
    import stat

    observed = []
    original_fsync = os.fsync

    def tracking_fsync(descriptor):
        mode = os.fstat(descriptor).st_mode
        observed.append("directory" if stat.S_ISDIR(mode) else "file")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", tracking_fsync)
    _write_json(tmp_path / "gate_result.json", {"verdict": "pass"})

    assert "file" in observed
    assert "directory" in observed


def test_verified_gate_is_durable_before_distillation(tmp_path, monkeypatch):
    plugin = tmp_path / "failing_plugin.py"
    plugin.write_text(
        "from experiments.example_plugins import embed, agent, harness\n"
        "def distill(transcript, verdict):\n"
        "    raise RuntimeError('distillation failed')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    tasks = tmp_path / "tasks"
    task = tasks / "task_a"
    task.mkdir(parents=True)
    (task / "TASK.md").write_text("PASS durable gate")
    config = {
        "mode": "chipmem",
        "domain": "example",
        "task_directory": str(tasks),
        "task_order": ["task_a"],
        "task_artifact": "TASK.md",
        "state_directory": str(tmp_path / "state"),
        "output_directory": str(tmp_path / "output"),
        "top_k": 2,
        "threshold": 0.0,
        "embedding_dimension": 8,
        "embedding_callable": "failing_plugin:embed",
        "agent_callable": "failing_plugin:agent",
        "harness_callable": "failing_plugin:harness",
        "distill_callable": "failing_plugin:distill",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    with pytest.raises(RuntimeError, match="distillation failed"):
        run_experiment(path)

    events = [
        json.loads(line)
        for line in (tmp_path / "output" / "session_log.jsonl").read_text().splitlines()
    ]
    assert events == [
        {
            "event": "gate",
            "index": 1,
            "mode": "chipmem",
            "retrieval_scores": [],
            "retrieved_skills": [],
            "skill_count_before": 0,
            "task_id": "task_a",
            "task_sha256": events[0]["task_sha256"],
            "verdict": "pass",
        }
    ]
    assert json.loads(
        (tmp_path / "output" / "sessions" / "0001_task_a" / "gate_result.json").read_text()
    )["verdict"] == "pass"


def test_identical_task_artifact_under_different_ids_learns_once(tmp_path):
    memory = ChipMEM(tmp_path / "state", "rtl", allow_empty=True)
    handle = memory.embed_task(
        "identical artifact",
        lambda _: [1.0, 0.0],
        identity={"model": "test", "dimension": 2, "provider": "local"},
    )
    calls = []

    first = memory.learn(
        [],
        source_task="task-a",
        verdict=memory.run_harness(handle, lambda: "pass"),
        distill=lambda *_: calls.append("first") or "skill",
        task_embedding=handle,
    )
    second = memory.learn(
        [],
        source_task="task-alias",
        verdict=memory.run_harness(handle, lambda: "pass"),
        distill=lambda *_: calls.append("second") or "duplicate",
        task_embedding=handle,
    )

    assert first == "skill_0001"
    assert second is None
    assert calls == ["first"]
    assert memory.store.list_skills() == ["skill_0001"]


def test_concurrent_same_task_distills_once(tmp_path):
    memory = ChipMEM(tmp_path / "state", "rtl", allow_empty=True)
    handle = memory.embed_task(
        "same task",
        lambda _: [1.0, 0.0],
        identity={"model": "test", "dimension": 2, "provider": "local"},
    )
    barrier = threading.Barrier(2)
    calls = []
    results = []

    def distill(*_):
        calls.append(1)
        return "complete skill"

    def worker():
        barrier.wait()
        results.append(
            memory.learn(
                [],
                source_task="same-task",
                verdict=memory.run_harness(handle, lambda: "pass"),
                distill=distill,
                task_embedding=handle,
            )
        )

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(calls) == 1
    assert sorted(result is None for result in results) == [False, True]
    assert memory.store.list_skills() == ["skill_0001"]
