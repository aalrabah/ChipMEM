import hashlib

from chipmem.memory import ChipMEM


def tree_hash(path):
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(item.relative_to(path).as_posix().encode())
            digest.update(item.read_bytes())
    return digest.hexdigest()


def test_learned_skill_persistence_fsyncs_state_chain(tmp_path, monkeypatch):
    import os
    import stat

    opened_directories = {}
    fsynced_directories = []
    regular_file_fsyncs = 0
    original_open = os.open
    original_close = os.close
    original_fsync = os.fsync

    def tracking_open(path, flags, *args, **kwargs):
        descriptor = original_open(path, flags, *args, **kwargs)
        if os.path.isdir(path):
            opened_directories[descriptor] = str(path)
        return descriptor

    def tracking_fsync(descriptor):
        nonlocal regular_file_fsyncs
        mode = os.fstat(descriptor).st_mode
        if stat.S_ISDIR(mode):
            fsynced_directories.append(opened_directories.get(descriptor, "unknown"))
        elif stat.S_ISREG(mode):
            regular_file_fsyncs += 1
        original_fsync(descriptor)

    def tracking_close(descriptor):
        original_close(descriptor)
        opened_directories.pop(descriptor, None)

    monkeypatch.setattr(os, "open", tracking_open)
    monkeypatch.setattr(os, "fsync", tracking_fsync)
    monkeypatch.setattr(os, "close", tracking_close)

    state = tmp_path / "state"
    memory = ChipMEM(state, "rtl", allow_empty=True)
    handle = memory.embed_task(
        "durable task",
        lambda _: [1.0, 0.0],
        identity={"model": "test", "dimension": 2, "provider": "local"},
    )
    created = memory.learn(
        [],
        source_task="task",
        verdict=memory.run_harness(handle, lambda: "pass"),
        distill=lambda *_: "durable skill",
        task_embedding=handle,
    )

    skill = state / "agents" / "rtl" / "memory" / created
    required_parents = {
        str(tmp_path),
        str(state),
        str(state / "agents"),
        str(state / "agents" / "rtl"),
        str(state / "agents" / "rtl" / "memory"),
        str(skill),
    }
    assert required_parents.issubset(set(fsynced_directories))
    assert regular_file_fsyncs >= 3


def test_incomplete_skill_directory_does_not_block_next_skill(tmp_path):
    memory = ChipMEM(tmp_path / "state", "rtl", allow_empty=True)
    store = memory.store
    incomplete = store.skill_dir("skill_0001")
    incomplete.mkdir()
    (incomplete / "SKILL.md").write_text("partial evidence")

    handle = memory.embed_task(
        "task artifact",
        lambda _: [1.0, 0.0],
        identity={"model": "test", "dimension": 2, "provider": "local"},
    )
    created = memory.learn(
        [],
        source_task="task",
        verdict=memory.run_harness(handle, lambda: "pass"),
        distill=lambda *_: "complete",
        task_embedding=handle,
    )

    assert created == "skill_0002"
    assert incomplete.is_dir()
    assert store.list_skills() == ["skill_0002"]


def test_duplicate_source_task_is_rejected_by_store(tmp_path):
    memory = ChipMEM(tmp_path / "state", "rtl", allow_empty=True)
    store = memory.store
    handle = memory.embed_task(
        "task artifact",
        lambda _: [1.0, 0.0],
        identity={"model": "test", "dimension": 2, "provider": "local"},
    )
    first = memory.learn(
        [],
        source_task="task",
        verdict=memory.run_harness(handle, lambda: "pass"),
        distill=lambda *_: "first",
        task_embedding=handle,
    )
    second = memory.learn(
        [],
        source_task="task-alias",
        verdict=memory.run_harness(handle, lambda: "pass"),
        distill=lambda *_: "second",
        task_embedding=handle,
    )

    assert first == "skill_0001"
    assert second is None
    assert store.list_skills() == ["skill_0001"]


def test_skill_files_are_not_overwritten(tmp_path):
    memory = ChipMEM(tmp_path / "state", "rtl", allow_empty=True)
    store = memory.store
    handle = memory.embed_task(
        "task artifact",
        lambda _: [1.0, 0.0],
        identity={"model": "test", "dimension": 2, "provider": "local"},
    )
    skill_id = memory.learn(
        [],
        source_task="task",
        verdict=memory.run_harness(handle, lambda: "pass"),
        distill=lambda *_: "original",
        task_embedding=handle,
    )
    before = tree_hash(store.skill_dir(skill_id))

    repeated = memory.learn(
        [],
        source_task="other",
        verdict=memory.run_harness(handle, lambda: "pass"),
        distill=lambda *_: "replacement",
        task_embedding=handle,
    )

    assert repeated is None
    assert tree_hash(store.skill_dir(skill_id)) == before
