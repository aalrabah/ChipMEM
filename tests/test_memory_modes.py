import json
from pathlib import Path

from chipmem.memory import ChipMEM
from chipmem.statistical.features import OpenFlowFeatures
from chipmem.statistical.online import StatisticalMemory
from experiments.run_experiment import run_experiment


def make_config(tmp_path, mode):
    tasks = tmp_path / "tasks"
    task = tasks / "task_a"
    task.mkdir(parents=True, exist_ok=True)
    (task / "TASK.md").write_text("PASS mode task")
    return {
        "mode": mode,
        "domain": "example",
        "task_directory": str(tasks),
        "task_order": ["task_a"],
        "task_artifact": "TASK.md",
        "state_directory": str(tmp_path / f"state-{mode}"),
        "output_directory": str(tmp_path / f"output-{mode}"),
        "top_k": 2,
        "threshold": 0.0,
        "embedding_dimension": 8,
        "agent_endpoint_env": None,
        "step_limit": 32,
        "session_timeout_seconds": 600,
        "nudge_threshold": 0.20,
        "max_nudges": None,
        "embedding_callable": "experiments.example_plugins:embed",
        "agent_callable": "experiments.example_plugins:agent",
        "harness_callable": "experiments.example_plugins:harness",
        "distill_callable": "experiments.example_plugins:distill",
        "features_callable": "chipmem.statistical.features:OpenFlowFeatures",
    }


def run_mode(tmp_path, mode):
    config = make_config(tmp_path, mode)
    path = tmp_path / f"{mode}.json"
    path.write_text(json.dumps(config))
    return run_experiment(path)[0], config


def test_nonstatistical_modes_preserve_five_argument_agent_contract(tmp_path):
    for mode in ("memory_off", "procedural_only"):
        config = make_config(tmp_path, mode)
        config["agent_callable"] = "experiments.example_plugins:legacy_agent"
        path = tmp_path / f"legacy-{mode}.json"
        path.write_text(json.dumps(config))

        row = run_experiment(path)[0]

        assert row["verdict"] == "pass"
        assert row["statistical_updates"] == 0


def test_four_mode_component_isolation(tmp_path):
    rows = {}
    configs = {}
    for mode in (
        "memory_off",
        "procedural_only",
        "statistical_only",
        "chipmem",
    ):
        rows[mode], configs[mode] = run_mode(tmp_path, mode)

    assert rows["memory_off"]["procedural_enabled"] is False
    assert rows["memory_off"]["statistical_enabled"] is False
    assert rows["memory_off"]["created_skill"] is None
    assert rows["memory_off"]["statistical_updates"] == 0

    assert rows["procedural_only"]["procedural_enabled"] is True
    assert rows["procedural_only"]["statistical_enabled"] is False
    assert rows["procedural_only"]["created_skill"] == "skill_0001"
    assert rows["procedural_only"]["statistical_updates"] == 0

    assert rows["statistical_only"]["procedural_enabled"] is False
    assert rows["statistical_only"]["statistical_enabled"] is True
    assert rows["statistical_only"]["created_skill"] is None
    assert rows["statistical_only"]["statistical_updates"] == 3

    assert rows["chipmem"]["procedural_enabled"] is True
    assert rows["chipmem"]["statistical_enabled"] is True
    assert rows["chipmem"]["created_skill"] == "skill_0001"
    assert rows["chipmem"]["statistical_updates"] == 3

    assert len({config["state_directory"] for config in configs.values()}) == 4


def test_full_mode_records_both_state_hashes(tmp_path):
    row, _ = run_mode(tmp_path, "chipmem")

    assert len(row["procedural_state_before_sha256"]) == 64
    assert len(row["procedural_state_after_sha256"]) == 64
    assert len(row["statistical_state_before_sha256"]) == 64
    assert len(row["statistical_state_after_sha256"]) == 64
    assert row["statistical_state_before_sha256"] != row[
        "statistical_state_after_sha256"
    ]


def test_full_mode_carries_both_memories_to_next_task(tmp_path):
    config = make_config(tmp_path, "chipmem")
    second = Path(config["task_directory"]) / "task_b"
    second.mkdir()
    (second / "TASK.md").write_text("FAIL second task")
    config["task_order"] = ["task_a", "task_b"]
    path = tmp_path / "combined.json"
    path.write_text(json.dumps(config))

    first, second_row = run_experiment(path)

    assert first["created_skill"] == "skill_0001"
    assert second_row["retrieved_skills"] == ["skill_0001"]
    assert first["statistical_updates"] == 3
    assert second_row["statistical_updates"] == 3
    assert second_row["procedural_state_before_sha256"] == first[
        "procedural_state_after_sha256"
    ]
    assert second_row["statistical_state_before_sha256"] == first[
        "statistical_state_after_sha256"
    ]


def test_loaded_statistical_state_is_copied_not_modified(tmp_path):
    seed = tmp_path / "seed"
    source = StatisticalMemory(seed, "example", OpenFlowFeatures())
    source.before_tool_call("seed-session", "yosys", {"script": "synth"})
    source.after_tool_call(
        "seed-session",
        "yosys",
        {"script": "synth"},
        "command failed with return code 1",
    )
    source_bytes = {
        path.name: path.read_bytes()
        for path in source.memory_dir.iterdir()
        if path.name in {"model.json", "nudge_experience.jsonl", "recovery_experience.jsonl"}
    }
    config = make_config(tmp_path, "statistical_only")
    config["statistical_seed_directory"] = str(seed)
    path = tmp_path / "loaded.json"
    path.write_text(json.dumps(config))

    row = run_experiment(path)[0]

    assert row["statistical_updates"] == 3
    assert {
        file.name: file.read_bytes()
        for file in source.memory_dir.iterdir()
        if file.name in source_bytes
    } == source_bytes
    destination = (
        Path(config["state_directory"])
        / "statistical"
        / "agents"
        / "example"
        / "memory"
    )
    persisted = json.loads((destination / "model.json").read_text())
    assert persisted["metadata"]["training_rows"] == 4


def test_historical_combined_policy_keeps_loaded_procedural_bank_read_only(
    tmp_path,
):
    seed_root = tmp_path / "procedural-seed"
    seed = ChipMEM(seed_root, "example", threshold=0.0, allow_empty=True)
    handle = seed.embed_task(
        "seed task",
        lambda _: [1.0] * 8,
        identity={"model": "seed", "dimension": 8, "provider": "local"},
    )
    seed.learn(
        [],
        source_task="seed",
        verdict=seed.run_harness(handle, lambda: "pass"),
        distill=lambda *_: "verified seed skill",
        task_embedding=handle,
    )
    source_files = {
        path.relative_to(seed_root): path.read_bytes()
        for path in seed_root.rglob("*")
        if path.is_file() and not path.name.endswith(".lock")
    }
    config = make_config(tmp_path, "chipmem")
    config["procedural_policy"] = "read_only"
    config["procedural_seed_directory"] = str(seed_root)
    path = tmp_path / "historical-combined.json"
    path.write_text(json.dumps(config))

    row = run_experiment(path)[0]

    assert row["retrieved_skills"] == ["skill_0001"]
    assert row["created_skill"] is None
    assert row["skill_count_before"] == row["skill_count_after"] == 1
    assert row["procedural_policy"] == "read_only"
    assert row["procedural_state_before_sha256"] == row[
        "procedural_state_after_sha256"
    ]
    assert row["statistical_updates"] == 3
    assert {
        file.relative_to(seed_root): file.read_bytes()
        for file in seed_root.rglob("*")
        if file.is_file() and not file.name.endswith(".lock")
    } == source_files
