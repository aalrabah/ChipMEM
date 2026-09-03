import json
import subprocess
import sys
from pathlib import Path

import pytest

import experiments.run_experiment as runner_module
from chipmem.memory import ChipMEM
from chipmem.statistical.features import OpenFlowFeatures
from chipmem.statistical.online import StatisticalMemory
from experiments.run_experiment import run_experiment


def test_synthetic_evolving_run_records_pass_only_learning(tmp_path):
    tasks = tmp_path / "tasks"
    for task_id, text in [("task_a", "PASS task A"), ("task_b", "FAIL task B")]:
        directory = tasks / task_id
        directory.mkdir(parents=True)
        (directory / "TASK.md").write_text(text)

    config = {
        "mode": "chipmem",
        "domain": "example",
        "task_directory": str(tasks),
        "task_order": ["task_a", "task_b"],
        "task_artifact": "TASK.md",
        "state_directory": str(tmp_path / "state"),
        "output_directory": str(tmp_path / "output"),
        "top_k": 2,
        "threshold": 0.0,
        "embedding_dimension": 8,
        "embedding_callable": "experiments.example_plugins:embed",
        "agent_callable": "experiments.example_plugins:agent",
        "harness_callable": "experiments.example_plugins:harness",
        "distill_callable": "experiments.example_plugins:distill",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    rows = run_experiment(config_path)

    assert [row["verdict"] for row in rows] == ["pass", "fail"]
    assert rows[0]["created_skill"] == "skill_0001"
    assert rows[1]["created_skill"] is None
    assert rows[1]["retrieved_skills"] == ["skill_0001"]
    assert rows[0]["skill_count_after"] == 1
    assert rows[1]["skill_count_after"] == 1
    assert (tmp_path / "output" / "session_log.jsonl").is_file()
    first_session = tmp_path / "output" / "sessions" / "0001_task_a"
    assert json.loads((first_session / "transcript.json").read_text()) == {
        "transcript": [
            {"role": "user", "content": "PASS task A"},
            {"role": "memory", "content": ""},
            {"role": "assistant", "content": "synthetic completion"},
        ]
    }
    assert json.loads((first_session / "agent_result.json").read_text())["artifact"] == "PASS task A"


def test_cli_script_runs_from_repository_root(tmp_path):
    tasks = tmp_path / "tasks"
    task = tasks / "task_a"
    task.mkdir(parents=True)
    (task / "TASK.md").write_text("PASS CLI task")
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
        "embedding_callable": "experiments.example_plugins:embed",
        "agent_callable": "experiments.example_plugins:agent",
        "harness_callable": "experiments.example_plugins:harness",
        "distill_callable": "experiments.example_plugins:distill",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    repository = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, "experiments/run_experiment.py", "--config", str(path)],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "fail": 0,
        "invalid": 0,
        "pass": 1,
        "tasks": 1,
    }


def test_runner_rejects_state_inside_task_directory(tmp_path):
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    config = {
        "mode": "chipmem",
        "domain": "example",
        "task_directory": str(tasks),
        "task_order": [],
        "task_artifact": "TASK.md",
        "state_directory": str(tasks / "state"),
        "output_directory": str(tmp_path / "output"),
        "top_k": 2,
        "threshold": 0.6,
        "embedding_dimension": 8,
        "embedding_callable": "experiments.example_plugins:embed",
        "agent_callable": "experiments.example_plugins:agent",
        "harness_callable": "experiments.example_plugins:harness",
        "distill_callable": "experiments.example_plugins:distill",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    try:
        run_experiment(path)
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("runner allowed state inside task directory")

    assert not (tasks / "state").exists()
    assert not (tmp_path / "output").exists()


def test_runner_refuses_existing_output_directory(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    config = {
        "mode": "memory_off",
        "domain": "example",
        "task_directory": str(tmp_path / "tasks"),
        "task_order": [],
        "task_artifact": "TASK.md",
        "state_directory": str(tmp_path / "state"),
        "output_directory": str(output),
        "top_k": 2,
        "threshold": 0.6,
        "embedding_dimension": 8,
        "embedding_callable": "experiments.example_plugins:embed",
        "agent_callable": "experiments.example_plugins:agent",
        "harness_callable": "experiments.example_plugins:harness",
        "distill_callable": "experiments.example_plugins:distill",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    try:
        run_experiment(path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("runner reused an existing output directory")


def test_runner_refuses_output_created_after_preflight(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    task = tasks / "task_a"
    task.mkdir(parents=True)
    (task / "TASK.md").write_text("PASS race task")
    output = tmp_path / "nested" / "output"
    config = {
        "mode": "memory_off",
        "domain": "example",
        "task_directory": str(tasks),
        "task_order": ["task_a"],
        "task_artifact": "TASK.md",
        "state_directory": str(tmp_path / "state"),
        "output_directory": str(output),
        "top_k": 2,
        "threshold": 0.6,
        "embedding_dimension": 8,
        "embedding_callable": "experiments.example_plugins:embed",
        "agent_callable": "experiments.example_plugins:agent",
        "harness_callable": "experiments.example_plugins:harness",
        "distill_callable": "experiments.example_plugins:distill",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    original = runner_module._configuration_context

    def racing_context(*args, **kwargs):
        result = original(*args, **kwargs)
        result[4].mkdir(parents=True)
        (result[4] / "preexisting.txt").write_text("preserve me")
        return result

    monkeypatch.setattr(runner_module, "_configuration_context", racing_context)

    with pytest.raises(FileExistsError):
        run_experiment(path)

    assert (output / "preexisting.txt").read_text() == "preserve me"
    assert not (output / "config.json").exists()
    assert not (output / "sessions").exists()


def test_direct_runner_preflights_malformed_seed_before_writing(tmp_path):
    tasks = tmp_path / "tasks"
    task = tasks / "task_a"
    task.mkdir(parents=True)
    (task / "TASK.md").write_text("PASS malformed seed task")
    seed = tmp_path / "seed" / "agents" / "example" / "memory"
    seed.mkdir(parents=True)
    (seed / "model.json").write_text("not-json")
    (seed / "nudge_experience.jsonl").write_text("")
    (seed / "recovery_experience.jsonl").write_text("")
    config = {
        "mode": "statistical_only",
        "domain": "example",
        "task_directory": str(tasks),
        "task_order": ["task_a"],
        "task_artifact": "TASK.md",
        "state_directory": str(tmp_path / "state"),
        "output_directory": str(tmp_path / "output"),
        "top_k": 2,
        "threshold": 0.6,
        "embedding_dimension": 8,
        "embedding_callable": "experiments.example_plugins:embed",
        "agent_callable": "experiments.example_plugins:agent",
        "harness_callable": "experiments.example_plugins:harness",
        "distill_callable": "experiments.example_plugins:distill",
        "statistical_seed_directory": str(tmp_path / "seed"),
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="statistical model"):
        run_experiment(path)

    assert not (tmp_path / "state").exists()
    assert not (tmp_path / "output").exists()


def test_runner_rejects_every_seed_and_writable_path_overlap(tmp_path):
    def config_for(root, mode):
        task = root / "tasks" / "task_a"
        task.mkdir(parents=True)
        (task / "TASK.md").write_text("PASS overlap task")
        return {
            "mode": mode,
            "domain": "example",
            "task_directory": str(root / "tasks"),
            "task_order": ["task_a"],
            "task_artifact": "TASK.md",
            "state_directory": str(root / "state"),
            "output_directory": str(root / "output"),
            "top_k": 2,
            "threshold": 0.0,
            "embedding_dimension": 8,
            "embedding_callable": "experiments.example_plugins:embed",
            "agent_callable": "experiments.example_plugins:agent",
            "harness_callable": "experiments.example_plugins:harness",
            "distill_callable": "experiments.example_plugins:distill",
        }

    def procedural_seed(root):
        memory = ChipMEM(root, "example", threshold=0.0, allow_empty=True)
        handle = memory.embed_task(
            "seed",
            lambda _: [1.0] * 8,
            identity={"model": "seed", "dimension": 8, "provider": "local"},
        )
        memory.learn(
            [],
            source_task="seed",
            verdict=memory.run_harness(handle, lambda: "pass"),
            distill=lambda *_: "seed skill",
            task_embedding=handle,
        )

    def statistical_seed(root):
        memory = StatisticalMemory(root, "example", OpenFlowFeatures())
        arguments = {"script": "synth"}
        memory.before_tool_call("seed", "yosys", arguments)
        memory.after_tool_call(
            "seed",
            "yosys",
            arguments,
            "command failed with return code 1",
        )

    def snapshot(root):
        return {
            path.relative_to(root): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file() and not path.name.endswith(".lock")
        }

    cases = []
    root = tmp_path / "output-in-statistical"
    root.mkdir()
    config = config_for(root, "statistical_only")
    seed = root / "seed"
    statistical_seed(seed)
    config["statistical_seed_directory"] = str(seed)
    config["output_directory"] = str(seed / "run-output")
    cases.append((root, seed, config))

    root = tmp_path / "output-in-procedural"
    root.mkdir()
    config = config_for(root, "procedural_only")
    seed = root / "seed"
    procedural_seed(seed)
    config["procedural_policy"] = "read_only"
    config["procedural_seed_directory"] = str(seed)
    config["output_directory"] = str(seed / "run-output")
    cases.append((root, seed, config))

    root = tmp_path / "procedural-at-statistical"
    root.mkdir()
    config = config_for(root, "chipmem")
    seed = root / "state" / "statistical"
    procedural_seed(seed)
    config["procedural_policy"] = "read_only"
    config["procedural_seed_directory"] = str(seed)
    cases.append((root, seed, config))

    root = tmp_path / "statistical-at-procedural"
    root.mkdir()
    config = config_for(root, "chipmem")
    seed = root / "state" / "procedural"
    statistical_seed(seed)
    config["statistical_seed_directory"] = str(seed)
    cases.append((root, seed, config))

    for root, seed, config in cases:
        before = snapshot(seed)
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="overlap"):
            run_experiment(config_path)

        assert snapshot(seed) == before
        output = Path(config["output_directory"])
        assert not output.exists()
