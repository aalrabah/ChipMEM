import json
import subprocess
import sys
from pathlib import Path

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
