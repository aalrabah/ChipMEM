import json
from pathlib import Path

from experiments.run_experiment import run_experiment


def test_shipped_example_config_is_self_contained(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    source_config = repository / "experiments" / "config.example.json"
    config = json.loads(source_config.read_text())
    task_root = (source_config.parent / config["task_directory"]).resolve()

    assert (task_root / "task_a" / config["task_artifact"]).is_file()
    assert (task_root / "task_b" / config["task_artifact"]).is_file()

    config["task_directory"] = str(task_root)
    config["state_directory"] = str(tmp_path / "state")
    config["output_directory"] = str(tmp_path / "output")
    runnable = tmp_path / "config.json"
    runnable.write_text(json.dumps(config))

    rows = run_experiment(runnable)

    assert [row["verdict"] for row in rows] == ["pass", "fail"]
    assert [row["created_skill"] for row in rows] == ["skill_0001", None]
    agent_result = json.loads(
        (tmp_path / "output" / "sessions" / "0001_task_a" / "agent_result.json").read_text()
    )
    assert agent_result["execution"] == {
        "endpoint": None,
        "session_timeout_seconds": 600,
        "step_limit": 32,
    }
