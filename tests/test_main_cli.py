import json
import subprocess
import sys
from pathlib import Path

from chipmem.statistical.model import AgentModel

REPOSITORY = Path(__file__).resolve().parents[1]


def run_cli(*arguments, cwd=None):
    return subprocess.run(
        [sys.executable, str(REPOSITORY / "main.py"), *arguments],
        cwd=cwd or REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )


def write_config(tmp_path, *, mode="chipmem"):
    tasks = tmp_path / "tasks"
    task = tasks / "task_a"
    task.mkdir(parents=True)
    (task / "TASK.md").write_text("PASS CLI task")
    config = {
        "mode": mode,
        "domain": "example",
        "task_directory": str(tasks),
        "task_order": ["task_a"],
        "task_artifact": "TASK.md",
        "state_directory": str(tmp_path / "state"),
        "output_directory": str(tmp_path / "output"),
        "top_k": 2,
        "threshold": 0.6,
        "embedding_dimension": 8,
        "agent_endpoint_env": None,
        "step_limit": 32,
        "session_timeout_seconds": 600,
        "nudge_threshold": 0.20,
        "max_nudges": None,
        "features_callable": "chipmem.statistical.features:OpenFlowFeatures",
        "planner_callable": None,
        "procedural_seed_directory": None,
        "statistical_seed_directory": None,
        "embedding_callable": "experiments.example_plugins:embed",
        "agent_callable": "experiments.example_plugins:agent",
        "harness_callable": "experiments.example_plugins:harness",
        "distill_callable": "experiments.example_plugins:distill",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def test_list_knobs_json_exposes_every_runner_setting():
    completed = run_cli("list-knobs", "--json")

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    names = {item["name"] for item in payload["knobs"]}
    assert names == {
        "agent_callable",
        "agent_endpoint_env",
        "distill_callable",
        "domain",
        "embedding_callable",
        "embedding_dimension",
        "features_callable",
        "harness_callable",
        "max_nudges",
        "mode",
        "nudge_threshold",
        "output_directory",
        "planner_callable",
        "procedural_policy",
        "procedural_seed_directory",
        "session_timeout_seconds",
        "state_directory",
        "statistical_seed_directory",
        "step_limit",
        "task_artifact",
        "task_directory",
        "task_order",
        "threshold",
        "top_k",
    }
    shipped = json.loads(
        (REPOSITORY / "experiments" / "config.example.json").read_text()
    )
    assert set(shipped) == names
    by_name = {item["name"]: item for item in payload["knobs"]}
    assert by_name["mode"]["choices"] == [
        "memory_off",
        "procedural_only",
        "statistical_only",
        "chipmem",
    ]
    assert by_name["nudge_threshold"]["default"] == 0.20
    assert by_name["nudge_threshold"]["methodology_default"] is True
    assert by_name["max_nudges"]["default"] is None
    assert by_name["max_nudges"]["methodology_default"] is True


def test_print_config_applies_named_and_generic_overrides(tmp_path):
    config = write_config(tmp_path)
    completed = run_cli(
        "print-config",
        "--config",
        str(config),
        "--mode",
        "procedural_only",
        "--state-directory",
        "cli-state",
        "--output-directory",
        "cli-output",
        "--task-order",
        "task_a",
        "--top-k",
        "4",
        "--set",
        'agent_endpoint_env=null',
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["mode"] == "procedural_only"
    assert payload["state_directory"] == str((tmp_path / "cli-state").resolve())
    assert payload["output_directory"] == str((tmp_path / "cli-output").resolve())
    assert payload["task_directory"] == str((tmp_path / "tasks").resolve())
    assert payload["task_order"] == ["task_a"]
    assert payload["top_k"] == 4
    assert payload["agent_endpoint_env"] is None
    assert payload["procedural_policy"] == "evolving"


def test_validate_checks_configuration_without_creating_state_or_output(tmp_path):
    config = write_config(tmp_path)

    completed = run_cli("validate", "--config", str(config))

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "valid"
    assert payload["mode"] == "chipmem"
    assert payload["task_count"] == 1
    assert payload["tasks"] == ["task_a"]
    assert not (tmp_path / "state").exists()
    assert not (tmp_path / "output").exists()


def test_run_dry_run_performs_preflight_without_writing(tmp_path):
    config = write_config(tmp_path)

    completed = run_cli("run", "--config", str(config), "--dry-run")

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "valid"
    assert payload["dry_run"] is True
    assert payload["task_count"] == 1
    assert not (tmp_path / "state").exists()
    assert not (tmp_path / "output").exists()


def test_run_executes_with_cli_overrides_and_records_resolved_config(tmp_path):
    config = write_config(tmp_path)
    state = tmp_path / "run-state"
    output = tmp_path / "run-output"

    completed = run_cli(
        "run",
        "--config",
        str(config),
        "--state-directory",
        str(state),
        "--output-directory",
        str(output),
        "--retrieval-threshold",
        "0.0",
        "--top-k",
        "3",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload == {
        "fail": 0,
        "invalid": 0,
        "mode": "chipmem",
        "output_directory": str(output),
        "pass": 1,
        "state_directory": str(state),
        "status": "completed",
        "tasks": 1,
    }
    persisted = json.loads((output / "config.json").read_text())
    assert persisted["output_directory"] == str(output)
    assert persisted["state_directory"] == str(state)
    assert persisted["threshold"] == 0.0
    assert persisted["top_k"] == 3


def test_methodology_deviation_requires_explicit_acknowledgement(tmp_path):
    config = write_config(tmp_path)

    refused = run_cli(
        "print-config",
        "--config",
        str(config),
        "--nudge-threshold",
        "0.30",
    )
    allowed = run_cli(
        "print-config",
        "--config",
        str(config),
        "--nudge-threshold",
        "0.30",
        "--allow-methodology-overrides",
    )

    assert refused.returncode == 2
    assert "--allow-methodology-overrides" in refused.stderr
    assert "Traceback" not in refused.stderr
    assert allowed.returncode == 0, allowed.stderr
    assert json.loads(allowed.stdout)["nudge_threshold"] == 0.30


def test_invalid_generic_override_returns_concise_error(tmp_path):
    config = write_config(tmp_path)

    completed = run_cli(
        "validate",
        "--config",
        str(config),
        "--set",
        "not_a_knob=true",
    )

    assert completed.returncode == 2
    assert "unknown config fields" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_default_base_config_can_be_used_with_only_path_overrides(tmp_path):
    output = tmp_path / "output"
    state = tmp_path / "state"
    completed = run_cli(
        "validate",
        "--task-directory",
        str(REPOSITORY / "dataset" / "synthetic"),
        "--task-order",
        "task_a",
        "--state-directory",
        str(state),
        "--output-directory",
        str(output),
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "valid"
    assert payload["tasks"] == ["task_a"]
    assert payload["state_directory"] == str(state)
    assert payload["output_directory"] == str(output)


def test_shipped_config_allows_mode_only_overrides(tmp_path):
    for mode in (
        "memory_off",
        "procedural_only",
        "statistical_only",
        "chipmem",
    ):
        completed = run_cli(
            "validate",
            "--mode",
            mode,
            "--state-directory",
            str(tmp_path / f"state-{mode}"),
            "--output-directory",
            str(tmp_path / f"output-{mode}"),
        )
        assert completed.returncode == 0, completed.stderr


def test_run_help_exposes_every_named_knob():
    completed = run_cli("run", "--help")

    assert completed.returncode == 0, completed.stderr
    for knob in json.loads(run_cli("list-knobs", "--json").stdout)["knobs"]:
        assert knob["flag"] in completed.stdout


def test_list_knobs_reports_fixed_methodology_constants():
    completed = run_cli("list-knobs", "--json")

    assert completed.returncode == 0, completed.stderr
    fixed = json.loads(completed.stdout)["fixed_methodology"]
    assert fixed == {
        "parent_strength": 8.0,
        "parent_strength_grid": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
        "planner_max_steps": 3,
        "prior_alpha": 1.0,
        "prior_beta": 1.0,
        "recovery_horizon": 8,
        "recovery_max_excerpts": 3,
        "retry_buckets": ["0", "1", "2", "3+"],
    }


def test_main_cli_runs_all_four_modes(tmp_path):
    config = write_config(tmp_path)
    expected = {
        "memory_off": (False, False, 0, None),
        "procedural_only": (True, False, 0, "skill_0001"),
        "statistical_only": (False, True, 3, None),
        "chipmem": (True, True, 3, "skill_0001"),
    }
    for mode, contract in expected.items():
        state = tmp_path / f"state-{mode}"
        output = tmp_path / f"output-{mode}"
        completed = run_cli(
            "run",
            "--config",
            str(config),
            "--mode",
            mode,
            "--state-directory",
            str(state),
            "--output-directory",
            str(output),
            "--retrieval-threshold",
            "0.0",
        )

        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout)["mode"] == mode
        events = [
            json.loads(line)
            for line in (output / "session_log.jsonl").read_text().splitlines()
        ]
        row = next(event for event in events if event["event"] == "finalized")
        assert (
            row["procedural_enabled"],
            row["statistical_enabled"],
            row["statistical_updates"],
            row["created_skill"],
        ) == contract


def test_run_persists_the_same_resolved_config_printed_by_cli(tmp_path):
    base = tmp_path / "experiment"
    task = base / "tasks" / "task_a"
    task.mkdir(parents=True)
    (task / "TASK.md").write_text("PASS relative-path task")
    config = json.loads(
        (REPOSITORY / "experiments" / "config.example.json").read_text()
    )
    config.update(
        {
            "task_directory": "tasks",
            "task_order": ["task_a"],
            "state_directory": "state",
            "output_directory": "output",
        }
    )
    config_path = base / "config.json"
    config_path.write_text(json.dumps(config))

    printed = run_cli("print-config", "--config", str(config_path))
    completed = run_cli("run", "--config", str(config_path))

    assert printed.returncode == 0, printed.stderr
    assert completed.returncode == 0, completed.stderr
    expected = json.loads(printed.stdout)
    persisted = json.loads((base / "output" / "config.json").read_text())
    assert persisted == expected


def test_validate_rejects_invalid_generic_knob_values_before_writing(tmp_path):
    config = write_config(tmp_path)
    for assignment in (
        "top_k=0",
        "embedding_dimension=0",
        'threshold="not-a-number"',
        'domain="../shared"',
    ):
        completed = run_cli(
            "validate",
            "--config",
            str(config),
            "--set",
            assignment,
        )
        assert completed.returncode == 2
        assert "Traceback" not in completed.stderr
    assert not (tmp_path / "state").exists()
    assert not (tmp_path / "output").exists()


def test_run_creates_missing_output_parents_without_overwriting(tmp_path):
    config = write_config(tmp_path)
    parent = tmp_path / "runs" / "nested"
    output = parent / "output"
    state = parent / "state"

    first = run_cli(
        "run",
        "--config",
        str(config),
        "--state-directory",
        str(state),
        "--output-directory",
        str(output),
    )
    second = run_cli(
        "run",
        "--config",
        str(config),
        "--state-directory",
        str(state),
        "--output-directory",
        str(output),
    )

    assert first.returncode == 0, first.stderr
    assert output.is_dir()
    assert second.returncode == 2
    assert "output directory already exists" in second.stderr
    assert "Traceback" not in second.stderr


def test_validate_rejects_malformed_seed_contents_without_writing(tmp_path):
    procedural = tmp_path / "procedural"
    procedural.mkdir()
    procedural_config = write_config(procedural, mode="procedural_only")
    procedural_seed = (
        procedural
        / "seed"
        / "agents"
        / "example"
        / "memory"
        / "skill_0001"
    )
    procedural_seed.mkdir(parents=True)
    (procedural_seed / "SKILL.md").write_text("# plausible")
    (procedural_seed / "skill.json").write_text("not-json")
    (procedural_seed / "embeddings.json").write_text("not-json")
    payload = json.loads(procedural_config.read_text())
    payload["procedural_policy"] = "read_only"
    payload["procedural_seed_directory"] = str(procedural / "seed")
    procedural_config.write_text(json.dumps(payload))

    statistical = tmp_path / "statistical"
    statistical.mkdir()
    statistical_config = write_config(statistical, mode="statistical_only")
    statistical_seed = (
        statistical / "seed" / "agents" / "example" / "memory"
    )
    statistical_seed.mkdir(parents=True)
    (statistical_seed / "model.json").write_text("not-json")
    (statistical_seed / "nudge_experience.jsonl").write_text("")
    (statistical_seed / "recovery_experience.jsonl").write_text("")
    payload = json.loads(statistical_config.read_text())
    payload["statistical_seed_directory"] = str(statistical / "seed")
    statistical_config.write_text(json.dumps(payload))

    procedural_result = run_cli(
        "validate", "--config", str(procedural_config)
    )
    statistical_result = run_cli(
        "validate", "--config", str(statistical_config)
    )

    assert procedural_result.returncode == 2
    assert statistical_result.returncode == 2
    assert "Traceback" not in procedural_result.stderr
    assert "Traceback" not in statistical_result.stderr
    for case in (procedural, statistical):
        assert not (case / "state").exists()
        assert not (case / "output").exists()


def statistical_case(tmp_path, name, *, model=None, nudge=None, recovery=None):
    case = tmp_path / name
    case.mkdir()
    config = write_config(case, mode="statistical_only")
    memory = case / "seed" / "agents" / "example" / "memory"
    memory.mkdir(parents=True)
    (memory / "model.json").write_text(
        json.dumps(AgentModel().to_dict() if model is None else model)
    )
    (memory / "nudge_experience.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in (nudge or []))
    )
    (memory / "recovery_experience.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in (recovery or []))
    )
    payload = json.loads(config.read_text())
    payload["statistical_seed_directory"] = str(case / "seed")
    config.write_text(json.dumps(payload))
    return case, config


def valid_retry_row(**overrides):
    row = {
        "session": "session-1",
        "step": 0,
        "ct": "yosys_synth",
        "rc": 0,
        "pr": "none",
        "pec": "none",
        "te": 0,
        "success": 1,
        "error_class": "ok",
    }
    row.update(overrides)
    return row


def valid_recovery_row(**overrides):
    row = {
        "record_type": "recovery",
        "session": "session-1",
        "step": 1,
        "failure_idx": 0,
        "error": "timing-fail",
        "stage": "synth",
        "strategy": "inspect_logs",
        "recovered": 1,
    }
    row.update(overrides)
    return row


def test_validate_rejects_semantically_invalid_statistical_state(tmp_path):
    scenarios = {
        "invalid-success": {"nudge": [valid_retry_row(success=2)]},
        "negative-step": {"nudge": [valid_retry_row(step=-1)]},
        "missing-field": {
            "nudge": [
                {
                    key: value
                    for key, value in valid_retry_row().items()
                    if key != "error_class"
                }
            ]
        },
        "duplicate-step": {
            "nudge": [valid_retry_row(), valid_retry_row(success=0)]
        },
        "invalid-recovered": {
            "recovery": [valid_recovery_row(recovered=2)]
        },
        "duplicate-recovery": {
            "recovery": [valid_recovery_row(), valid_recovery_row(recovered=0)]
        },
        "invalid-counts": {"model": {"counts": []}},
        "negative-count": {
            "model": {
                **AgentModel().to_dict(),
                "counts": {"global": [-1.0, 0.0]},
            }
        },
    }
    for name, values in scenarios.items():
        case, config = statistical_case(tmp_path, name, **values)
        completed = run_cli("validate", "--config", str(config))
        assert completed.returncode == 2, (name, completed.stdout, completed.stderr)
        assert "Traceback" not in completed.stderr
        assert not (case / "state").exists()
        assert not (case / "output").exists()


def test_validate_rejects_seed_destination_overlap_without_output(tmp_path):
    case = tmp_path / "overlap"
    case.mkdir()
    config = write_config(case, mode="statistical_only")
    seed_root = case / "state" / "statistical" / "seed"
    memory = seed_root / "agents" / "example" / "memory"
    memory.mkdir(parents=True)
    (memory / "model.json").write_text(json.dumps(AgentModel().to_dict()))
    (memory / "nudge_experience.jsonl").write_text("")
    (memory / "recovery_experience.jsonl").write_text("")
    payload = json.loads(config.read_text())
    payload["statistical_seed_directory"] = str(seed_root)
    config.write_text(json.dumps(payload))

    completed = run_cli("validate", "--config", str(config))

    assert completed.returncode == 2
    assert "overlap" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not (case / "output").exists()
    assert (memory / "model.json").is_file()


def test_missing_statistical_hooks_is_a_concise_contract_error(tmp_path):
    config = write_config(tmp_path, mode="statistical_only")
    payload = json.loads(config.read_text())
    payload["agent_callable"] = "experiments.example_plugins:agent_without_hooks"
    config.write_text(json.dumps(payload))

    completed = run_cli("run", "--config", str(config))

    assert completed.returncode == 2
    assert "requires live tool-call hooks" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_nullable_flags_and_auto_policy_are_consistent(tmp_path):
    config = write_config(tmp_path)
    for flag in (
        "--agent-endpoint-env",
        "--planner-callable",
        "--procedural-seed-directory",
        "--statistical-seed-directory",
    ):
        for value in ("none", "null"):
            completed = run_cli(
                "print-config", "--config", str(config), flag, value
            )
            assert completed.returncode == 0, (flag, value, completed.stderr)
            name = flag[2:].replace("-", "_")
            assert json.loads(completed.stdout)[name] is None
    for value in ("auto", "none", "null"):
        completed = run_cli(
            "print-config",
            "--config",
            str(config),
            "--mode",
            "memory_off",
            "--procedural-policy",
            value,
        )
        assert completed.returncode == 0, (value, completed.stderr)
        assert json.loads(completed.stdout)["procedural_policy"] == "disabled"
