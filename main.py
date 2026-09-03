from __future__ import annotations

import argparse
import json
from pathlib import Path

from chipmem.statistical.model import (
    DEFAULT_PARENT_STRENGTH,
    PARENT_STRENGTH_GRID,
    PLAN_MAX_STEPS,
    RECOVERY_HORIZON,
    RECOVERY_MAX_EXCERPTS,
)
from experiments.run_experiment import (
    load_config,
    preflight_experiment,
    run_experiment,
)

REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPOSITORY_ROOT / "experiments" / "config.example.json"

KNOBS = (
    {
        "name": "mode",
        "flag": "--mode",
        "default": "chipmem",
        "choices": [
            "memory_off",
            "procedural_only",
            "statistical_only",
            "chipmem",
        ],
        "methodology_default": False,
        "description": "Memory mode to execute.",
    },
    {
        "name": "domain",
        "flag": "--domain",
        "default": "example",
        "methodology_default": False,
        "description": "Domain-private memory namespace.",
    },
    {
        "name": "task_directory",
        "flag": "--task-directory",
        "default": "dataset/synthetic",
        "methodology_default": False,
        "description": "Directory containing one folder per task.",
    },
    {
        "name": "task_order",
        "flag": "--task-order",
        "default": ["task_a", "task_b"],
        "methodology_default": False,
        "description": "Ordered task identifiers.",
    },
    {
        "name": "task_artifact",
        "flag": "--task-artifact",
        "default": "TASK.md",
        "methodology_default": False,
        "description": "Exact task artifact used for retrieval.",
    },
    {
        "name": "state_directory",
        "flag": "--state-directory",
        "default": "state",
        "methodology_default": False,
        "description": "Persistent memory-state directory.",
    },
    {
        "name": "output_directory",
        "flag": "--output-directory",
        "default": "output",
        "methodology_default": False,
        "description": "New experiment output directory.",
    },
    {
        "name": "top_k",
        "flag": "--top-k",
        "default": 2,
        "methodology_default": False,
        "description": "Maximum number of procedural skills retrieved.",
    },
    {
        "name": "threshold",
        "flag": "--retrieval-threshold",
        "default": 0.6,
        "methodology_default": False,
        "description": "Minimum cosine similarity for retrieval.",
    },
    {
        "name": "embedding_dimension",
        "flag": "--embedding-dimension",
        "default": 8,
        "methodology_default": False,
        "description": "Expected task-embedding dimension.",
    },
    {
        "name": "embedding_callable",
        "flag": "--embedding-callable",
        "default": "experiments.example_plugins:embed",
        "methodology_default": False,
        "description": "Task embedding callable as module:function.",
    },
    {
        "name": "agent_callable",
        "flag": "--agent-callable",
        "default": "experiments.example_plugins:agent",
        "methodology_default": False,
        "description": "Agent callable as module:function.",
    },
    {
        "name": "harness_callable",
        "flag": "--harness-callable",
        "default": "experiments.example_plugins:harness",
        "methodology_default": False,
        "description": "Deterministic evaluation callable as module:function.",
    },
    {
        "name": "distill_callable",
        "flag": "--distill-callable",
        "default": "experiments.example_plugins:distill",
        "methodology_default": False,
        "description": "PASS-only skill distillation callable as module:function.",
    },
    {
        "name": "agent_endpoint_env",
        "flag": "--agent-endpoint-env",
        "default": None,
        "methodology_default": False,
        "description": "Environment-variable name containing an agent endpoint.",
    },
    {
        "name": "step_limit",
        "flag": "--step-limit",
        "default": 32,
        "methodology_default": False,
        "description": "Maximum agent steps passed to the adapter.",
    },
    {
        "name": "session_timeout_seconds",
        "flag": "--session-timeout-seconds",
        "default": 600,
        "methodology_default": False,
        "description": "Per-session timeout passed to the adapter.",
    },
    {
        "name": "procedural_policy",
        "flag": "--procedural-policy",
        "default": "auto",
        "choices": ["auto", "disabled", "evolving", "read_only"],
        "methodology_default": False,
        "description": "Procedural-state policy; default depends on mode.",
    },
    {
        "name": "procedural_seed_directory",
        "flag": "--procedural-seed-directory",
        "default": None,
        "methodology_default": False,
        "description": "External verified procedural bank copied before execution.",
    },
    {
        "name": "features_callable",
        "flag": "--features-callable",
        "default": "chipmem.statistical.features:OpenFlowFeatures",
        "methodology_default": False,
        "description": "Statistical feature adapter as module:class.",
    },
    {
        "name": "statistical_seed_directory",
        "flag": "--statistical-seed-directory",
        "default": None,
        "methodology_default": False,
        "description": "External statistical state copied before execution.",
    },
    {
        "name": "nudge_threshold",
        "flag": "--nudge-threshold",
        "default": 0.20,
        "methodology_default": True,
        "description": "Warn when retry success is strictly below this value.",
    },
    {
        "name": "max_nudges",
        "flag": "--max-nudges",
        "default": None,
        "methodology_default": True,
        "description": "Optional warning cap; reported methodology is uncapped.",
    },
    {
        "name": "planner_callable",
        "flag": "--planner-callable",
        "default": None,
        "methodology_default": False,
        "description": "Optional recovery planner as module:function.",
    },
)

PATH_FIELDS = {
    "output_directory",
    "procedural_seed_directory",
    "state_directory",
    "statistical_seed_directory",
    "task_directory",
}

FIXED_METHODOLOGY = {
    "parent_strength": DEFAULT_PARENT_STRENGTH,
    "parent_strength_grid": list(PARENT_STRENGTH_GRID),
    "planner_max_steps": PLAN_MAX_STEPS,
    "prior_alpha": 1.0,
    "prior_beta": 1.0,
    "recovery_horizon": RECOVERY_HORIZON,
    "recovery_max_excerpts": RECOVERY_MAX_EXCERPTS,
    "retry_buckets": ["0", "1", "2", "3+"],
}


def _nullable_string(value: str) -> str | None:
    return None if value.lower() in {"none", "null"} else value


def _nullable_integer(value: str) -> int | None:
    if value.lower() in {"none", "null", "unlimited", "uncapped"}:
        return None
    return int(value)


def _procedural_policy(value: str) -> str | None:
    normalized = value.lower()
    if normalized in {"auto", "none", "null"}:
        return None
    if normalized not in {"disabled", "evolving", "read_only"}:
        raise argparse.ArgumentTypeError(
            "procedural policy must be auto, disabled, evolving, or read_only"
        )
    return normalized


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    suppress = argparse.SUPPRESS
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Base JSON configuration (default: experiments/config.example.json).",
    )
    parser.add_argument("--mode", choices=KNOBS[0]["choices"], default=suppress)
    parser.add_argument("--domain", default=suppress)
    parser.add_argument("--task-directory", default=suppress)
    parser.add_argument("--task-order", nargs="+", default=suppress)
    parser.add_argument("--task-artifact", default=suppress)
    parser.add_argument("--state-directory", default=suppress)
    parser.add_argument("--output-directory", default=suppress)
    parser.add_argument("--top-k", type=int, default=suppress)
    parser.add_argument(
        "--retrieval-threshold",
        dest="threshold",
        type=float,
        default=suppress,
    )
    parser.add_argument("--embedding-dimension", type=int, default=suppress)
    parser.add_argument("--embedding-callable", default=suppress)
    parser.add_argument("--agent-callable", default=suppress)
    parser.add_argument("--harness-callable", default=suppress)
    parser.add_argument("--distill-callable", default=suppress)
    parser.add_argument(
        "--agent-endpoint-env",
        type=_nullable_string,
        default=suppress,
    )
    parser.add_argument("--step-limit", type=int, default=suppress)
    parser.add_argument(
        "--session-timeout-seconds",
        type=int,
        default=suppress,
    )
    parser.add_argument(
        "--procedural-policy",
        type=_procedural_policy,
        metavar="{auto,disabled,evolving,read_only}",
        default=suppress,
    )
    parser.add_argument(
        "--procedural-seed-directory",
        type=_nullable_string,
        default=suppress,
    )
    parser.add_argument("--features-callable", default=suppress)
    parser.add_argument(
        "--statistical-seed-directory",
        type=_nullable_string,
        default=suppress,
    )
    parser.add_argument("--nudge-threshold", type=float, default=suppress)
    parser.add_argument(
        "--max-nudges",
        type=_nullable_integer,
        default=suppress,
    )
    parser.add_argument(
        "--planner-callable",
        type=_nullable_string,
        default=suppress,
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="Set any supported configuration key using a JSON value.",
    )
    parser.add_argument(
        "--allow-methodology-overrides",
        action="store_true",
        help="Allow changes to reported-methodology defaults.",
    )


def _generic_overrides(values: list[str]) -> dict:
    result = {}
    for assignment in values:
        if "=" not in assignment:
            raise ValueError("--set values must use KEY=JSON")
        name, raw = assignment.split("=", 1)
        if not name:
            raise ValueError("--set requires a non-empty key")
        try:
            result[name] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--set value for {name} must be valid JSON") from exc
    return result


def _collect_overrides(arguments: argparse.Namespace) -> dict:
    values = vars(arguments)
    overrides = _generic_overrides(values.get("set", []))
    knob_names = {item["name"] for item in KNOBS}
    for name in knob_names:
        if name in values:
            overrides[name] = values[name]
    for name in PATH_FIELDS & overrides.keys():
        value = overrides[name]
        if value is not None:
            overrides[name] = str(Path(value).resolve())
    return overrides


def _resolved_config(arguments: argparse.Namespace) -> tuple[Path, dict, dict]:
    config_path = Path(arguments.config).resolve()
    overrides = _collect_overrides(arguments)
    config = load_config(config_path, overrides)
    for name in PATH_FIELDS:
        value = config.get(name)
        if value is not None:
            path = Path(value)
            config[name] = str(
                (path if path.is_absolute() else config_path.parent / path).resolve()
            )
    return config_path, config, overrides


def _enforce_methodology_defaults(
    config: dict,
    *,
    allow_overrides: bool,
) -> None:
    deviations = []
    if float(config["nudge_threshold"]) != 0.20:
        deviations.append(
            f"nudge_threshold={config['nudge_threshold']} (reported default: 0.20)"
        )
    if config["max_nudges"] is not None:
        deviations.append(
            f"max_nudges={config['max_nudges']} (reported default: uncapped)"
        )
    if deviations and not allow_overrides:
        detail = "; ".join(deviations)
        raise ValueError(
            "methodology defaults changed: "
            f"{detail}; pass --allow-methodology-overrides to continue"
        )


def _print_knobs(as_json: bool) -> None:
    if as_json:
        print(
            json.dumps(
                {"fixed_methodology": FIXED_METHODOLOGY, "knobs": KNOBS},
                indent=2,
                sort_keys=True,
            )
        )
        return
    print("ChipMEM experiment knobs")
    print("CLI values override the JSON configuration.\n")
    for item in KNOBS:
        default = json.dumps(item["default"])
        choices = item.get("choices")
        choice_text = f" choices={','.join(choices)}" if choices else ""
        fixed = " methodology-default" if item["methodology_default"] else ""
        print(f"{item['flag']:<32} default={default}{choice_text}{fixed}")
        print(f"  {item['description']}")
    print("\nFixed methodology constants (informational, not CLI knobs)")
    for name, value in FIXED_METHODOLOGY.items():
        print(f"{name:<32} {json.dumps(value)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and inspect ChipMEM experiments without editing Python code."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    knobs = subparsers.add_parser("list-knobs", help="List every supported setting.")
    knobs.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    printed = subparsers.add_parser(
        "print-config",
        help="Print the fully resolved configuration without running.",
    )
    _add_config_arguments(printed)
    validated = subparsers.add_parser(
        "validate",
        help="Validate configuration, tasks, adapters, and paths without writing.",
    )
    _add_config_arguments(validated)
    run = subparsers.add_parser("run", help="Run an experiment.")
    _add_config_arguments(run)
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform validation only; create no state or output.",
    )
    return parser


def _dispatch(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> None:
    if arguments.command == "list-knobs":
        _print_knobs(arguments.json)
        return
    if arguments.command == "print-config":
        _, config, _ = _resolved_config(arguments)
        _enforce_methodology_defaults(
            config,
            allow_overrides=arguments.allow_methodology_overrides,
        )
        print(json.dumps(config, indent=2, sort_keys=True))
        return
    if arguments.command in {"validate", "run"}:
        config_path, config, _ = _resolved_config(arguments)
        _enforce_methodology_defaults(
            config,
            allow_overrides=arguments.allow_methodology_overrides,
        )
        report = preflight_experiment(config_path, config)
        if arguments.command == "validate" or arguments.dry_run:
            if arguments.command == "run":
                report["dry_run"] = True
            print(json.dumps(report, indent=2, sort_keys=True))
            return
        rows = run_experiment(config_path, config)
        summary = {
            "fail": sum(row["verdict"] == "fail" for row in rows),
            "invalid": sum(row["verdict"] == "invalid" for row in rows),
            "mode": config["mode"],
            "output_directory": config["output_directory"],
            "pass": sum(row["verdict"] == "pass" for row in rows),
            "state_directory": config["state_directory"],
            "status": "completed",
            "tasks": len(rows),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    parser.error(f"unsupported command: {arguments.command}")


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        _dispatch(parser, arguments)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
