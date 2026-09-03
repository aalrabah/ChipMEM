# ChipMEM

ChipMEM combines task-indexed procedural memory with step-online Bayesian statistical memory for tool-using agents. It retrieves complete verified skills, predicts retry success before tool calls, updates from every completed tool call, and learns new skills only after a deterministic evaluation harness returns `PASS`.

This repository contains the public procedural and statistical-memory implementation and a generic Python experiment runner. It does not contain paper results, raw trajectories, trained memory states, proprietary agents, private prompts, commercial datasets, or commercial tool configurations.

## Core behavior

1. Read only the benchmark-declared task artifact for retrieval.
2. Compute and cache one task embedding for that exact artifact.
3. Use the task embedding as the retrieval key.
4. Retrieve the top matching skills above a configurable threshold.
5. Inject complete `SKILL.md` files without rewriting or summarizing them.
6. Run the configured agent and deterministic evaluation harness.
7. Create at most one immutable skill after a verified `PASS`.
8. Create no skill after `FAIL` or `INVALID`.
9. Predict before each tool call and update statistical memory after every completed call.
10. Provide advisory recovery guidance when predicted retry success is below `0.20`.
11. Carry procedural and statistical state forward only within the selected mode.

## Repository layout

```text
ChipMEM/
├── main.py
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── chipmem/
│   ├── __init__.py
│   ├── dataset.py
│   ├── embedding.py
│   ├── memory.py
│   ├── retrieval.py
│   ├── learning.py
│   ├── store.py
│   ├── injector.py
│   ├── task_document.py
│   └── statistical/
│       ├── model.py
│       ├── features.py
│       ├── extraction.py
│       ├── nudge.py
│       ├── online.py
│       └── hooks.py
├── dataset/
│   ├── README.md
│   └── synthetic/
│       ├── task_a/TASK.md
│       └── task_b/TASK.md
├── examples/
│   └── tasks/
│       ├── task_a/TASK.md
│       └── task_b/TASK.md
├── experiments/
│   ├── __init__.py
│   ├── run_experiment.py
│   ├── example_plugins.py
│   └── config.example.json
└── tests/
```

## Installation

ChipMEM currently supports Python 3.10 or newer on Linux and macOS. The
implementation uses standard-library file locks for atomic state updates.

```text
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The runtime implementation uses only the Python standard library. `pytest` is included for verification.

## Running experiments with `main.py`

`main.py` is the primary experiment interface. Users select modes, paths,
adapters, retrieval settings, statistical settings, and execution limits from
the command line or a JSON configuration; no Python source edits are required.

Show the commands:

```text
python3 main.py --help
```

List every supported experiment knob and every fixed methodology constant:

```text
python3 main.py list-knobs
python3 main.py list-knobs --json
```

### Recommended workflow

First inspect the effective configuration:

```text
python3 main.py print-config \
  --config experiments/config.example.json
```

Then validate the configuration, task artifacts, paths, adapters, optional
seed states, and endpoint environment without creating state or output:

```text
python3 main.py validate \
  --config experiments/config.example.json \
  --state-directory runs/example-state \
  --output-directory runs/example-output
```

`run --dry-run` performs the same side-effect-free preflight:

```text
python3 main.py run \
  --config experiments/config.example.json \
  --state-directory runs/example-state \
  --output-directory runs/example-output \
  --dry-run
```

Run after validation:

```text
python3 main.py run \
  --config experiments/config.example.json \
  --state-directory runs/example-state \
  --output-directory runs/example-output
```

Output directories must be new. The runner refuses to overwrite an existing
output directory.

### Selecting a memory mode

Use one of the four explicit modes:

```text
python3 main.py run --mode memory_off       --state-directory runs/off-state  --output-directory runs/off-output
python3 main.py run --mode procedural_only  --state-directory runs/proc-state --output-directory runs/proc-output
python3 main.py run --mode statistical_only --state-directory runs/stat-state --output-directory runs/stat-output
python3 main.py run --mode chipmem           --state-directory runs/full-state --output-directory runs/full-output
```

These commands use `experiments/config.example.json` as the base configuration.
Supply `--config PATH` to use a different base file.

### Overriding experiment settings

Every runner setting has a named flag. For example:

```text
python3 main.py run \
  --config experiments/config.example.json \
  --mode chipmem \
  --domain rtlopt \
  --task-directory /path/to/tasks \
  --task-order design_a design_b design_c \
  --task-artifact TASK.md \
  --state-directory /path/to/state \
  --output-directory /path/to/output \
  --top-k 2 \
  --retrieval-threshold 0.60 \
  --embedding-dimension 1024 \
  --embedding-callable package.adapters:embed \
  --agent-callable package.adapters:agent \
  --harness-callable package.adapters:harness \
  --distill-callable package.adapters:distill \
  --features-callable package.features:Features \
  --step-limit 32 \
  --session-timeout-seconds 600 \
  --nudge-threshold 0.20 \
  --max-nudges none
```

`--agent-endpoint-env`, `--planner-callable`,
`--procedural-seed-directory`, and `--statistical-seed-directory` accept
`none` or `null`. `--max-nudges` accepts `none`, `null`, `unlimited`, or
`uncapped` for the reported uncapped behavior.

`--procedural-policy auto` selects `evolving` for `procedural_only` and
`chipmem`, and `disabled` for `memory_off` and `statistical_only`. The aliases
`none` and `null` also select `auto`.

For future or programmatic settings, `--set KEY=JSON` is available:

```text
python3 main.py print-config \
  --set 'task_order=["task_a","task_b"]' \
  --set 'agent_endpoint_env=null'
```

Precedence is:

```text
named CLI flag > --set KEY=JSON > JSON configuration
```

Paths supplied on the command line resolve from the current working directory.
Paths stored in a JSON configuration resolve from that configuration file's
directory. `print-config` displays absolute resolved paths.

The reported statistical methodology uses `nudge_threshold=0.20` and no
warning cap. Changing either requires an explicit acknowledgement:

```text
python3 main.py run \
  --nudge-threshold 0.30 \
  --allow-methodology-overrides \
  --state-directory runs/variant-state \
  --output-directory runs/variant-output
```

The resolved effective configuration is stored as `output/config.json` for
every completed run.

## Task format

The experiment runner loads the ordered dataset through `chipmem.dataset.load_dataset()` and expects one directory per task:

```text
dataset/<dataset-name>/
├── task_a/
│   └── TASK.md
└── task_b/
    └── TASK.md
```

`TASK.md` is the exact UTF-8 retrieval document. Invalid UTF-8 is rejected rather than replaced. Agent instructions, tool syntax, generated files, tests, evaluation artifacts, and skill text must not be added to it.

For RTL benchmarks, applications may instead call `chipmem.task_document.rtl_task_document()` to use the exact top-level `.v` or `.sv` source.

## Adapter interface

The runner loads four Python callables declared as `module:function` strings in the JSON configuration. These imports execute Python code, so use only trusted configuration files and adapter modules.

### Embedding callable

```python
def embed(task_document: str, dimension: int) -> list[float]:
    ...
```

### Agent callable

```python
def agent(task_document, memory_context, task_directory, session_directory, execution, hooks=None) -> dict:
    # execution contains endpoint, step_limit, and session_timeout_seconds.
    return {
        "artifact": ...,
        "transcript": [...],
    }
```

The agent result must be a JSON-serializable dictionary containing a JSON-serializable `transcript` list. This allows the runner to preserve the exact adapter result before evaluation.

For `memory_off` and `procedural_only`, the runner preserves the original five-argument adapter call and does not pass `hooks`. For `statistical_only` and `chipmem`, the sixth argument is required. The adapter must call `hooks.before_tool_call(tool, arguments)` immediately before every tool call and `hooks.after_tool_call(tool, arguments, output)` immediately after every completed call. A returned warning is advisory: the already-selected call still executes, and the advice may influence only the next decision. Statistical modes fail closed when an adapter completes without exercising these hooks.

### Evaluation-harness callable

```python
def harness(task_document, agent_result, task_directory, session_directory) -> str:
    return "pass"  # or "fail" or "invalid"
```

The harness must return an actual string with exactly `pass`, `fail`, or `invalid` after case normalization. The runner invokes it through `ChipMEM.run_harness(task_embedding, harness, ...)`, which returns an integrity-sealed, memory-bound, task-bound, single-use verified-verdict capability. `ChipMEM.learn()` accepts only that capability; a caller-supplied string, mutated verdict, replayed verdict, or verdict issued for another task cannot authorize skill creation. The harness must evaluate the produced artifact independently. The agent's own claim of success is not a valid verdict.

### Distillation callable

```python
def distill(transcript: list[dict], verdict: str) -> str:
    ...
```

The runner calls the distiller only after `PASS`. Existing verified skills are never distilled again or modified.

Production adapters may call a private or self-hosted model endpoint. Set `agent_endpoint_env` to the name of an environment variable containing the endpoint. The runner resolves that variable at runtime and passes its value to the agent together with `step_limit` and `session_timeout_seconds`. The bundled synthetic configuration uses `null`, so it requires no endpoint. Credentials must be supplied through environment variables or an external secret manager, never committed to configuration files.

A deterministic external evaluation command can be wrapped by the configured Python harness callable using `subprocess.run()` with an argument list and `shell=False`. ChipMEM does not require or ship Bash wrappers.

## Configuration

Copy `experiments/config.example.json` and update the dataset, state, output, and adapter fields. Relative paths are resolved from the configuration file. The bundled configuration reads privacy-safe tasks from `../dataset/synthetic` and writes `../state` and `../output` at the repository root. Output directories must be new. The runner refuses to overwrite an existing output directory.

Run with the primary interface:

```text
python3 main.py run --config experiments/config.example.json
```

The original entry point remains available for backward compatibility:

```text
python3 experiments/run_experiment.py --config experiments/config.example.json
```

The runner supports four explicit modes:

| Mode | Procedural | Statistical |
|---|---:|---:|
| `memory_off` | No | No |
| `procedural_only` | Yes | No |
| `statistical_only` | No | Yes |
| `chipmem` | Yes | Yes |

`chipmem` is the full system. Statistical modes use a fixed default nudge threshold of `0.20` and four retry buckets (`0`, `1`, `2`, `3+`). Eligible warnings are not capped by default. The optional recovery planner may return at most three steps.

Procedural modes default to `procedural_policy: "evolving"`, where a verified PASS may add one skill. Set `procedural_policy: "read_only"` with `procedural_seed_directory` to reproduce a loaded fixed-bank evaluation. The source bank is copied into private mode state and is never modified. Query embeddings use a separate runtime cache, so the copied procedural tree remains byte-stable.

## Output

Each run creates:

```text
output/
├── config.json
├── session_log.jsonl
└── sessions/
    └── 0001_<task_id>/
        ├── agent_result.json
        ├── transcript.json
        ├── gate_result.json
        ├── task_sha256.txt
        └── retrieved_skills/  # present only when retrieval is non-empty
```

`session_log.jsonl` is event-based. Each task first appends a durable `gate` event immediately after the harness verdict and before distillation. A successful completion then appends a `finalized` event with learning and post-state fields. If distillation or storage fails after a verified verdict, the gate event and per-session `gate_result.json` remain available.

Gate and finalized events collectively record:

- Task ID, order, and SHA-256
- Mode
- Retrieved skill IDs and similarity scores
- Skill counts before and after the task
- Created skill ID, if any
- Final `pass`, `fail`, or `invalid` verdict
- Memory-state hashes before and after the task
- Procedural and statistical state hashes
- Statistical update and nudge counts

The separate state directory contains complete immutable skill files, cached task embeddings, and mode-private statistical state.

## Skill format

```text
state/procedural/agents/<domain>/memory/skill_0001/
├── SKILL.md
├── skill.json
└── embeddings.json
```

`SKILL.md` contains the complete reusable procedure. `skill.json` records the source-task label, exact source-artifact SHA-256, and verified-PASS provenance. `embeddings.json` stores the vector carried by the task-bound `TaskEmbedding` handle, not an embedding of the skill text.

## Statistical state

Statistical modes persist three runtime-generated files:

```text
state/statistical/agents/<domain>/memory/
├── model.json
├── nudge_experience.jsonl
└── recovery_experience.jsonl
```

`model.json` stores the hierarchical retry and recovery counts. The experience files preserve idempotent step and recovery observations. To start from an external trained state without modifying it, set `statistical_seed_directory`; the runner copies and byte-verifies the three canonical files into the mode-private destination before execution.

## Tests

```text
.venv/bin/python -m pytest -q
```

The tests cover:

- PASS-only skill creation
- No learning after FAIL or INVALID
- One immutable skill per exact source-task artifact, including concurrent and aliased task IDs
- Task-only embedding handles, identity-aware caching, and strict UTF-8 artifact handling
- Domain-private top-K retrieval
- Complete skill injection and materialization containment
- Strict harness verdict and JSON adapter contracts
- Durable gate evidence before distillation
- Refusal to overwrite skills, materialized memory, or run outputs
- A self-contained synthetic end-to-end evolving-memory run
- Four-mode component isolation
- Differential parity with the executed Bayesian mathematics
- Prediction-before-call and update-after-call ordering
- Retry buckets, hierarchical probabilities, recovery ranking, and threshold behavior
- Idempotent online updates and loaded-state copying
- Combined procedural and statistical carry-forward

## Synthetic adapters

`experiments/example_plugins.py`, `dataset/synthetic/`, and `examples/tasks/` contain deterministic privacy-safe fixtures used by examples and tests. They demonstrate the adapter signatures and dataset layout but are not benchmark agents, paper datasets, results, or evaluation tools.

## Release scope

This repository intentionally excludes:

- Existing experiment results or paper tables
- Raw session transcripts
- Existing skill libraries and embeddings
- Trained statistical state and historical observations
- Proprietary agent implementations and prompts
- Private or commercial datasets
- Commercial EDA scripts and licenses
- Internal endpoints, hostnames, credentials, and absolute project paths
- Paper source and figures
- Bash scripts
