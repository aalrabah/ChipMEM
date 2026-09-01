# ChipMEM

ChipMEM is a task-indexed procedural memory layer for tool-using agents. It retrieves complete verified skills for later tasks and learns new skills only after a deterministic evaluation harness returns `PASS`.

This repository contains the public procedural-memory implementation and a generic Python experiment runner. It does not contain paper results, raw trajectories, trained memory states, proprietary agents, private prompts, commercial datasets, or commercial tool configurations.

## Core behavior

1. Read only the benchmark-declared task artifact for retrieval.
2. Compute and cache one task embedding for that exact artifact.
3. Use the task embedding as the retrieval key.
4. Retrieve the top matching skills above a configurable threshold.
5. Inject complete `SKILL.md` files without rewriting or summarizing them.
6. Run the configured agent and deterministic evaluation harness.
7. Create at most one immutable skill after a verified `PASS`.
8. Create no skill after `FAIL` or `INVALID`.
9. Carry verified skills forward to later tasks in `chipmem` mode.

## Repository layout

```text
ChipMEM/
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
│   └── task_document.py
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
def agent(task_document, memory_context, task_directory, session_directory, execution) -> dict:
    # execution contains endpoint, step_limit, and session_timeout_seconds.
    return {
        "artifact": ...,
        "transcript": [...],
    }
```

The agent result must be a JSON-serializable dictionary containing a JSON-serializable `transcript` list. This allows the runner to preserve the exact adapter result before evaluation.

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

Run with:

```text
python3 experiments/run_experiment.py --config experiments/config.example.json
```

Use `"mode": "memory_off"` for the no-memory control and `"mode": "chipmem"` for retrieval and PASS-only continual learning.

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

The separate state directory contains complete immutable skill files and cached task embeddings.

## Skill format

```text
state/agents/<domain>/memory/skill_0001/
├── SKILL.md
├── skill.json
└── embeddings.json
```

`SKILL.md` contains the complete reusable procedure. `skill.json` records the source-task label, exact source-artifact SHA-256, and verified-PASS provenance. `embeddings.json` stores the vector carried by the task-bound `TaskEmbedding` handle, not an embedding of the skill text.

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

## Synthetic adapters

`experiments/example_plugins.py`, `dataset/synthetic/`, and `examples/tasks/` contain deterministic privacy-safe fixtures used by examples and tests. They demonstrate the adapter signatures and dataset layout but are not benchmark agents, paper datasets, results, or evaluation tools.

## Release scope

This repository intentionally excludes:

- Existing experiment results or paper tables
- Raw session transcripts
- Existing skill libraries and embeddings
- Bayesian state
- Proprietary agent implementations and prompts
- Private or commercial datasets
- Commercial EDA scripts and licenses
- Internal endpoints, hostnames, credentials, and absolute project paths
- Paper source and figures
- Bash scripts
