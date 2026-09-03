from __future__ import annotations

import hashlib
import math
from pathlib import Path


def embed(document: str, dimension: int) -> list[float]:
    """Deterministic synthetic embedding used only by tests and examples."""
    values = [0.0] * dimension
    for index, byte in enumerate(hashlib.sha256(document.encode()).digest()):
        values[index % dimension] += byte / 255.0
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def agent(
    task_document: str,
    memory_context: str,
    task_directory: Path,
    session_directory: Path,
    execution: dict,
    hooks=None,
) -> dict:
    tool_trace = []
    advice = []
    arguments = {"script": "synth"}
    if hooks is not None:
        for output in (
            "command failed with return code 1",
            "command failed with return code 1",
            "synthesis completed",
        ):
            note = hooks.before_tool_call("yosys", arguments)
            if note:
                advice.append(note)
            hooks.after_tool_call("yosys", arguments, output)
            tool_trace.append(
                {"tool": "yosys", "arguments": arguments, "output": output}
            )
    return {
        "advice": advice,
        "artifact": task_document,
        "execution": execution,
        "transcript": [
            {"role": "user", "content": task_document},
            {"role": "memory", "content": memory_context},
            {"role": "assistant", "content": "synthetic completion"},
        ],
        "tool_trace": tool_trace,
    }


def legacy_agent(
    task_document: str,
    memory_context: str,
    task_directory: Path,
    session_directory: Path,
    execution: dict,
) -> dict:
    return agent(
        task_document,
        memory_context,
        task_directory,
        session_directory,
        execution,
    )


def harness(
    task_document: str,
    agent_result: dict,
    task_directory: Path,
    session_directory: Path,
) -> str:
    return "pass" if task_document.startswith("PASS") else "fail"


def distill(transcript: list[dict], verdict: str) -> str:
    return "## DO\n- Reuse only strategies accepted by the final harness.\n\n## AVOID\n- Treating agent self-report as verification.\n"
