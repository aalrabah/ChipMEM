from __future__ import annotations

import re
from typing import ClassVar


class OpenFlowFeatures:
    stage_order: ClassVar[list[str]] = [
        "read_libs",
        "read_hdl",
        "elaborate",
        "synth",
        "report",
    ]

    @staticmethod
    def _shell_program(arguments) -> str:
        command = (
            arguments
            if isinstance(arguments, str)
            else str((arguments or {}).get("command", ""))
        )
        for token in command.strip().split():
            if "=" in token and not token.startswith(("/", ".")):
                continue
            if token in ("sudo", "env", "nohup", "time"):
                continue
            return token.rsplit("/", 1)[-1]
        return "shell"

    def command_type(self, tool: str, arguments) -> str:
        if tool == "equivalence":
            return "equivalence_check"
        if tool in {"yosys", "orfs_synth"}:
            if tool == "orfs_synth":
                return "yosys_synth"
            script = (
                arguments
                if isinstance(arguments, str)
                else str((arguments or {}).get("script", ""))
            )
            return "yosys_synth" if "synth" in script else "yosys_read"
        if tool == "sta":
            script = str(arguments)
            return "sta_power" if "power" in script.lower() else "sta_timing"
        if tool == "iverilog":
            return "iverilog_compile"
        if tool == "vvp":
            return "vvp_run"
        if tool == "shell":
            return f"sh_{self._shell_program(arguments)}"
        return tool

    @staticmethod
    def classify(tool: str, output_text: str) -> tuple[int, str]:
        low = (output_text or "").lower()
        if re.search(
            r"\b[a-z0-9_]+_observation:\s*area-non-improvement\b",
            low,
        ):
            return 0, "area-non-improvement"
        if re.search(
            r"\b[a-z0-9_]+_observation:\s*equivalence-fail\b",
            low,
        ):
            return 0, "equivalence-fail"
        if "syntax error" in low:
            return 0, "syntax-error"
        if "is not part of the design" in low or "unknown module" in low:
            return 0, "unknown-module"
        if "liberty" in low and ("cannot" in low or "error" in low):
            return 0, "liberty-missing"
        if "timing" in low and ("violated" in low or "fail" in low):
            return 0, "timing-fail"
        if "error" in low or "command failed with return code" in low:
            return 0, "rc-nonzero"
        return 1, "ok"

    @staticmethod
    def readable_command(command_type: str) -> str:
        readable = {
            "yosys_synth": "yosys synthesis",
            "yosys_read": "yosys read",
            "sta_timing": "timing analysis",
            "sta_power": "power analysis",
            "iverilog_compile": "iverilog compile",
            "vvp_run": "simulation",
            "equivalence_check": "RTL equivalence check",
        }
        if command_type.startswith("sh_"):
            return command_type[3:]
        return readable.get(command_type, command_type)

    @staticmethod
    def strategies() -> dict[str, str]:
        return {
            "absolute_hdl_paths": "Reissue the read with absolute file paths.",
            "verify_top_module": "Confirm the declared top module matches the sources.",
            "rebuild_filelist": "Rebuild the source file list from the workspace.",
            "inspect_logs": "Read the full log around the first error before editing.",
            "change_tool_setup": "Adjust tool options rather than the design.",
            "simplify_edit": "Return to the last passing artifact and apply a smaller edit.",
            "unchanged_retry": "Retry once when a transient failure is plausible.",
            "revert_non_improving_edit": (
                "Restore the last equivalent lower-area candidate or try a smaller edit."
            ),
            "inspect_equivalence_failure": (
                "Inspect widths, signedness, and control behavior before retrying."
            ),
        }

    @staticmethod
    def stage_of(command_type: str) -> str:
        return {
            "yosys_read": "read_hdl",
            "yosys_synth": "synth",
            "sta_timing": "report",
            "sta_power": "report",
            "iverilog_compile": "elaborate",
            "vvp_run": "report",
            "equivalence_check": "report",
        }.get(command_type, command_type)
