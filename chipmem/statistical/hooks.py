from __future__ import annotations

from chipmem.statistical.online import StatisticalMemory


class ToolHooks:
    """Enforce predict-before and update-after ordering for live tool calls."""

    def __init__(
        self,
        statistical_memory: StatisticalMemory | None,
        session_id: str,
        *,
        required: bool,
    ):
        self.statistical_memory = statistical_memory
        self.session_id = session_id
        self.required = bool(required)
        self.completed_calls = 0
        self._pending: tuple[str, object] | None = None

    @property
    def update_count(self) -> int:
        if self.statistical_memory is None:
            return 0
        return self.statistical_memory.update_count

    @property
    def nudge_count(self) -> int:
        if self.statistical_memory is None:
            return 0
        return self.statistical_memory.nudge_count

    def before_tool_call(self, tool: str, arguments) -> str | None:
        if self._pending is not None:
            raise RuntimeError("previous tool call has not completed")
        self._pending = (tool, arguments)
        if self.statistical_memory is None:
            return None
        return self.statistical_memory.before_tool_call(
            self.session_id,
            tool,
            arguments,
        )

    def after_tool_call(self, tool: str, arguments, output: str) -> None:
        if self._pending is None:
            raise RuntimeError("before_tool_call must run before after_tool_call")
        if self._pending != (tool, arguments):
            raise RuntimeError("completed tool call does not match pending call")
        if self.statistical_memory is not None:
            self.statistical_memory.after_tool_call(
                self.session_id,
                tool,
                arguments,
                output,
            )
        self._pending = None
        self.completed_calls += 1

    def finalize(self) -> None:
        if self._pending is not None:
            raise RuntimeError("session ended with an incomplete tool call")
        if self.statistical_memory is not None:
            self.statistical_memory.finalize_session(
                self.session_id,
                require_calls=self.required,
            )
        elif self.required:
            raise RuntimeError("statistical mode requires live tool-call hooks")
