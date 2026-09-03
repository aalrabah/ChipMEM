import pytest

from chipmem.statistical.features import OpenFlowFeatures
from chipmem.statistical.hooks import ToolHooks
from chipmem.statistical.online import StatisticalMemory


def test_hooks_enforce_before_execute_after_order(tmp_path):
    statistical = StatisticalMemory(
        tmp_path / "state",
        "example",
        OpenFlowFeatures(),
        threshold=0.30,
    )
    hooks = ToolHooks(statistical, "session", required=True)

    advice = hooks.before_tool_call("yosys", {"script": "synth"})
    assert advice is None
    hooks.after_tool_call(
        "yosys",
        {"script": "synth"},
        "command failed with return code 1",
    )
    advice = hooks.before_tool_call("yosys", {"script": "synth"})
    assert advice is not None
    hooks.after_tool_call("yosys", {"script": "synth"}, "completed")
    hooks.finalize()

    assert hooks.completed_calls == 2
    assert hooks.update_count == 2
    assert hooks.nudge_count == 1


def test_hooks_reject_after_without_before(tmp_path):
    statistical = StatisticalMemory(
        tmp_path / "state", "example", OpenFlowFeatures()
    )
    hooks = ToolHooks(statistical, "session", required=True)
    with pytest.raises(RuntimeError, match="before_tool_call"):
        hooks.after_tool_call("yosys", {}, "completed")


def test_statistical_mode_rejects_adapter_that_never_uses_hooks(tmp_path):
    statistical = StatisticalMemory(
        tmp_path / "state", "example", OpenFlowFeatures()
    )
    hooks = ToolHooks(statistical, "session", required=True)
    with pytest.raises(RuntimeError, match="requires live tool-call hooks"):
        hooks.finalize()


def test_disabled_hooks_have_no_statistical_activity():
    hooks = ToolHooks(None, "session", required=False)
    assert hooks.before_tool_call("yosys", {}) is None
    hooks.after_tool_call("yosys", {}, "completed")
    hooks.finalize()
    assert hooks.completed_calls == 1
    assert hooks.update_count == 0
    assert hooks.nudge_count == 0
