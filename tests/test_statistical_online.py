import json

import pytest

import chipmem.statistical.online as online_module
from chipmem.statistical.features import OpenFlowFeatures
from chipmem.statistical.model import AgentModel, Feat
from chipmem.statistical.nudge import WarningGenerator
from chipmem.statistical.online import StatisticalMemory


def test_warning_matches_executed_reference_text():
    feature = Feat(
        ct="yosys_synth",
        rc=1,
        pr="fail",
        pec="timing-fail",
        te=3,
    )
    model = AgentModel()
    for _ in range(20):
        model.retry.update(feature, 0)

    warning = WarningGenerator(
        model,
        OpenFlowFeatures(),
        threshold=0.20,
    ).maybe_warn(feature, stage="synth")

    assert warning == (
        "You have run yosys synthesis 1 times, failing with timing-fail each "
        "time. Historically, retries in this situation succeed only 0 percent "
        "of the time. Consider a different approach."
    )


def test_warning_requires_a_failed_previous_same_type_call():
    generator = WarningGenerator(AgentModel(), OpenFlowFeatures(), threshold=1.0)
    assert generator.maybe_warn(Feat(ct="synth", rc=0, pr="none")) is None
    assert generator.maybe_warn(Feat(ct="synth", rc=2, pr="success")) is None


def test_exact_threshold_does_not_warn():
    class FixedRetry:
        @staticmethod
        def predict(_feature):
            return 0.20, 10

    model = AgentModel()
    model.retry = FixedRetry()
    generator = WarningGenerator(model, OpenFlowFeatures(), threshold=0.20)
    assert generator.maybe_warn(
        Feat(ct="yosys_synth", rc=2, pr="fail", pec="failure")
    ) is None


def test_planner_contract_and_fallback():
    feature = Feat(ct="yosys_synth", rc=2, pr="fail", pec="failure")
    model = AgentModel()
    for _ in range(20):
        model.retry.update(feature, 0)
    model.recovery.update("failure", "synth", "inspect_logs", 1, "evidence")
    valid = WarningGenerator(
        model,
        OpenFlowFeatures(),
        threshold=0.20,
        planner=lambda _prompt: (
            "1. Inspect the first error.\n"
            "SUCCESS CHECK: the next call succeeds\n"
            "STOP IF: the same failure repeats"
        ),
    )
    invalid = WarningGenerator(
        model,
        OpenFlowFeatures(),
        threshold=0.20,
        planner=lambda _prompt: "unstructured advice",
    )

    assert valid.maybe_warn(feature, stage="synth").startswith("Recovery plan")
    assert invalid.maybe_warn(feature, stage="synth").startswith("You have run")


def test_default_warning_count_is_not_capped_at_three(tmp_path):
    memory = StatisticalMemory(
        tmp_path / "state", "example", OpenFlowFeatures(), threshold=0.20
    )
    for _ in range(6):
        arguments = {"script": "synth"}
        memory.before_tool_call("session", "yosys", arguments)
        memory.after_tool_call(
            "session",
            "yosys",
            arguments,
            "command failed with return code 1",
        )

    assert memory.nudge_count == 4


def test_online_step_updates_once_and_persists(tmp_path):
    memory = StatisticalMemory(
        tmp_path / "state",
        "example",
        OpenFlowFeatures(),
        threshold=0.20,
    )
    session = "synthetic-session"

    first_advice = memory.before_tool_call(
        session,
        "yosys",
        {"script": "synth"},
    )
    memory.after_tool_call(
        session,
        "yosys",
        {"script": "synth"},
        "command failed with return code 1",
    )
    retry_advice = memory.before_tool_call(
        session,
        "yosys",
        {"script": "synth"},
    )
    assert retry_advice is None
    memory.after_tool_call(
        session,
        "yosys",
        {"script": "synth"},
        "command failed with return code 1",
    )
    third_advice = memory.before_tool_call(
        session,
        "yosys",
        {"script": "synth"},
    )
    memory.after_tool_call(
        session,
        "yosys",
        {"script": "synth"},
        "synthesis completed",
    )

    assert first_advice is None
    assert third_advice is not None
    assert memory.update_count == 3
    experience = [
        json.loads(line)
        for line in memory.experience_path.read_text().splitlines()
    ]
    assert len(experience) == 3
    assert [row["step"] for row in experience] == [0, 1, 2]
    assert memory.model_path.is_file()

    memory.reconcile(session)
    assert memory.update_count == 3
    assert len(memory.experience_path.read_text().splitlines()) == 3


def test_recovery_success_and_right_censoring(tmp_path):
    recovered = StatisticalMemory(
        tmp_path / "recovered", "example", OpenFlowFeatures()
    )
    calls = [
        ("iverilog", {"files": ["design.v"]}, "syntax error near line 4"),
        ("read_file", {"path": "compile.log"}, "missing semicolon"),
        ("iverilog", {"files": ["design.v"]}, "compile completed"),
    ]
    for tool, arguments, output in calls:
        recovered.before_tool_call("session", tool, arguments)
        recovered.after_tool_call("session", tool, arguments, output)
    rows = [
        json.loads(line)
        for line in recovered.recovery_path.read_text().splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["strategy"] == "inspect_logs"
    assert rows[0]["recovered"] == 1

    censored = StatisticalMemory(
        tmp_path / "censored", "example", OpenFlowFeatures()
    )
    for tool, arguments, output in calls[:2]:
        censored.before_tool_call("session", tool, arguments)
        censored.after_tool_call("session", tool, arguments, output)
    assert not censored.recovery_path.read_text().strip()


def test_unrecovered_strategy_is_negative_only_after_full_horizon(tmp_path):
    memory = StatisticalMemory(
        tmp_path / "state", "example", OpenFlowFeatures()
    )
    failure = ("iverilog", {"files": ["design.v"]}, "syntax error")
    memory.before_tool_call("session", failure[0], failure[1])
    memory.after_tool_call("session", *failure)
    for index in range(8):
        arguments = {"path": f"log-{index}"}
        memory.before_tool_call("session", "read_file", arguments)
        memory.after_tool_call("session", "read_file", arguments, "inspection")

    rows = [
        json.loads(line)
        for line in memory.recovery_path.read_text().splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["strategy"] == "inspect_logs"
    assert rows[0]["recovered"] == 0


def test_transaction_recovers_after_interrupted_apply(tmp_path, monkeypatch):
    memory = StatisticalMemory(
        tmp_path / "state", "example", OpenFlowFeatures()
    )
    memory.before_tool_call("session", "yosys", {"script": "synth"})

    def crash(_transaction):
        raise RuntimeError("injected crash")

    monkeypatch.setattr(memory, "_apply_transaction", crash)
    with pytest.raises(RuntimeError, match="injected crash"):
        memory.after_tool_call(
            "session", "yosys", {"script": "synth"}, "synthesis completed"
        )
    assert memory.transaction_path.is_file()

    recovered = StatisticalMemory(
        tmp_path / "state", "example", OpenFlowFeatures()
    )
    assert recovered.update_count == 1
    assert recovered.model.retry.counts["global"] == [1.0, 0.0]
    assert not recovered.transaction_path.exists()


def test_transaction_rejects_checksum_mismatch(tmp_path):
    import hashlib

    memory = StatisticalMemory(
        tmp_path / "state", "example", OpenFlowFeatures()
    )
    model = b'{}'
    experience = b''
    recovery = b''
    transaction = {
        "version": 1,
        "model": model.decode(),
        "experience": experience.decode(),
        "recovery": recovery.decode(),
        "model_sha256": hashlib.sha256(model).hexdigest(),
        "experience_sha256": hashlib.sha256(experience).hexdigest(),
        "recovery_sha256": "0" * 64,
    }
    memory.transaction_path.write_text(json.dumps(transaction))

    with pytest.raises(ValueError, match="checksum mismatch"):
        StatisticalMemory(tmp_path / "state", "example", OpenFlowFeatures())


def test_constructor_recovers_transaction_under_domain_lock(tmp_path, monkeypatch):
    lock_acquired = {"value": False}
    real_recover = StatisticalMemory._recover_transaction

    def observe_lock(_descriptor, _operation):
        lock_acquired["value"] = True

    def checked_recover(self):
        assert lock_acquired["value"], (
            "transaction recovery must occur while holding the domain lock"
        )
        return real_recover(self)

    monkeypatch.setattr(online_module.fcntl, "flock", observe_lock)
    monkeypatch.setattr(StatisticalMemory, "_recover_transaction", checked_recover)

    StatisticalMemory(tmp_path / "state", "example", OpenFlowFeatures())


def test_reconciliation_rejects_conflicting_duplicate_step(tmp_path):
    memory = StatisticalMemory(
        tmp_path / "state", "example", OpenFlowFeatures()
    )
    memory.before_tool_call("session", "yosys", {"script": "synth"})
    memory.after_tool_call(
        "session", "yosys", {"script": "synth"}, "synthesis completed"
    )

    with pytest.raises(ValueError, match="conflicting duplicate"):
        memory.reconcile(
            "session",
            [
                {
                    "tool": "yosys",
                    "arguments": {"script": "synth"},
                    "output": "command failed with return code 1",
                }
            ],
        )


def test_after_hook_requires_matching_before_hook(tmp_path):
    memory = StatisticalMemory(
        tmp_path / "state",
        "example",
        OpenFlowFeatures(),
    )
    with pytest.raises(RuntimeError, match="before_tool_call"):
        memory.after_tool_call("session", "yosys", {}, "completed")
