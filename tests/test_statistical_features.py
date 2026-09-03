from chipmem.statistical.extraction import extract_rows, pre_call_feat
from chipmem.statistical.features import OpenFlowFeatures
from chipmem.statistical.model import Feat


def synthetic_transcript():
    return [
        {
            "tool": "yosys",
            "arguments": {"script": "read"},
            "output": "syntax error",
        },
        {
            "tool": "yosys",
            "arguments": {"script": "synth"},
            "output": "command failed with return code 1",
        },
        {
            "tool": "yosys",
            "arguments": {"script": "synth"},
            "output": "synthesis completed",
        },
    ]


def test_feature_rows_match_executed_reference():
    rows = extract_rows(synthetic_transcript(), OpenFlowFeatures())

    assert rows == [
        {
            "feat": Feat(ct="yosys_read", rc=0, pr="none", pec="none", te=0),
            "success": 0,
            "error_class": "syntax-error",
            "cmd_type": "yosys_read",
        },
        {
            "feat": Feat(ct="yosys_synth", rc=0, pr="none", pec="none", te=1),
            "success": 0,
            "error_class": "rc-nonzero",
            "cmd_type": "yosys_synth",
        },
        {
            "feat": Feat(
                ct="yosys_synth",
                rc=1,
                pr="fail",
                pec="rc-nonzero",
                te=2,
            ),
            "success": 1,
            "error_class": "ok",
            "cmd_type": "yosys_synth",
        },
    ]


def test_pre_call_feature_matches_reference():
    feature = pre_call_feat(
        synthetic_transcript(),
        "yosys",
        {"script": "synth"},
        OpenFlowFeatures(),
    )

    assert feature == Feat(
        ct="yosys_synth",
        rc=2,
        pr="success",
        pec="ok",
        te=3,
    )


def test_structured_objective_markers_are_prefix_agnostic():
    features = OpenFlowFeatures()
    for prefix in ("CHIPMEM", "LEGACY_PIPELINE"):
        assert features.classify(
            "yosys", f"{prefix}_OBSERVATION: area-non-improvement"
        ) == (0, "area-non-improvement")
        assert features.classify(
            "equivalence", f"{prefix}_OBSERVATION: equivalence-fail"
        ) == (0, "equivalence-fail")


def test_rtl_optimization_objective_failures_are_observable():
    features = OpenFlowFeatures()
    assert features.command_type("equivalence", {}) == "equivalence_check"
    assert features.stage_of("equivalence_check") == "report"
    assert features.classify(
        "yosys", "CHIPMEM_OBSERVATION: area-non-improvement"
    ) == (0, "area-non-improvement")
    assert features.classify(
        "equivalence", "CHIPMEM_OBSERVATION: equivalence-fail"
    ) == (0, "equivalence-fail")
    assert "revert_non_improving_edit" in features.strategies()
    assert "inspect_equivalence_failure" in features.strategies()


def test_open_flow_classification_and_stages():
    features = OpenFlowFeatures()
    assert features.classify("yosys", "timing violated") == (0, "timing-fail")
    assert features.classify("yosys", "completed") == (1, "ok")
    assert features.stage_of("yosys_synth") == "synth"
    assert "inspect_logs" in features.strategies()
