
import pytest

from chipmem.statistical.model import (
    AgentModel,
    Feat,
    HierarchicalRetry,
    RecoveryAdvisor,
    retry_bucket,
)


def test_retry_bucket_matches_executed_method():
    assert [retry_bucket(value) for value in (0, 1, 2, 3, 4, 20)] == [
        0,
        1,
        2,
        3,
        3,
        3,
    ]


def test_hierarchical_retry_matches_reference_predictions():
    feature = Feat(
        ct="yosys_synth",
        rc=1,
        pr="fail",
        pec="timing-fail",
        te=3,
    )
    model = HierarchicalRetry(parent_strength=8.0)

    assert model.predict(feature) == (0.5, 0)
    model.update(feature, 0)
    probability, count = model.predict(feature)
    assert probability == pytest.approx(0.23411065386374025)
    assert count == 1
    model.update(feature, 1)
    assert model.predict(feature) == pytest.approx((0.5, 2))


def test_recovery_ranking_matches_reference_probabilities():
    advisor = RecoveryAdvisor(parent_strength=8.0)
    advisor.update(
        "timing-fail",
        "synth",
        "inspect_logs",
        1,
        excerpt="read first error",
    )
    advisor.update("timing-fail", "synth", "unchanged_retry", 0)

    ranked = advisor.rank(
        "timing-fail",
        "synth",
        {
            "inspect_logs": "inspect",
            "unchanged_retry": "retry",
        },
    )

    assert [row["strategy"] for row in ranked] == [
        "inspect_logs",
        "unchanged_retry",
    ]
    assert ranked[0]["probability"] == pytest.approx(0.6488340192043895)
    assert ranked[1]["probability"] == pytest.approx(0.3511659807956104)
    assert ranked[0]["excerpts"] == ["read first error"]


def test_model_round_trip_preserves_counts_and_parameters():
    model = AgentModel()
    feature = Feat(ct="synth", rc=2, pr="fail", pec="timing", te=4)
    model.retry.update(feature, 1)
    model.recovery.update("timing", "synth", "inspect", 1, excerpt="evidence")
    model.metadata["training_rows"] = 1

    restored = AgentModel.from_dict(model.to_dict())

    assert restored.to_dict() == model.to_dict()


def test_model_rejects_invalid_success_label():
    model = HierarchicalRetry()
    with pytest.raises(ValueError, match="0 or 1"):
        model.update(Feat(ct="synth"), 2)


def test_sequential_failure_probabilities_match_reference():
    model = HierarchicalRetry(parent_strength=8.0)
    features = [
        Feat(ct="synth", rc=0, pr="none", pec="none", te=0),
        Feat(ct="synth", rc=1, pr="fail", pec="failure", te=1),
        Feat(ct="synth", rc=2, pr="fail", pec="failure", te=2),
    ]
    probabilities = []
    for feature in features:
        probabilities.append(model.predict(feature)[0])
        model.update(feature, 0)

    assert probabilities == pytest.approx(
        [0.5, 0.2962962962962963, 0.17777777777777778]
    )


def test_parent_strength_grid_selection_matches_reference():
    rows = [
        (Feat(ct="synth", rc=index % 5, pr="fail", pec="failure", te=index), index % 2)
        for index in range(400)
    ]
    model = HierarchicalRetry(parent_strength=8.0)

    assert model.retune_parent_strength(rows) == 1.0


def test_parent_strength_retains_on_tuning_error():
    model = HierarchicalRetry(parent_strength=8.0)
    rows = [(Feat(ct="synth"), 1)] * 40
    assert model.retune_parent_strength(rows, holdout_fraction=0.0) == 8.0


def test_parent_strength_is_unchanged_for_thin_data():
    model = HierarchicalRetry(parent_strength=8.0)
    rows = [(Feat(ct="synth"), 1)] * 39
    assert model.retune_parent_strength(rows) == 8.0
