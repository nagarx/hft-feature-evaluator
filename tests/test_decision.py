"""Tests for decision logic: classify_feature, compute_best_p, tier boundaries."""

import pytest
from hft_evaluator.decision import (
    Tier, PathResult, classify_feature, compute_best_p,
)
from hft_evaluator.config import EvaluationConfig


@pytest.fixture
def config():
    return EvaluationConfig.from_dict({
        "export_dir": "/tmp",
        "stability": {
            "stable_threshold": 0.6,
            "investigate_threshold": 0.4,
        },
        "classification": {
            "strong_keep_p": 0.01,
        },
    })


class TestClassifyFeature:
    def test_discard_no_paths(self, config):
        assert classify_feature([], 0.8, 0.001, True, config) == Tier.DISCARD

    def test_discard_low_stability(self, config):
        assert classify_feature(
            ["linear_signal"], 0.3, 0.001, True, config
        ) == Tier.DISCARD

    def test_investigate_mid_stability(self, config):
        assert classify_feature(
            ["linear_signal"], 0.5, 0.001, True, config
        ) == Tier.INVESTIGATE

    def test_keep_stable_high_p(self, config):
        assert classify_feature(
            ["linear_signal"], 0.7, 0.03, True, config
        ) == Tier.KEEP

    def test_strong_keep(self, config):
        assert classify_feature(
            ["linear_signal"], 0.8, 0.005, True, config
        ) == Tier.STRONG_KEEP

    def test_strong_keep_fails_holdout_downgrade(self, config):
        """STRONG-KEEP candidate that fails holdout → KEEP."""
        assert classify_feature(
            ["linear_signal"], 0.8, 0.005, False, config
        ) == Tier.KEEP

    def test_phase2_no_stability(self, config):
        """stability_pct=None → skip stability checks."""
        assert classify_feature(
            ["linear_signal"], None, 0.005, True, config
        ) == Tier.STRONG_KEEP

    def test_phase2_no_stability_no_paths(self, config):
        assert classify_feature([], None, 1.0, False, config) == Tier.DISCARD

    def test_phase2_no_stability_high_p(self, config):
        assert classify_feature(
            ["linear_signal"], None, 0.1, False, config
        ) == Tier.KEEP

    def test_boundary_stability_exact_60pct(self, config):
        """stability=0.6 exactly (== stable_threshold) → NOT INVESTIGATE."""
        # 0.6 is NOT < 0.6, so it doesn't trigger INVESTIGATE
        result = classify_feature(
            ["linear_signal"], 0.6, 0.03, True, config
        )
        assert result == Tier.KEEP

    def test_boundary_stability_exact_40pct(self, config):
        """stability=0.4 exactly (== investigate_threshold) → NOT DISCARD."""
        # 0.4 is NOT < 0.4, so it doesn't trigger DISCARD
        result = classify_feature(
            ["linear_signal"], 0.4, 0.03, True, config
        )
        assert result == Tier.INVESTIGATE

    def test_boundary_stability_just_below_40(self, config):
        result = classify_feature(
            ["linear_signal"], 0.39, 0.03, True, config
        )
        assert result == Tier.DISCARD


class TestComputeBestP:
    def test_multiple_passing(self):
        results = [
            PathResult("a", 1, "ic", 0.1, 0.02, 0.0, 0.2, True),
            PathResult("b", 1, "ic", 0.2, 0.001, 0.1, 0.3, True),
            PathResult("c", 1, "ic", 0.05, 0.05, -0.1, 0.2, True),
        ]
        assert compute_best_p(results) == 0.001

    def test_no_passing(self):
        results = [
            PathResult("a", 1, "ic", 0.01, 0.5, -0.1, 0.1, False),
            PathResult("b", 1, "ic", 0.02, 0.3, -0.1, 0.1, False),
        ]
        assert compute_best_p(results) == 1.0

    def test_mixed_passing(self):
        results = [
            PathResult("a", 1, "ic", 0.1, 0.02, 0.0, 0.2, True),
            PathResult("b", 1, "ic", 0.01, 0.5, -0.1, 0.1, False),
        ]
        assert compute_best_p(results) == 0.02

    def test_empty_results(self):
        assert compute_best_p([]) == 1.0


class TestTierEnum:
    def test_values(self):
        assert Tier.STRONG_KEEP.value == "STRONG-KEEP"
        assert Tier.KEEP.value == "KEEP"
        assert Tier.INVESTIGATE.value == "INVESTIGATE"
        assert Tier.DISCARD.value == "DISCARD"

    def test_string_comparison(self):
        assert Tier.STRONG_KEEP == "STRONG-KEEP"
        assert Tier.DISCARD == "DISCARD"
