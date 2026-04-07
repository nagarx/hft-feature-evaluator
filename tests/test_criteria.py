"""Tests for SelectionCriteria and select_features()."""

import math
import pytest
from dataclasses import FrozenInstanceError

from hft_evaluator.criteria import SelectionCriteria, select_features
from hft_evaluator.profile import FeatureProfile, StabilityDetail


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _stability(combined=0.8):
    return StabilityDetail(
        path1_stability=combined,
        path2_stability=0.0,
        path3a_stability=0.0,
        combined_stability=combined,
        path4_ci_coverage=0.0,
        path5_jmi_stability=0.0,
    )


def _profile(
    name="feat_a",
    passing=("linear_signal",),
    stability=0.8,
    best_value=0.15,
    best_p=0.001,
    cf_ratio=None,
    cf_class="forward",
    vif=None,
    max_corr=None,
    best_horizon=10,
):
    return FeatureProfile(
        feature_name=name,
        feature_index=0,
        best_horizon=best_horizon,
        best_metric="forward_ic",
        best_value=best_value,
        best_p=best_p,
        passing_paths=tuple(passing),
        stability=_stability(stability),
        concurrent_forward_ratio=cf_ratio,
        cf_classification=cf_class,
        redundancy_cluster_id=0,
        max_pairwise_correlation=max_corr,
        vif=vif,
        ic_acf_half_life=5,
    )


def _profiles():
    """Build a small profile set for testing."""
    return {
        "strong": _profile("strong", passing=("linear_signal", "nonlinear_signal"),
                           stability=1.0, best_p=0.001),
        "weak": _profile("weak", passing=("regime_conditional",),
                         stability=0.5, best_p=float("nan"), best_value=0.03),
        "contemp": _profile("contemp", passing=("linear_signal",),
                            cf_ratio=15.0, cf_class="contemporaneous"),
        "redundant": _profile("redundant", passing=("linear_signal",),
                              vif=12.0, max_corr=0.96),
        "noise": _profile("noise", passing=(), stability=0.0),
    }


# ---------------------------------------------------------------------------
# SelectionCriteria
# ---------------------------------------------------------------------------


class TestSelectionCriteria:
    def test_frozen(self):
        c = SelectionCriteria()
        with pytest.raises(FrozenInstanceError):
            c.name = "other"

    def test_defaults(self):
        c = SelectionCriteria()
        assert c.name == "default"
        assert c.min_passing_paths == 1
        assert c.min_combined_stability == 0.6
        assert c.required_paths == ()

    def test_custom_name(self):
        c = SelectionCriteria(name="momentum_hft")
        assert c.name == "momentum_hft"


# ---------------------------------------------------------------------------
# select_features
# ---------------------------------------------------------------------------


class TestSelectFeatures:
    def test_default_criteria_excludes_noise(self):
        profiles = _profiles()
        selected = select_features(profiles, SelectionCriteria())
        assert "noise" not in selected

    def test_default_criteria_includes_strong(self):
        profiles = _profiles()
        selected = select_features(profiles, SelectionCriteria())
        assert "strong" in selected

    def test_default_excludes_low_stability(self):
        profiles = _profiles()
        selected = select_features(profiles, SelectionCriteria())
        # "weak" has stability=0.5, below default 0.6
        assert "weak" not in selected

    def test_lower_stability_threshold(self):
        profiles = _profiles()
        c = SelectionCriteria(min_combined_stability=0.4)
        selected = select_features(profiles, c)
        assert "weak" in selected

    def test_cf_gating(self):
        profiles = _profiles()
        c = SelectionCriteria(max_cf_ratio=10.0)
        selected = select_features(profiles, c)
        assert "contemp" not in selected
        assert "strong" in selected

    def test_allowed_cf_classes(self):
        profiles = _profiles()
        c = SelectionCriteria(
            allowed_cf_classes=("forward", "partially_forward")
        )
        selected = select_features(profiles, c)
        assert "contemp" not in selected

    def test_vif_filter(self):
        profiles = _profiles()
        c = SelectionCriteria(max_vif=10.0)
        selected = select_features(profiles, c)
        assert "redundant" not in selected

    def test_max_pairwise_corr(self):
        profiles = _profiles()
        c = SelectionCriteria(max_pairwise_corr=0.95)
        selected = select_features(profiles, c)
        assert "redundant" not in selected

    def test_required_paths(self):
        profiles = _profiles()
        c = SelectionCriteria(
            required_paths=("linear_signal", "nonlinear_signal")
        )
        selected = select_features(profiles, c)
        assert selected == ["strong"]

    def test_allowed_horizons(self):
        profiles = {
            "h10": _profile("h10", best_horizon=10),
            "h60": _profile("h60", best_horizon=60),
        }
        c = SelectionCriteria(allowed_horizons=(60,))
        selected = select_features(profiles, c)
        assert selected == ["h60"]

    def test_explicit_include(self):
        profiles = _profiles()
        c = SelectionCriteria(include_names=("noise", "weak"))
        selected = select_features(profiles, c)
        assert set(selected) == {"noise", "weak"}

    def test_explicit_exclude(self):
        profiles = _profiles()
        c = SelectionCriteria(exclude_names=("strong",))
        selected = select_features(profiles, c)
        assert "strong" not in selected

    def test_sorted_output(self):
        profiles = _profiles()
        selected = select_features(profiles, SelectionCriteria())
        assert selected == sorted(selected)

    def test_min_abs_metric(self):
        profiles = _profiles()
        c = SelectionCriteria(min_abs_metric=0.10)
        selected = select_features(profiles, c)
        # "weak" has best_value=0.03, below 0.10
        assert "weak" not in selected

    def test_max_p_value(self):
        profiles = _profiles()
        c = SelectionCriteria(
            max_p_value=0.01, min_combined_stability=0.0
        )
        selected = select_features(profiles, c)
        assert "strong" in selected
        # "weak" has NaN p — isfinite check: NaN p doesn't fail max_p
        # (we don't penalize features without p-values)
