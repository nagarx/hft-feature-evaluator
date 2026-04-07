"""Tests for FeatureProfile, PathEvidence, StabilityDetail, compute_tier."""

import math
import pytest
from dataclasses import FrozenInstanceError

from hft_evaluator.profile import (
    FeatureProfile,
    PathEvidence,
    StabilityDetail,
    compute_tier,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_stability(combined=0.8, p1=0.8, p2=0.0, p3a=0.0, p4=0.0, p5=0.0):
    return StabilityDetail(
        path1_stability=p1,
        path2_stability=p2,
        path3a_stability=p3a,
        combined_stability=combined,
        path4_ci_coverage=p4,
        path5_jmi_stability=p5,
    )


def _make_profile(
    name="test_feature",
    index=0,
    passing_paths=("linear_signal",),
    best_p=0.001,
    stability=None,
    cf_ratio=None,
    cf_class=None,
    holdout=False,
    vif=None,
    max_corr=None,
):
    return FeatureProfile(
        feature_name=name,
        feature_index=index,
        best_horizon=10,
        best_metric="forward_ic",
        best_value=0.15,
        best_p=best_p,
        passing_paths=tuple(passing_paths),
        stability=stability or _make_stability(),
        concurrent_forward_ratio=cf_ratio,
        cf_classification=cf_class,
        redundancy_cluster_id=0,
        max_pairwise_correlation=max_corr,
        vif=vif,
        ic_acf_half_life=5,
        holdout_confirmed=holdout,
    )


# ---------------------------------------------------------------------------
# PathEvidence
# ---------------------------------------------------------------------------


class TestPathEvidence:
    def test_frozen(self):
        ev = PathEvidence(
            path_name="linear_signal", horizon=10, metric_name="forward_ic",
            metric_value=0.15, p_value=0.001, ci_lower=0.05, ci_upper=0.25,
            passes=True, is_informational=False,
        )
        with pytest.raises(FrozenInstanceError):
            ev.passes = False

    def test_informational_flag(self):
        ev = PathEvidence(
            path_name="temporal_value", horizon=10, metric_name="te_L1",
            metric_value=0.02, p_value=0.5, ci_lower=float("nan"),
            ci_upper=float("nan"), passes=False, is_informational=True,
        )
        assert ev.is_informational is True


# ---------------------------------------------------------------------------
# StabilityDetail
# ---------------------------------------------------------------------------


class TestStabilityDetail:
    def test_frozen(self):
        sd = _make_stability()
        with pytest.raises(FrozenInstanceError):
            sd.combined_stability = 0.0

    def test_combined_is_max(self):
        sd = StabilityDetail(
            path1_stability=0.3, path2_stability=0.7,
            path3a_stability=0.5, combined_stability=0.7,
            path4_ci_coverage=0.0, path5_jmi_stability=0.0,
        )
        assert sd.combined_stability == max(
            sd.path1_stability, sd.path2_stability, sd.path3a_stability
        )


# ---------------------------------------------------------------------------
# FeatureProfile
# ---------------------------------------------------------------------------


class TestFeatureProfile:
    def test_frozen(self):
        p = _make_profile()
        with pytest.raises(FrozenInstanceError):
            p.best_p = 0.5

    def test_default_holdout_false(self):
        p = _make_profile()
        assert p.holdout_confirmed is False

    def test_default_all_evidence_empty(self):
        p = _make_profile()
        assert p.all_evidence == ()


# ---------------------------------------------------------------------------
# compute_tier
# ---------------------------------------------------------------------------


class TestComputeTier:
    def test_no_passing_paths_discard(self):
        p = _make_profile(passing_paths=())
        assert compute_tier(p) == "DISCARD"

    def test_low_stability_discard(self):
        p = _make_profile(stability=_make_stability(combined=0.3))
        assert compute_tier(p) == "DISCARD"

    def test_mid_stability_investigate(self):
        p = _make_profile(stability=_make_stability(combined=0.5))
        assert compute_tier(p) == "INVESTIGATE"

    def test_high_stability_keep(self):
        p = _make_profile(stability=_make_stability(combined=0.8))
        assert compute_tier(p) == "KEEP"

    def test_strong_keep_requires_holdout(self):
        p = _make_profile(
            best_p=0.001,
            stability=_make_stability(combined=0.8),
            holdout=False,
        )
        assert compute_tier(p) == "KEEP"

    def test_strong_keep_with_holdout(self):
        p = _make_profile(
            best_p=0.001,
            stability=_make_stability(combined=0.8),
            holdout=True,
        )
        assert compute_tier(p) == "STRONG-KEEP"

    def test_nan_p_prevents_strong_keep(self):
        p = _make_profile(
            best_p=float("nan"),
            stability=_make_stability(combined=0.8),
            holdout=True,
        )
        assert compute_tier(p) == "KEEP"

    def test_boundary_stability_060_is_keep(self):
        p = _make_profile(stability=_make_stability(combined=0.6))
        assert compute_tier(p) == "KEEP"

    def test_boundary_stability_040_is_investigate(self):
        p = _make_profile(stability=_make_stability(combined=0.4))
        assert compute_tier(p) == "INVESTIGATE"

    def test_custom_thresholds(self):
        p = _make_profile(stability=_make_stability(combined=0.5))
        assert compute_tier(p, stable_threshold=0.5) == "KEEP"
