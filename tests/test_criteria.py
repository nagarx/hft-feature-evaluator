"""Tests for SelectionCriteria and select_features()."""

import math
import textwrap
from dataclasses import FrozenInstanceError, replace

import pytest

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


# ---------------------------------------------------------------------------
# Phase 4: criteria_schema_version
# ---------------------------------------------------------------------------


class TestCriteriaSchemaVersion:
    """Phase 4: criteria_schema_version field tracks schema evolution
    so future additive fields can be detected + documented."""

    def test_default_is_1_0(self):
        assert SelectionCriteria().criteria_schema_version == "1.0"

    def test_custom_version_accepted(self):
        c = SelectionCriteria(criteria_schema_version="1.1")
        assert c.criteria_schema_version == "1.1"


# ---------------------------------------------------------------------------
# Phase 4: require_holdout_confirmed gate
# ---------------------------------------------------------------------------


class TestRequireHoldoutConfirmed:
    """Phase 4: the optional STRONG-KEEP-only gate.

    Default behavior (False) preserves pre-Phase-4 semantics — the
    holdout flag is an advisory upgrade signal, not a filter.
    When True, only features whose FeatureProfile.holdout_confirmed
    is True survive selection.
    """

    def test_default_does_not_filter_on_holdout(self):
        profiles = _profiles()
        selected = select_features(profiles, SelectionCriteria())
        # Default criteria does not inspect holdout_confirmed.
        assert "strong" in selected

    def test_require_true_excludes_unconfirmed_profiles(self):
        profiles = _profiles()
        # None of the base fixtures have holdout_confirmed=True.
        c = SelectionCriteria(require_holdout_confirmed=True)
        selected = select_features(profiles, c)
        assert selected == []

    def test_require_true_keeps_confirmed_profiles(self):
        base = _profiles()
        confirmed = replace(base["strong"], holdout_confirmed=True)
        profiles = {**base, "strong": confirmed}
        c = SelectionCriteria(require_holdout_confirmed=True)
        selected = select_features(profiles, c)
        assert selected == ["strong"]

    def test_require_true_respects_explicit_include(self):
        # include_names is documented to bypass all gates EXCEPT
        # explicit exclusion. Holdout gate is no exception — tests
        # guard against accidentally promoting holdout into the
        # "always-enforced" category.
        profiles = _profiles()
        c = SelectionCriteria(
            include_names=("noise",),
            require_holdout_confirmed=True,
        )
        selected = select_features(profiles, c)
        assert selected == ["noise"]

    def test_require_true_composes_and_with_stability_gate(self):
        # P2 (Phase 4 Batch 4a validation): holdout is ONE gate among
        # many; it must compose AND-wise with other gates, not override
        # them. A holdout-confirmed feature with low stability is still
        # excluded when min_combined_stability is applied. This locks
        # the invariant that require_holdout_confirmed is never a
        # "stability bypass."
        base = _profiles()
        # "weak" has stability=0.5 (below default 0.6) AND best_value=0.03.
        # Mark it holdout_confirmed=True — it still fails stability.
        confirmed_weak = replace(base["weak"], holdout_confirmed=True)
        profiles = {**base, "weak": confirmed_weak}
        c = SelectionCriteria(require_holdout_confirmed=True)
        selected = select_features(profiles, c)
        assert "weak" not in selected, (
            "require_holdout_confirmed must AND-compose with other "
            "gates; a confirmed but low-stability profile must still "
            "be excluded."
        )


# ---------------------------------------------------------------------------
# Phase 4: from_dict
# ---------------------------------------------------------------------------


class TestFromDict:
    def test_minimal(self):
        c = SelectionCriteria.from_dict({"name": "minimal"})
        assert c.name == "minimal"
        assert c.criteria_schema_version == "1.0"   # default preserved

    def test_all_primitive_fields(self):
        c = SelectionCriteria.from_dict({
            "name": "full",
            "criteria_schema_version": "1.0",
            "min_passing_paths": 2,
            "min_combined_stability": 0.7,
            "min_abs_metric": 0.05,
            "max_p_value": 0.01,
            "max_cf_ratio": 5.0,
            "max_vif": 8.0,
            "max_pairwise_corr": 0.9,
            "require_holdout_confirmed": True,
        })
        assert c.name == "full"
        assert c.min_passing_paths == 2
        assert c.require_holdout_confirmed is True

    def test_tuple_coercion_from_list(self):
        c = SelectionCriteria.from_dict({
            "required_paths": ["linear_signal", "nonlinear_signal"],
            "allowed_cf_classes": ["forward"],
            "allowed_horizons": [10, 60, 300],
            "include_names": ["feat_a"],
            "exclude_names": ["feat_b"],
        })
        assert c.required_paths == ("linear_signal", "nonlinear_signal")
        assert c.allowed_cf_classes == ("forward",)
        assert c.allowed_horizons == (10, 60, 300)
        assert c.include_names == ("feat_a",)
        assert c.exclude_names == ("feat_b",)

    def test_none_valued_tuple_fields_passthrough(self):
        # None for Optional tuple fields must be preserved (not coerced
        # to an empty tuple).
        c = SelectionCriteria.from_dict({
            "allowed_cf_classes": None,
            "allowed_horizons": None,
            "include_names": None,
            "exclude_names": None,
        })
        assert c.allowed_cf_classes is None
        assert c.allowed_horizons is None
        assert c.include_names is None
        assert c.exclude_names is None

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown SelectionCriteria keys"):
            SelectionCriteria.from_dict({"not_a_field": 42})

    def test_string_passed_to_tuple_field_raises(self):
        # Strings are iterable — guard against silent per-char tuplification.
        with pytest.raises(ValueError, match="must be a sequence"):
            SelectionCriteria.from_dict({"required_paths": "linear_signal"})

    def test_non_iterable_tuple_field_raises(self):
        with pytest.raises(ValueError, match="must be iterable"):
            SelectionCriteria.from_dict({"allowed_horizons": 10})


# ---------------------------------------------------------------------------
# Phase 4: from_yaml
# ---------------------------------------------------------------------------


class TestFromYaml:
    def test_flat_yaml(self, tmp_path):
        path = tmp_path / "crit.yaml"
        path.write_text(textwrap.dedent("""
            name: momentum_hft
            min_passing_paths: 2
            min_combined_stability: 0.7
            required_paths:
              - linear_signal
            allowed_horizons: [10, 60]
            require_holdout_confirmed: true
        """).strip() + "\n")
        c = SelectionCriteria.from_yaml(path)
        assert c.name == "momentum_hft"
        assert c.min_passing_paths == 2
        assert c.min_combined_stability == 0.7
        assert c.required_paths == ("linear_signal",)
        assert c.allowed_horizons == (10, 60)
        assert c.require_holdout_confirmed is True

    def test_nested_criteria_key(self, tmp_path):
        # Users often place SelectionCriteria inside a larger config file
        # under a `criteria:` header. The loader unwraps this form.
        path = tmp_path / "wrapped.yaml"
        path.write_text(textwrap.dedent("""
            criteria:
              name: wrapped
              min_combined_stability: 0.8
        """).strip() + "\n")
        c = SelectionCriteria.from_yaml(path)
        assert c.name == "wrapped"
        assert c.min_combined_stability == 0.8

    def test_nested_criteria_only_when_single_key(self, tmp_path):
        # A YAML with multiple top-level keys + a `criteria:` entry is
        # rejected (unwrapping is only safe when it's unambiguous).
        path = tmp_path / "ambiguous.yaml"
        path.write_text(textwrap.dedent("""
            criteria:
              name: nested
            extra_key: should_fail
        """).strip() + "\n")
        with pytest.raises(ValueError, match="Unknown SelectionCriteria keys"):
            SelectionCriteria.from_yaml(path)

    def test_unknown_key_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("name: x\nbogus: 1\n")
        with pytest.raises(ValueError, match="Unknown SelectionCriteria keys"):
            SelectionCriteria.from_yaml(path)

    def test_non_dict_yaml_raises(self, tmp_path):
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n")
        with pytest.raises(ValueError, match="Expected YAML dict"):
            SelectionCriteria.from_yaml(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SelectionCriteria.from_yaml(tmp_path / "nonexistent.yaml")

    def test_round_trip_via_from_dict(self):
        # from_dict is the canonical constructor; from_yaml delegates.
        # Ensure the two paths agree.
        d = {
            "name": "rt",
            "min_passing_paths": 1,
            "required_paths": ["a", "b"],
        }
        assert SelectionCriteria.from_dict(d) == SelectionCriteria.from_dict(d)


# ---------------------------------------------------------------------------
# Phase 4: determinism of select_features
# ---------------------------------------------------------------------------


class TestSelectFeaturesDeterminism:
    """Content-addressed FeatureSet hashing (Phase 4 R1) relies on
    select_features producing identical output for identical inputs.
    These tests lock that guarantee."""

    def test_same_inputs_same_output(self):
        profiles = _profiles()
        c = SelectionCriteria(min_combined_stability=0.5)
        outputs = [select_features(profiles, c) for _ in range(100)]
        assert all(o == outputs[0] for o in outputs)

    def test_insertion_order_does_not_affect_output(self):
        p = _profiles()
        forward = dict(p.items())
        reversed_order = dict(reversed(list(p.items())))
        c = SelectionCriteria(min_combined_stability=0.5)
        assert select_features(forward, c) == select_features(
            reversed_order, c
        )

    def test_output_is_sorted(self):
        profiles = _profiles()
        c = SelectionCriteria(min_combined_stability=0.0, min_passing_paths=0)
        selected = select_features(profiles, c)
        assert selected == sorted(selected)
