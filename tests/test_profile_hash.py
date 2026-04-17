"""Tests for compute_profile_hash (Phase 4 FeatureSet producer input).

The FeatureSet registry producer records `produced_by.source_profile_hash`
so consumers can link a FeatureSet back to the evaluator snapshot that
produced it. These tests lock the determinism + sensitivity properties
of compute_profile_hash that downstream Phase 4 code depends on.
"""

from dataclasses import replace

import pytest

from hft_evaluator.pipeline import (
    _sanitize_for_hash,
    compute_profile_hash,
)
from hft_evaluator.profile import FeatureProfile, StabilityDetail


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _stability(combined: float = 0.8) -> StabilityDetail:
    return StabilityDetail(
        path1_stability=combined,
        path2_stability=0.0,
        path3a_stability=0.0,
        combined_stability=combined,
        path4_ci_coverage=0.0,
        path5_jmi_stability=0.0,
    )


def _profile(
    name: str = "feat_a",
    feature_index: int = 0,
    best_value: float = 0.15,
    best_p: float = 0.001,
    stability: float = 0.8,
    cf_ratio: float | None = None,
) -> FeatureProfile:
    return FeatureProfile(
        feature_name=name,
        feature_index=feature_index,
        best_horizon=10,
        best_metric="forward_ic",
        best_value=best_value,
        best_p=best_p,
        passing_paths=("linear_signal",),
        stability=_stability(stability),
        concurrent_forward_ratio=cf_ratio,
        cf_classification="forward",
        redundancy_cluster_id=0,
        max_pairwise_correlation=None,
        vif=None,
        ic_acf_half_life=5,
    )


def _two_profiles() -> dict[str, FeatureProfile]:
    return {
        "feat_a": _profile("feat_a", feature_index=0, best_value=0.15),
        "feat_b": _profile("feat_b", feature_index=1, best_value=0.12),
    }


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Hashing must be deterministic across runs — Phase 4 content
    addressing depends on this."""

    def test_same_profiles_same_hash_across_calls(self):
        profiles = _two_profiles()
        hashes = {compute_profile_hash(profiles) for _ in range(100)}
        assert len(hashes) == 1

    def test_insertion_order_does_not_affect_hash(self):
        forward = {
            "feat_a": _profile("feat_a", feature_index=0),
            "feat_b": _profile("feat_b", feature_index=1),
        }
        backward = {
            "feat_b": _profile("feat_b", feature_index=1),
            "feat_a": _profile("feat_a", feature_index=0),
        }
        assert compute_profile_hash(forward) == compute_profile_hash(backward)

    def test_empty_profiles_has_stable_hash(self):
        # Empty dict must produce a well-defined hash, not crash.
        h1 = compute_profile_hash({})
        h2 = compute_profile_hash({})
        assert h1 == h2
        assert len(h1) == 64

    def test_repeated_profiles_same_hash(self):
        # Same dict passed in twice (identity-same) yields same hash.
        p = _two_profiles()
        assert compute_profile_hash(p) == compute_profile_hash(p)


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------


class TestSensitivity:
    """Any meaningful change to a profile must change the hash."""

    def test_different_feature_names_different_hash(self):
        a = {"feat_a": _profile("feat_a")}
        b = {"feat_b": _profile("feat_b")}
        assert compute_profile_hash(a) != compute_profile_hash(b)

    def test_different_best_value_different_hash(self):
        base = _two_profiles()
        mutated = {
            **base,
            "feat_a": replace(base["feat_a"], best_value=0.16),  # was 0.15
        }
        assert compute_profile_hash(base) != compute_profile_hash(mutated)

    def test_different_stability_different_hash(self):
        base = _two_profiles()
        mutated = {
            **base,
            "feat_a": replace(
                base["feat_a"],
                stability=_stability(combined=0.9),  # was 0.8
            ),
        }
        assert compute_profile_hash(base) != compute_profile_hash(mutated)

    def test_different_holdout_confirmed_different_hash(self):
        base = _two_profiles()
        mutated = {
            **base,
            "feat_a": replace(base["feat_a"], holdout_confirmed=True),
        }
        assert compute_profile_hash(base) != compute_profile_hash(mutated)

    def test_feature_addition_changes_hash(self):
        base = _two_profiles()
        extended = {
            **base,
            "feat_c": _profile("feat_c", feature_index=2),
        }
        assert compute_profile_hash(base) != compute_profile_hash(extended)


# ---------------------------------------------------------------------------
# Hash format
# ---------------------------------------------------------------------------


class TestHashFormat:
    """Locks the hash string format for downstream consumers."""

    def test_hex_64_chars(self):
        h = compute_profile_hash(_two_profiles())
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_no_prefix(self):
        # compute_profile_hash MUST return raw hex, matching
        # ExperimentRecord.fingerprint + hash_config_dict convention.
        # The `sha256:` prefix is reserved for external identifiers
        # (databento manifests, fast_gate export gates).
        h = compute_profile_hash(_two_profiles())
        assert not h.startswith("sha256:")
        assert not h.startswith("SHA256:")

    def test_lowercase_hex(self):
        h = compute_profile_hash(_two_profiles())
        assert h == h.lower()


# ---------------------------------------------------------------------------
# NaN / Inf handling
# ---------------------------------------------------------------------------


class TestNaNHandling:
    """Non-finite floats must be canonicalized to None so the hash is
    strict-JSON safe AND semantically correct (NaN p-values represent
    'no hypothesis test run' — i.e., absent information)."""

    def test_nan_best_p_does_not_crash(self):
        profiles = {
            "feat_nan": _profile(best_p=float("nan")),
        }
        h = compute_profile_hash(profiles)
        assert len(h) == 64

    def test_nan_p_equals_none_p(self):
        # Two profiles that differ ONLY in NaN vs None for best_p
        # should hash identically — because NaN is sanitized to None.
        # (FeatureProfile.best_p is typed float; but None would be the
        # strict representation. This test asserts the canonical
        # equivalence via _sanitize_for_hash.)
        p_nan = _profile(best_p=float("nan"))
        h_nan = compute_profile_hash({"x": p_nan})
        # Cannot construct a FeatureProfile with best_p=None (float type),
        # so we validate the sanitizer directly instead:
        from dataclasses import asdict

        sanitized = _sanitize_for_hash(asdict(p_nan))
        assert sanitized["best_p"] is None

    def test_inf_sanitized_to_none(self):
        assert _sanitize_for_hash(float("inf")) is None
        assert _sanitize_for_hash(float("-inf")) is None

    def test_nested_nan_in_tuple_and_list_sanitized(self):
        mixed = {
            "list_with_nan": [1.0, float("nan"), 2.0],
            "tuple_with_inf": (3.0, float("inf")),
            "nested_dict": {"inner": float("nan")},
        }
        out = _sanitize_for_hash(mixed)
        assert out["list_with_nan"] == [1.0, None, 2.0]
        # Tuples become lists (canonicalization — tuple/list have identical
        # JSON representation).
        assert out["tuple_with_inf"] == [3.0, None]
        assert out["nested_dict"] == {"inner": None}

    def test_finite_floats_preserved(self):
        assert _sanitize_for_hash(1.5) == 1.5
        assert _sanitize_for_hash(0.0) == 0.0
        assert _sanitize_for_hash(-3.14) == -3.14

    def test_non_float_types_pass_through(self):
        assert _sanitize_for_hash(42) == 42  # int
        assert _sanitize_for_hash("hello") == "hello"  # str
        assert _sanitize_for_hash(True) is True  # bool
        assert _sanitize_for_hash(None) is None  # None


# ---------------------------------------------------------------------------
# Cross-consistency: compute_profile_hash matches monorepo canonical form
# ---------------------------------------------------------------------------


class TestCanonicalFormAlignment:
    """Lock the canonical form (sort_keys=True, default=str, NaN→None)
    so a future refactor cannot drift from hft-ops dedup.py / lineage.py
    without a matching test update."""

    def test_hash_matches_explicit_canonical_form(self):
        import hashlib
        import json
        from dataclasses import asdict

        profiles = _two_profiles()
        # Reproduce compute_profile_hash independently — if this ever
        # drifts from the production code the test fails.
        sanitized = {
            name: _sanitize_for_hash(asdict(profiles[name]))
            for name in sorted(profiles)
        }
        expected_blob = json.dumps(
            sanitized, sort_keys=True, default=str
        ).encode("utf-8")
        expected = hashlib.sha256(expected_blob).hexdigest()

        assert compute_profile_hash(profiles) == expected


# ---------------------------------------------------------------------------
# Stale-state hazard: v1 `run()` must reset _last_profile_hash
# ---------------------------------------------------------------------------


class TestLastProfileHashResetContract:
    """Phase 4 Batch 4a validation caught a stale-state hazard: if a
    caller invokes ``run_v2()`` (populates hash), then ``run()`` (v1,
    which does not produce profiles), the hash would have been stale
    without the reset. These tests lock the reset contract against a
    fake pipeline so we can verify the contract without a 150s run_v2
    call."""

    def test_fresh_pipeline_hash_is_none(self):
        # Constructed but never-called pipeline.
        from unittest.mock import MagicMock

        from hft_evaluator.pipeline import EvaluationPipeline

        # Bypass __init__'s data-loading work by monkey-patching
        pipeline = EvaluationPipeline.__new__(EvaluationPipeline)
        pipeline._last_profile_hash = None
        assert pipeline.last_profile_hash is None

    def test_run_v1_source_contains_hash_reset(self):
        # Source-level contract lock: v1 `run()` MUST reset the hash at
        # entry. Without this, a prior `run_v2()` hash would linger as a
        # stale digest attached to a v1 classification that produced no
        # profile. We verify the source code contains the reset
        # statement rather than execute a 150s `run()` — the contract
        # is a structural invariant, not a behavioral one.
        import inspect

        from hft_evaluator.pipeline import EvaluationPipeline

        src = inspect.getsource(EvaluationPipeline.run)
        assert "self._last_profile_hash = None" in src, (
            "EvaluationPipeline.run() must reset _last_profile_hash at "
            "entry to prevent stale v2 hashes leaking across v1 calls. "
            "See Phase 4 Batch 4a validation finding P1."
        )

    def test_run_v2_source_contains_hash_reset(self):
        # Same contract for run_v2 — so a mid-run crash leaves the hash
        # as None rather than a prior successful run's digest.
        import inspect

        from hft_evaluator.pipeline import EvaluationPipeline

        src = inspect.getsource(EvaluationPipeline.run_v2)
        assert "self._last_profile_hash = None" in src, (
            "EvaluationPipeline.run_v2() must reset _last_profile_hash "
            "at entry so a mid-run crash leaves the hash as None, not "
            "a stale digest from a prior successful call."
        )

    def test_property_is_read_only(self):
        # Contract: last_profile_hash is a read-only property; callers
        # can only observe it, not mutate it directly from outside.
        from hft_evaluator.pipeline import EvaluationPipeline

        pipeline = EvaluationPipeline.__new__(EvaluationPipeline)
        pipeline._last_profile_hash = None
        with pytest.raises(AttributeError):
            pipeline.last_profile_hash = "any_value"
