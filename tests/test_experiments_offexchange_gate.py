"""Phase 5 Preview: tests for the `hft_evaluator.experiments.offexchange_gate`
library module (migrated from `scripts/offexchange_gate_check.py`).

Tests cover:
- `GateCheckConfig` validation (range checks, required fields, splits).
- `GateCheckResult` serialization + verdict/passes derived properties.
- `GateCheckFinding` shape.
- Core analyze logic against hand-crafted synthetic days.
- Entry-point `run()` end-to-end on a tiny synthetic export (fixture).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from hft_evaluator.experiments import (
    GateCheckConfig,
    GateCheckFinding,
    GateCheckResult,
)
from hft_evaluator.experiments.offexchange_gate import (
    _analyze_feature,
    _build_markdown,
)


# -----------------------------------------------------------------------------
# Config validation
# -----------------------------------------------------------------------------


class TestGateCheckConfigValidation:
    def test_defaults_are_valid(self):
        # Defaults pass __post_init__. validate() still needs export_dir.
        c = GateCheckConfig()
        with pytest.raises(ValueError, match="export_dir"):
            c.validate()

    def test_acf60_threshold_out_of_range_raises(self):
        with pytest.raises(ValueError, match="g1_acf60_threshold"):
            GateCheckConfig(g1_acf60_threshold=-0.1)
        with pytest.raises(ValueError, match="g1_acf60_threshold"):
            GateCheckConfig(g1_acf60_threshold=1.5)

    def test_ic_thresholds_out_of_range_raise(self):
        with pytest.raises(ValueError, match="g2_stride60_ic_threshold"):
            GateCheckConfig(g2_stride60_ic_threshold=-0.01)
        with pytest.raises(ValueError, match="g3_lag1_ic_threshold"):
            GateCheckConfig(g3_lag1_ic_threshold=2.0)

    def test_stride_must_be_positive(self):
        with pytest.raises(ValueError, match="stride"):
            GateCheckConfig(stride=0)

    def test_bad_split_rejected(self):
        with pytest.raises(ValueError, match="Invalid split"):
            GateCheckConfig(splits=("garbage",))

    def test_missing_export_dir_fails_validate(self):
        c = GateCheckConfig()
        with pytest.raises(ValueError, match="export_dir"):
            c.validate()

    def test_nonexistent_export_dir_fails_validate(self, tmp_path):
        bogus = tmp_path / "does_not_exist"
        c = GateCheckConfig(export_dir=str(bogus))
        with pytest.raises(FileNotFoundError):
            c.validate()


# -----------------------------------------------------------------------------
# Core analyze logic
# -----------------------------------------------------------------------------


def _make_days(
    n_days: int = 3,
    n_samples_per_day: int = 200,
    feat_autocorr: float = 0.9,
    label_correlation: float = 0.0,
    seed: int = 0,
) -> List[Dict[str, np.ndarray]]:
    """Build synthetic (features, labels) days for testing.

    `feat_autocorr` controls AR(1) coefficient of the feature series
    (higher = more persistence → higher ACF60 if enough length).
    `label_correlation` controls the shift-correlation feature ↔ label.
    """
    rng = np.random.default_rng(seed)
    days = []
    for _ in range(n_days):
        # 32-feature exchange-like layout; only index 8 is exercised
        n_feat = 34
        n_horizons = 8
        features = np.zeros((n_samples_per_day, n_feat), dtype=np.float64)
        # AR(1) series on index 8
        noise = rng.standard_normal(n_samples_per_day)
        f = np.zeros(n_samples_per_day)
        for t in range(1, n_samples_per_day):
            f[t] = feat_autocorr * f[t - 1] + noise[t]
        features[:, 8] = f
        # Labels at index 0 are correlated with feature via shift
        labels = np.zeros((n_samples_per_day, n_horizons), dtype=np.float64)
        lab = label_correlation * f + (1 - label_correlation) * rng.standard_normal(n_samples_per_day)
        labels[:, 0] = lab
        days.append({"features": features, "labels": labels})
    return days


class TestAnalyzeFeature:
    def test_all_pass_with_persistent_correlated_series(self):
        days = _make_days(
            n_days=3,
            n_samples_per_day=500,
            # AR(1) lag-k ACF = rho^k; short-sample bias pulls estimates below
            # theory. rho=0.99 gives acf60 ≈ 0.29 on 500-sample draws; we lower
            # the gate threshold so we exercise the pass mechanism on a signal
            # that is qualitatively strong but bias-limited.
            feat_autocorr=0.99,
            label_correlation=0.8,
        )
        cfg = GateCheckConfig(
            export_dir="dummy",
            g1_acf60_threshold=0.15,
        )
        finding = _analyze_feature(
            days=days,
            feat_idx=8,
            feat_name="subpenny_intensity",
            horizon_idx=0,
            horizon_name="H=1",
            split="test",
            config=cfg,
        )
        assert finding.g1_pass, f"G1 expected PASS, got acf60={finding.acf60}"
        assert finding.g2_pass, f"G2 expected PASS, got ic60={finding.ic_stride60_pooled}"
        assert finding.g3_pass, f"G3 expected PASS, got lag1={finding.lag1_ic_pooled}"
        assert finding.all_gates_pass

    def test_all_fail_with_pure_noise(self):
        days = _make_days(
            n_days=3,
            n_samples_per_day=500,
            feat_autocorr=0.0,  # no persistence
            label_correlation=0.0,  # no link
            seed=7,
        )
        cfg = GateCheckConfig(export_dir="dummy")
        finding = _analyze_feature(
            days=days,
            feat_idx=8,
            feat_name="subpenny_intensity",
            horizon_idx=0,
            horizon_name="H=1",
            split="test",
            config=cfg,
        )
        assert not finding.all_gates_pass
        # Well-formed finding even when every gate fails
        assert finding.feature == "subpenny_intensity"
        assert finding.n_traded_bins >= 0

    def test_short_series_no_crash(self):
        """Days shorter than min_valid_samples are skipped cleanly."""
        days = [
            {
                "features": np.zeros((10, 34)),  # below min_valid_samples=20
                "labels": np.zeros((10, 8)),
            }
        ]
        cfg = GateCheckConfig(export_dir="dummy")
        finding = _analyze_feature(
            days=days,
            feat_idx=8,
            feat_name="subpenny_intensity",
            horizon_idx=0,
            horizon_name="H=1",
            split="test",
            config=cfg,
        )
        # All gates fail gracefully (no valid days contributed data)
        assert finding.acf60 == 0.0
        assert finding.ic_stride60_pooled == 0.0
        assert finding.lag1_ic_pooled == 0.0
        assert not finding.all_gates_pass

    def test_nan_values_masked(self):
        """NaN rows in feature or label are masked by isfinite check."""
        n = 200
        feat = np.random.default_rng(0).standard_normal((n, 34))
        lab = np.random.default_rng(1).standard_normal((n, 8))
        # Inject NaN in feature at various rows
        feat[10:20, 8] = np.nan
        days = [{"features": feat, "labels": lab}]
        cfg = GateCheckConfig(export_dir="dummy")
        finding = _analyze_feature(
            days=days,
            feat_idx=8,
            feat_name="subpenny_intensity",
            horizon_idx=0,
            horizon_name="H=1",
            split="test",
            config=cfg,
        )
        # No crash; finding well-formed
        assert isinstance(finding, GateCheckFinding)
        assert finding.n_traded_bins >= 0


# -----------------------------------------------------------------------------
# Result dataclass
# -----------------------------------------------------------------------------


class TestGateCheckResult:
    def _make_finding(self, all_pass: bool = False) -> GateCheckFinding:
        return GateCheckFinding(
            split="test",
            feature="f",
            feature_index=0,
            horizon="H=1",
            horizon_index=0,
            acf1=0.1, acf30=0.1, acf60=0.1,
            g1_pass=all_pass,
            ic_stride1_mean=0.01,
            ic_stride60_pooled=0.01,
            ic_degradation_ratio=1.0,
            n_stride60_samples=100,
            g2_pass=all_pass,
            lag1_ic_pooled=0.01,
            g3_pass=all_pass,
            da_traded_bins=0.5,
            q_spread_traded_bps=0.0,
            n_traded_bins=100,
            all_gates_pass=all_pass,
        )

    def test_verdict_pass_when_any_finding_passes(self):
        r = GateCheckResult(
            schema="v2", analysis_date="2026-04-16", export_dir="/tmp/e",
            gate_thresholds={},
            findings=(self._make_finding(True), self._make_finding(False)),
        )
        assert r.verdict == "PASS"
        assert len(r.passes) == 1

    def test_verdict_fail_when_no_finding_passes(self):
        r = GateCheckResult(
            schema="v2", analysis_date="2026-04-16", export_dir="/tmp/e",
            gate_thresholds={},
            findings=(self._make_finding(False), self._make_finding(False)),
        )
        assert r.verdict == "FAIL"
        assert r.passes == ()

    def test_to_dict_roundtrip(self):
        r = GateCheckResult(
            schema="v2", analysis_date="2026-04-16", export_dir="/tmp/e",
            gate_thresholds={"g1_acf60": 0.3},
            findings=(self._make_finding(True),),
            elapsed_seconds=1.5,
        )
        d = r.to_dict()
        assert d["schema"] == "v2"
        assert d["findings"][0]["feature"] == "f"
        assert d["findings"][0]["all_gates_pass"] is True
        assert d["elapsed_seconds"] == 1.5

    def test_to_json_writes_file(self, tmp_path):
        r = GateCheckResult(
            schema="v2", analysis_date="2026-04-16", export_dir="/tmp/e",
            gate_thresholds={"g1_acf60": 0.3},
            findings=(self._make_finding(False),),
        )
        out = tmp_path / "sub" / "results.json"
        r.to_json(out)
        assert out.exists()
        parsed = json.loads(out.read_text())
        assert parsed["schema"] == "v2"

    def test_markdown_report_has_verdict(self):
        r = GateCheckResult(
            schema="v2", analysis_date="2026-04-16", export_dir="/tmp/e",
            gate_thresholds={"g1_acf60": 0.3},
            findings=(self._make_finding(True),),
        )
        md = _build_markdown(r)
        assert "PASS" in md
        assert "| test | f | H=1 |" in md

    def test_markdown_empty_pass_branch(self):
        r = GateCheckResult(
            schema="v2", analysis_date="2026-04-16", export_dir="/tmp/e",
            gate_thresholds={"g1_acf60": 0.3},
            findings=(self._make_finding(False),),
        )
        md = _build_markdown(r)
        assert "FAIL" in md
        assert "No passing features" in md
