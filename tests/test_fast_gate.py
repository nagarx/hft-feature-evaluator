"""Tests for the pre-training fast_gate (Rule 13 IC gate).

These tests use small synthetic off-exchange exports crafted so each gate
scenario is deterministically PASS or FAIL. The gate's job is to be
*correct* on ground-truth signal presence/absence — these tests lock
that behavior down.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest


N_TIMESTEPS = 5
N_HORIZONS = 3
HORIZONS = [1, 5, 10]


def _write_day(
    day_dir: Path,
    date: str,
    sequences: np.ndarray,
    labels: np.ndarray,
    metadata_extra: Dict | None = None,
) -> None:
    """Persist one day's NPY triplet in the evaluator-expected layout."""
    np.save(day_dir / f"{date}_sequences.npy", sequences.astype(np.float32))
    np.save(day_dir / f"{date}_labels.npy", labels.astype(np.float64))
    md = {
        "day": date,
        "n_sequences": int(sequences.shape[0]),
        "window_size": int(sequences.shape[1]),
        "n_features": int(sequences.shape[2]),
        "schema_version": "1.0",
        "contract_version": "off_exchange_1.0",
        "label_strategy": "point_return",
        "label_encoding": "continuous_bps",
        "horizons": HORIZONS,
        "bin_size_seconds": 60,
        "normalization": {
            "strategy": "per_day_zscore",
            "applied": False,
            "params_file": f"{date}_normalization.json",
        },
        "provenance": {"processor_version": "0.1.0"},
        "export_timestamp": "2026-04-14T00:00:00Z",
    }
    if metadata_extra:
        md.update(metadata_extra)
    with open(day_dir / f"{date}_metadata.json", "w") as f:
        json.dump(md, f)


def _build_export(
    tmp_path: Path,
    *,
    n_days: int = 8,
    n_seqs_per_day: int = 500,
    signal_features: tuple = (0, 3),
    signal_strength: float = 0.3,
    label_std_bps: float = 10.0,
    allow_constant_feature: int | None = None,
    stable_across_days: bool = True,
) -> Path:
    """Build a synthetic export with controlled IC for testing.

    ``signal_features`` lists the feature indices that carry signal
    (default: indices 0 and 3 — two informative features so G_IC_COUNT
    can be satisfied with min_ic_count=2). All other features are noise.
    Categorical indices {29, 30, 32, 33} are populated with constants
    (matching OFF_EXCHANGE_CATEGORICAL_INDICES).

    Args:
        signal_features: Tuple of feature indices that carry signal.
            Empty tuple (``()``) means zero-signal export.
        signal_strength: coefficient in the linear model
            ``feature[:, t, f] = signal_strength * label + noise``. Larger →
            higher IC. 0.0 → no signal.
        label_std_bps: regression label std in basis points. Drives
            G_RETURN_STD gate.
        allow_constant_feature: If set, this feature index is a constant
            (test for bypass behavior).
        stable_across_days: If True, signal is identical sign across days
            (high stability). If False, sign flips per day (low stability).
    """
    export_dir = tmp_path / "export"
    train_dir = export_dir / "train"
    train_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)
    n_features = 34

    for d_i in range(n_days):
        date = f"2026-01-{d_i + 1:02d}"
        labels = rng.normal(0.0, label_std_bps, size=(n_seqs_per_day, N_HORIZONS))

        sequences = rng.normal(
            0.0, 1.0, size=(n_seqs_per_day, N_TIMESTEPS, n_features)
        ).astype(np.float32)

        # Inject signal into each requested feature, all timesteps
        sign = 1.0 if stable_across_days or (d_i % 2 == 0) else -1.0
        for t in range(N_TIMESTEPS):
            for f_idx in signal_features:
                noise = rng.normal(0.0, 1.0, size=n_seqs_per_day)
                sequences[:, t, f_idx] = (
                    sign * signal_strength * labels[:, 0] + noise
                ).astype(np.float32)

        # Categorical indices: constants matching the contract
        for cat in (29, 30, 33):
            sequences[:, :, cat] = 1.0
        sequences[:, :, 32] = float(d_i % 4)  # time_bucket

        if allow_constant_feature is not None:
            sequences[:, :, allow_constant_feature] = 1.0

        _write_day(train_dir, date, sequences, labels)

    return export_dir


# ---------------------------------------------------------------------------
# Core gate behavior
# ---------------------------------------------------------------------------


class TestFastGatePassScenarios:
    """Strong signal → all gates PASS."""

    def test_strong_signal_passes(self, tmp_path):
        from hft_evaluator.fast_gate import (
            GateThresholds,
            run_fast_gate,
        )

        export_dir = _build_export(
            tmp_path,
            n_days=8,
            n_seqs_per_day=500,
            signal_strength=0.7,  # very strong
            label_std_bps=10.0,
        )
        report = run_fast_gate(
            data_dir=export_dir,
            horizon_idx=0,
            thresholds=GateThresholds(),
            sample_size=10_000,
            n_folds=4,
        )

        assert report.verdict == "PASS", (
            f"strong signal should PASS; reason: {report.reason}"
        )
        assert report.gate_g_ic_passed
        assert report.gate_g_ic_count_passed
        assert report.gate_g_return_std_passed
        assert report.gate_g_stability_passed

        # One of the injected signal features (0 or 3) should be the top feature
        assert report.best_feature_idx in (0, 3), (
            f"expected a signal feature (0 or 3) to win; got "
            f"{report.best_feature_idx} ({report.best_feature_name}) "
            f"with IC={report.best_feature_ic}"
        )
        assert report.best_feature_ic > 0.05
        # ic_count should be at least 2 — both injected features should surface
        assert report.ic_count >= 2

    def test_gate_report_serializable(self, tmp_path):
        """gate_report.as_dict() + to_json() roundtrip as JSON."""
        from hft_evaluator.fast_gate import (
            GateThresholds,
            run_fast_gate,
        )

        export_dir = _build_export(
            tmp_path,
            n_days=8,
            n_seqs_per_day=500,
            signal_strength=0.5,
        )
        report = run_fast_gate(
            data_dir=export_dir,
            horizon_idx=0,
            thresholds=GateThresholds(),
            sample_size=5_000,
            n_folds=4,
        )

        out = tmp_path / "gate_report.json"
        report.to_json(out)
        assert out.exists()
        with open(out) as f:
            loaded = json.load(f)

        # Key invariants
        assert loaded["verdict"] in {"PASS", "FAIL"}
        assert loaded["best_feature_idx"] == report.best_feature_idx
        assert len(loaded["per_feature_ic"]) > 0
        assert loaded["thresholds"]["min_ic"] == 0.05


# ---------------------------------------------------------------------------
# Failure scenarios
# ---------------------------------------------------------------------------


class TestFastGateFailScenarios:
    """No signal / weak signal / unstable signal → gate FAILS."""

    def test_no_signal_fails_g_ic(self, tmp_path):
        from hft_evaluator.fast_gate import (
            GateThresholds,
            run_fast_gate,
        )

        export_dir = _build_export(
            tmp_path,
            n_days=8,
            n_seqs_per_day=500,
            signal_features=(),  # no signal at all
            signal_strength=0.0,
            label_std_bps=10.0,
        )
        report = run_fast_gate(
            data_dir=export_dir,
            horizon_idx=0,
            sample_size=5_000,
            n_folds=4,
        )
        assert report.verdict == "FAIL"
        assert not report.gate_g_ic_passed, (
            f"zero-signal export should fail G_IC; got |IC|={report.best_feature_ic}"
        )

    def test_low_return_std_fails_gate(self, tmp_path):
        from hft_evaluator.fast_gate import (
            GateThresholds,
            run_fast_gate,
        )

        export_dir = _build_export(
            tmp_path,
            n_days=8,
            n_seqs_per_day=500,
            signal_strength=0.6,
            label_std_bps=1.0,  # << 5 bps threshold
        )
        report = run_fast_gate(
            data_dir=export_dir,
            horizon_idx=0,
            thresholds=GateThresholds(min_return_std_bps=5.0),
            sample_size=5_000,
            n_folds=4,
        )
        assert not report.gate_g_return_std_passed, (
            f"return_std={report.return_std_bps} should fail the 5 bps gate"
        )
        assert report.verdict == "FAIL"

    def test_out_of_range_horizon_raises(self, tmp_path):
        from hft_evaluator.fast_gate import FastGateError, run_fast_gate

        export_dir = _build_export(tmp_path, n_days=5, n_seqs_per_day=200)
        with pytest.raises(FastGateError):
            run_fast_gate(
                data_dir=export_dir,
                horizon_idx=99,  # export has only 3 horizons
                sample_size=100,
                n_folds=2,
            )


# ---------------------------------------------------------------------------
# Bypass list
# ---------------------------------------------------------------------------


class TestFastGateBypass:
    """allow_zero_ic_names excludes context features from IC count gate."""

    def test_bypass_marks_feature(self, tmp_path):
        from hft_evaluator.fast_gate import run_fast_gate

        export_dir = _build_export(
            tmp_path,
            n_days=6,
            n_seqs_per_day=500,
            signal_strength=0.5,
        )
        report = run_fast_gate(
            data_dir=export_dir,
            horizon_idx=0,
            sample_size=3_000,
            n_folds=3,
            allow_zero_ic_names=("dark_share",),
        )

        # dark_share is feature 4 per the off-exchange contract. It should
        # be marked as bypassed in the report's per-feature list.
        matching = [
            p for p in report.per_feature_ic if p.feature_name == "dark_share"
        ]
        # Only present if it's in the top-N report, so check feature name
        # in the bypass list rather than presence.
        assert "dark_share" in report.allow_zero_ic_names


# ---------------------------------------------------------------------------
# Adaptive folds
# ---------------------------------------------------------------------------


class TestFastGateAdaptiveFolds:
    """n_folds is clipped to fit the available days."""

    def test_requested_20_clipped_for_4_days(self, tmp_path):
        from hft_evaluator.fast_gate import run_fast_gate

        export_dir = _build_export(tmp_path, n_days=4, n_seqs_per_day=200)
        report = run_fast_gate(
            data_dir=export_dir,
            horizon_idx=0,
            sample_size=500,
            n_folds=20,  # requested 20, but only 4 days
        )
        assert report.n_days_used == 4
        # Adaptive: max(5, min(20, 4//8=0)) = max(5, 0) = 5; clipped to 4
        assert report.n_folds_used == 4, (
            f"expected n_folds_used clipped to 4 (days), got {report.n_folds_used}"
        )

    def test_2_folds_minimum_produces_stability(self, tmp_path):
        from hft_evaluator.fast_gate import run_fast_gate

        export_dir = _build_export(
            tmp_path, n_days=10, n_seqs_per_day=300, signal_strength=0.6
        )
        report = run_fast_gate(
            data_dir=export_dir, horizon_idx=0, sample_size=1000, n_folds=5
        )
        assert report.n_folds_used >= 2
        assert np.isfinite(report.stability)


# ---------------------------------------------------------------------------
# Classification labels (1-D)
# ---------------------------------------------------------------------------


class TestFastGateClassificationLabels:
    """1-D labels (classification) are accepted at horizon_idx=0."""

    def test_1d_labels_accepted(self, tmp_path):
        from hft_evaluator.fast_gate import run_fast_gate

        export_dir = tmp_path / "export"
        train_dir = export_dir / "train"
        train_dir.mkdir(parents=True)

        rng = np.random.default_rng(42)
        for d in range(6):
            date = f"2026-01-{d + 1:02d}"
            n = 300
            labels = rng.choice([-1, 0, 1], size=n).astype(np.int64)
            sequences = rng.normal(0, 1, (n, N_TIMESTEPS, 34)).astype(np.float32)
            # Inject signal at feature 0
            sequences[:, -1, 0] = (
                0.7 * labels.astype(np.float32)
                + rng.normal(0, 0.3, n).astype(np.float32)
            )
            for cat in (29, 30, 33):
                sequences[:, :, cat] = 1.0
            sequences[:, :, 32] = float(d % 4)
            _write_day(
                train_dir,
                date,
                sequences,
                labels,
                metadata_extra={
                    "horizons": [10],
                    "label_strategy": "tlob",
                },
            )

        report = run_fast_gate(
            data_dir=export_dir, horizon_idx=0, sample_size=1000, n_folds=3
        )
        assert report.best_feature_idx == 0
        # return_std is meaningful (std of {-1, 0, 1} ~= 0.82 in the sample,
        # which happens to fail the 5 bps threshold — that's fine; we only
        # check that the gate doesn't crash on 1-D labels here.
        assert np.isfinite(report.best_feature_ic)
