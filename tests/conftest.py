"""
Shared test fixtures for hft-feature-evaluator.

Provides a deterministic synthetic export matching the off-exchange contract
(34 features, categorical indices {29, 30, 32, 33}).

Test signal features:
- Feature 0 (trf_signed_imbalance): linear signal (IC ~ 0.3 with horizon 0)
- Feature 1 (mroib): zero variance (constant 1.0 — pre-screen catches)
- Feature 2 (inv_inst_direction): pure noise (IC ~ 0)
- Feature 3 (bvc_imbalance): regime-conditional (signal only in high-spread)
- Features 4-28: random noise (evaluable filler)
- Feature 12 (spread_bps): conditioning variable for regime IC
- Feature 27 (bin_trade_count): conditioning variable for regime IC
- Feature 29 (bin_valid): constant 1.0 (categorical)
- Feature 30 (bbo_valid): constant 1.0 (categorical)
- Feature 31 (session_progress): 0..1 ramp (evaluable conditioning var)
- Feature 32 (time_bucket): categorical integer
- Feature 33 (schema_version): constant 1.0 (categorical)

All random data generated with seed=42 for determinism.
"""

import json
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Constants matching off-exchange contract
# ---------------------------------------------------------------------------

N_DAYS = 5
N_SEQS = 50
N_TIMESTEPS = 5
N_FEATURES = 34          # Matches OFF_EXCHANGE_FEATURE_COUNT
N_HORIZONS = 3
HORIZONS = [1, 5, 10]
# Real off-exchange categoricals from hft-contracts
CATEGORICAL_INDICES = frozenset({29, 30, 32, 33})


def _make_day_data(date: str, rng: np.random.RandomState) -> dict:
    """Build one day's synthetic data with known statistical properties."""
    # Labels: point returns in bps (continuous)
    labels = rng.randn(N_SEQS, N_HORIZONS) * 5.0  # ~5 bps std

    # Build sequences [N, T, F]
    sequences = np.zeros((N_SEQS, N_TIMESTEPS, N_FEATURES), dtype=np.float32)

    for t in range(N_TIMESTEPS):
        # Feature 0 (trf_signed_imbalance): linear signal correlated with label[:,0]
        noise_0 = rng.randn(N_SEQS) * 0.95
        sequences[:, t, 0] = labels[:, 0] * 0.3 + noise_0

        # Feature 1 (mroib): zero variance (constant)
        sequences[:, t, 1] = 1.0

        # Feature 2 (inv_inst_direction): pure noise
        sequences[:, t, 2] = rng.randn(N_SEQS)

        # Feature 3 (bvc_imbalance): regime-conditional signal
        # Signal present only when spread_bps (index 12) is in HIGH tercile
        spread_values = rng.uniform(0.5, 5.0, N_SEQS)
        high_spread = spread_values > np.percentile(spread_values, 67)
        signal_3 = np.where(
            high_spread, labels[:, 0] * 0.5, rng.randn(N_SEQS) * 0.5
        )
        sequences[:, t, 3] = signal_3.astype(np.float32)

        # Features 4-11: evaluable noise filler
        for j in range(4, 12):
            sequences[:, t, j] = rng.randn(N_SEQS).astype(np.float32)

        # Feature 12 (spread_bps): conditioning variable — positive, varying
        sequences[:, t, 12] = spread_values.astype(np.float32)

        # Features 13-26: evaluable noise filler
        for j in range(13, 27):
            sequences[:, t, j] = rng.randn(N_SEQS).astype(np.float32)

        # Feature 27 (bin_trade_count): conditioning variable — count-like
        sequences[:, t, 27] = rng.poisson(50, N_SEQS).astype(np.float32)

        # Feature 28: evaluable noise
        sequences[:, t, 28] = rng.randn(N_SEQS).astype(np.float32)

        # Feature 29 (bin_valid): constant 1.0 — categorical
        sequences[:, t, 29] = 1.0

        # Feature 30 (bbo_valid): constant 1.0 — categorical
        sequences[:, t, 30] = 1.0

        # Feature 31 (session_progress): 0 to 1 ramp — evaluable conditioning var
        sequences[:, t, 31] = np.linspace(0.01, 0.99, N_SEQS).astype(np.float32)

        # Feature 32 (time_bucket): categorical integer
        sequences[:, t, 32] = float(int(date.replace("-", "")[-2:]) % 4)

        # Feature 33 (schema_version): constant 1.0 — categorical
        sequences[:, t, 33] = 1.0

    metadata = {
        "day": date,
        "n_sequences": N_SEQS,
        "window_size": N_TIMESTEPS,
        "n_features": N_FEATURES,
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
        "provenance": {
            "processor_version": "0.1.0",
            "export_timestamp_utc": "2026-03-26T00:00:00Z",
        },
        "export_timestamp": "2026-03-26T00:00:00Z",
    }

    return {
        "sequences": sequences,
        "labels": labels.astype(np.float64),
        "metadata": metadata,
    }


@pytest.fixture
def synthetic_export(tmp_path):
    """Create a synthetic 34-feature off-exchange export for testing.

    Returns the export directory path. Structure:
        tmp_path/train/  (5 days, 50 seqs each)
        tmp_path/dataset_manifest.json
    """
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    rng = np.random.RandomState(42)
    dates = [f"2025-01-{d:02d}" for d in range(1, N_DAYS + 1)]

    for date in dates:
        day_data = _make_day_data(date, rng)
        np.save(train_dir / f"{date}_sequences.npy", day_data["sequences"])
        np.save(train_dir / f"{date}_labels.npy", day_data["labels"])
        with open(train_dir / f"{date}_metadata.json", "w") as f:
            json.dump(day_data["metadata"], f)

    # Dataset manifest
    manifest = {
        "experiment": "synthetic_test",
        "feature_count": N_FEATURES,
        "schema_version": "1.0",
        "contract_version": "off_exchange_1.0",
        "days_processed": N_DAYS,
        "total_sequences": N_DAYS * N_SEQS,
        "sequence_length": N_TIMESTEPS,
        "horizons": HORIZONS,
        "bin_size_seconds": 60,
        "splits": {
            "train": {"days": dates, "total_sequences": N_DAYS * N_SEQS},
        },
    }
    with open(tmp_path / "dataset_manifest.json", "w") as f:
        json.dump(manifest, f)

    return str(tmp_path)


@pytest.fixture
def synthetic_config_dict(synthetic_export):
    """Config dict for the synthetic export."""
    return {
        "export_dir": synthetic_export,
        "split": "train",
        "holdout_days": 1,
        "seed": 42,
        "screening": {
            "horizons": HORIZONS,
            "bh_fdr_level": 0.05,
            "ic_threshold": 0.05,
            "dcor_permutations": 100,
            "dcor_subsample": 500,
            "mi_permutations": 50,
            "mi_k": 3,
        },
        "stability": {
            "n_bootstraps": 10,
            "subsample_fraction": 0.8,
            "stable_threshold": 0.6,
            "investigate_threshold": 0.4,
        },
        "classification": {
            "strong_keep_p": 0.01,
            "ic_ir_threshold": 0.5,
        },
        "regime": {
            "min_samples_per_bin": 100,  # Higher than ~67/tercile → skips bootstrap
        },
        "verbose": False,
    }
