"""
Concurrent vs forward IC decomposition.

Decomposes each feature's IC into concurrent (feature explains what IS
happening) and forward (feature predicts what WILL happen) components.

Convention (Framework §4.2, CODEBASE.md §5.2):
    Forward IC:    spearman_ic(feature[t], label[t])     → feature predicts forward return
    Concurrent IC: spearman_ic(feature[t], label[t-1])   → feature explains current return

Note: The concurrent IC is only purely concurrent at h=1. For h>1,
label[t-1] spans bins (t-1) to (t-1+h), mixing concurrent and forward.
The concurrent fraction is 1/h. The ratio is most meaningful at h=1.

Reference: CODEBASE.md Section 5.2, Framework Section 4.2-4.3
"""

from dataclasses import dataclass
from collections import defaultdict

import numpy as np

from hft_metrics.ic import spearman_ic
from hft_metrics._sanitize import EPS

from hft_evaluator.data.loader import ExportLoader


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConcurrentForwardResult:
    """Result for one (feature, horizon) pair from CF decomposition."""

    feature_name: str
    feature_index: int
    horizon: int
    forward_ic: float             # Avg daily forward IC
    concurrent_ic: float          # Avg daily concurrent IC
    ratio: float                  # |concurrent| / max(|forward|, EPS)
    classification: str           # "contemporaneous"|"partially_forward"|"forward"|"state_variable"
    n_days: int


# ---------------------------------------------------------------------------
# Main decomposition function
# ---------------------------------------------------------------------------


def decompose_concurrent_forward(
    loader: ExportLoader,
    evaluation_dates: list[str],
    evaluable_indices: list[int],
    horizons: list[int],
) -> dict[str, dict[int, ConcurrentForwardResult]]:
    """Concurrent vs forward IC decomposition for all features.

    Per-day averaging (not pooled): compute daily ICs and average across days.
    This avoids bias toward high-N days.

    Args:
        loader: ExportLoader with detected schema.
        evaluation_dates: Dates to evaluate on.
        evaluable_indices: Feature indices that passed pre-screening.
        horizons: Horizon values to decompose.

    Returns:
        dict[feature_name -> dict[horizon -> ConcurrentForwardResult]]
    """
    schema = loader.schema
    feature_names = schema.feature_names

    # Accumulate daily ICs: [j][h_idx] -> lists of (forward_ic, concurrent_ic)
    daily_forward: dict[int, dict[int, list[float]]] = {
        j: {h_idx: [] for h_idx in range(len(horizons))}
        for j in evaluable_indices
    }
    daily_concurrent: dict[int, dict[int, list[float]]] = {
        j: {h_idx: [] for h_idx in range(len(horizons))}
        for j in evaluable_indices
    }

    for bundle in loader.iter_days(evaluation_dates):
        n_samples = bundle.sequences.shape[0]
        if n_samples < 4:
            continue  # Need at least 4 samples for shift alignment

        features_2d = bundle.sequences[:, -1, :]  # [N, F]
        labels_2d = bundle.labels                   # [N, H]

        for j in evaluable_indices:
            feature_col = np.asarray(features_2d[:, j], dtype=np.float64)

            for h_idx in range(len(horizons)):
                if h_idx >= labels_2d.shape[1]:
                    continue
                label_col = np.asarray(labels_2d[:, h_idx], dtype=np.float64)

                # Forward IC: feature[t] vs label[t] (full N samples, matches Path 1)
                fwd_rho, fwd_p = spearman_ic(feature_col, label_col)

                # Concurrent IC: feature[t] vs label[t-1] (N-1 samples, shifted)
                conc_rho, conc_p = spearman_ic(feature_col[1:], label_col[:-1])

                # Skip degenerate returns
                if not (fwd_rho == 0.0 and fwd_p == 1.0):
                    daily_forward[j][h_idx].append(fwd_rho)
                if not (conc_rho == 0.0 and conc_p == 1.0):
                    daily_concurrent[j][h_idx].append(conc_rho)

    # Aggregate across days
    results: dict[str, dict[int, ConcurrentForwardResult]] = defaultdict(dict)

    for j in evaluable_indices:
        name = feature_names.get(j, f"feature_{j}")
        for h_idx in range(len(horizons)):
            fwd_array = np.array(daily_forward[j][h_idx])
            conc_array = np.array(daily_concurrent[j][h_idx])
            n_days = len(fwd_array)

            if n_days == 0:
                fwd_ic = 0.0
                conc_ic = 0.0
            else:
                fwd_ic = float(np.mean(fwd_array))
                conc_ic = float(np.mean(conc_array))

            # Classification (Framework §4.3)
            abs_conc = abs(conc_ic)
            abs_fwd = abs(fwd_ic)
            ratio = abs_conc / max(abs_fwd, EPS)

            if abs_conc < 0.01 and abs_fwd > 0.03:
                classification = "state_variable"
            elif ratio > 10:
                classification = "contemporaneous"
            elif ratio > 2:
                classification = "partially_forward"
            else:
                classification = "forward"

            results[name][horizons[h_idx]] = ConcurrentForwardResult(
                feature_name=name,
                feature_index=j,
                horizon=horizons[h_idx],
                forward_ic=fwd_ic,
                concurrent_ic=conc_ic,
                ratio=ratio,
                classification=classification,
                n_days=n_days,
            )

    return dict(results)


# ---------------------------------------------------------------------------
# Cache-based variant (Phase 1: no disk I/O)
# ---------------------------------------------------------------------------


def compute_cf_from_cache(
    cache: "DataCache",
    horizons: list[int],
) -> dict[str, dict[int, ConcurrentForwardResult]]:
    """CF decomposition from pre-computed daily IC cubes. No disk I/O.

    Args:
        cache: DataCache from build_cache().
        horizons: Horizon values.

    Returns:
        dict[feature_name -> dict[horizon -> ConcurrentForwardResult]]
    """
    from hft_evaluator.data.cache import DataCache

    schema = cache.schema
    feature_names = schema.feature_names

    results: dict[str, dict[int, ConcurrentForwardResult]] = defaultdict(dict)

    for pos, j in enumerate(cache.evaluable_indices):
        name = feature_names.get(j, f"feature_{j}")
        for h_idx in range(len(horizons)):
            fwd_series = cache.daily_forward_ic_cube[:, pos, h_idx]
            conc_series = cache.daily_concurrent_ic_cube[:, pos, h_idx]

            fwd_valid = fwd_series[np.isfinite(fwd_series)]
            conc_valid = conc_series[np.isfinite(conc_series)]
            n_days = len(fwd_valid)

            fwd_ic = float(np.mean(fwd_valid)) if n_days > 0 else 0.0
            conc_ic = float(np.mean(conc_valid)) if len(conc_valid) > 0 else 0.0

            abs_conc = abs(conc_ic)
            abs_fwd = abs(fwd_ic)
            ratio = abs_conc / max(abs_fwd, EPS)

            if abs_conc < 0.01 and abs_fwd > 0.03:
                classification = "state_variable"
            elif ratio > 10:
                classification = "contemporaneous"
            elif ratio > 2:
                classification = "partially_forward"
            else:
                classification = "forward"

            results[name][horizons[h_idx]] = ConcurrentForwardResult(
                feature_name=name,
                feature_index=j,
                horizon=horizons[h_idx],
                forward_ic=fwd_ic,
                concurrent_ic=conc_ic,
                ratio=ratio,
                classification=classification,
                n_days=n_days,
            )

    return dict(results)
