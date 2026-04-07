"""
Path 3a: Temporal IC — IC of rolling features across timesteps.

Computes rolling_mean, rolling_slope, and rate_of_change over the T-timestep
window, then evaluates IC of the rolling features against returns. This captures
features whose predictive power lies in their trajectory, not their current level.

Uses per-day streaming + t-test (valid because Spearman rho is unbiased under
independence — empirically verified: bias = -0.003 ≈ 0).

Reference: CODEBASE.md Section 6.1, Framework Section 2.3
"""

from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from scipy import stats as scipy_stats

from hft_metrics.ic import spearman_ic, ic_ir
from hft_metrics.bootstrap import block_bootstrap_ci
from hft_metrics.temporal import rolling_mean, rolling_slope, rate_of_change
from hft_metrics.testing import benjamini_hochberg

from hft_evaluator.config import EvaluationConfig
from hft_evaluator.data.loader import ExportLoader
from hft_evaluator.screening import bh_adjusted_pvalues


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalICResult:
    """Result for one feature from temporal IC analysis."""

    feature_name: str
    feature_index: int
    best_horizon: int
    rolling_mean_ic: float
    rolling_slope_ic: float
    rate_of_change_ic: float
    best_temporal_ic: float
    best_temporal_metric: str
    best_temporal_p: float       # BH-adjusted p for best rolling metric
    n_days: int
    passes_path3: bool


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def compute_temporal_ic(
    loader: ExportLoader,
    evaluation_dates: list[str],
    evaluable_indices: list[int],
    horizons: list[int],
    config: EvaluationConfig,
) -> dict[str, TemporalICResult]:
    """Temporal IC of rolling features for all evaluable features.

    Per-day streaming: for each day, compute rolling features on [N, T] tensor,
    extract last-timestep IC against labels, accumulate daily ICs, then t-test.

    Args:
        loader: ExportLoader with detected schema.
        evaluation_dates: Dates to evaluate on.
        evaluable_indices: Feature indices that passed pre-screening.
        horizons: Horizon values to evaluate.
        config: EvaluationConfig with temporal.rolling_window.

    Returns:
        dict[feature_name -> TemporalICResult]
    """
    schema = loader.schema
    feature_names = schema.feature_names
    window = config.temporal.rolling_window

    # Accumulate daily ICs for 3 rolling metrics × horizons
    # daily_ics[j][(metric_name, h_idx)] = list[float]
    daily_ics: dict[int, dict[tuple[str, int], list[float]]] = {
        j: {
            (metric, h_idx): []
            for metric in ("rolling_mean", "rolling_slope", "rate_of_change")
            for h_idx in range(len(horizons))
        }
        for j in evaluable_indices
    }

    # Phase 1: Per-day computation
    for bundle in loader.iter_days(evaluation_dates):
        if bundle.sequences.shape[0] < 3:
            continue

        labels_2d = bundle.labels

        for j in evaluable_indices:
            # Use [N, T, 1] shape to trigger the 3D code path in rolling
            # functions (dim 1 = timesteps). A bare [N, T] would be
            # interpreted as [T, F] (dim 0 = time), rolling across samples.
            seq_j = np.asarray(
                bundle.sequences[:, :, j:j+1], dtype=np.float64
            )  # [N, T, 1]

            # Compute rolling features along T dimension → [N, T, 1]
            rm = rolling_mean(seq_j, window=window)
            rs = rolling_slope(seq_j, window=window)
            roc = rate_of_change(seq_j, lag=window)

            # Extract last timestep of each rolling feature → [N]
            rm_last = rm[:, -1, 0]
            rs_last = rs[:, -1, 0]
            roc_last = roc[:, -1, 0]

            for h_idx in range(len(horizons)):
                if h_idx >= labels_2d.shape[1]:
                    continue
                label_col = np.asarray(labels_2d[:, h_idx], dtype=np.float64)

                # IC of each rolling feature vs label
                for metric_name, feat_vals in [
                    ("rolling_mean", rm_last),
                    ("rolling_slope", rs_last),
                    ("rate_of_change", roc_last),
                ]:
                    # Filter NaN (first window-1 entries are NaN from rolling)
                    valid = np.isfinite(feat_vals) & np.isfinite(label_col)
                    if valid.sum() < 3:
                        continue
                    rho, p = spearman_ic(feat_vals[valid], label_col[valid])
                    if not (rho == 0.0 and p == 1.0):
                        daily_ics[j][(metric_name, h_idx)].append(rho)

    # Phase 2: Aggregate — find best rolling metric per feature
    # Collect all (j, metric, h_idx) triplets for BH
    all_keys: list[tuple[int, str, int]] = []
    all_raw_p: list[float] = []
    all_ic_mean: dict[tuple[int, str, int], float] = {}

    for j in evaluable_indices:
        for metric in ("rolling_mean", "rolling_slope", "rate_of_change"):
            for h_idx in range(len(horizons)):
                ic_array = np.array(daily_ics[j].get((metric, h_idx), []))
                if len(ic_array) < 3:
                    all_keys.append((j, metric, h_idx))
                    all_raw_p.append(1.0)
                    all_ic_mean[(j, metric, h_idx)] = 0.0
                    continue
                mean_ic = float(np.mean(ic_array))
                _, p_val = scipy_stats.ttest_1samp(ic_array, 0.0)
                all_keys.append((j, metric, h_idx))
                all_raw_p.append(float(p_val))
                all_ic_mean[(j, metric, h_idx)] = mean_ic

    # Phase 3: BH correction across all temporal IC tests
    raw_p_array = np.array(all_raw_p)
    bh_mask = benjamini_hochberg(raw_p_array, q=config.screening.bh_fdr_level)
    adj_p = bh_adjusted_pvalues(raw_p_array)

    # Build lookup: (j, metric, h_idx) -> (bh_rejected, adj_p)
    bh_lookup = {}
    for idx, key in enumerate(all_keys):
        bh_lookup[key] = (bool(bh_mask[idx]), float(adj_p[idx]))

    # Phase 4: Pick best rolling metric per feature
    results: dict[str, TemporalICResult] = {}

    for j in evaluable_indices:
        name = feature_names.get(j, f"feature_{j}")

        best_ic = 0.0
        best_metric = ""
        best_horizon = horizons[0] if horizons else 0
        best_p = 1.0
        best_passes = False

        rm_ic = 0.0
        rs_ic = 0.0
        roc_ic = 0.0

        for metric in ("rolling_mean", "rolling_slope", "rate_of_change"):
            for h_idx, h in enumerate(horizons):
                key = (j, metric, h_idx)
                mean_ic = all_ic_mean.get(key, 0.0)
                rejected, adj = bh_lookup.get(key, (False, 1.0))

                if abs(mean_ic) > abs(best_ic):
                    best_ic = mean_ic
                    best_metric = metric
                    best_horizon = h
                    best_p = adj
                    best_passes = (
                        abs(mean_ic) > config.screening.ic_threshold
                        and rejected
                    )

                # Store per-metric IC at best horizon (latest update wins)
                if metric == "rolling_mean" and abs(mean_ic) > abs(rm_ic):
                    rm_ic = mean_ic
                elif metric == "rolling_slope" and abs(mean_ic) > abs(rs_ic):
                    rs_ic = mean_ic
                elif metric == "rate_of_change" and abs(mean_ic) > abs(roc_ic):
                    roc_ic = mean_ic

        n_days = len(daily_ics[j].get(("rolling_mean", 0), []))

        results[name] = TemporalICResult(
            feature_name=name,
            feature_index=j,
            best_horizon=best_horizon,
            rolling_mean_ic=rm_ic,
            rolling_slope_ic=rs_ic,
            rate_of_change_ic=roc_ic,
            best_temporal_ic=best_ic,
            best_temporal_metric=best_metric,
            best_temporal_p=best_p,
            n_days=n_days,
            passes_path3=best_passes,
        )

    return results


# ---------------------------------------------------------------------------
# Cache-based variant (Phase 1: no disk I/O)
# ---------------------------------------------------------------------------


def compute_temporal_ic_from_cache(
    cache: "DataCache",
    config: EvaluationConfig,
) -> dict[str, TemporalICResult]:
    """Temporal IC from pre-computed daily temporal IC cubes. No disk I/O.

    Args:
        cache: DataCache from build_cache().
        config: EvaluationConfig with temporal.rolling_window.

    Returns:
        dict[feature_name -> TemporalICResult]
    """
    from hft_evaluator.data.cache import DataCache, TEMPORAL_METRICS

    schema = cache.schema
    feature_names = schema.feature_names
    horizons = list(cache.horizons)

    # Phase 2: Aggregate — collect all (j, metric, h_idx) for BH
    all_keys: list[tuple[int, str, int]] = []
    all_raw_p: list[float] = []
    all_ic_mean: dict[tuple[int, str, int], float] = {}

    for pos, j in enumerate(cache.evaluable_indices):
        for metric in TEMPORAL_METRICS:
            cube = cache.daily_temporal_cubes[metric]
            for h_idx in range(len(horizons)):
                ic_series = cube[:, pos, h_idx]
                ic_array = ic_series[np.isfinite(ic_series)]

                if len(ic_array) < 3:
                    all_keys.append((j, metric, h_idx))
                    all_raw_p.append(1.0)
                    all_ic_mean[(j, metric, h_idx)] = 0.0
                    continue

                mean_ic = float(np.mean(ic_array))
                _, p_val = scipy_stats.ttest_1samp(ic_array, 0.0)
                all_keys.append((j, metric, h_idx))
                all_raw_p.append(float(p_val))
                all_ic_mean[(j, metric, h_idx)] = mean_ic

    # Phase 3: BH correction
    raw_p_array = np.array(all_raw_p)
    bh_mask = benjamini_hochberg(raw_p_array, q=config.screening.bh_fdr_level)
    adj_p = bh_adjusted_pvalues(raw_p_array)

    bh_lookup = {}
    for idx, key in enumerate(all_keys):
        bh_lookup[key] = (bool(bh_mask[idx]), float(adj_p[idx]))

    # Phase 4: Best metric per feature
    results: dict[str, TemporalICResult] = {}

    for pos, j in enumerate(cache.evaluable_indices):
        name = feature_names.get(j, f"feature_{j}")

        best_ic = 0.0
        best_metric = ""
        best_horizon = horizons[0] if horizons else 0
        best_p = 1.0
        best_passes = False
        rm_ic = 0.0
        rs_ic = 0.0
        roc_ic = 0.0

        for metric in TEMPORAL_METRICS:
            for h_idx, h in enumerate(horizons):
                key = (j, metric, h_idx)
                mean_ic = all_ic_mean.get(key, 0.0)
                rejected, adj = bh_lookup.get(key, (False, 1.0))

                if abs(mean_ic) > abs(best_ic):
                    best_ic = mean_ic
                    best_metric = metric
                    best_horizon = h
                    best_p = adj
                    best_passes = (
                        abs(mean_ic) > config.screening.ic_threshold
                        and rejected
                    )

                if metric == "rolling_mean" and abs(mean_ic) > abs(rm_ic):
                    rm_ic = mean_ic
                elif metric == "rolling_slope" and abs(mean_ic) > abs(rs_ic):
                    rs_ic = mean_ic
                elif metric == "rate_of_change" and abs(mean_ic) > abs(roc_ic):
                    roc_ic = mean_ic

        n_days_series = cache.daily_temporal_cubes["rolling_mean"][:, pos, 0]
        n_days = int(np.sum(np.isfinite(n_days_series)))

        results[name] = TemporalICResult(
            feature_name=name,
            feature_index=j,
            best_horizon=best_horizon,
            rolling_mean_ic=rm_ic,
            rolling_slope_ic=rs_ic,
            rate_of_change_ic=roc_ic,
            best_temporal_ic=best_ic,
            best_temporal_metric=best_metric,
            best_temporal_p=best_p,
            n_days=n_days,
            passes_path3=best_passes,
        )

    return results
