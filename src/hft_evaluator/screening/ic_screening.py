"""
Path 1: Forward IC screening with per-day aggregation.

Computes Spearman IC per day, aggregates across days with block bootstrap CI,
applies BH correction, and classifies features as passing/failing Path 1.

CRITICAL: Uses per-day IC aggregation, NOT pooled-sample CI. Within-day
samples have ACF ~0.97 (stride=1 overlap), making pooled CIs unreliable.

Reference: CODEBASE.md Section 4.1, Framework Section 6.2 Path 1
"""

from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from scipy import stats as scipy_stats

from hft_metrics.ic import spearman_ic, ic_ir
from hft_metrics.bootstrap import block_bootstrap_ci
from hft_metrics.testing import benjamini_hochberg

from hft_evaluator.config import EvaluationConfig
from hft_evaluator.data.loader import ExportLoader
from hft_evaluator.screening import bh_adjusted_pvalues


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ICScreeningResult:
    """Result for one (feature, horizon) pair from IC screening."""

    feature_name: str
    feature_index: int
    horizon: int
    ic_mean: float         # Mean of daily ICs
    raw_p: float           # t-test p-value (H0: mean daily IC = 0)
    bh_adjusted_p: float   # BH-adjusted p-value
    ci_lower: float        # Block bootstrap 95% CI lower bound
    ci_upper: float        # Block bootstrap 95% CI upper bound
    ic_ir: float           # IC Information Ratio: mean / std
    n_days: int            # Number of days with valid IC
    bh_rejected: bool      # True if BH-corrected p < fdr_level
    passes_path1: bool     # Full Path 1 pass criteria


# ---------------------------------------------------------------------------
# Main screening function
# ---------------------------------------------------------------------------


def screen_ic(
    loader: ExportLoader,
    evaluation_dates: list[str],
    evaluable_indices: list[int],
    config: EvaluationConfig,
) -> dict[str, dict[int, ICScreeningResult]]:
    """Forward IC screening for all evaluable features at all horizons.

    Algorithm:
        Phase 1: Per-day IC computation (streaming, O(1 day) memory)
        Phase 2: Cross-day aggregation (bootstrap CI, IC_IR, t-test p-value)
        Phase 3: BH correction + adjusted p-values
        Phase 4: Pass/fail classification

    Args:
        loader: ExportLoader with detected schema.
        evaluation_dates: Dates to evaluate on (excludes holdout).
        evaluable_indices: Feature indices that passed pre-screening.
        config: EvaluationConfig with screening and classification params.

    Returns:
        dict[feature_name -> dict[horizon -> ICScreeningResult]]
    """
    schema = loader.schema
    horizons = list(config.screening.horizons)
    feature_names = schema.feature_names

    # Phase 1: Stream through days, compute per-day ICs
    # daily_ics[j][h_idx] is a list of daily IC values
    daily_ics: dict[int, dict[int, list[float]]] = {
        j: {h_idx: [] for h_idx in range(len(horizons))}
        for j in evaluable_indices
    }

    for bundle in loader.iter_days(evaluation_dates):
        if bundle.sequences.shape[0] < 3:
            continue  # Skip days with too few samples

        features_2d = bundle.sequences[:, -1, :]  # [N, F] last timestep
        labels_2d = bundle.labels                   # [N, H]

        for j in evaluable_indices:
            feature_col = np.asarray(features_2d[:, j], dtype=np.float64)
            for h_idx in range(len(horizons)):
                if h_idx >= labels_2d.shape[1]:
                    continue
                label_col = np.asarray(labels_2d[:, h_idx], dtype=np.float64)
                rho, p = spearman_ic(feature_col, label_col)
                # Skip degenerate returns (zero-variance or insufficient samples).
                # spearman_ic returns (0.0, 1.0) for these cases.
                if not (rho == 0.0 and p == 1.0):
                    daily_ics[j][h_idx].append(rho)

    # Phase 2: Cross-day aggregation
    # Collect all (j, h_idx) -> aggregated results
    pair_keys: list[tuple[int, int]] = []  # (j, h_idx) ordering
    raw_pvalues: list[float] = []
    aggregated: dict[tuple[int, int], dict] = {}

    for j in evaluable_indices:
        for h_idx in range(len(horizons)):
            ic_array = np.array(daily_ics[j][h_idx], dtype=np.float64)
            n_days = len(ic_array)

            if n_days < 2:
                # Not enough days for meaningful statistics
                aggregated[(j, h_idx)] = {
                    "ic_mean": 0.0, "raw_p": 1.0,
                    "ci_lower": -1.0, "ci_upper": 1.0,
                    "ic_ir_val": 0.0, "n_days": n_days,
                }
                pair_keys.append((j, h_idx))
                raw_pvalues.append(1.0)
                continue

            ic_mean = float(np.mean(ic_array))
            ic_ir_val = ic_ir(ic_array)

            # Block bootstrap CI on daily IC series
            # statistic_fn takes (x, y) -> float; y is unused
            try:
                _, ci_lo, ci_hi = block_bootstrap_ci(
                    statistic_fn=lambda x, _y: float(np.mean(x)),
                    x=ic_array,
                    y=np.zeros_like(ic_array),
                    n_bootstraps=1000,
                    ci=0.95,
                    seed=config.seed,
                )
            except Exception:
                ci_lo, ci_hi = -1.0, 1.0

            # T-test p-value: H0: mean(daily_ics) = 0
            if n_days >= 3 and np.std(ic_array) > 0:
                _, raw_p = scipy_stats.ttest_1samp(ic_array, 0.0)
                raw_p = float(raw_p)
            else:
                raw_p = 1.0

            aggregated[(j, h_idx)] = {
                "ic_mean": ic_mean,
                "raw_p": raw_p,
                "ci_lower": float(ci_lo),
                "ci_upper": float(ci_hi),
                "ic_ir_val": float(ic_ir_val),
                "n_days": n_days,
            }
            pair_keys.append((j, h_idx))
            raw_pvalues.append(raw_p)

    # Phase 3: BH correction
    raw_p_array = np.array(raw_pvalues, dtype=np.float64)
    bh_mask = benjamini_hochberg(raw_p_array, q=config.screening.bh_fdr_level)
    bh_adjusted = bh_adjusted_pvalues(raw_p_array)

    # Phase 4: Build results with pass/fail
    results: dict[str, dict[int, ICScreeningResult]] = defaultdict(dict)

    for idx, (j, h_idx) in enumerate(pair_keys):
        agg = aggregated[(j, h_idx)]
        horizon = horizons[h_idx]
        name = feature_names.get(j, f"feature_{j}")

        ci_lo = agg["ci_lower"]
        ci_hi = agg["ci_upper"]
        ci_excludes_zero = (ci_lo > 0) or (ci_hi < 0)

        bh_rejected = bool(bh_mask[idx])
        adj_p = float(bh_adjusted[idx])

        passes = (
            abs(agg["ic_mean"]) > config.screening.ic_threshold
            and ci_excludes_zero
            and agg["ic_ir_val"] > config.classification.ic_ir_threshold
            and bh_rejected
        )

        results[name][horizon] = ICScreeningResult(
            feature_name=name,
            feature_index=j,
            horizon=horizon,
            ic_mean=agg["ic_mean"],
            raw_p=agg["raw_p"],
            bh_adjusted_p=adj_p,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            ic_ir=agg["ic_ir_val"],
            n_days=agg["n_days"],
            bh_rejected=bh_rejected,
            passes_path1=passes,
        )

    return dict(results)


# ---------------------------------------------------------------------------
# Cache-based variant (Phase 1: no disk I/O)
# ---------------------------------------------------------------------------


def screen_ic_from_cache(
    cache: "DataCache",
    config: EvaluationConfig,
) -> dict[str, dict[int, ICScreeningResult]]:
    """IC screening from pre-computed daily IC cube. No disk I/O.

    Uses cache.daily_ic_cube[D, F_eval, H] where NaN marks invalid days.
    Aggregation logic (Phase 2-4) is identical to screen_ic().

    Args:
        cache: DataCache from build_cache().
        config: EvaluationConfig with screening params.

    Returns:
        dict[feature_name -> dict[horizon -> ICScreeningResult]]
    """
    from hft_evaluator.data.cache import DataCache  # late import

    schema = cache.schema
    horizons = list(cache.horizons)
    feature_names = schema.feature_names

    # Phase 2: Cross-day aggregation from cube
    pair_keys: list[tuple[int, int]] = []
    raw_pvalues: list[float] = []
    aggregated: dict[tuple[int, int], dict] = {}

    for pos, j in enumerate(cache.evaluable_indices):
        for h_idx in range(len(horizons)):
            ic_series = cache.daily_ic_cube[:, pos, h_idx]
            ic_array = ic_series[np.isfinite(ic_series)]
            n_days = len(ic_array)

            if n_days < 2:
                aggregated[(j, h_idx)] = {
                    "ic_mean": 0.0, "raw_p": 1.0,
                    "ci_lower": -1.0, "ci_upper": 1.0,
                    "ic_ir_val": 0.0, "n_days": n_days,
                }
                pair_keys.append((j, h_idx))
                raw_pvalues.append(1.0)
                continue

            ic_mean = float(np.mean(ic_array))
            ic_ir_val = ic_ir(ic_array)

            try:
                _, ci_lo, ci_hi = block_bootstrap_ci(
                    statistic_fn=lambda x, _y: float(np.mean(x)),
                    x=ic_array,
                    y=np.zeros_like(ic_array),
                    n_bootstraps=1000,
                    ci=0.95,
                    seed=config.seed,
                )
            except (ValueError, np.linalg.LinAlgError):
                ci_lo, ci_hi = -1.0, 1.0

            if n_days >= 3 and np.std(ic_array) > 0:
                _, raw_p = scipy_stats.ttest_1samp(ic_array, 0.0)
                raw_p = float(raw_p)
            else:
                raw_p = 1.0

            aggregated[(j, h_idx)] = {
                "ic_mean": ic_mean,
                "raw_p": raw_p,
                "ci_lower": float(ci_lo),
                "ci_upper": float(ci_hi),
                "ic_ir_val": float(ic_ir_val),
                "n_days": n_days,
            }
            pair_keys.append((j, h_idx))
            raw_pvalues.append(raw_p)

    # Phase 3: BH correction
    raw_p_array = np.array(raw_pvalues, dtype=np.float64)
    bh_mask = benjamini_hochberg(raw_p_array, q=config.screening.bh_fdr_level)
    bh_adjusted = bh_adjusted_pvalues(raw_p_array)

    # Phase 4: Build results
    results: dict[str, dict[int, ICScreeningResult]] = defaultdict(dict)

    for idx, (j, h_idx) in enumerate(pair_keys):
        agg = aggregated[(j, h_idx)]
        horizon = horizons[h_idx]
        name = feature_names.get(j, f"feature_{j}")

        ci_lo = agg["ci_lower"]
        ci_hi = agg["ci_upper"]
        ci_excludes_zero = (ci_lo > 0) or (ci_hi < 0)

        bh_rejected = bool(bh_mask[idx])
        adj_p = float(bh_adjusted[idx])

        passes = (
            abs(agg["ic_mean"]) > config.screening.ic_threshold
            and ci_excludes_zero
            and agg["ic_ir_val"] > config.classification.ic_ir_threshold
            and bh_rejected
        )

        results[name][horizon] = ICScreeningResult(
            feature_name=name,
            feature_index=j,
            horizon=horizon,
            ic_mean=agg["ic_mean"],
            raw_p=agg["raw_p"],
            bh_adjusted_p=adj_p,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            ic_ir=agg["ic_ir_val"],
            n_days=agg["n_days"],
            bh_rejected=bh_rejected,
            passes_path1=passes,
        )

    return dict(results)
