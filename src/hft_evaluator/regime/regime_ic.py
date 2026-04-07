"""
Path 4: Regime-conditional IC (spread/time/activity terciles).

Computes IC within each tercile of conditioning variables, then checks
whether signal persists across market regimes.

Sample budget (Framework §9):
    Single-variable tercile: ~22,962/cell (7.3x required n=3,138)
    Two-variable cross (9 cells): ~7,654/cell (2.4x)
    Three-variable cross: INFEASIBLE. DO NOT USE.

Reference: CODEBASE.md Section 7.1, Framework Section 6.2 Path 4
"""

from dataclasses import dataclass
from collections import defaultdict

import numpy as np

from hft_metrics.ic import windowed_ic
from hft_metrics.discretization import quantile_buckets

from hft_evaluator.config import EvaluationConfig
from hft_evaluator.data.loader import ExportLoader


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeTercileResult:
    """IC result for one tercile of one conditioning variable."""

    tercile_name: str       # "LOW", "MEDIUM", "HIGH"
    ic: float
    ci_lower: float
    ci_upper: float
    n_samples: int
    ci_excludes_zero: bool


@dataclass(frozen=True)
class RegimeICResult:
    """Result for one (feature, horizon, conditioning variable) triplet."""

    feature_name: str
    feature_index: int
    horizon: int
    conditioning_variable: str
    per_tercile: tuple[RegimeTercileResult, ...]
    best_tercile: str
    best_tercile_ic: float
    passes_path4: bool     # CI excludes zero in ANY tercile


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def compute_regime_ic(
    loader: ExportLoader,
    evaluation_dates: list[str],
    evaluable_indices: list[int],
    horizons: list[int],
    conditioning_indices: dict[str, int],
    config: EvaluationConfig,
) -> dict[str, dict[int, list[RegimeICResult]]]:
    """Regime-conditional IC for all features at all horizons.

    Two passes:
        Pass 1: Stream through eval_dates, pool features and labels.
        Pass 2: For each conditioning variable, compute tercile bins,
                 then windowed_ic per evaluable feature and horizon.

    Args:
        loader: ExportLoader with detected schema.
        evaluation_dates: Dates to evaluate on.
        evaluable_indices: Feature indices that passed pre-screening.
        horizons: Horizon values to evaluate.
        conditioning_indices: {name: feature_index} for conditioning vars.
        config: EvaluationConfig with regime params.

    Returns:
        dict[feature_name -> dict[horizon -> list[RegimeICResult]]]
        The list has one entry per conditioning variable.
    """
    schema = loader.schema
    feature_names = schema.feature_names

    # Pass 1: Pool all features and labels across evaluation days
    all_features_list: list[np.ndarray] = []  # Each [N, F]
    all_labels_list: list[np.ndarray] = []    # Each [N, H]

    for bundle in loader.iter_days(evaluation_dates):
        if bundle.sequences.shape[0] < 1:
            continue
        features_2d = bundle.sequences[:, -1, :]  # [N, F] last timestep
        all_features_list.append(np.asarray(features_2d, dtype=np.float64))
        all_labels_list.append(np.asarray(bundle.labels, dtype=np.float64))

    if not all_features_list:
        return {}

    pooled_features = np.concatenate(all_features_list, axis=0)  # [N_total, F]
    pooled_labels = np.concatenate(all_labels_list, axis=0)      # [N_total, H]

    # Pass 2: Per conditioning variable
    results: dict[str, dict[int, list[RegimeICResult]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for cond_name, cond_idx in conditioning_indices.items():
        if cond_idx >= pooled_features.shape[1]:
            continue

        cond_values = pooled_features[:, cond_idx]

        # Compute tercile bins
        bins, bin_names = quantile_buckets(cond_values, n_bins=config.regime.n_bins)

        for j in evaluable_indices:
            feature_col = pooled_features[:, j]
            name = feature_names.get(j, f"feature_{j}")

            for h_idx, horizon in enumerate(horizons):
                if h_idx >= pooled_labels.shape[1]:
                    continue
                label_col = pooled_labels[:, h_idx]

                # windowed_ic returns list[(bin_id, ic, ci_lo, ci_hi, n)]
                ic_results = windowed_ic(
                    feature_col,
                    label_col,
                    bins.astype(np.int32),
                    min_samples=config.regime.min_samples_per_bin,
                    ci=0.95,
                    seed=config.seed,
                )

                # Build per-tercile results
                tercile_results = []
                for bin_id, ic, ci_lo, ci_hi, n_samples in ic_results:
                    tercile_name = bin_names.get(bin_id, f"bin_{bin_id}")
                    ci_excludes_zero = (ci_lo > 0) or (ci_hi < 0)
                    tercile_results.append(RegimeTercileResult(
                        tercile_name=tercile_name,
                        ic=ic,
                        ci_lower=ci_lo,
                        ci_upper=ci_hi,
                        n_samples=n_samples,
                        ci_excludes_zero=ci_excludes_zero,
                    ))

                # Path 4 pass: ANY tercile has CI excluding zero
                passes = any(tr.ci_excludes_zero for tr in tercile_results)

                # Best tercile (by absolute IC)
                if tercile_results:
                    best = max(tercile_results, key=lambda tr: abs(tr.ic))
                    best_name = best.tercile_name
                    best_ic = best.ic
                else:
                    best_name = ""
                    best_ic = 0.0
                    passes = False

                results[name][horizon].append(RegimeICResult(
                    feature_name=name,
                    feature_index=j,
                    horizon=horizon,
                    conditioning_variable=cond_name,
                    per_tercile=tuple(tercile_results),
                    best_tercile=best_name,
                    best_tercile_ic=best_ic,
                    passes_path4=passes,
                ))

    # Convert nested defaultdicts to plain dicts
    return {k: dict(v) for k, v in results.items()}


# ---------------------------------------------------------------------------
# Cache-based variant (Phase 1: no disk I/O)
# ---------------------------------------------------------------------------


def compute_regime_ic_from_cache(
    cache: "DataCache",
    horizons: list[int],
    conditioning_indices: dict[str, int],
    config: EvaluationConfig,
) -> dict[str, dict[int, list[RegimeICResult]]]:
    """Regime IC from pre-pooled cache data. No disk I/O.

    Args:
        cache: DataCache from build_cache().
        horizons: Horizon values.
        conditioning_indices: {name: feature_index} for conditioning.
        config: EvaluationConfig with regime params.

    Returns:
        dict[feature_name -> dict[horizon -> list[RegimeICResult]]]
    """
    from hft_evaluator.data.cache import DataCache

    schema = cache.schema
    feature_names = schema.feature_names

    results: dict[str, dict[int, list[RegimeICResult]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for cond_name, cond_idx in conditioning_indices.items():
        if cond_idx >= cache.pooled_features.shape[1]:
            continue

        cond_values = cache.pooled_features[:, cond_idx]
        bins, bin_names = quantile_buckets(
            cond_values, n_bins=config.regime.n_bins
        )

        for j in cache.evaluable_indices:
            feature_col = cache.pooled_features[:, j]
            name = feature_names.get(j, f"feature_{j}")

            for h_idx, horizon in enumerate(horizons):
                if h_idx >= cache.pooled_labels.shape[1]:
                    continue
                label_col = cache.pooled_labels[:, h_idx]

                ic_results = windowed_ic(
                    feature_col, label_col,
                    bins.astype(np.int32),
                    min_samples=config.regime.min_samples_per_bin,
                    ci=0.95, seed=config.seed,
                )

                tercile_results = []
                for bin_id, ic, ci_lo, ci_hi, n_samples in ic_results:
                    tercile_name = bin_names.get(bin_id, f"bin_{bin_id}")
                    ci_excludes_zero = (ci_lo > 0) or (ci_hi < 0)
                    tercile_results.append(RegimeTercileResult(
                        tercile_name=tercile_name,
                        ic=ic, ci_lower=ci_lo, ci_upper=ci_hi,
                        n_samples=n_samples,
                        ci_excludes_zero=ci_excludes_zero,
                    ))

                passes = any(tr.ci_excludes_zero for tr in tercile_results)
                if tercile_results:
                    best = max(tercile_results, key=lambda tr: abs(tr.ic))
                    best_name = best.tercile_name
                    best_ic = best.ic
                else:
                    best_name = ""
                    best_ic = 0.0
                    passes = False

                results[name][horizon].append(RegimeICResult(
                    feature_name=name,
                    feature_index=j,
                    horizon=horizon,
                    conditioning_variable=cond_name,
                    per_tercile=tuple(tercile_results),
                    best_tercile=best_name,
                    best_tercile_ic=best_ic,
                    passes_path4=passes,
                ))

    return {k: dict(v) for k, v in results.items()}
