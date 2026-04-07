"""
Path 3b: Transfer entropy screening with BH correction.

Uses subsampled pooled transfer_entropy_test (NOT per-day t-test) because TE
has positive bias under independence (mean=+0.051 at n=308).

CRITICAL: Uses horizon=1 in transfer_entropy call. Labels are already forward
returns at horizon h — using horizon=h would double-count the lookahead.

Reference: CODEBASE.md Section 6.2, Framework Section 2.3
"""

from dataclasses import dataclass
from collections import defaultdict

import numpy as np

from hft_metrics.transfer_entropy import transfer_entropy_test
from hft_metrics.testing import benjamini_hochberg

from hft_evaluator.config import EvaluationConfig
from hft_evaluator.data.loader import ExportLoader
from hft_evaluator.screening import bh_adjusted_pvalues, _test_seed

_TE_SUBSAMPLE = 1000  # Smaller than dCor because conditional MI is more expensive


@dataclass(frozen=True)
class TEScreeningResult:
    """Result for one (feature, horizon) pair from TE screening."""

    feature_name: str
    feature_index: int
    horizon: int
    best_lag: int
    te_value: float
    bh_adjusted_p: float
    n_subsample: int
    passes_te: bool


def screen_transfer_entropy(
    loader: ExportLoader,
    evaluation_dates: list[str],
    evaluable_indices: list[int],
    horizons: list[int],
    config: EvaluationConfig,
) -> dict[str, dict[int, TEScreeningResult]]:
    """Transfer entropy screening via subsampled pooled permutation tests.

    Algorithm:
        Phase 1: Pool all bin-level data, subsample to ~1000.
        Phase 2: For each (feature, horizon, lag): transfer_entropy_test with horizon=1.
        Phase 3: BH correction across all (feature, horizon, lag) triplets.
        Phase 4: Best lag selection per (feature, horizon).

    Returns:
        dict[feature_name -> dict[horizon -> TEScreeningResult]]
    """
    schema = loader.schema
    feature_names = schema.feature_names
    te_lags = list(config.temporal.te_lags)

    # Phase 1: Pool and subsample
    all_features_list: list[np.ndarray] = []
    all_labels_list: list[np.ndarray] = []

    for bundle in loader.iter_days(evaluation_dates):
        if bundle.sequences.shape[0] < 1:
            continue
        features_2d = bundle.sequences[:, -1, :]
        all_features_list.append(np.asarray(features_2d, dtype=np.float64))
        all_labels_list.append(np.asarray(bundle.labels, dtype=np.float64))

    if not all_features_list:
        return {}

    pooled_features = np.concatenate(all_features_list, axis=0)
    pooled_labels = np.concatenate(all_labels_list, axis=0)
    n_total = pooled_features.shape[0]

    rng = np.random.RandomState(config.seed)
    if n_total > _TE_SUBSAMPLE:
        sub_idx = rng.choice(n_total, size=_TE_SUBSAMPLE, replace=False)
        sub_features = pooled_features[sub_idx]
        sub_labels = pooled_labels[sub_idx]
        actual_n = _TE_SUBSAMPLE
    else:
        sub_features = pooled_features
        sub_labels = pooled_labels
        actual_n = n_total

    # Phase 2: Permutation tests per (feature, horizon, lag)
    triplet_keys: list[tuple[int, int, int]] = []  # (j, h_idx, lag)
    raw_p: list[float] = []
    te_values: list[float] = []

    for j in evaluable_indices:
        feature_col = sub_features[:, j]
        for h_idx in range(len(horizons)):
            if h_idx >= sub_labels.shape[1]:
                continue
            label_col = sub_labels[:, h_idx]

            for lag_idx, lag in enumerate(te_lags):
                # Unique seed per (feature, horizon, lag) triplet
                te_seed = _test_seed(config.seed, j, h_idx, lag_idx)
                te_val, te_p = transfer_entropy_test(
                    feature_col, label_col,
                    lag=lag, horizon=1,  # horizon=1: labels already forward-looking
                    k=config.screening.mi_k,
                    n_permutations=100,  # Informational only — cannot survive BH
                    seed=te_seed,
                )
                triplet_keys.append((j, h_idx, lag))
                te_values.append(te_val)
                raw_p.append(te_p)

    if not triplet_keys:
        return {}

    # Phase 3: BH correction across all triplets
    raw_p_array = np.array(raw_p)
    bh_mask = benjamini_hochberg(raw_p_array, q=config.screening.bh_fdr_level)
    adj_p = bh_adjusted_pvalues(raw_p_array)

    # Phase 4: Best lag per (feature, horizon)
    # Build lookup: (j, h_idx, lag) -> index
    idx_lookup = {key: idx for idx, key in enumerate(triplet_keys)}

    results: dict[str, dict[int, TEScreeningResult]] = defaultdict(dict)

    for j in evaluable_indices:
        name = feature_names.get(j, f"feature_{j}")
        for h_idx, h in enumerate(horizons):
            # Find best lag for this (feature, horizon)
            best_lag = te_lags[0]
            best_te = 0.0
            best_adj_p = 1.0
            best_rejected = False

            for lag in te_lags:
                key = (j, h_idx, lag)
                if key not in idx_lookup:
                    continue
                idx = idx_lookup[key]
                if adj_p[idx] < best_adj_p:
                    best_lag = lag
                    best_te = te_values[idx]
                    best_adj_p = float(adj_p[idx])
                    best_rejected = bool(bh_mask[idx])

            results[name][h] = TEScreeningResult(
                feature_name=name,
                feature_index=j,
                horizon=h,
                best_lag=best_lag,
                te_value=best_te,
                bh_adjusted_p=best_adj_p,
                n_subsample=actual_n,
                passes_te=best_rejected,
            )

    return dict(results)


# ---------------------------------------------------------------------------
# Cache-based variant (Phase 1: no disk I/O)
# ---------------------------------------------------------------------------


def screen_te_from_cache(
    cache: "DataCache",
    horizons: list[int],
    config: EvaluationConfig,
) -> dict[str, dict[int, TEScreeningResult]]:
    """TE screening from pre-pooled cache data. No disk I/O.

    Informational only — TE p-values cannot survive BH correction
    with current permutation counts.

    Args:
        cache: DataCache from build_cache().
        horizons: Horizon values.
        config: EvaluationConfig.

    Returns:
        dict[feature_name -> dict[horizon -> TEScreeningResult]]
    """
    from hft_evaluator.data.cache import DataCache

    schema = cache.schema
    feature_names = schema.feature_names
    te_lags = list(config.temporal.te_lags)

    # Subsample for TE (smaller than dCor)
    n_total = cache.pooled_features.shape[0]
    rng = np.random.RandomState(config.seed)
    if n_total > _TE_SUBSAMPLE:
        sub_idx = rng.choice(n_total, size=_TE_SUBSAMPLE, replace=False)
        sub_features = cache.pooled_features[sub_idx]
        sub_labels = cache.pooled_labels[sub_idx]
        actual_n = _TE_SUBSAMPLE
    else:
        sub_features = cache.pooled_features
        sub_labels = cache.pooled_labels
        actual_n = n_total

    if actual_n == 0:
        return {}

    # Phase 2: Permutation tests
    triplet_keys: list[tuple[int, int, int]] = []
    raw_p: list[float] = []
    te_values: list[float] = []

    for j in cache.evaluable_indices:
        feature_col = sub_features[:, j]
        for h_idx in range(len(horizons)):
            if h_idx >= sub_labels.shape[1]:
                continue
            label_col = sub_labels[:, h_idx]

            for lag_idx, lag in enumerate(te_lags):
                te_seed = _test_seed(config.seed, j, h_idx, lag_idx)
                te_val, te_p = transfer_entropy_test(
                    feature_col, label_col,
                    lag=lag, horizon=1,
                    k=config.screening.mi_k,
                    n_permutations=100,
                    seed=te_seed,
                )
                triplet_keys.append((j, h_idx, lag))
                te_values.append(te_val)
                raw_p.append(te_p)

    if not triplet_keys:
        return {}

    # Phase 3: BH correction
    raw_p_array = np.array(raw_p)
    bh_mask = benjamini_hochberg(raw_p_array, q=config.screening.bh_fdr_level)
    adj_p = bh_adjusted_pvalues(raw_p_array)

    # Phase 4: Best lag per (feature, horizon)
    idx_lookup = {key: idx for idx, key in enumerate(triplet_keys)}
    results: dict[str, dict[int, TEScreeningResult]] = defaultdict(dict)

    for j in cache.evaluable_indices:
        name = feature_names.get(j, f"feature_{j}")
        for h_idx, h in enumerate(horizons):
            best_lag = te_lags[0]
            best_te = 0.0
            best_adj_p = 1.0
            best_rejected = False

            for lag in te_lags:
                key = (j, h_idx, lag)
                if key not in idx_lookup:
                    continue
                idx = idx_lookup[key]
                if adj_p[idx] < best_adj_p:
                    best_lag = lag
                    best_te = te_values[idx]
                    best_adj_p = float(adj_p[idx])
                    best_rejected = bool(bh_mask[idx])

            results[name][h] = TEScreeningResult(
                feature_name=name,
                feature_index=j,
                horizon=h,
                best_lag=best_lag,
                te_value=best_te,
                bh_adjusted_p=best_adj_p,
                n_subsample=actual_n,
                passes_te=best_rejected,
            )

    return dict(results)
