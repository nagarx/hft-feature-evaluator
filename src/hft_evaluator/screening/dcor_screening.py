"""
Path 2: dCor + MI non-linear independence screening.

Uses subsampled pooled permutation tests (NOT per-day t-test) because dCor has
massive positive bias under independence (mean=0.102 at n=308). T-testing daily
dCor values against 0 produces 100% false positive rate.

KSG MI is approximately unbiased, but for consistency both dCor and MI use
subsampled pooled permutation tests on the same subsample.

dCor and MI are separate BH test families. A feature passes Path 2 only if
BOTH dCor AND MI are BH-significant at any horizon.

Reference: CODEBASE.md Section 4.2, Framework Section 6.2 Path 2
"""

from dataclasses import dataclass
from collections import defaultdict

import numpy as np

from hft_metrics.dcor import dcor_test
from hft_metrics.mi import ksg_mi_test
from hft_metrics.testing import benjamini_hochberg

from hft_evaluator.config import EvaluationConfig
from hft_evaluator.data.loader import ExportLoader
from hft_evaluator.screening import bh_adjusted_pvalues, _test_seed


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DcorScreeningResult:
    """Result for one (feature, horizon) pair from dCor+MI screening."""

    feature_name: str
    feature_index: int
    horizon: int
    dcor_value: float
    dcor_p: float           # BH-adjusted from permutation test
    mi_value: float
    mi_p: float             # BH-adjusted from permutation test
    n_subsample: int
    dcor_bh_rejected: bool
    mi_bh_rejected: bool
    passes_path2: bool      # BOTH dcor AND mi must be BH-rejected


# ---------------------------------------------------------------------------
# Main screening function
# ---------------------------------------------------------------------------


def screen_dcor(
    loader: ExportLoader,
    evaluation_dates: list[str],
    evaluable_indices: list[int],
    config: EvaluationConfig,
) -> dict[str, dict[int, DcorScreeningResult]]:
    """dCor + MI screening via subsampled pooled permutation tests.

    Algorithm:
        Phase 1: Pool all data across evaluation days, subsample.
        Phase 2: For each (feature, horizon): run dcor_test + ksg_mi_test on subsample.
        Phase 3: BH correction — separate families for dCor and MI.
        Phase 4: Pass/fail — BOTH dCor AND MI must be BH-significant.

    Args:
        loader: ExportLoader with detected schema.
        evaluation_dates: Dates to evaluate on.
        evaluable_indices: Feature indices that passed pre-screening.
        config: EvaluationConfig with screening params.

    Returns:
        dict[feature_name -> dict[horizon -> DcorScreeningResult]]
    """
    schema = loader.schema
    horizons = list(config.screening.horizons)
    feature_names = schema.feature_names
    n_subsample = config.screening.dcor_subsample

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

    # Subsample for computational feasibility
    rng = np.random.RandomState(config.seed)
    if n_total > n_subsample:
        sub_idx = rng.choice(n_total, size=n_subsample, replace=False)
        sub_features = pooled_features[sub_idx]
        sub_labels = pooled_labels[sub_idx]
        actual_n = n_subsample
    else:
        sub_features = pooled_features
        sub_labels = pooled_labels
        actual_n = n_total

    # Phase 2: Permutation tests per (feature, horizon)
    pair_keys: list[tuple[int, int]] = []
    dcor_raw_p: list[float] = []
    mi_raw_p: list[float] = []
    dcor_values: list[float] = []
    mi_values: list[float] = []

    for j in evaluable_indices:
        feature_col = sub_features[:, j]
        for h_idx in range(len(horizons)):
            if h_idx >= sub_labels.shape[1]:
                continue
            label_col = sub_labels[:, h_idx]

            # dCor permutation test — unique seed per (feature, horizon)
            pair_seed = _test_seed(config.seed, j, h_idx)
            dc_val, dc_p = dcor_test(
                feature_col, label_col,
                n_permutations=config.screening.dcor_permutations,
                seed=pair_seed,
            )

            # MI permutation test — offset seed to avoid overlap with dCor
            mi_val, mi_p = ksg_mi_test(
                feature_col, label_col,
                k=config.screening.mi_k,
                n_permutations=config.screening.mi_permutations,
                seed=pair_seed + 1_000_000,
            )

            pair_keys.append((j, h_idx))
            dcor_values.append(dc_val)
            dcor_raw_p.append(dc_p)
            mi_values.append(mi_val)
            mi_raw_p.append(mi_p)

    if not pair_keys:
        return {}

    # Phase 3: BH correction — SEPARATE families
    dcor_p_array = np.array(dcor_raw_p)
    mi_p_array = np.array(mi_raw_p)

    dcor_bh_mask = benjamini_hochberg(dcor_p_array, q=config.screening.bh_fdr_level)
    mi_bh_mask = benjamini_hochberg(mi_p_array, q=config.screening.bh_fdr_level)

    dcor_adj = bh_adjusted_pvalues(dcor_p_array)
    mi_adj = bh_adjusted_pvalues(mi_p_array)

    # Phase 4: Build results
    results: dict[str, dict[int, DcorScreeningResult]] = defaultdict(dict)

    for idx, (j, h_idx) in enumerate(pair_keys):
        horizon = horizons[h_idx]
        name = feature_names.get(j, f"feature_{j}")

        dcor_rejected = bool(dcor_bh_mask[idx])
        mi_rejected = bool(mi_bh_mask[idx])

        results[name][horizon] = DcorScreeningResult(
            feature_name=name,
            feature_index=j,
            horizon=horizon,
            dcor_value=dcor_values[idx],
            dcor_p=float(dcor_adj[idx]),
            mi_value=mi_values[idx],
            mi_p=float(mi_adj[idx]),
            n_subsample=actual_n,
            dcor_bh_rejected=dcor_rejected,
            mi_bh_rejected=mi_rejected,
            passes_path2=(dcor_rejected and mi_rejected),
        )

    return dict(results)


# ---------------------------------------------------------------------------
# Cache-based variant (Phase 1: no disk I/O)
# ---------------------------------------------------------------------------


def screen_dcor_from_cache(
    cache: "DataCache",
    config: EvaluationConfig,
) -> dict[str, dict[int, DcorScreeningResult]]:
    """dCor + MI screening from pre-pooled cache data. No disk I/O.

    Subsamples from cache.pooled_features/labels using its own fresh RNG
    (identical behavior to screen_dcor).

    Args:
        cache: DataCache from build_cache().
        config: EvaluationConfig with screening params.

    Returns:
        dict[feature_name -> dict[horizon -> DcorScreeningResult]]
    """
    from hft_evaluator.data.cache import DataCache  # late import

    schema = cache.schema
    horizons = list(cache.horizons)
    feature_names = schema.feature_names
    n_subsample = config.screening.dcor_subsample

    # Subsample from pooled cache
    n_total = cache.pooled_features.shape[0]
    rng = np.random.RandomState(config.seed)

    if n_total > n_subsample:
        sub_idx = rng.choice(n_total, size=n_subsample, replace=False)
        sub_features = cache.pooled_features[sub_idx]
        sub_labels = cache.pooled_labels[sub_idx]
        actual_n = n_subsample
    else:
        sub_features = cache.pooled_features
        sub_labels = cache.pooled_labels
        actual_n = n_total

    if actual_n == 0:
        return {}

    # Phase 2: Permutation tests (identical to screen_dcor)
    pair_keys: list[tuple[int, int]] = []
    dcor_raw_p: list[float] = []
    mi_raw_p: list[float] = []
    dcor_values: list[float] = []
    mi_values: list[float] = []

    for j in cache.evaluable_indices:
        feature_col = sub_features[:, j]
        for h_idx in range(len(horizons)):
            if h_idx >= sub_labels.shape[1]:
                continue
            label_col = sub_labels[:, h_idx]

            pair_seed = _test_seed(config.seed, j, h_idx)
            dc_val, dc_p = dcor_test(
                feature_col, label_col,
                n_permutations=config.screening.dcor_permutations,
                seed=pair_seed,
            )

            mi_val, mi_p = ksg_mi_test(
                feature_col, label_col,
                k=config.screening.mi_k,
                n_permutations=config.screening.mi_permutations,
                seed=pair_seed + 1_000_000,
            )

            pair_keys.append((j, h_idx))
            dcor_values.append(dc_val)
            dcor_raw_p.append(dc_p)
            mi_values.append(mi_val)
            mi_raw_p.append(mi_p)

    if not pair_keys:
        return {}

    # Phase 3: BH correction
    dcor_p_array = np.array(dcor_raw_p)
    mi_p_array = np.array(mi_raw_p)

    dcor_bh_mask = benjamini_hochberg(dcor_p_array, q=config.screening.bh_fdr_level)
    mi_bh_mask = benjamini_hochberg(mi_p_array, q=config.screening.bh_fdr_level)

    dcor_adj = bh_adjusted_pvalues(dcor_p_array)
    mi_adj = bh_adjusted_pvalues(mi_p_array)

    # Phase 4: Build results
    results: dict[str, dict[int, DcorScreeningResult]] = defaultdict(dict)

    for idx, (j, h_idx) in enumerate(pair_keys):
        horizon = horizons[h_idx]
        name = feature_names.get(j, f"feature_{j}")

        dcor_rejected = bool(dcor_bh_mask[idx])
        mi_rejected = bool(mi_bh_mask[idx])

        results[name][horizon] = DcorScreeningResult(
            feature_name=name,
            feature_index=j,
            horizon=horizon,
            dcor_value=dcor_values[idx],
            dcor_p=float(dcor_adj[idx]),
            mi_value=mi_values[idx],
            mi_p=float(mi_adj[idx]),
            n_subsample=actual_n,
            dcor_bh_rejected=dcor_rejected,
            mi_bh_rejected=mi_rejected,
            passes_path2=(dcor_rejected and mi_rejected),
        )

    return dict(results)
