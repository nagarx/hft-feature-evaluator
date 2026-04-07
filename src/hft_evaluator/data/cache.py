"""
Unified data cache: single load, all paths read from memory.

Eliminates 49-109 redundant disk scans by loading data in two passes:
    Pass 1 (pre-screen): Lightweight streaming for variance → evaluable indices.
    Pass 2 (main): One streaming pass computes per-day ICs (Path 1, 3a, CF),
                    accumulates pooled data (Paths 2, 3b, 4, 5).

After construction, all evaluation paths read from this cache.
No path touches the loader directly.

Reference: Plan Phase 1 — Unified Data Cache
"""

import logging
from dataclasses import dataclass

import numpy as np

from hft_metrics.welford import StreamingColumnStats
from hft_metrics.ic import spearman_ic
from hft_metrics.temporal import rolling_mean, rolling_slope, rate_of_change

from hft_evaluator.config import EvaluationConfig
from hft_evaluator.data.loader import ExportLoader, ExportSchema

logger = logging.getLogger(__name__)

# Temporal metrics computed in the cache
TEMPORAL_METRICS = ("rolling_mean", "rolling_slope", "rate_of_change")


@dataclass
class DataCache:
    """Shared data cache populated in two streaming passes.

    All evaluation paths read from this cache instead of the loader.

    Cube layout:
        daily_ic_cube[d, f, h] = Spearman IC for day d, evaluable feature f, horizon h.
        NaN where IC could not be computed (degenerate day/feature).
        Feature axis maps to evaluable_indices[f].
        Day axis maps to evaluation_dates[d].

    Pooled layout:
        pooled_features[n, :] = last-timestep features for sample n (ALL features).
        pooled_labels[n, :] = labels for sample n.
        Each _from_cache path subsamples independently for its own needs.

    Attrs:
        schema: Detected export schema.
        evaluation_dates: Dates used for evaluation (excludes holdout).
        evaluable_indices: Sorted feature indices that passed pre-screening.
        excluded_features: {feature_name: exclusion_reason}.
        horizons: Horizon values from config.
        seed: Global random seed from config.
        daily_ic_cube: [D, F_eval, H] float64 — Path 1 per-day ICs.
        daily_temporal_cubes: {metric_name: [D, F_eval, H]} — Path 3a per-day ICs.
        daily_forward_ic_cube: [D, F_eval, H] float64 — CF forward ICs.
        daily_concurrent_ic_cube: [D, F_eval, H] float64 — CF concurrent ICs.
        pooled_features: [N_total, F_all] float64 — full pool (not subsampled).
        pooled_labels: [N_total, H] float64.
        n_total_samples: Total samples before any subsampling.
    """

    schema: ExportSchema
    evaluation_dates: tuple[str, ...]
    evaluable_indices: tuple[int, ...]
    excluded_features: dict[str, str]
    horizons: tuple[int, ...]
    seed: int

    # Per-day IC cubes — NaN where invalid
    daily_ic_cube: np.ndarray                       # [D, F_eval, H]
    daily_temporal_cubes: dict[str, np.ndarray]     # metric -> [D, F_eval, H]
    daily_forward_ic_cube: np.ndarray               # [D, F_eval, H]
    daily_concurrent_ic_cube: np.ndarray            # [D, F_eval, H]

    # Pooled data
    pooled_features: np.ndarray                     # [N_total, F_all]
    pooled_labels: np.ndarray                       # [N_total, H]
    pooled_date_indices: np.ndarray                 # [N_total] int32 — day index per sample
    n_total_samples: int


def build_cache(
    loader: ExportLoader,
    evaluation_dates: list[str],
    config: EvaluationConfig,
) -> DataCache:
    """Build unified data cache in two streaming passes.

    Pass 1 (pre-screen): Stream all days, compute column variance via
        StreamingColumnStats. Determine evaluable indices from variance
        + categorical exclusion. Lightweight: O(N×F) per day, no Spearman.

    Pass 2 (main): Stream all days again. For each day:
        - Compute per-day Spearman IC for all (evaluable_feature, horizon).
        - Compute per-day temporal rolling ICs (3D vectorized rolling).
        - Compute per-day forward/concurrent ICs (CF decomposition).
        - Accumulate last-timestep features + labels for pooling.
        With mmap, Pass 2 finds pages cached from Pass 1 → effective I/O ≈ 1 pass.

    Args:
        loader: ExportLoader with detected schema.
        evaluation_dates: Dates to evaluate (excludes holdout).
        config: EvaluationConfig with screening params.

    Returns:
        DataCache with all per-day cubes and pooled data.
    """
    schema = loader.schema
    horizons = list(config.screening.horizons)
    n_horizons = len(horizons)
    rolling_window = config.temporal.rolling_window

    # ---------------------------------------------------------------
    # Pass 1: Pre-screen (variance + categorical exclusion)
    # ---------------------------------------------------------------
    streaming_stats = StreamingColumnStats(n_columns=schema.n_features)

    for bundle in loader.iter_days(evaluation_dates):
        features_2d = bundle.sequences[:, -1, :]
        streaming_stats.update_batch(
            np.asarray(features_2d, dtype=np.float64)
        )

    summary = streaming_stats.get_summary()
    excluded: dict[str, str] = {}
    evaluable: list[int] = []

    for j in range(schema.n_features):
        name = schema.feature_names.get(j, f"feature_{j}")
        if j in schema.categorical_indices:
            excluded[name] = "categorical"
        elif summary[j]["n"] == 0:
            excluded[name] = "no_data"
        elif summary[j]["std"] ** 2 < 1e-10:
            excluded[name] = "zero_variance"
        else:
            evaluable.append(j)

    n_eval = len(evaluable)
    n_days = len(evaluation_dates)

    # Build evaluable index → cube position mapping
    eval_idx_to_pos = {j: pos for pos, j in enumerate(evaluable)}

    # Pre-allocate cubes — filled with NaN
    daily_ic_cube = np.full((n_days, n_eval, n_horizons), np.nan)
    daily_temporal_cubes = {
        metric: np.full((n_days, n_eval, n_horizons), np.nan)
        for metric in TEMPORAL_METRICS
    }
    daily_forward_cube = np.full((n_days, n_eval, n_horizons), np.nan)
    daily_concurrent_cube = np.full((n_days, n_eval, n_horizons), np.nan)

    # Lists for pooling
    all_features_list: list[np.ndarray] = []
    all_labels_list: list[np.ndarray] = []
    all_date_idx_list: list[np.ndarray] = []  # Track day index per sample

    # ---------------------------------------------------------------
    # Pass 2: Main computation + pooling
    # ---------------------------------------------------------------
    for day_idx, bundle in enumerate(loader.iter_days(evaluation_dates)):
        n_samples = bundle.sequences.shape[0]

        # Last-timestep features for IC + pooling
        features_2d = np.asarray(
            bundle.sequences[:, -1, :], dtype=np.float64
        )  # [N, F_all]
        labels_2d = np.asarray(bundle.labels, dtype=np.float64)  # [N, H]

        # Accumulate for pooling
        all_features_list.append(features_2d)
        all_labels_list.append(labels_2d)
        all_date_idx_list.append(
            np.full(n_samples, day_idx, dtype=np.int32)
        )

        if n_samples < 3:
            continue  # Too few for Spearman

        # --- Per-day Path 1 ICs ---
        for j in evaluable:
            pos = eval_idx_to_pos[j]
            feature_col = features_2d[:, j]

            for h_idx in range(n_horizons):
                if h_idx >= labels_2d.shape[1]:
                    continue
                label_col = labels_2d[:, h_idx]
                rho, p = spearman_ic(feature_col, label_col)
                if not (rho == 0.0 and p == 1.0):
                    daily_ic_cube[day_idx, pos, h_idx] = rho

        # --- Per-day CF decomposition ICs ---
        if n_samples >= 4:
            for j in evaluable:
                pos = eval_idx_to_pos[j]
                feature_col = features_2d[:, j]

                for h_idx in range(n_horizons):
                    if h_idx >= labels_2d.shape[1]:
                        continue
                    label_col = labels_2d[:, h_idx]

                    # Forward IC: feature[t] vs label[t]
                    fwd_rho, fwd_p = spearman_ic(feature_col, label_col)
                    if not (fwd_rho == 0.0 and fwd_p == 1.0):
                        daily_forward_cube[day_idx, pos, h_idx] = fwd_rho

                    # Concurrent IC: feature[t] vs label[t-1]
                    conc_rho, conc_p = spearman_ic(
                        feature_col[1:], label_col[:-1]
                    )
                    if not (conc_rho == 0.0 and conc_p == 1.0):
                        daily_concurrent_cube[day_idx, pos, h_idx] = conc_rho

        # --- Per-day temporal rolling ICs ---
        # Vectorized 3D rolling across all features at once
        seq_3d = np.asarray(bundle.sequences, dtype=np.float64)  # [N, T, F]
        rm_3d = rolling_mean(seq_3d, window=rolling_window)      # [N, T, F]
        rs_3d = rolling_slope(seq_3d, window=rolling_window)     # [N, T, F]
        roc_3d = rate_of_change(seq_3d, lag=rolling_window)      # [N, T, F]

        # Extract last timestep of rolling features → [N, F]
        rm_last = rm_3d[:, -1, :]
        rs_last = rs_3d[:, -1, :]
        roc_last = roc_3d[:, -1, :]

        temporal_data = {
            "rolling_mean": rm_last,
            "rolling_slope": rs_last,
            "rate_of_change": roc_last,
        }

        for metric_name, feat_2d in temporal_data.items():
            for j in evaluable:
                pos = eval_idx_to_pos[j]
                feat_vals = feat_2d[:, j]

                for h_idx in range(n_horizons):
                    if h_idx >= labels_2d.shape[1]:
                        continue
                    label_col = labels_2d[:, h_idx]

                    # Filter NaN from rolling edge effects
                    valid = np.isfinite(feat_vals) & np.isfinite(label_col)
                    if valid.sum() < 3:
                        continue
                    rho, p = spearman_ic(feat_vals[valid], label_col[valid])
                    if not (rho == 0.0 and p == 1.0):
                        daily_temporal_cubes[metric_name][
                            day_idx, pos, h_idx
                        ] = rho

    # ---------------------------------------------------------------
    # Pool all data
    # ---------------------------------------------------------------
    if all_features_list:
        pooled_features = np.concatenate(all_features_list, axis=0)
        pooled_labels = np.concatenate(all_labels_list, axis=0)
        pooled_date_indices = np.concatenate(all_date_idx_list, axis=0)
        n_total = pooled_features.shape[0]
    else:
        pooled_features = np.empty((0, schema.n_features), dtype=np.float64)
        pooled_labels = np.empty((0, n_horizons), dtype=np.float64)
        pooled_date_indices = np.empty(0, dtype=np.int32)
        n_total = 0

    return DataCache(
        schema=schema,
        evaluation_dates=tuple(evaluation_dates),
        evaluable_indices=tuple(evaluable),
        excluded_features=excluded,
        horizons=tuple(horizons),
        seed=config.seed,
        daily_ic_cube=daily_ic_cube,
        daily_temporal_cubes=daily_temporal_cubes,
        daily_forward_ic_cube=daily_forward_cube,
        daily_concurrent_ic_cube=daily_concurrent_cube,
        pooled_features=pooled_features,
        pooled_labels=pooled_labels,
        pooled_date_indices=pooled_date_indices,
        n_total_samples=n_total,
    )
