"""
Path 5: JMI (Joint Mutual Information) greedy forward selection.

Selects features that provide UNIQUE information about the target when
combined, penalizing redundant features and rewarding conditional dependence.

Three-term decomposition (Brown et al. 2012, Eq. 17-18):
    J_JMI(X_k) = I(X_k;Y) - (1/|S|) Σ I(X_k;X_j) + (1/|S|) Σ I(X_k;X_j|Y)

Elbow detection: stop when gain_k / gain_1 < jmi_elbow_threshold.

Reference: CODEBASE.md Section 5.1, Framework Section 2.2
"""

from dataclasses import dataclass

import numpy as np

from hft_metrics.mi import ksg_mutual_information, conditional_mi_ksg
from hft_metrics._sanitize import EPS

from hft_evaluator.config import EvaluationConfig
from hft_evaluator.data.loader import ExportLoader

_JMI_SUBSAMPLE = 3000  # Subsample for MI/CMI feasibility


def jmi_forward_selection(
    loader: ExportLoader,
    evaluation_dates: list[str],
    evaluable_indices: list[int],
    horizon: int,
    config: EvaluationConfig,
    max_features: int | None = None,
) -> list[tuple[str, float]]:
    """JMI greedy forward selection on pooled + subsampled data.

    Args:
        loader: ExportLoader with detected schema.
        evaluation_dates: Dates to evaluate on.
        evaluable_indices: Feature indices eligible for selection.
        horizon: SINGLE horizon to run JMI on (best from IC screening).
        config: EvaluationConfig with selection params.
        max_features: Maximum features to select (None = elbow detection).

    Returns:
        Ordered list of (feature_name, jmi_score), best first.
        Features in the top-k are considered passing Path 5.
    """
    schema = loader.schema
    feature_names = schema.feature_names
    horizons = list(config.screening.horizons)

    # Find horizon index
    if horizon not in horizons:
        return []
    h_idx = horizons.index(horizon)

    # Pool all data
    all_features_list: list[np.ndarray] = []
    all_labels_list: list[np.ndarray] = []

    for bundle in loader.iter_days(evaluation_dates):
        if bundle.sequences.shape[0] < 1:
            continue
        features_2d = bundle.sequences[:, -1, :]
        all_features_list.append(np.asarray(features_2d, dtype=np.float64))
        all_labels_list.append(np.asarray(bundle.labels, dtype=np.float64))

    if not all_features_list:
        return []

    pooled_features = np.concatenate(all_features_list, axis=0)
    pooled_labels = np.concatenate(all_labels_list, axis=0)

    if h_idx >= pooled_labels.shape[1]:
        return []

    target = pooled_labels[:, h_idx]
    n_total = pooled_features.shape[0]

    # Subsample for MI/CMI feasibility
    rng = np.random.RandomState(config.seed)
    if n_total > _JMI_SUBSAMPLE:
        sub_idx = rng.choice(n_total, size=_JMI_SUBSAMPLE, replace=False)
        sub_features = pooled_features[sub_idx]
        sub_target = target[sub_idx]
    else:
        sub_features = pooled_features
        sub_target = target

    # Pre-compute relevancy: I(X_j; Y) for all features
    relevancy = {}
    for j in evaluable_indices:
        relevancy[j] = ksg_mutual_information(
            sub_features[:, j], sub_target, k=config.screening.mi_k
        )

    # Greedy forward selection
    selected: list[int] = []
    gains: list[float] = []
    remaining = list(evaluable_indices)

    effective_max = max_features or config.selection.jmi_max_features or len(remaining)

    while remaining and len(selected) < effective_max:
        best_j = None
        best_score = -np.inf

        for j in remaining:
            # Relevancy term
            rel = relevancy[j]

            # Redundancy and conditional redundancy terms
            red = 0.0
            cond_red = 0.0

            for s in selected:
                # I(X_j; X_s) — redundancy
                red += ksg_mutual_information(
                    sub_features[:, j], sub_features[:, s],
                    k=config.screening.mi_k,
                )
                # I(X_j; X_s | Y) — conditional redundancy
                cond_red += conditional_mi_ksg(
                    sub_features[:, j], sub_features[:, s], sub_target,
                    k=config.screening.mi_k,
                )

            if selected:
                red /= len(selected)
                cond_red /= len(selected)

            jmi_score = rel - red + cond_red

            if jmi_score > best_score:
                best_j = j
                best_score = jmi_score

        if best_j is None:
            break

        selected.append(best_j)
        remaining.remove(best_j)
        gains.append(best_score)

        # Elbow detection: stop when gain drops below threshold relative to first
        if len(gains) > 1 and gains[0] > EPS:
            relative_gain = gains[-1] / gains[0]
            if relative_gain < config.selection.jmi_elbow_threshold:
                break

    # Build output
    return [
        (feature_names.get(j, f"feature_{j}"), score)
        for j, score in zip(selected, gains)
    ]


# ---------------------------------------------------------------------------
# Cache-based variant (Phase 1: no disk I/O)
# ---------------------------------------------------------------------------


def jmi_from_cache(
    cache: "DataCache",
    horizon: int,
    config: EvaluationConfig,
    max_features: int | None = None,
) -> list[tuple[str, float]]:
    """JMI forward selection from pre-pooled cache data. No disk I/O.

    Args:
        cache: DataCache from build_cache().
        horizon: Single horizon to run JMI on.
        config: EvaluationConfig with selection params.
        max_features: Maximum features to select (None = elbow).

    Returns:
        Ordered list of (feature_name, jmi_score), best first.
    """
    from hft_evaluator.data.cache import DataCache

    schema = cache.schema
    feature_names = schema.feature_names
    horizons = list(cache.horizons)

    if horizon not in horizons:
        return []
    h_idx = horizons.index(horizon)

    n_total = cache.pooled_features.shape[0]
    if n_total == 0 or h_idx >= cache.pooled_labels.shape[1]:
        return []

    target = cache.pooled_labels[:, h_idx]

    # Subsample (identical to jmi_forward_selection)
    rng = np.random.RandomState(config.seed)
    if n_total > _JMI_SUBSAMPLE:
        sub_idx = rng.choice(n_total, size=_JMI_SUBSAMPLE, replace=False)
        sub_features = cache.pooled_features[sub_idx]
        sub_target = target[sub_idx]
    else:
        sub_features = cache.pooled_features
        sub_target = target

    evaluable_list = list(cache.evaluable_indices)

    # Pre-compute relevancy
    relevancy = {}
    for j in evaluable_list:
        relevancy[j] = ksg_mutual_information(
            sub_features[:, j], sub_target, k=config.screening.mi_k
        )

    # Greedy forward selection (identical logic)
    selected: list[int] = []
    gains: list[float] = []
    remaining = list(evaluable_list)
    effective_max = (
        max_features or config.selection.jmi_max_features or len(remaining)
    )

    while remaining and len(selected) < effective_max:
        best_j = None
        best_score = -np.inf

        for j in remaining:
            rel = relevancy[j]
            red = 0.0
            cond_red = 0.0

            for s in selected:
                red += ksg_mutual_information(
                    sub_features[:, j], sub_features[:, s],
                    k=config.screening.mi_k,
                )
                cond_red += conditional_mi_ksg(
                    sub_features[:, j], sub_features[:, s], sub_target,
                    k=config.screening.mi_k,
                )

            if selected:
                red /= len(selected)
                cond_red /= len(selected)

            jmi_score = rel - red + cond_red

            if jmi_score > best_score:
                best_j = j
                best_score = jmi_score

        if best_j is None:
            break

        selected.append(best_j)
        remaining.remove(best_j)
        gains.append(best_score)

        if len(gains) > 1 and gains[0] > EPS:
            relative_gain = gains[-1] / gains[0]
            if relative_gain < config.selection.jmi_elbow_threshold:
                break

    return [
        (feature_names.get(j, f"feature_{j}"), score)
        for j, score in zip(selected, gains)
    ]
