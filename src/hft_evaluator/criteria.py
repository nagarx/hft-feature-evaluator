"""
SelectionCriteria: declarative feature selection from profiles.

Replaces the hard-coded classify_feature() verdict with configurable
criteria matching. Each experiment defines its criteria via YAML config.
The same profiles serve multiple experiments without re-evaluation.

Reference: Plan Phase 2 — SelectionCriteria
"""

import math
from dataclasses import dataclass, field

from hft_evaluator.profile import FeatureProfile


# ---------------------------------------------------------------------------
# Selection criteria
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionCriteria:
    """Declarative criteria for selecting features from profiles.

    Frozen dataclass, YAML-serializable. All fields are optional filters.
    A feature must satisfy ALL specified criteria to be selected.
    None/unset fields are not checked.

    Example YAML:
        criteria:
            name: "momentum_hft"
            min_passing_paths: 1
            min_combined_stability: 0.6
            exclude_contemporaneous: true
    """

    name: str = "default"

    # Path requirements
    min_passing_paths: int = 1
    required_paths: tuple[str, ...] = ()

    # Stability
    min_combined_stability: float = 0.6

    # Signal strength
    min_abs_metric: float | None = None
    max_p_value: float | None = None

    # CF decomposition gating
    max_cf_ratio: float | None = None
    allowed_cf_classes: tuple[str, ...] | None = None

    # Redundancy
    max_vif: float | None = None
    max_pairwise_corr: float | None = None

    # Horizon constraints
    allowed_horizons: tuple[int, ...] | None = None

    # Explicit inclusion/exclusion
    include_names: tuple[str, ...] | None = None
    exclude_names: tuple[str, ...] | None = None


# ---------------------------------------------------------------------------
# Selection function
# ---------------------------------------------------------------------------


def select_features(
    profiles: dict[str, FeatureProfile],
    criteria: SelectionCriteria,
) -> list[str]:
    """Apply selection criteria to profiles.

    Returns feature names that satisfy ALL specified criteria, sorted
    alphabetically for determinism.

    Args:
        profiles: {feature_name: FeatureProfile} from the pipeline.
        criteria: SelectionCriteria to match against.

    Returns:
        Sorted list of selected feature names.
    """
    selected = []
    for name, profile in profiles.items():
        if _matches(profile, criteria):
            selected.append(name)
    return sorted(selected)


def _matches(profile: FeatureProfile, c: SelectionCriteria) -> bool:
    """Check if a profile satisfies all criteria requirements."""
    # Explicit exclusion
    if c.exclude_names is not None and profile.feature_name in c.exclude_names:
        return False

    # Explicit inclusion overrides all other criteria
    if c.include_names is not None:
        return profile.feature_name in c.include_names

    # Path requirements
    if len(profile.passing_paths) < c.min_passing_paths:
        return False

    for req_path in c.required_paths:
        if req_path not in profile.passing_paths:
            return False

    # Stability
    if profile.stability.combined_stability < c.min_combined_stability:
        return False

    # Signal strength
    if c.min_abs_metric is not None:
        if abs(profile.best_value) < c.min_abs_metric:
            return False

    if c.max_p_value is not None:
        if math.isfinite(profile.best_p) and profile.best_p > c.max_p_value:
            return False

    # CF decomposition
    if c.max_cf_ratio is not None and profile.concurrent_forward_ratio is not None:
        if profile.concurrent_forward_ratio > c.max_cf_ratio:
            return False

    if c.allowed_cf_classes is not None and profile.cf_classification is not None:
        if profile.cf_classification not in c.allowed_cf_classes:
            return False

    # Redundancy
    if c.max_vif is not None and profile.vif is not None:
        if profile.vif > c.max_vif:
            return False

    if c.max_pairwise_corr is not None and profile.max_pairwise_correlation is not None:
        if profile.max_pairwise_correlation > c.max_pairwise_corr:
            return False

    # Horizon
    if c.allowed_horizons is not None:
        if profile.best_horizon not in c.allowed_horizons:
            return False

    return True
