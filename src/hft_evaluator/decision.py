"""
Decision logic: 5-path aggregation into 4-tier classification.

Aggregates results from all evaluation paths and classifies each feature
into STRONG-KEEP / KEEP / INVESTIGATE / DISCARD.

Reference: CODEBASE.md Section 10, Framework Section 6.6
"""

import math
from dataclasses import dataclass, field
from enum import Enum

from hft_evaluator.config import EvaluationConfig
from hft_evaluator.data.loader import ExportSchema


# ---------------------------------------------------------------------------
# Tier enum
# ---------------------------------------------------------------------------


class Tier(str, Enum):
    """4-tier feature classification."""

    STRONG_KEEP = "STRONG-KEEP"
    KEEP = "KEEP"
    INVESTIGATE = "INVESTIGATE"
    DISCARD = "DISCARD"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathResult:
    """Result of one evaluation path for one feature at one horizon.

    Stored in FeatureTier.all_path_results for full traceability.
    """

    path_name: str         # "linear_signal", "regime_conditional", etc.
    horizon: int
    metric_name: str       # "forward_ic", "regime_ic", etc.
    metric_value: float
    p_value: float         # BH-adjusted p-value (within this path's test family)
    ci_lower: float
    ci_upper: float
    passes: bool


@dataclass(frozen=True)
class FeatureTier:
    """Per-feature 4-tier classification with full provenance."""

    tier: Tier
    passing_paths: tuple[str, ...]
    best_horizon: int
    best_metric: str
    best_value: float
    best_p: float
    stability_pct: float | None      # None in Phase 2 (no stability yet)
    concurrent_forward_ratio: float | None
    all_path_results: tuple[PathResult, ...]


@dataclass(frozen=True)
class HoldoutReport:
    """Results of holdout validation for STRONG-KEEP candidates."""

    holdout_dates: tuple[str, ...]
    n_holdout_days: int
    candidates_tested: int
    candidates_confirmed: int
    per_feature: dict[str, bool]     # feature_name -> holdout_confirmed


@dataclass
class FeatureClassification:
    """Complete evaluation output."""

    per_feature: dict[str, FeatureTier]
    config: EvaluationConfig
    excluded_features: dict[str, str]  # feature_name -> exclusion reason
    schema: ExportSchema
    holdout: HoldoutReport | None = None


# ---------------------------------------------------------------------------
# Classification functions
# ---------------------------------------------------------------------------


def classify_feature(
    passing_paths: list[str],
    stability_pct: float | None,
    best_p: float,
    holdout_confirmed: bool,
    config: EvaluationConfig,
) -> Tier:
    """4-tier classification per Framework Section 6.6.

    Algorithm:
        if len(passing_paths) == 0:
            return DISCARD
        if stability_pct is not None and stability_pct < investigate_threshold:
            return DISCARD
        if stability_pct is not None and stability_pct < stable_threshold:
            return INVESTIGATE
        if best_p < strong_keep_p and holdout_confirmed:
            return STRONG_KEEP
        return KEEP

    When stability_pct is None (Phase 2): skip stability checks.
    STRONG-KEEP candidates that fail holdout downgrade to KEEP.

    Args:
        passing_paths: Path names this feature passed.
        stability_pct: % of bootstraps where feature passes (None if not yet computed).
        best_p: Best BH-adjusted p-value across all passing paths.
        holdout_confirmed: Whether signal maintained on holdout data.
        config: EvaluationConfig with stability and classification params.

    Returns:
        Tier classification.
    """
    if len(passing_paths) == 0:
        return Tier.DISCARD

    if stability_pct is not None:
        if stability_pct < config.stability.investigate_threshold:
            return Tier.DISCARD
        if stability_pct < config.stability.stable_threshold:
            return Tier.INVESTIGATE

    if best_p < config.classification.strong_keep_p and holdout_confirmed:
        return Tier.STRONG_KEEP

    return Tier.KEEP


def compute_best_p(path_results: list[PathResult]) -> float:
    """Minimum BH-adjusted p-value across all passing PathResults.

    If no paths pass, returns 1.0.

    Args:
        path_results: All PathResults for a feature across all paths and horizons.

    Returns:
        Minimum p-value among passing results, or 1.0 if none pass.
    """
    passing_ps = [
        r.p_value for r in path_results
        if r.passes and math.isfinite(r.p_value)
    ]
    return min(passing_ps) if passing_ps else 1.0
