"""
FeatureProfile: comprehensive per-feature characterization.

Replaces FeatureTier as the primary output of the evaluation pipeline.
A profile is NOT a verdict — it is a rich record of all measurements.
Selection criteria (criteria.py) match against profiles to select features.

Reference: Plan Phase 2 — FeatureProfile + SelectionCriteria
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Path evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathEvidence:
    """Evidence from a single evaluation path at one horizon.

    Replaces PathResult with an explicit is_informational flag for
    paths that store results but don't participate in classification
    (e.g., transfer entropy).
    """

    path_name: str          # "linear_signal", "nonlinear_signal", etc.
    horizon: int
    metric_name: str        # "forward_ic", "dcor", "rolling_slope", etc.
    metric_value: float
    p_value: float          # NaN for paths without hypothesis tests (regime, JMI)
    ci_lower: float         # NaN if not available
    ci_upper: float         # NaN if not available
    passes: bool
    is_informational: bool  # True = stored but NOT counted in passing_paths


# ---------------------------------------------------------------------------
# Stability breakdown
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StabilityDetail:
    """Per-path stability breakdown.

    Each field is a fraction in [0.0, 1.0] representing the proportion
    of bootstrap subsamples where the feature passes that path.

    combined_stability = max(path1, path2, path3a) — a feature is stable
    if stable on ANY screening path. Used as the primary stability gate.
    """

    path1_stability: float       # IC screening stability (date bootstrap)
    path2_stability: float       # dCor+MI stability (subsample bootstrap)
    path3a_stability: float      # Temporal IC stability (date bootstrap)
    combined_stability: float    # max(path1, path2, path3a)
    path4_ci_coverage: float     # Regime IC CI coverage fraction
    path5_jmi_stability: float   # JMI selection stability (subsample bootstrap)


# ---------------------------------------------------------------------------
# Feature profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureProfile:
    """Complete characterization of a single feature.

    Not a tier — a rich profile that selection criteria match against.
    The tier property is derived on demand via compute_tier() for
    backward compatibility with v1 consumers.

    Frozen after construction. Use dataclasses.replace() to update
    holdout_confirmed after holdout validation.
    """

    feature_name: str
    feature_index: int

    # Signal characterization
    best_horizon: int
    best_metric: str
    best_value: float
    best_p: float               # NaN if no hypothesis-tested path passes

    # Passing paths (only non-informational paths that survived their gate)
    passing_paths: tuple[str, ...]

    # Stability
    stability: StabilityDetail

    # Concurrent/forward decomposition
    concurrent_forward_ratio: float | None
    cf_classification: str | None   # "forward", "partially_forward",
    #                                 "contemporaneous", "state_variable"

    # Redundancy (from hft-metrics correlation analysis)
    redundancy_cluster_id: int | None
    max_pairwise_correlation: float | None
    vif: float | None

    # Temporal dynamics (ACF of daily IC series)
    ic_acf_half_life: int | None

    # Holdout validation
    holdout_confirmed: bool = False

    # All evidence (for traceability, includes informational)
    all_evidence: tuple[PathEvidence, ...] = ()


# ---------------------------------------------------------------------------
# Tier derivation (backward-compatible)
# ---------------------------------------------------------------------------


def compute_tier(
    profile: FeatureProfile,
    stable_threshold: float = 0.6,
    investigate_threshold: float = 0.4,
    strong_keep_p: float = 0.01,
) -> str:
    """Backward-compatible tier derivation from profile.

    Uses combined_stability (max across Paths 1/2/3a) as the stability
    gate — NOT path1-only, which was the original bug.

    Args:
        profile: FeatureProfile to classify.
        stable_threshold: Minimum stability for KEEP (default 0.6).
        investigate_threshold: Minimum stability for INVESTIGATE (default 0.4).
        strong_keep_p: Maximum p-value for STRONG-KEEP (default 0.01).

    Returns:
        Tier string: "STRONG-KEEP", "KEEP", "INVESTIGATE", or "DISCARD".
    """
    import math

    if len(profile.passing_paths) == 0:
        return "DISCARD"

    stability = profile.stability.combined_stability

    if stability < investigate_threshold:
        return "DISCARD"
    if stability < stable_threshold:
        return "INVESTIGATE"

    if (
        math.isfinite(profile.best_p)
        and profile.best_p < strong_keep_p
        and profile.holdout_confirmed
    ):
        return "STRONG-KEEP"

    return "KEEP"
