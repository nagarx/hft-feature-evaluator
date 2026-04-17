"""
SelectionCriteria: declarative feature selection from profiles.

Replaces the hard-coded classify_feature() verdict with configurable
criteria matching. Each experiment defines its criteria via YAML config.
The same profiles serve multiple experiments without re-evaluation.

Reference: Plan Phase 2 — SelectionCriteria. Phase 4 (2026-04-15)
added ``from_yaml`` / ``from_dict``, ``criteria_schema_version``, and
``require_holdout_confirmed`` to support the FeatureSet registry producer.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

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
            criteria_schema_version: "1.0"
            min_passing_paths: 1
            min_combined_stability: 0.6
            require_holdout_confirmed: true
    """

    name: str = "default"

    # Schema version of this criteria declaration. Bumped on any ADDITIVE
    # field extension that could change hash output across evaluator
    # versions (see Phase 4 FeatureSet content-hash policy). Downstream
    # registry producers include this in produced_by provenance so
    # consumers can re-validate criteria compatibility.
    criteria_schema_version: str = "1.0"

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

    # Holdout verification. When True, only features whose
    # ``FeatureProfile.holdout_confirmed`` is True survive selection —
    # i.e., the STRONG-KEEP criterion that passed out-of-sample. When
    # False (default), the holdout flag is ignored.
    require_holdout_confirmed: bool = False

    # Explicit inclusion/exclusion
    include_names: tuple[str, ...] | None = None
    exclude_names: tuple[str, ...] | None = None

    # -------------------------------------------------------------------
    # YAML loaders (Phase 4)
    # -------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SelectionCriteria":
        """Parse SelectionCriteria from a YAML file.

        Mirrors ``EvaluationConfig.from_yaml``. Unknown top-level keys
        raise ValueError. If the YAML top level is a single-key dict
        ``{criteria: {...}}``, the inner dict is used (supports
        composability with multi-section config files).

        Tuple-typed fields (``required_paths``, ``allowed_cf_classes``,
        ``allowed_horizons``, ``include_names``, ``exclude_names``)
        accept YAML sequences; values are coerced to tuples to preserve
        frozen-dataclass hashability.

        Args:
            path: Path to YAML file.

        Returns:
            SelectionCriteria.

        Raises:
            FileNotFoundError: If path does not exist.
            ValueError: If YAML top level is not a dict, or contains
                unknown keys.
        """
        with open(path) as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(
                f"Expected YAML dict, got {type(raw).__name__} "
                f"(path={path})"
            )
        # Accept single-key {criteria: {...}} wrapper for composability
        if len(raw) == 1 and "criteria" in raw and isinstance(
            raw["criteria"], dict
        ):
            raw = raw["criteria"]
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SelectionCriteria":
        """Construct from a plain dict (e.g., from yaml.safe_load).

        Validates the key set and coerces list-valued tuple fields.

        Raises:
            ValueError: If unknown keys are present, or a tuple field
                receives a non-sequence value.
        """
        d = dict(d)  # shallow copy
        unknown = set(d.keys()) - _KNOWN_CRITERIA_KEYS
        if unknown:
            raise ValueError(
                f"Unknown SelectionCriteria keys: {sorted(unknown)}. "
                f"Known keys: {sorted(_KNOWN_CRITERIA_KEYS)}"
            )
        for field_name in _TUPLE_CRITERIA_FIELDS:
            if field_name in d and d[field_name] is not None:
                value = d[field_name]
                if isinstance(value, (str, bytes)):
                    raise ValueError(
                        f"SelectionCriteria.{field_name} must be a sequence "
                        f"of values (not a string), got {type(value).__name__}"
                    )
                try:
                    d[field_name] = tuple(value)
                except TypeError as exc:
                    raise ValueError(
                        f"SelectionCriteria.{field_name} must be iterable, "
                        f"got {type(value).__name__}"
                    ) from exc
        return cls(**d)


# Known keys used by ``SelectionCriteria.from_dict`` validation. Kept as a
# module-level constant so future field additions fail loudly if this
# set is not updated in lockstep (see ``criteria_schema_version``).
_KNOWN_CRITERIA_KEYS: frozenset[str] = frozenset(
    {
        "name",
        "criteria_schema_version",
        "min_passing_paths",
        "required_paths",
        "min_combined_stability",
        "min_abs_metric",
        "max_p_value",
        "max_cf_ratio",
        "allowed_cf_classes",
        "max_vif",
        "max_pairwise_corr",
        "allowed_horizons",
        "require_holdout_confirmed",
        "include_names",
        "exclude_names",
    }
)

_TUPLE_CRITERIA_FIELDS: tuple[str, ...] = (
    "required_paths",
    "allowed_cf_classes",
    "allowed_horizons",
    "include_names",
    "exclude_names",
)


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

    # Holdout verification (Phase 4). Only gate when explicitly required —
    # default behavior (False) preserves pre-Phase-4 semantics where
    # holdout confirmation was an advisory STRONG-KEEP upgrade signal.
    if c.require_holdout_confirmed and not profile.holdout_confirmed:
        return False

    return True
