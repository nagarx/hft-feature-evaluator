"""
Evaluation configuration: YAML-driven, immutable, fully validated.

All thresholds and parameters for the 5-path evaluation framework.
Invalid values raise ValueError with descriptive messages at parse time.

Reference: FEATURE_EVALUATION_FRAMEWORK.md Sections 6.2, 6.3, 6.6
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Nested config sections (all frozen)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScreeningConfig:
    """Parameters for Layer 1 screening (IC, dCor, MI)."""

    horizons: tuple[int, ...]          # e.g., (1, 2, 3, 5, 10, 20, 30, 60)
    bh_fdr_level: float = 0.05        # Benjamini-Hochberg FDR level
    ic_threshold: float = 0.05        # Forward IC pass threshold (Framework §6.2)
    dcor_permutations: int = 500      # Permutations for dCor test
    dcor_subsample: int = 3000        # Subsample size for pooled dCor/MI (bias mitigation)
    mi_permutations: int = 200        # Permutations for ksg_mi_test
    mi_k: int = 5                     # k for KSG MI estimator


@dataclass(frozen=True)
class StabilityConfig:
    """Parameters for bootstrap stability selection (Framework §6.3)."""

    n_bootstraps: int = 50            # Number of bootstrap subsamples
    subsample_fraction: float = 0.8   # Fraction of training days per subsample
    stable_threshold: float = 0.6     # >= this → KEEP/STRONG-KEEP eligible
    investigate_threshold: float = 0.4  # >= this but < stable → INVESTIGATE


@dataclass(frozen=True)
class ClassificationConfig:
    """Parameters for 4-tier classification (Framework §6.6)."""

    strong_keep_p: float = 0.01       # Best BH-adjusted p < this for STRONG-KEEP
    ic_ir_threshold: float = 0.5      # IC_IR > this for Path 1


@dataclass(frozen=True)
class RegimeConfig:
    """Parameters for regime-conditional IC (Framework Path 4)."""

    n_bins: int = 3                   # Number of conditioning bins (terciles)
    min_samples_per_bin: int = 30     # Minimum samples per regime cell
    conditioning_indices: dict[str, int] | None = field(default=None)
    # None → auto-detect from schema. Off-exchange: {spread_bps: 12, ...}


@dataclass(frozen=True)
class TemporalConfig:
    """Parameters for temporal IC and transfer entropy (Framework Path 3)."""

    rolling_window: int = 5              # K for rolling_mean_K, rolling_slope_K
    te_lags: tuple[int, ...] = (1, 2, 3)  # Lag range L for transfer entropy


@dataclass(frozen=True)
class SelectionConfig:
    """Parameters for JMI forward selection (Framework Path 5)."""

    jmi_max_features: int | None = None  # None = use elbow detection
    jmi_elbow_threshold: float = 0.05    # Relative gain threshold for stopping


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationConfig:
    """Top-level evaluation configuration. Immutable after construction.

    Construct via from_yaml() or from_dict(). Always call validate() before use.
    """

    export_dir: str
    split: str = "train"
    holdout_days: int = 20
    seed: int = 42
    screening: ScreeningConfig = field(default_factory=lambda: ScreeningConfig(
        horizons=(1, 2, 3, 5, 10, 20, 30, 60),
    ))
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    verbose: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "EvaluationConfig":
        """Parse config from a YAML file.

        Args:
            path: Path to YAML file.

        Returns:
            Validated EvaluationConfig.

        Raises:
            FileNotFoundError: If path does not exist.
            ValueError: If YAML contains unknown keys or invalid values.
        """
        with open(path) as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Expected YAML dict, got {type(raw).__name__}")
        config = cls.from_dict(raw)
        config.validate()
        return config

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvaluationConfig":
        """Construct from a plain dict (e.g., from yaml.safe_load).

        Raises:
            ValueError: If unknown top-level or nested keys are present.
        """
        d = dict(d)  # shallow copy

        _KNOWN_TOP = {
            "export_dir", "split", "holdout_days", "seed",
            "screening", "stability", "classification", "regime",
            "temporal", "selection", "verbose",
        }
        unknown = set(d.keys()) - _KNOWN_TOP
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")

        # Parse nested sections
        screening_raw = d.pop("screening", {})
        stability_raw = d.pop("stability", {})
        classification_raw = d.pop("classification", {})
        regime_raw = d.pop("regime", {})
        temporal_raw = d.pop("temporal", {})
        selection_raw = d.pop("selection", {})

        screening = _parse_screening(screening_raw)
        stability = _parse_stability(stability_raw)
        classification = _parse_classification(classification_raw)
        regime = _parse_regime(regime_raw)
        temporal = _parse_temporal(temporal_raw)
        selection = _parse_selection(selection_raw)

        return cls(
            screening=screening,
            stability=stability,
            classification=classification,
            regime=regime,
            temporal=temporal,
            selection=selection,
            **d,
        )

    def validate(self) -> None:
        """Validate all config parameters. Raises ValueError on violation."""
        # Top-level
        if not self.export_dir:
            raise ValueError("export_dir must be non-empty")
        if self.split not in {"train", "val", "test"}:
            raise ValueError(
                f"split must be 'train', 'val', or 'test', got '{self.split}'"
            )
        if not (0 <= self.holdout_days <= 100):
            raise ValueError(
                f"holdout_days must be in [0, 100], got {self.holdout_days}"
            )
        if not (0 <= self.seed <= 2**31 - 1):
            raise ValueError(f"seed must be in [0, 2^31-1], got {self.seed}")

        # Screening
        s = self.screening
        if not s.horizons:
            raise ValueError("screening.horizons must be non-empty")
        if any(h <= 0 for h in s.horizons):
            raise ValueError("screening.horizons must all be positive")
        if list(s.horizons) != sorted(s.horizons):
            raise ValueError(
                f"screening.horizons must be sorted ascending, got {s.horizons}"
            )
        if not (0 < s.bh_fdr_level < 1):
            raise ValueError(
                f"screening.bh_fdr_level must be in (0, 1), got {s.bh_fdr_level}"
            )
        if not (0 < s.ic_threshold < 1):
            raise ValueError(
                f"screening.ic_threshold must be in (0, 1), got {s.ic_threshold}"
            )
        if not (100 <= s.dcor_permutations <= 10000):
            raise ValueError(
                f"screening.dcor_permutations must be in [100, 10000], "
                f"got {s.dcor_permutations}"
            )
        if not (500 <= s.dcor_subsample <= 10000):
            raise ValueError(
                f"screening.dcor_subsample must be in [500, 10000], "
                f"got {s.dcor_subsample}"
            )
        if not (50 <= s.mi_permutations <= 5000):
            raise ValueError(
                f"screening.mi_permutations must be in [50, 5000], "
                f"got {s.mi_permutations}"
            )
        if not (1 <= s.mi_k <= 20):
            raise ValueError(
                f"screening.mi_k must be in [1, 20], got {s.mi_k}"
            )

        # Stability
        st = self.stability
        if not (10 <= st.n_bootstraps <= 500):
            raise ValueError(
                f"stability.n_bootstraps must be in [10, 500], got {st.n_bootstraps}"
            )
        if not (0.5 < st.subsample_fraction < 0.99):
            raise ValueError(
                f"stability.subsample_fraction must be in (0.5, 0.99), "
                f"got {st.subsample_fraction}"
            )
        if not (0 < st.investigate_threshold < 1):
            raise ValueError(
                f"stability.investigate_threshold must be in (0, 1), "
                f"got {st.investigate_threshold}"
            )
        if not (0 < st.stable_threshold < 1):
            raise ValueError(
                f"stability.stable_threshold must be in (0, 1), "
                f"got {st.stable_threshold}"
            )
        if st.stable_threshold <= st.investigate_threshold:
            raise ValueError(
                f"stability.stable_threshold ({st.stable_threshold}) must be > "
                f"stability.investigate_threshold ({st.investigate_threshold})"
            )

        # Classification
        c = self.classification
        if not (0 < c.strong_keep_p < 1):
            raise ValueError(
                f"classification.strong_keep_p must be in (0, 1), "
                f"got {c.strong_keep_p}"
            )
        if c.strong_keep_p >= s.bh_fdr_level:
            raise ValueError(
                f"classification.strong_keep_p ({c.strong_keep_p}) must be < "
                f"screening.bh_fdr_level ({s.bh_fdr_level})"
            )
        if not (0 < c.ic_ir_threshold < 10):
            raise ValueError(
                f"classification.ic_ir_threshold must be in (0, 10), "
                f"got {c.ic_ir_threshold}"
            )

        # Regime
        r = self.regime
        if not (2 <= r.n_bins <= 5):
            raise ValueError(
                f"regime.n_bins must be in [2, 5], got {r.n_bins}"
            )
        if r.min_samples_per_bin < 10:
            raise ValueError(
                f"regime.min_samples_per_bin must be >= 10, "
                f"got {r.min_samples_per_bin}"
            )

        # Temporal
        t = self.temporal
        if not (2 <= t.rolling_window <= 10):
            raise ValueError(
                f"temporal.rolling_window must be in [2, 10], "
                f"got {t.rolling_window}"
            )
        if not t.te_lags:
            raise ValueError("temporal.te_lags must be non-empty")
        if any(lag < 1 for lag in t.te_lags):
            raise ValueError("temporal.te_lags must all be >= 1")
        if list(t.te_lags) != sorted(t.te_lags):
            raise ValueError(
                f"temporal.te_lags must be sorted ascending, got {t.te_lags}"
            )

        # Selection
        sel = self.selection
        if not (0 < sel.jmi_elbow_threshold < 1):
            raise ValueError(
                f"selection.jmi_elbow_threshold must be in (0, 1), "
                f"got {sel.jmi_elbow_threshold}"
            )
        if sel.jmi_max_features is not None and sel.jmi_max_features < 1:
            raise ValueError(
                f"selection.jmi_max_features must be >= 1 or None, "
                f"got {sel.jmi_max_features}"
            )


# ---------------------------------------------------------------------------
# Internal parsers for nested sections
# ---------------------------------------------------------------------------

_SCREENING_KEYS = {
    "horizons", "bh_fdr_level", "ic_threshold",
    "dcor_permutations", "dcor_subsample", "mi_permutations", "mi_k",
}
_STABILITY_KEYS = {
    "n_bootstraps", "subsample_fraction", "stable_threshold", "investigate_threshold",
}
_CLASSIFICATION_KEYS = {"strong_keep_p", "ic_ir_threshold"}
_REGIME_KEYS = {"n_bins", "min_samples_per_bin", "conditioning_indices"}


def _check_unknown(raw: dict, known: set, section: str) -> None:
    unknown = set(raw.keys()) - known
    if unknown:
        raise ValueError(f"Unknown keys in {section}: {unknown}")


def _parse_screening(raw: dict) -> ScreeningConfig:
    _check_unknown(raw, _SCREENING_KEYS, "screening")
    raw = dict(raw)
    if "horizons" in raw:
        raw["horizons"] = tuple(raw["horizons"])
    else:
        raw["horizons"] = (1, 2, 3, 5, 10, 20, 30, 60)
    return ScreeningConfig(**raw)


def _parse_stability(raw: dict) -> StabilityConfig:
    _check_unknown(raw, _STABILITY_KEYS, "stability")
    return StabilityConfig(**raw)


def _parse_classification(raw: dict) -> ClassificationConfig:
    _check_unknown(raw, _CLASSIFICATION_KEYS, "classification")
    return ClassificationConfig(**raw)


_TEMPORAL_KEYS = {"rolling_window", "te_lags"}
_SELECTION_KEYS = {"jmi_max_features", "jmi_elbow_threshold"}


def _parse_regime(raw: dict) -> RegimeConfig:
    _check_unknown(raw, _REGIME_KEYS, "regime")
    return RegimeConfig(**raw)


def _parse_temporal(raw: dict) -> TemporalConfig:
    _check_unknown(raw, _TEMPORAL_KEYS, "temporal")
    raw = dict(raw)
    if "te_lags" in raw:
        raw["te_lags"] = tuple(raw["te_lags"])
    return TemporalConfig(**raw)


def _parse_selection(raw: dict) -> SelectionConfig:
    _check_unknown(raw, _SELECTION_KEYS, "selection")
    return SelectionConfig(**raw)
