"""
Pre-training IC gate (fast_gate) — lightweight signal-quality check.

This is the Rule-13 gate promoted from the ``validate-signal`` skill into a
library function that ``hft-ops`` can call directly (no subprocess).

**What the gate answers**: "Does any feature have measurable signal against
the target horizon's labels, and is that signal stable over time?"

**What the gate does NOT answer**: Interaction/context/temporal patterns
that only emerge in a trained model. The evaluator CLAUDE.md explicitly
warns against using this check as a HARD filter — individual-feature IC
misses interaction, temporal, and conditioning-feature value. Use the
``allow_zero_ic_names`` bypass list for features known to carry value
through other mechanisms (e.g., ``time_regime``, ``dark_share``).

**Default disposition is WARN, not ABORT** — see the plan rationale and
the evaluator's "Mandatory safeguards" section. The gate surfaces
failures; researchers decide whether to proceed. Confidence can be
escalated to ``abort`` on a per-experiment basis once the gate's
configuration is proven suitable for the workload.

**Library import, not subprocess** (Phase 2b architectural decision): this
module is imported directly by ``hft_ops.stages.validation``. Subprocess
invocation was considered and rejected (adds 500ms–2s overhead per run,
fragments test coverage, complicates error propagation).

Formulas and thresholds:
- Per-feature Spearman IC: ``hft_metrics.spearman_ic`` (rho, p).
- Walk-forward stability: mean(best-feature IC per fold) / std(...).
  Adaptive folds = ``max(5, min(n_folds, n_days // 8))`` to avoid tiny
  folds when the training set has few days.
- Return std (bps): ``std(labels[:, horizon_idx])``. The contract emits
  regression labels in basis points; see pipeline_contract.toml.
- Gate thresholds: see ``GateThresholds``. Sources documented inline.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from hft_evaluator.data.loader import ExportLoader, ExportSchema

logger = logging.getLogger(__name__)


__all__ = [
    "GateThresholds",
    "PerFeatureIC",
    "GateReport",
    "run_fast_gate",
    "FastGateError",
]


class FastGateError(RuntimeError):
    """Fast-gate runtime error (unrecoverable data/schema problem).

    Gate ``verdict: FAIL`` is NOT an exception — it is an expected outcome
    recorded in the report. This exception is raised only for conditions
    that prevent the gate from running at all (missing data, empty export,
    horizon out of range).
    """


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateThresholds:
    """Rule-13 gate thresholds.

    Sources (see ``.claude/skills/validate-signal/SKILL.md``):
    - ``min_ic = 0.05``: Rule §13 floor for "any feature has signal".
    - ``min_ic_count = 2``: guard against spurious single-feature signal.
    - ``min_return_std_bps = 5.0``: label variance must exceed typical
      round-trip cost band (ATM call breakeven ≈ 4.9 bps).
    - ``min_stability = 2.0``: mean(IC)/std(IC) across folds — prevents
      regime-dependent signals from passing.
    """

    min_ic: float = 0.05
    min_ic_count: int = 2
    min_return_std_bps: float = 5.0
    min_stability: float = 2.0


@dataclass
class PerFeatureIC:
    """Per-feature IC measurement."""

    feature_idx: int
    feature_name: str
    ic: float
    p_value: float
    bypassed: bool = False  # True if feature is in allow_zero_ic_names

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GateReport:
    """Structured output of one fast_gate run.

    Persisted to ``gate_report.json`` and stored in the experiment ledger
    as ``validation_report`` for cross-experiment comparison.
    """

    verdict: str            # "PASS" or "FAIL" (factual; caller applies on_fail)
    reason: str             # Human-readable summary
    data_dir: str
    split: str
    horizon_idx: int
    horizon_value: Optional[int]
    n_sequences_sampled: int
    n_days_used: int

    # Aggregate gate outcomes
    best_feature_ic: float          # |IC| of top feature
    best_feature_idx: int
    best_feature_name: str
    ic_count: int                   # # features with |IC| > min_ic
    return_std_bps: float           # std of labels[:, horizon_idx]; NaN if unavailable
    stability: float                # mean(fold_IC) / std(fold_IC); NaN if <2 folds
    n_folds_used: int

    # Per-gate pass/fail
    gate_g_ic_passed: bool
    gate_g_ic_count_passed: bool
    gate_g_return_std_passed: bool
    gate_g_stability_passed: bool

    # Inputs recorded for traceability
    thresholds: Dict[str, float] = field(default_factory=dict)
    allow_zero_ic_names: List[str] = field(default_factory=list)

    # Top-N features ranked by |IC| descending
    per_feature_ic: List[PerFeatureIC] = field(default_factory=list)

    # Provenance
    contract_version: str = ""
    schema_version: str = ""
    duration_seconds: float = 0.0
    profile_ref_hash: str = ""  # SHA-256 of profile_ref if provided

    def as_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (JSON-safe)."""
        d = asdict(self)
        d["per_feature_ic"] = [p.as_dict() for p in self.per_feature_ic]
        return d

    def to_json(self, path: Path) -> None:
        """Write the report as formatted JSON to ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.as_dict(), f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Sampling / fold utilities
# ---------------------------------------------------------------------------


def _adaptive_n_folds(n_days: int, requested: int) -> int:
    """Clip the requested fold count to keep fold sizes meaningful.

    Formula (from the validate-signal skill): ``max(5, min(requested, n_days // 8))``.
    Rationale: a fold of <8 days loses statistical power; the 5-fold floor
    keeps the stability estimate non-degenerate.
    """
    if n_days <= 0:
        return 0
    adaptive = max(5, min(requested, max(1, n_days // 8)))
    # Never exceed n_days (defensive)
    return min(adaptive, n_days)


def _sample_last_timesteps(
    loader: ExportLoader,
    dates: Sequence[str],
    horizon_idx: int,
    sample_size: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load sequences across days and return ``(features[N,F], labels[N])``.

    Takes the last timestep per sequence (``sequences[:, -1, :]``) — the
    representation most evaluator paths use for IC (Framework §6.2 Path 1).
    Pooled sampling across days prevents early-day bias.
    """
    rng = np.random.default_rng(seed)

    # First pass: count sequences per day for quota allocation
    per_day_counts: List[int] = []
    for date in dates:
        bundle = loader.load_day(date)
        per_day_counts.append(bundle.sequences.shape[0])

    total = sum(per_day_counts)
    if total == 0:
        raise FastGateError(
            f"No sequences found across {len(dates)} day(s) in split."
        )

    # Allocate a per-day quota proportional to day size, capped at sample_size
    target = min(sample_size, total)
    quotas: List[int] = []
    if target >= total:
        quotas = list(per_day_counts)
    else:
        for count in per_day_counts:
            q = int(round(target * (count / total)))
            quotas.append(min(q, count))
        # Fix rounding drift
        drift = target - sum(quotas)
        idx = 0
        while drift > 0 and idx < len(quotas):
            room = per_day_counts[idx] - quotas[idx]
            if room > 0:
                delta = min(room, drift)
                quotas[idx] += delta
                drift -= delta
            idx += 1

    feature_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []

    for date, quota in zip(dates, quotas):
        if quota <= 0:
            continue
        bundle = loader.load_day(date)
        n = bundle.sequences.shape[0]

        # Validate horizon_idx fits
        if bundle.labels.ndim == 1:
            # Classification label (1D) — no horizon dim. horizon_idx must be 0.
            if horizon_idx != 0:
                raise FastGateError(
                    f"Labels for {date} are 1-D but horizon_idx={horizon_idx}. "
                    f"Regression labels required for multi-horizon fast_gate."
                )
            day_labels = bundle.labels.astype(np.float64)
        else:
            n_horizons = bundle.labels.shape[1]
            if horizon_idx >= n_horizons:
                raise FastGateError(
                    f"horizon_idx={horizon_idx} out of range; {date} has "
                    f"{n_horizons} horizons."
                )
            day_labels = bundle.labels[:, horizon_idx].astype(np.float64)

        # Last-timestep snapshot (float32 → float64 for IC)
        day_features = np.asarray(
            bundle.sequences[:, -1, :], dtype=np.float64
        )

        if quota >= n:
            feature_chunks.append(day_features)
            label_chunks.append(day_labels)
        else:
            sel = rng.choice(n, size=quota, replace=False)
            sel.sort()
            feature_chunks.append(day_features[sel])
            label_chunks.append(day_labels[sel])

    features = np.concatenate(feature_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    return features, labels, features.shape[0]


def _compute_per_feature_ic(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: Dict[int, str],
    allow_zero_ic_names: Sequence[str],
    categorical_indices: Sequence[int] = (),
) -> List[PerFeatureIC]:
    """Per-feature IC, sorted by |IC| descending.

    Categorical indices get IC=0 (their IC is not meaningful via Spearman);
    features in ``allow_zero_ic_names`` are marked ``bypassed=True`` so
    ``ic_count`` excludes them from the "must exceed min_ic" requirement.
    """
    # Import inside function to avoid a heavy import at module load time.
    from hft_metrics.ic import spearman_ic

    allow_set = set(allow_zero_ic_names)
    categorical_set = frozenset(int(i) for i in categorical_indices)
    n_features = features.shape[1]

    out: List[PerFeatureIC] = []
    for f_idx in range(n_features):
        name = feature_names.get(f_idx, f"feature_{f_idx}")
        bypassed = name in allow_set
        if f_idx in categorical_set:
            ic, p = 0.0, 1.0
        else:
            ic, p = spearman_ic(features[:, f_idx], labels)
        out.append(
            PerFeatureIC(
                feature_idx=f_idx,
                feature_name=name,
                ic=float(ic),
                p_value=float(p),
                bypassed=bypassed,
            )
        )
    out.sort(key=lambda p: abs(p.ic), reverse=True)
    return out


def _compute_walk_forward_stability(
    loader: ExportLoader,
    dates: Sequence[str],
    horizon_idx: int,
    best_feature_idx: int,
    n_folds: int,
) -> tuple[float, int, List[float]]:
    """Walk-forward stability of the top-IC feature.

    Partitions dates into ``n_folds`` consecutive groups, computes the
    best-feature IC on each fold, and returns mean/std ratio.

    Returns:
        (stability, n_folds_used, per_fold_ic_list)
        stability is NaN if fewer than 2 folds or std below numerical noise.
    """
    from hft_metrics.ic import spearman_ic

    n_days = len(dates)
    n_folds_eff = _adaptive_n_folds(n_days, n_folds)
    if n_folds_eff < 2:
        return float("nan"), 0, []

    # Partition dates into contiguous folds
    fold_sizes = [n_days // n_folds_eff] * n_folds_eff
    for i in range(n_days % n_folds_eff):
        fold_sizes[i] += 1

    per_fold_ic: List[float] = []
    cursor = 0
    for fs in fold_sizes:
        fold_dates = dates[cursor : cursor + fs]
        cursor += fs

        feature_chunks: List[np.ndarray] = []
        label_chunks: List[np.ndarray] = []
        for date in fold_dates:
            bundle = loader.load_day(date)
            if bundle.labels.ndim == 1:
                y = bundle.labels.astype(np.float64)
            else:
                y = bundle.labels[:, horizon_idx].astype(np.float64)
            x = np.asarray(
                bundle.sequences[:, -1, best_feature_idx], dtype=np.float64
            )
            feature_chunks.append(x)
            label_chunks.append(y)

        x_all = np.concatenate(feature_chunks, axis=0)
        y_all = np.concatenate(label_chunks, axis=0)
        ic, _ = spearman_ic(x_all, y_all)
        per_fold_ic.append(float(ic))

    arr = np.asarray(per_fold_ic, dtype=np.float64)
    if arr.size < 2:
        return float("nan"), n_folds_eff, per_fold_ic
    std = float(np.std(arr, ddof=1))
    if std < 1e-12:
        return float("nan"), n_folds_eff, per_fold_ic
    mean_abs = float(abs(np.mean(arr)))
    return mean_abs / std, n_folds_eff, per_fold_ic


def _hash_file(path: Path) -> str:
    """SHA-256 of a file's bytes. Returns '' if the file doesn't exist."""
    import hashlib

    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_fast_gate(
    data_dir: Path | str,
    horizon_idx: int,
    *,
    split: str = "train",
    horizon_value: Optional[int] = None,
    thresholds: Optional[GateThresholds] = None,
    sample_size: int = 200_000,
    n_folds: int = 20,
    allow_zero_ic_names: Sequence[str] = (),
    top_n_report: int = 20,
    seed: int = 42,
    profile_ref: Optional[Path | str] = None,
) -> GateReport:
    """Run the pre-training fast IC gate on an export.

    Args:
        data_dir: Export root (must contain ``<split>/`` subdirectory).
        horizon_idx: Index into the export's horizons list.
        split: Split to sample from (``train`` by default).
        horizon_value: Actual horizon value (for reporting only).
        thresholds: Gate thresholds. If None, ``GateThresholds()`` is used.
        sample_size: Max sequences sampled from the split (default 200k).
        n_folds: Walk-forward fold count; clipped adaptively per fold sizing.
        allow_zero_ic_names: Feature names that bypass the IC check (context
            features — ``time_regime``, ``dark_share``, etc.).
        top_n_report: Number of top features to include in ``per_feature_ic``.
        seed: RNG seed for sampling determinism.
        profile_ref: Optional path to a precomputed ``feature_profiles.json``;
            captured as a provenance hash in the report. (Post-processing
            the profile itself is deferred — not needed for fast_gate.)

    Returns:
        GateReport with verdict and per-gate outcomes.

    Raises:
        FastGateError: Export cannot be loaded, no sequences, or horizon_idx
            out of range. Gate ``FAIL`` is NOT an exception — it is a
            recorded verdict.
    """
    t0 = time.monotonic()
    data_dir = Path(data_dir)
    thresholds = thresholds or GateThresholds()

    loader = ExportLoader(str(data_dir), split=split)
    schema: ExportSchema = loader.schema
    dates = loader.list_dates()
    if not dates:
        raise FastGateError(f"No dates found in {data_dir}/{split}")

    # Validate horizon_idx against schema
    if schema.horizons and horizon_idx >= len(schema.horizons):
        raise FastGateError(
            f"horizon_idx={horizon_idx} out of range; schema has "
            f"{len(schema.horizons)} horizons: {schema.horizons}"
        )

    # Sample and compute per-feature IC
    features, labels, n_samples = _sample_last_timesteps(
        loader, dates, horizon_idx, sample_size, seed=seed
    )

    categorical = tuple(sorted(int(i) for i in schema.categorical_indices))
    per_feature = _compute_per_feature_ic(
        features,
        labels,
        schema.feature_names,
        allow_zero_ic_names=allow_zero_ic_names,
        categorical_indices=categorical,
    )

    # Gate G_IC & G_IC_COUNT (exclude bypassed features from ic_count).
    best = per_feature[0]
    ic_count = sum(
        1 for p in per_feature if (not p.bypassed) and abs(p.ic) > thresholds.min_ic
    )
    gate_ic = abs(best.ic) > thresholds.min_ic
    gate_ic_count = ic_count >= thresholds.min_ic_count

    # Gate G_RETURN_STD. Regression labels are emitted in bps (contract 2.2).
    # If std is below threshold OR NaN (edge case), fail the gate.
    finite_labels = labels[np.isfinite(labels)]
    if finite_labels.size == 0:
        return_std_bps = float("nan")
    else:
        return_std_bps = float(np.std(finite_labels, ddof=1))
    gate_return_std = bool(
        np.isfinite(return_std_bps) and return_std_bps > thresholds.min_return_std_bps
    )

    # Gate G_STABILITY — best-feature IC across walk-forward folds.
    stability, n_folds_used, _ = _compute_walk_forward_stability(
        loader,
        dates,
        horizon_idx,
        best.feature_idx,
        n_folds,
    )
    gate_stability = bool(np.isfinite(stability) and stability > thresholds.min_stability)

    passed = gate_ic and gate_ic_count and gate_return_std and gate_stability
    verdict = "PASS" if passed else "FAIL"

    failure_reasons: List[str] = []
    if not gate_ic:
        failure_reasons.append(
            f"best |IC|={abs(best.ic):.4f} <= {thresholds.min_ic}"
        )
    if not gate_ic_count:
        failure_reasons.append(
            f"ic_count={ic_count} < {thresholds.min_ic_count}"
        )
    if not gate_return_std:
        failure_reasons.append(
            f"return_std_bps={return_std_bps:.3f} <= {thresholds.min_return_std_bps}"
        )
    if not gate_stability:
        failure_reasons.append(
            f"stability={stability:.3f} <= {thresholds.min_stability}"
        )
    reason = (
        "all gates passed"
        if passed
        else "gates failed: " + "; ".join(failure_reasons)
    )

    # Hash profile_ref if provided (for traceability)
    profile_hash = ""
    if profile_ref:
        profile_hash = _hash_file(Path(profile_ref))

    duration = time.monotonic() - t0

    return GateReport(
        verdict=verdict,
        reason=reason,
        data_dir=str(data_dir),
        split=split,
        horizon_idx=horizon_idx,
        horizon_value=horizon_value,
        n_sequences_sampled=n_samples,
        n_days_used=len(dates),
        best_feature_ic=float(abs(best.ic)),
        best_feature_idx=best.feature_idx,
        best_feature_name=best.feature_name,
        ic_count=ic_count,
        return_std_bps=return_std_bps,
        stability=stability,
        n_folds_used=n_folds_used,
        gate_g_ic_passed=gate_ic,
        gate_g_ic_count_passed=gate_ic_count,
        gate_g_return_std_passed=gate_return_std,
        gate_g_stability_passed=gate_stability,
        thresholds={
            "min_ic": thresholds.min_ic,
            "min_ic_count": float(thresholds.min_ic_count),
            "min_return_std_bps": thresholds.min_return_std_bps,
            "min_stability": thresholds.min_stability,
        },
        allow_zero_ic_names=list(allow_zero_ic_names),
        per_feature_ic=per_feature[:top_n_report],
        contract_version=schema.contract_version,
        schema_version=schema.schema_version,
        duration_seconds=duration,
        profile_ref_hash=profile_hash,
    )
