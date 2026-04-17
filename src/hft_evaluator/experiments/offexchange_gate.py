"""Off-exchange signal gate check (E14) — library port.

Applies three statistical gates to each configured (feature, horizon) pair
across the specified splits:

- **G1 (Persistence)**: mean per-day ``ACF(lag=60)`` > ``g1_acf60_threshold``.
- **G2 (Stride-60 IC)**: ``|pooled Spearman IC at stride=60|`` >
  ``g2_stride60_ic_threshold``.
- **G3 (Lag-1 IC)**: ``|pooled Spearman IC(feat[t], ret[t+1])|`` >
  ``g3_lag1_ic_threshold``.

If no feature passes all three gates, directional LOB/off-exchange trading
is NOT viable at the configured cadence.

Phase 5 Preview (2026-04-16) migration notes:

- Migrated from ``scripts/offexchange_gate_check.py`` (347 LOC, 0 CLI flags,
  all module-level constants).
- Core analyze-feature logic preserved byte-identically so E14's historical
  findings (EXPERIMENT_INDEX.md:1618) remain reproducible. The original
  script becomes a thin deprecation shim that delegates to ``run()`` here.
- Schema version bump: the legacy ``results.json`` wrote
  ``"schema": "offexchange_gate_check_v1"``; this library writes
  ``"schema": "offexchange_gate_check_v2"`` signalling the port. Downstream
  consumers key on schema version.

Reference:
    E14 experiment record — ``lob-model-trainer/EXPERIMENT_INDEX.md:1618``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

from hft_evaluator.data.loader import ExportLoader

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Default candidate set (mirrors the legacy script constants for E14 parity)
# -----------------------------------------------------------------------------

_DEFAULT_CANDIDATES: Tuple[Tuple[str, int], ...] = (
    ("subpenny_intensity", 8),
    ("trf_signed_imbalance", 0),
    ("dark_share", 4),
    ("spread_bps", 12),
    ("quote_imbalance", 16),
    ("bbo_update_rate", 15),
)

_DEFAULT_HORIZONS: Tuple[Tuple[int, str], ...] = (
    (0, "H=1"),
    (4, "H=10"),
    (7, "H=60"),
)


# -----------------------------------------------------------------------------
# Config dataclass
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class GateCheckConfig:
    """Configuration for ``run()``.

    All fields keyword-only — the legacy script had zero CLI flags (module-
    level constants), so any value that varied between runs was captured here
    as a parameter rather than inferred from the environment.

    Attributes:
        name: Experiment identity (for JSON/report headers).
        schema_version: Result schema version; bump for additive changes.
        export_dir: Absolute or working-dir-relative path to an NPY export
            produced by ``basic-quote-processor``. Required.
        splits: Which splits to evaluate. Each produces its own findings.
        candidates: Tuple of ``(feature_name, feature_index)`` pairs to probe.
            Feature indices reference the off-exchange 34-feature schema.
        horizons: Tuple of ``(label_column_index, horizon_label)`` pairs.
        g1_acf60_threshold: Pass bar for G1 persistence gate.
        g2_stride60_ic_threshold: Pass bar for G2 stride-60 IC gate.
        g3_lag1_ic_threshold: Pass bar for G3 lag-1 IC gate.
        stride: Bins per trading step (default 60).
        min_valid_samples: Minimum per-day finite samples to include the day.
        min_acf1_samples: Minimum per-day samples for ACF(1) compute.
        min_acf30_samples: Minimum for ACF(30) — also used for ACF(60).
        min_da_nonzero_samples: Minimum non-zero demeaned points for DA.
        output_dir: If set, write ``results.json`` and/or ``REPORT.md`` there.
            If None, library caller receives only the returned dataclass.
        write_json: Gate the JSON write.
        write_markdown: Gate the markdown report write.
        verbose: Emit per-feature progress logs.
    """

    name: str = "offexchange_gate_check"
    schema_version: str = "offexchange_gate_check_v2"
    export_dir: str = ""
    splits: Tuple[str, ...] = ("val", "test")
    candidates: Tuple[Tuple[str, int], ...] = _DEFAULT_CANDIDATES
    horizons: Tuple[Tuple[int, str], ...] = _DEFAULT_HORIZONS
    g1_acf60_threshold: float = 0.30
    g2_stride60_ic_threshold: float = 0.05
    g3_lag1_ic_threshold: float = 0.03
    stride: int = 60
    min_valid_samples: int = 20
    min_acf1_samples: int = 65
    min_acf30_samples: int = 35
    min_da_nonzero_samples: int = 10
    output_dir: Optional[str] = None
    write_json: bool = True
    write_markdown: bool = True
    verbose: bool = True

    def __post_init__(self) -> None:
        # Sanity bounds (thresholds are magnitudes — must be non-negative).
        if self.g1_acf60_threshold < 0 or self.g1_acf60_threshold > 1:
            raise ValueError(
                f"g1_acf60_threshold must be in [0, 1]; got {self.g1_acf60_threshold}"
            )
        if self.g2_stride60_ic_threshold < 0 or self.g2_stride60_ic_threshold > 1:
            raise ValueError(
                f"g2_stride60_ic_threshold must be in [0, 1]; got {self.g2_stride60_ic_threshold}"
            )
        if self.g3_lag1_ic_threshold < 0 or self.g3_lag1_ic_threshold > 1:
            raise ValueError(
                f"g3_lag1_ic_threshold must be in [0, 1]; got {self.g3_lag1_ic_threshold}"
            )
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1; got {self.stride}")
        if self.min_valid_samples < 1:
            raise ValueError("min_valid_samples must be >= 1")
        for split in self.splits:
            if split not in ("train", "val", "test"):
                raise ValueError(f"Invalid split {split!r}; expected train|val|test")

    def validate(self) -> None:
        """Extra validation that requires external state (dir existence)."""
        if not self.export_dir:
            raise ValueError("GateCheckConfig.export_dir is required")
        if not Path(self.export_dir).expanduser().exists():
            raise FileNotFoundError(
                f"Export directory does not exist: {self.export_dir}"
            )


# -----------------------------------------------------------------------------
# Result dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class GateCheckFinding:
    """One row of the gate check — a (split, feature, horizon) triple."""

    split: str
    feature: str
    feature_index: int
    horizon: str
    horizon_index: int
    # G1
    acf1: float
    acf30: float
    acf60: float
    g1_pass: bool
    # G2
    ic_stride1_mean: float
    ic_stride60_pooled: float
    ic_degradation_ratio: float
    n_stride60_samples: int
    g2_pass: bool
    # G3
    lag1_ic_pooled: float
    g3_pass: bool
    # Tradability diagnostics
    da_traded_bins: float
    q_spread_traded_bps: float
    n_traded_bins: int
    # Overall
    all_gates_pass: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateCheckResult:
    """Top-level result of ``run()``. JSON-serializable via ``to_dict()``."""

    schema: str
    analysis_date: str
    export_dir: str
    gate_thresholds: Dict[str, float]
    findings: Tuple[GateCheckFinding, ...]
    elapsed_seconds: float = 0.0

    @property
    def passes(self) -> Tuple[GateCheckFinding, ...]:
        return tuple(f for f in self.findings if f.all_gates_pass)

    @property
    def verdict(self) -> str:
        return "PASS" if self.passes else "FAIL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "analysis_date": self.analysis_date,
            "export_dir": self.export_dir,
            "gate_thresholds": dict(self.gate_thresholds),
            "findings": [f.to_dict() for f in self.findings],
            "elapsed_seconds": self.elapsed_seconds,
        }

    def to_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


# -----------------------------------------------------------------------------
# Core analysis — logic preserved byte-for-byte from legacy script for E14
# reproducibility. Do NOT refactor without re-baselining against the legacy
# `outputs/offexchange_gate_check/results.json` fixture.
# -----------------------------------------------------------------------------


def _analyze_feature(
    days: List[Dict[str, np.ndarray]],
    feat_idx: int,
    feat_name: str,
    horizon_idx: int,
    horizon_name: str,
    split: str,
    config: GateCheckConfig,
) -> GateCheckFinding:
    """Run all three gates for one feature at one horizon.

    Mirrors the legacy ``analyze_feature`` function. The mean-of-per-day ACF
    aggregation is deliberately preserved (statistically unusual but
    E14-defining; see migration note in module docstring).
    """
    acf1_list, acf30_list, acf60_list = [], [], []
    per_day_ic_s1: List[float] = []
    pooled_feat_s60: List[float] = []
    pooled_lab_s60: List[float] = []
    pooled_feat_lag: List[float] = []
    pooled_lab_lag: List[float] = []
    pooled_feat_traded: List[float] = []
    pooled_ret_traded: List[float] = []

    stride = config.stride

    for day in days:
        feat = day["features"][:, feat_idx]
        lab = day["labels"][:, horizon_idx]
        valid = np.isfinite(feat) & np.isfinite(lab)

        if valid.sum() < config.min_valid_samples:
            continue

        f, r = feat[valid], lab[valid]
        nv = len(f)

        # G1 ACFs
        if nv > config.min_acf1_samples:
            acf1_list.append(float(np.corrcoef(f[:-1], f[1:])[0, 1]))
            if nv > config.min_acf30_samples:
                acf30_list.append(float(np.corrcoef(f[:-30], f[30:])[0, 1]))
            if nv > config.min_acf1_samples:  # >65 for ACF(60) too
                acf60_list.append(float(np.corrcoef(f[:-60], f[60:])[0, 1]))

        rho, _ = spearmanr(f, r)
        if np.isfinite(rho):
            per_day_ic_s1.append(float(rho))

        s60_idx = np.arange(stride, nv, stride)
        if len(s60_idx) > 0:
            pooled_feat_s60.extend(f[s60_idx].tolist())
            pooled_lab_s60.extend(r[s60_idx].tolist())

        if nv > 1:
            pooled_feat_lag.extend(f[:-1].tolist())
            pooled_lab_lag.extend(r[1:].tolist())

        traded_idx = np.arange(stride, nv, stride)
        if len(traded_idx) > 0:
            pooled_feat_traded.extend(f[traded_idx].tolist())
            pooled_ret_traded.extend(r[traded_idx].tolist())

    # G1
    acf1 = float(np.mean(acf1_list)) if acf1_list else 0.0
    acf30 = float(np.mean(acf30_list)) if acf30_list else 0.0
    acf60 = float(np.mean(acf60_list)) if acf60_list else 0.0
    g1_pass = acf60 > config.g1_acf60_threshold

    # G2
    pf_s60 = np.array(pooled_feat_s60)
    pl_s60 = np.array(pooled_lab_s60)
    if len(pf_s60) >= config.min_valid_samples:
        pooled_ic_s60_raw, _ = spearmanr(pf_s60, pl_s60)
        pooled_ic_s60 = float(pooled_ic_s60_raw) if np.isfinite(pooled_ic_s60_raw) else 0.0
    else:
        pooled_ic_s60 = 0.0
    g2_pass = abs(pooled_ic_s60) > config.g2_stride60_ic_threshold

    # G3
    pf_lag = np.array(pooled_feat_lag)
    pl_lag = np.array(pooled_lab_lag)
    if len(pf_lag) >= config.min_valid_samples:
        pooled_lag1_raw, _ = spearmanr(pf_lag, pl_lag)
        pooled_lag1_ic = float(pooled_lag1_raw) if np.isfinite(pooled_lag1_raw) else 0.0
    else:
        pooled_lag1_ic = 0.0
    g3_pass = abs(pooled_lag1_ic) > config.g3_lag1_ic_threshold

    mean_ic_s1 = float(np.mean(per_day_ic_s1)) if per_day_ic_s1 else 0.0
    ic_degradation = (
        abs(pooled_ic_s60) / max(abs(mean_ic_s1), 1e-10)
        if mean_ic_s1 != 0
        else 0.0
    )

    # Tradability
    pf_t = np.array(pooled_feat_traded)
    pr_t = np.array(pooled_ret_traded)
    da_traded = 0.0
    q_spread = 0.0
    n_traded = len(pf_t)
    if n_traded >= config.min_valid_samples:
        dm_f = pf_t - np.mean(pf_t)
        dm_r = pr_t - np.mean(pr_t)
        nonzero = (dm_f != 0) & (dm_r != 0)
        if nonzero.sum() > config.min_da_nonzero_samples:
            da_traded = float(np.mean(np.sign(dm_f[nonzero]) == np.sign(dm_r[nonzero])))
        q20 = np.percentile(pf_t, 20)
        q80 = np.percentile(pf_t, 80)
        bot = pr_t[pf_t <= q20]
        top = pr_t[pf_t >= q80]
        if len(bot) > 0 and len(top) > 0:
            q_spread = float(np.mean(top) - np.mean(bot))

    all_pass = g1_pass and g2_pass and g3_pass

    return GateCheckFinding(
        split=split,
        feature=feat_name,
        feature_index=feat_idx,
        horizon=horizon_name,
        horizon_index=horizon_idx,
        acf1=round(acf1, 4),
        acf30=round(acf30, 4),
        acf60=round(acf60, 4),
        g1_pass=bool(g1_pass),
        ic_stride1_mean=round(mean_ic_s1, 4),
        ic_stride60_pooled=round(pooled_ic_s60, 4),
        ic_degradation_ratio=round(ic_degradation, 4),
        n_stride60_samples=int(n_traded),
        g2_pass=bool(g2_pass),
        lag1_ic_pooled=round(pooled_lag1_ic, 4),
        g3_pass=bool(g3_pass),
        da_traded_bins=round(da_traded, 4),
        q_spread_traded_bps=round(q_spread, 2),
        n_traded_bins=int(n_traded),
        all_gates_pass=bool(all_pass),
    )


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def _load_split_days(export_dir: str, split: str) -> List[Dict[str, np.ndarray]]:
    """Load all days for a split into an in-memory list of dicts.

    Shape mirrors the legacy script: each day has ``features`` (N, n_feat)
    and ``labels`` (N, n_horizons), extracted from the export's sequence
    tensor's last timestep.
    """
    loader = ExportLoader(Path(export_dir), split)
    days = []
    for date in loader.list_dates():
        b = loader.load_day(date)
        days.append({
            "features": b.sequences[:, -1, :],
            "labels": b.labels,
        })
    return days


def run(config: GateCheckConfig) -> GateCheckResult:
    """Execute the off-exchange signal gate check.

    Args:
        config: Validated ``GateCheckConfig``. If the caller constructed the
            dataclass directly (not via a YAML loader), ``config.validate()``
            is invoked at the top of ``run()`` — caller need not pre-validate.

    Returns:
        ``GateCheckResult`` with one ``GateCheckFinding`` per
        (split × feature × horizon) combination.

    Side effects (only when ``config.output_dir`` is not None):
        - Writes ``results.json`` (gated by ``config.write_json``).
        - Writes ``REPORT.md`` (gated by ``config.write_markdown``).

    Raises:
        ValueError: on invalid config (range-checked fields).
        FileNotFoundError: on missing ``export_dir``.
    """
    config.validate()

    start = time.monotonic()
    findings: List[GateCheckFinding] = []

    for split in config.splits:
        if config.verbose:
            logger.info("loading split=%s from %s", split, config.export_dir)
        days = _load_split_days(config.export_dir, split)
        if config.verbose:
            logger.info("split=%s: %d days loaded", split, len(days))
        for feat_name, feat_idx in config.candidates:
            for horizon_idx, horizon_name in config.horizons:
                finding = _analyze_feature(
                    days=days,
                    feat_idx=feat_idx,
                    feat_name=feat_name,
                    horizon_idx=horizon_idx,
                    horizon_name=horizon_name,
                    split=split,
                    config=config,
                )
                findings.append(finding)
                if config.verbose and finding.all_gates_pass:
                    logger.info(
                        "PASS %s %s %s: acf60=%.4f ic60=%.4f lag1=%.4f",
                        split, feat_name, horizon_name,
                        finding.acf60, finding.ic_stride60_pooled,
                        finding.lag1_ic_pooled,
                    )

    elapsed = time.monotonic() - start

    # Phase 6 6A.10 (2026-04-17): datetime.utcnow() is deprecated in Python
    # 3.12+. Use timezone-aware now(UTC) for forward compatibility.
    from datetime import timezone as _tz
    result = GateCheckResult(
        schema=config.schema_version,
        analysis_date=datetime.now(_tz.utc).strftime("%Y-%m-%d"),
        export_dir=str(config.export_dir),
        gate_thresholds={
            "g1_acf60": config.g1_acf60_threshold,
            "g2_stride60_ic": config.g2_stride60_ic_threshold,
            "g3_lag1_ic": config.g3_lag1_ic_threshold,
        },
        findings=tuple(findings),
        elapsed_seconds=elapsed,
    )

    if config.output_dir is not None:
        out = Path(config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if config.write_json:
            result.to_json(out / "results.json")
        if config.write_markdown:
            (out / "REPORT.md").write_text(_build_markdown(result))

    return result


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------


def _build_markdown(result: GateCheckResult) -> str:
    """Produce REPORT.md content matching the legacy script's layout."""
    lines = [
        f"# Off-Exchange Gate Check — {result.verdict}",
        "",
        f"- Analysis date: {result.analysis_date}",
        f"- Export: `{result.export_dir}`",
        f"- Schema: `{result.schema}`",
        f"- Elapsed: {result.elapsed_seconds:.2f}s",
        "",
        "## Gate thresholds",
        "",
    ]
    for k, v in result.gate_thresholds.items():
        lines.append(f"- `{k}`: {v}")

    lines.extend([
        "",
        "## Summary table",
        "",
        "| Split | Feature | Horizon | ACF60 | IC60 | Lag1IC | Verdict |",
        "|---|---|---|---|---|---|---|",
    ])
    for f in result.findings:
        verdict = "PASS" if f.all_gates_pass else "FAIL"
        lines.append(
            f"| {f.split} | {f.feature} | {f.horizon} | "
            f"{f.acf60:.4f} | {f.ic_stride60_pooled:.4f} | "
            f"{f.lag1_ic_pooled:.4f} | {verdict} |"
        )

    if result.passes:
        lines.extend(["", "## Passes detail", ""])
        for f in result.passes:
            lines.append(
                f"- **{f.split} / {f.feature} / {f.horizon}**: "
                f"ACF60={f.acf60:.4f} IC60={f.ic_stride60_pooled:.4f} "
                f"Lag1IC={f.lag1_ic_pooled:.4f} "
                f"DA_traded={f.da_traded_bins:.4f} "
                f"Q-spread_bps={f.q_spread_traded_bps:.2f}"
            )
    else:
        lines.extend([
            "",
            "## No passing features",
            "",
            "Zero features cleared all three gates. Directional trading "
            "NOT viable at this cadence/horizon set.",
        ])

    return "\n".join(lines) + "\n"
