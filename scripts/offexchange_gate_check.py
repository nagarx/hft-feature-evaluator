#!/usr/bin/env python3
"""
Off-Exchange Signal Gate Check — E14 (E13 Lesson 33 Filter).

Applies three gates to off-exchange features BEFORE any backtesting:
  G1: Persistence — ACF(60) > 0.30 (signal persists at trading cadence)
  G2: Stride-60 IC — POOLED IC at stride=60 > 0.05 (IC survives at trading cadence)
  G3: Lag-1 IC — POOLED IC(feat[t], ret[t+1]) > 0.03 (temporal, not just contemporaneous)

If no feature passes all three gates, directional LOB/off-exchange trading is NOT viable.

Usage:
    python scripts/offexchange_gate_check.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from hft_evaluator.data.loader import ExportLoader


# =============================================================================
# Constants
# =============================================================================

EXPORT_DIR = "../data/exports/basic_nvda_60s"
OUTPUT_DIR = "outputs/offexchange_gate_check"

# Candidate features
CANDIDATES = {
    "subpenny_intensity": 8,
    "trf_signed_imbalance": 0,
    "dark_share": 4,
    "spread_bps": 12,
    "quote_imbalance": 16,
    "bbo_update_rate": 15,
}

# Horizons to test: index in labels array → horizon in minutes
HORIZONS = {0: "H=1", 4: "H=10", 7: "H=60"}

# Gate thresholds
G1_ACF60_THRESHOLD = 0.30
G2_STRIDE60_IC_THRESHOLD = 0.05
G3_LAG1_IC_THRESHOLD = 0.03

STRIDE = 60  # trading cadence: 1 trade per 60 bins


def log(msg: str) -> None:
    print(f"[gate] {msg}", flush=True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# =============================================================================
# Analysis
# =============================================================================

def analyze_feature(days, feat_idx, feat_name, horizon_idx, horizon_name):
    """Run all three gates for one feature at one horizon."""
    # Collect per-day ACFs
    acf1_list, acf30_list, acf60_list = [], [], []
    # Collect per-day stride=1 ICs
    per_day_ic_s1 = []
    # Pooled arrays for stride=60 and lag-1
    pooled_feat_s60, pooled_lab_s60 = [], []
    pooled_feat_lag, pooled_lab_lag = [], []
    pooled_feat_traded, pooled_ret_traded = [], []

    for day in days:
        feat = day["features"][:, feat_idx]
        lab = day["labels"][:, horizon_idx]
        n = len(feat)
        valid = np.isfinite(feat) & np.isfinite(lab)

        if valid.sum() < 20:
            continue

        f, r = feat[valid], lab[valid]
        nv = len(f)

        # G1: ACF per day
        if nv > 65:
            acf1_list.append(float(np.corrcoef(f[:-1], f[1:])[0, 1]))
            if nv > 35:
                acf30_list.append(float(np.corrcoef(f[:-30], f[30:])[0, 1]))
            if nv > 65:
                acf60_list.append(float(np.corrcoef(f[:-60], f[60:])[0, 1]))

        # Per-day IC at stride=1
        rho, _ = spearmanr(f, r)
        if np.isfinite(rho):
            per_day_ic_s1.append(float(rho))

        # Pooled stride=60 samples
        s60_idx = np.arange(STRIDE, nv, STRIDE)
        if len(s60_idx) > 0:
            pooled_feat_s60.extend(f[s60_idx].tolist())
            pooled_lab_s60.extend(r[s60_idx].tolist())

        # Pooled lag-1 samples
        if nv > 1:
            pooled_feat_lag.extend(f[:-1].tolist())
            pooled_lab_lag.extend(r[1:].tolist())

        # Pooled traded-bin samples (entry at stride intervals after warmup)
        traded_idx = np.arange(STRIDE, nv, STRIDE)
        if len(traded_idx) > 0:
            pooled_feat_traded.extend(f[traded_idx].tolist())
            pooled_ret_traded.extend(r[traded_idx].tolist())

    # === G1: Persistence ===
    acf1 = float(np.mean(acf1_list)) if acf1_list else 0.0
    acf30 = float(np.mean(acf30_list)) if acf30_list else 0.0
    acf60 = float(np.mean(acf60_list)) if acf60_list else 0.0
    g1_pass = acf60 > G1_ACF60_THRESHOLD

    # === G2: Pooled stride-60 IC ===
    pf_s60 = np.array(pooled_feat_s60)
    pl_s60 = np.array(pooled_lab_s60)
    if len(pf_s60) >= 20:
        pooled_ic_s60, _ = spearmanr(pf_s60, pl_s60)
        pooled_ic_s60 = float(pooled_ic_s60) if np.isfinite(pooled_ic_s60) else 0.0
    else:
        pooled_ic_s60 = 0.0
    g2_pass = abs(pooled_ic_s60) > G2_STRIDE60_IC_THRESHOLD

    # === G3: Pooled lag-1 IC ===
    pf_lag = np.array(pooled_feat_lag)
    pl_lag = np.array(pooled_lab_lag)
    if len(pf_lag) >= 20:
        pooled_lag1_ic, _ = spearmanr(pf_lag, pl_lag)
        pooled_lag1_ic = float(pooled_lag1_ic) if np.isfinite(pooled_lag1_ic) else 0.0
    else:
        pooled_lag1_ic = 0.0
    g3_pass = abs(pooled_lag1_ic) > G3_LAG1_IC_THRESHOLD

    # Per-day IC at stride=1
    mean_ic_s1 = float(np.mean(per_day_ic_s1)) if per_day_ic_s1 else 0.0
    ic_degradation = abs(pooled_ic_s60) / max(abs(mean_ic_s1), 1e-10) if mean_ic_s1 != 0 else 0.0

    # Direction accuracy and quintile spread at traded bins
    pf_t = np.array(pooled_feat_traded)
    pr_t = np.array(pooled_ret_traded)
    da_traded = 0.0
    q_spread = 0.0
    n_traded = len(pf_t)
    if n_traded >= 20:
        # Demeaned DA (feature always positive → raw DA meaningless)
        dm_f = pf_t - np.mean(pf_t)
        dm_r = pr_t - np.mean(pr_t)
        nonzero = (dm_f != 0) & (dm_r != 0)
        if nonzero.sum() > 10:
            da_traded = float(np.mean(np.sign(dm_f[nonzero]) == np.sign(dm_r[nonzero])))

        # Quintile spread
        q20 = np.percentile(pf_t, 20)
        q80 = np.percentile(pf_t, 80)
        bot = pr_t[pf_t <= q20]
        top = pr_t[pf_t >= q80]
        if len(bot) > 0 and len(top) > 0:
            q_spread = float(np.mean(top) - np.mean(bot))

    all_pass = g1_pass and g2_pass and g3_pass

    return {
        "feature": feat_name,
        "feature_index": feat_idx,
        "horizon": horizon_name,
        "horizon_index": horizon_idx,
        # G1
        "acf1": round(acf1, 4),
        "acf30": round(acf30, 4),
        "acf60": round(acf60, 4),
        "g1_pass": g1_pass,
        # G2
        "ic_stride1_mean": round(mean_ic_s1, 4),
        "ic_stride60_pooled": round(pooled_ic_s60, 4),
        "ic_degradation_ratio": round(ic_degradation, 4),
        "n_stride60_samples": n_traded,
        "g2_pass": g2_pass,
        # G3
        "lag1_ic_pooled": round(pooled_lag1_ic, 4),
        "g3_pass": g3_pass,
        # Tradability
        "da_traded_bins": round(da_traded, 4),
        "q_spread_traded_bps": round(q_spread, 2),
        "n_traded_bins": n_traded,
        # Overall
        "all_gates_pass": all_pass,
    }


# =============================================================================
# Report
# =============================================================================

def build_report(results):
    lines = []
    w = lines.append

    w("# Off-Exchange Signal Gate Check (E14)")
    w(f"\n> Date: {results.get('analysis_date', 'N/A')}")
    w(f"> Gate criteria: G1(ACF60>{G1_ACF60_THRESHOLD}), "
      f"G2(stride-60 IC>{G2_STRIDE60_IC_THRESHOLD}), "
      f"G3(lag-1 IC>{G3_LAG1_IC_THRESHOLD})")

    # Any passes?
    all_checks = results.get("checks", [])
    passes = [c for c in all_checks if c["all_gates_pass"]]
    w(f"\n> **Verdict: {'GATE PASS — backtest warranted' if passes else 'ALL FAIL — no tradeable signal'}**")

    # Summary table
    w("\n## Gate Results\n")
    w("| Feature | Horizon | ACF(60) | G1 | IC(s=1) | IC(s=60) | G2 | Lag-1 IC | G3 | **ALL** |")
    w("|---------|---------|---------|----|---------|---------|----|----------|----|---------|")
    for c in all_checks:
        g1 = "PASS" if c["g1_pass"] else "fail"
        g2 = "PASS" if c["g2_pass"] else "fail"
        g3 = "PASS" if c["g3_pass"] else "fail"
        verdict = "**PASS**" if c["all_gates_pass"] else "fail"
        w(f"| {c['feature']} | {c['horizon']} | {c['acf60']:.3f} | {g1} | "
          f"{c['ic_stride1_mean']:+.4f} | {c['ic_stride60_pooled']:+.4f} | {g2} | "
          f"{c['lag1_ic_pooled']:+.4f} | {g3} | {verdict} |")

    # Detail for passing features
    if passes:
        w("\n## Passing Features (Detail)\n")
        for c in passes:
            w(f"### {c['feature']} @ {c['horizon']}\n")
            w(f"- ACF: (1)={c['acf1']:.3f}, (30)={c['acf30']:.3f}, (60)={c['acf60']:.3f}")
            w(f"- IC at stride=1 (per-day mean): {c['ic_stride1_mean']:+.4f}")
            w(f"- IC at stride=60 (pooled): {c['ic_stride60_pooled']:+.4f} "
              f"(degradation: {c['ic_degradation_ratio']:.1%})")
            w(f"- Lag-1 IC (pooled): {c['lag1_ic_pooled']:+.4f}")
            w(f"- DA at traded bins: {c['da_traded_bins']:.1%}")
            w(f"- Quintile spread at traded bins: {c['q_spread_traded_bps']:+.2f} bps")
            w(f"- Traded samples: {c['n_traded_bins']}")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    start = time.time()
    log("Off-Exchange Signal Gate Check (E14) starting...")

    script_dir = Path(__file__).resolve().parent.parent
    export_path = (script_dir / EXPORT_DIR).resolve()
    output_path = (script_dir / OUTPUT_DIR).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    all_checks = []
    for split_name in ["val", "test"]:
        loader = ExportLoader(str(export_path), split_name)
        dates = loader.list_dates()
        log(f"Loading {split_name}: {len(dates)} days...")

        days = []
        for d in dates:
            b = loader.load_day(d)
            days.append({
                "date": d,
                "features": np.asarray(b.sequences[:, -1, :], dtype=np.float64),
                "labels": np.asarray(b.labels, dtype=np.float64),
                "n": b.sequences.shape[0],
            })

        log(f"Analyzing {len(CANDIDATES)} features × {len(HORIZONS)} horizons on {split_name}...")
        for feat_name, feat_idx in CANDIDATES.items():
            for h_idx, h_name in HORIZONS.items():
                result = analyze_feature(days, feat_idx, feat_name, h_idx, h_name)
                result["split"] = split_name
                all_checks.append(result)

                status = "PASS" if result["all_gates_pass"] else "    "
                log(f"  {status} {feat_name:25s} {h_name:5s} "
                    f"ACF60={result['acf60']:.3f} "
                    f"IC(s1)={result['ic_stride1_mean']:+.4f} "
                    f"IC(s60)={result['ic_stride60_pooled']:+.4f} "
                    f"lag1={result['lag1_ic_pooled']:+.4f}")

    # Assemble results
    results = {
        "schema": "offexchange_gate_check_v1",
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "export_dir": str(export_path),
        "gates": {
            "G1_acf60_threshold": G1_ACF60_THRESHOLD,
            "G2_stride60_ic_threshold": G2_STRIDE60_IC_THRESHOLD,
            "G3_lag1_ic_threshold": G3_LAG1_IC_THRESHOLD,
        },
        "checks": all_checks,
    }

    # Write
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    report = build_report(results)
    with open(output_path / "REPORT.md", "w") as f:
        f.write(report)

    elapsed = time.time() - start
    log(f"\nTotal: {elapsed:.1f}s")

    # Summary
    passes = [c for c in all_checks if c["all_gates_pass"]]
    log(f"\n{'='*60}")
    if passes:
        log(f"GATE PASS: {len(passes)} feature×horizon combinations passed all 3 gates")
        for p in passes:
            log(f"  {p['feature']} @ {p['horizon']} ({p['split']}): "
                f"ACF60={p['acf60']:.3f}, IC(s60)={p['ic_stride60_pooled']:+.4f}, "
                f"lag1={p['lag1_ic_pooled']:+.4f}")
    else:
        log("ALL FAIL: No feature passes all 3 gates.")
        log("Directional LOB/off-exchange trading is NOT viable with this pipeline.")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
