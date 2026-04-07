#!/usr/bin/env python3
"""
Feature OOS Signal Analysis — E13 Phase 7.

Computes per-day OOS IC for all 5 core features individually + Ridge model,
with tradability metrics (quintile spread, Grinold E[r]), UP/DOWN conditioning,
concept drift, and head-to-head comparison.

Decision gate: spread_bps OOS per-day IC determines backtester strategy.
  > 0.30 → SPREAD_PRIMARY (spread_bps rank is the signal)
  0.10-0.30 → DUAL_TEST (test both spread_bps and Ridge)
  < 0.10 → RIDGE_ONLY (Ridge model IC=0.07 is the only viable signal)

Usage:
    python scripts/feature_oos_analysis.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from hft_evaluator.data.loader import ExportLoader
from hft_metrics import spearman_ic, EPS


# =============================================================================
# Constants
# =============================================================================

SEED = 42
EXPORT_DIR = "../data/exports/e5_timebased_60s_point_return"
OUTPUT_DIR = "outputs/feature_oos_analysis"
TARGET_HORIZON_IDX = 7  # H=60

# Features to analyze individually (Subset A from ridge_walkforward_mbo.py)
FEATURES = {
    "spread_bps": 42,
    "total_ask_volume": 44,
    "volume_imbalance": 45,
    "true_ofi": 84,
    "depth_norm_ofi": 85,
}
ALL_INDICES = sorted(FEATURES.values())

RIDGE_ALPHA = 1000.0
COST_BPS = 1.4
MIN_SAMPLES_IC = 10   # minimum samples per day for IC computation
MIN_SAMPLES_Q = 20    # minimum samples per day for quintile analysis


def log(msg: str) -> None:
    print(f"[feat_oos] {msg}", flush=True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _safe_ic(x, y):
    """Spearman IC with NaN guard."""
    rho, p = spearman_ic(x, y)
    if rho == 0.0 and p == 1.0:
        return float("nan")
    return float(rho)


# =============================================================================
# Data Loading
# =============================================================================

def load_all_data(loader, dates, feature_indices):
    """Load all days into per-day feature + label arrays."""
    day_data = []
    for d in dates:
        b = loader.load_day(d)
        feat = np.asarray(b.sequences[:, -1, :][:, feature_indices], dtype=np.float64)
        lab = np.asarray(b.labels[:, TARGET_HORIZON_IDX], dtype=np.float64)
        day_data.append({"date": d, "features": feat, "labels": lab, "n": feat.shape[0]})
    return day_data


# =============================================================================
# Ridge Model (reused from ridge_walkforward_mbo.py)
# =============================================================================

def ridge_fit_predict(X_train, y_train, X_test, alpha=1.0):
    """Standardized Ridge regression. Returns (predictions, beta, mu, sigma)."""
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    sigma[sigma < 1e-10] = 1.0

    X_tr = (X_train - mu) / sigma
    X_te = (X_test - mu) / sigma

    n, d = X_tr.shape
    X_tr_i = np.column_stack([np.ones(n), X_tr])
    X_te_i = np.column_stack([np.ones(X_te.shape[0]), X_te])
    I = np.eye(d + 1)
    I[0, 0] = 0.0  # don't regularize intercept

    try:
        beta = np.linalg.solve(X_tr_i.T @ X_tr_i + alpha * I, X_tr_i.T @ y_train)
        preds = X_te_i @ beta
        return preds, beta, mu, sigma
    except np.linalg.LinAlgError:
        return np.full(X_test.shape[0], np.mean(y_train)), np.zeros(d + 1), mu, sigma


# =============================================================================
# Per-Day Signal Analysis
# =============================================================================

def analyze_signal_per_day(signal_values, returns, n_min_ic=MIN_SAMPLES_IC,
                           n_min_q=MIN_SAMPLES_Q):
    """Compute IC, demeaned DA, and quintile spread for a single signal on one day.

    Args:
        signal_values: [N] array of signal values for this day
        returns: [N] array of point returns (bps) for this day
        n_min_ic: minimum samples for IC computation
        n_min_q: minimum samples for quintile analysis

    Returns:
        dict with per-day metrics, or None if insufficient data.
    """
    valid = np.isfinite(signal_values) & np.isfinite(returns)
    n_valid = int(valid.sum())
    if n_valid < n_min_ic:
        return None

    sig = signal_values[valid]
    ret = returns[valid]

    # Spearman IC (signed — sign indicates trading direction)
    ic = _safe_ic(sig, ret)
    if not np.isfinite(ic):
        return None

    # Demeaned DA: deviation-predicts-deviation
    # (works for always-positive features like spread_bps)
    dm_sig = sig - np.mean(sig)
    dm_ret = ret - np.mean(ret)
    nonzero = (dm_sig != 0) & (dm_ret != 0)
    demeaned_da = float(np.mean(np.sign(dm_sig[nonzero]) == np.sign(dm_ret[nonzero]))) if nonzero.sum() > 10 else 0.5

    result = {
        "ic": float(ic),
        "n_valid": n_valid,
        "return_std_bps": float(np.std(ret)),
        "return_mean_bps": float(np.mean(ret)),
        "demeaned_da": demeaned_da,
    }

    # Quintile spread (rank-based tradability)
    if n_valid >= n_min_q:
        q20 = np.percentile(sig, 20)
        q80 = np.percentile(sig, 80)
        bot_mask = sig <= q20
        top_mask = sig >= q80
        if bot_mask.sum() > 0 and top_mask.sum() > 0:
            q1_ret = float(np.mean(ret[bot_mask]))
            q5_ret = float(np.mean(ret[top_mask]))
            result["q1_return_bps"] = q1_ret
            result["q5_return_bps"] = q5_ret
            result["q_spread_bps"] = q5_ret - q1_ret

    return result


# =============================================================================
# Aggregation
# =============================================================================

def aggregate_per_day(per_day_results):
    """Aggregate per-day results into summary statistics.

    Args:
        per_day_results: list of dicts from analyze_signal_per_day

    Returns:
        dict with aggregated metrics
    """
    if not per_day_results:
        return {"n_days": 0, "ic_per_day_mean": 0.0}

    ics = np.array([d["ic"] for d in per_day_results])
    n_days = len(ics)

    # IC distribution
    ic_mean = float(np.mean(ics))
    ic_std = float(np.std(ics))  # ddof=0 consistent with walk_forward
    ic_t_raw = ic_mean / max(ic_std / np.sqrt(n_days), EPS) if n_days >= 2 else 0.0

    # ACF(1) for t-stat correction
    acf1 = 0.0
    if n_days > 5:
        acf1 = float(np.corrcoef(ics[:-1], ics[1:])[0, 1])
    n_eff = n_days
    if acf1 > 0:
        n_eff = n_days * (1 - acf1) / (1 + acf1)
    n_eff = max(n_eff, 2)
    ic_t_corrected = ic_mean / max(ic_std / np.sqrt(n_eff), EPS) if n_days >= 2 else 0.0

    # Demeaned DA
    das = np.array([d["demeaned_da"] for d in per_day_results])

    # Quintile spreads
    q_spreads = np.array([d["q_spread_bps"] for d in per_day_results if "q_spread_bps" in d])
    q1_rets = np.array([d["q1_return_bps"] for d in per_day_results if "q1_return_bps" in d])
    q5_rets = np.array([d["q5_return_bps"] for d in per_day_results if "q5_return_bps" in d])

    # Grinold E[r] = IC * sigma_r (per day, then averaged)
    return_stds = np.array([d["return_std_bps"] for d in per_day_results])
    mean_return_std = float(np.mean(return_stds))
    grinold = ic_mean * mean_return_std
    grinold_net = grinold - COST_BPS

    # UP-day vs DOWN-day IC
    up_ics = [d["ic"] for d in per_day_results if d["return_mean_bps"] > 0]
    down_ics = [d["ic"] for d in per_day_results if d["return_mean_bps"] <= 0]

    # Concept drift (linear regression IC vs day index)
    drift_slope, drift_p = 0.0, 1.0
    if n_days >= 10:
        slope, intercept, r, p, se = scipy_stats.linregress(np.arange(n_days), ics)
        drift_slope, drift_p = float(slope), float(p)

    return {
        "n_days": n_days,
        # IC distribution
        "ic_per_day_mean": ic_mean,
        "ic_per_day_std": ic_std,
        "ic_per_day_median": float(np.median(ics)),
        "ic_per_day_q25": float(np.percentile(ics, 25)),
        "ic_per_day_q75": float(np.percentile(ics, 75)),
        "ic_per_day_min": float(np.min(ics)),
        "ic_per_day_max": float(np.max(ics)),
        "ic_per_day_frac_positive": float(np.mean(ics > 0)),
        "ic_per_day_t_raw": float(ic_t_raw),
        "ic_per_day_acf1": float(acf1),
        "ic_per_day_n_eff": float(n_eff),
        "ic_per_day_t_corrected": float(ic_t_corrected),
        # Demeaned DA
        "demeaned_da_mean": float(np.mean(das)),
        # Quintile tradability
        "q_spread_mean_bps": float(np.mean(q_spreads)) if len(q_spreads) > 0 else 0.0,
        "q1_return_mean_bps": float(np.mean(q1_rets)) if len(q1_rets) > 0 else 0.0,
        "q5_return_mean_bps": float(np.mean(q5_rets)) if len(q5_rets) > 0 else 0.0,
        # Grinold
        "grinold_expected_return_bps": float(grinold),
        "grinold_net_of_cost_bps": float(grinold_net),
        "mean_return_std_bps": mean_return_std,
        # UP/DOWN conditioning
        "ic_up_day_mean": float(np.mean(up_ics)) if len(up_ics) >= 3 else None,
        "ic_up_day_n": len(up_ics),
        "ic_down_day_mean": float(np.mean(down_ics)) if len(down_ics) >= 3 else None,
        "ic_down_day_n": len(down_ics),
        # Concept drift
        "ic_drift_slope": drift_slope,
        "ic_drift_p": drift_p,
        "ic_drift_interpretation": "stable" if drift_p > 0.10 else (
            "degrading" if drift_slope < 0 else "improving"
        ),
    }


# =============================================================================
# Report Builder
# =============================================================================

def build_report(results):
    """Generate markdown report."""
    lines = []
    w = lines.append

    w("# Feature OOS Signal Analysis Report (E13 Phase 7)")
    w(f"\n> **Date**: {results.get('analysis_date', 'N/A')}")
    w(f"> **Export**: {results.get('export_dir', 'N/A')}")

    decision = results.get("decision", {})
    w(f"\n> **Verdict**: **{decision.get('verdict', 'N/A')}** "
      f"(spread_bps val IC={decision.get('spread_bps_val_ic', 0):.4f})")

    # Ranking table
    w("\n## Feature Ranking (Val Per-Day |IC|)\n")
    w("| Rank | Signal | Val Per-Day IC | t (corrected) | Frac+ | "
      "Q-Spread (bps) | Grinold Net (bps) | Drift |")
    w("|------|--------|---------------|---------------|-------|"
      "---------------|-------------------|-------|")
    for i, entry in enumerate(results.get("ranking_table", []), 1):
        ic_val = entry.get("val_ic", 0)
        w(f"| {i} | {entry['signal']} | "
          f"{'**' if abs(ic_val) > 0.30 else ''}{ic_val:+.4f}{'**' if abs(ic_val) > 0.30 else ''} | "
          f"{entry.get('val_t_corrected', 0):.2f} | "
          f"{entry.get('val_frac_pos', 0):.0%} | "
          f"{entry.get('val_q_spread', 0):+.2f} | "
          f"{entry.get('val_grinold_net', 0):+.2f} | "
          f"{entry.get('val_drift', 'N/A')} |")

    # Detailed per-signal results
    signals = results.get("signals", {})
    for sig_name in ["spread_bps", "total_ask_volume", "volume_imbalance",
                     "true_ofi", "depth_norm_ofi", "ridge_5feat"]:
        sig = signals.get(sig_name, {})
        if not sig:
            continue
        w(f"\n## {sig_name}\n")
        for split in ["val", "test"]:
            s = sig.get(split, {})
            if not s or s.get("n_days", 0) == 0:
                continue
            w(f"**{split}** ({s['n_days']} days):")
            w(f"- Per-day IC: **{s['ic_per_day_mean']:+.4f}** "
              f"(t_raw={s.get('ic_per_day_t_raw', 0):.2f}, "
              f"t_corrected={s.get('ic_per_day_t_corrected', 0):.2f}, "
              f"ACF={s.get('ic_per_day_acf1', 0):.3f})")
            w(f"- IC distribution: median={s.get('ic_per_day_median', 0):+.4f}, "
              f"[Q25={s.get('ic_per_day_q25', 0):+.3f}, Q75={s.get('ic_per_day_q75', 0):+.3f}], "
              f"frac+={s.get('ic_per_day_frac_positive', 0):.0%}")
            w(f"- Demeaned DA: {s.get('demeaned_da_mean', 0):.4f}")
            w(f"- Quintile spread: {s.get('q_spread_mean_bps', 0):+.2f} bps "
              f"(Q1={s.get('q1_return_mean_bps', 0):+.2f}, Q5={s.get('q5_return_mean_bps', 0):+.2f})")
            w(f"- Grinold E[r]: {s.get('grinold_expected_return_bps', 0):.2f} bps "
              f"(net: {s.get('grinold_net_of_cost_bps', 0):+.2f} bps)")
            up_ic = s.get("ic_up_day_mean")
            down_ic = s.get("ic_down_day_mean")
            w(f"- UP-day IC: {up_ic if up_ic is not None else 'N/A'}"
              f" ({s.get('ic_up_day_n', 0)}d), "
              f"DOWN-day IC: {down_ic if down_ic is not None else 'N/A'}"
              f" ({s.get('ic_down_day_n', 0)}d)")
            w(f"- Drift: {s.get('ic_drift_interpretation', 'N/A')} "
              f"(slope={s.get('ic_drift_slope', 0):.6f}, p={s.get('ic_drift_p', 1):.3f})")
            w("")

    # Head-to-head
    h2h = results.get("head_to_head", {})
    w("\n## Head-to-Head\n")
    w(f"- spread_bps wins over Ridge: "
      f"**{h2h.get('spread_vs_ridge_val', 0):.0%}** (val), "
      f"**{h2h.get('spread_vs_ridge_test', 0):.0%}** (test)")
    for split in ["val", "test"]:
        wins = h2h.get(f"win_fractions_{split}", {})
        if wins:
            w(f"- {split} winner distribution: " +
              ", ".join(f"{k}={v:.0%}" for k, v in sorted(wins.items(), key=lambda x: -x[1])))

    # Decision
    w(f"\n## Decision Gate\n")
    w(f"- spread_bps val per-day IC: **{decision.get('spread_bps_val_ic', 0):.4f}**")
    w(f"- Threshold: > 0.30 → SPREAD_PRIMARY, 0.10-0.30 → DUAL_TEST, < 0.10 → RIDGE_ONLY")
    w(f"- **Verdict: {decision.get('verdict', 'N/A')}**")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Feature OOS Signal Analysis")
    parser.add_argument("--export-dir", default=EXPORT_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    np.random.seed(SEED)
    start = time.time()
    log("Feature OOS Signal Analysis starting...")

    script_dir = Path(__file__).resolve().parent.parent
    export_path = (script_dir / args.export_dir).resolve()
    output_path = (script_dir / args.output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # === Load Data ===
    log("Loading data...")
    train_loader = ExportLoader(str(export_path), "train")
    val_loader = ExportLoader(str(export_path), "val")
    test_loader = ExportLoader(str(export_path), "test")

    train_dates = train_loader.list_dates()
    val_dates = val_loader.list_dates()
    test_dates = test_loader.list_dates()
    log(f"Train: {len(train_dates)}, Val: {len(val_dates)}, Test: {len(test_dates)}")

    train_data = load_all_data(train_loader, train_dates, ALL_INDICES)
    val_data = load_all_data(val_loader, val_dates, ALL_INDICES)
    test_data = load_all_data(test_loader, test_dates, ALL_INDICES)

    # Feature name → column index in loaded data (position within ALL_INDICES)
    feat_col = {name: ALL_INDICES.index(idx) for name, idx in FEATURES.items()}

    # === Fit Ridge Model (all train → predict val/test) ===
    log("Fitting Ridge model...")
    X_train = np.vstack([d["features"] for d in train_data])
    y_train = np.concatenate([d["labels"] for d in train_data])
    tr_valid = np.all(np.isfinite(X_train), axis=1) & np.isfinite(y_train)
    X_tr, y_tr = X_train[tr_valid], y_train[tr_valid]

    # Predict on val and test day-by-day (store predictions per day)
    ridge_mu = np.mean(X_tr, axis=0)
    ridge_sigma = np.std(X_tr, axis=0)
    ridge_sigma[ridge_sigma < 1e-10] = 1.0
    X_tr_std = (X_tr - ridge_mu) / ridge_sigma
    n, d = X_tr_std.shape
    X_tr_i = np.column_stack([np.ones(n), X_tr_std])
    I = np.eye(d + 1)
    I[0, 0] = 0.0
    ridge_beta = np.linalg.solve(X_tr_i.T @ X_tr_i + RIDGE_ALPHA * I, X_tr_i.T @ y_tr)

    def ridge_predict_day(day_features):
        """Predict using fitted Ridge for one day's features."""
        X = np.asarray(day_features, dtype=np.float64)
        X_std = (X - ridge_mu) / ridge_sigma
        X_i = np.column_stack([np.ones(X_std.shape[0]), X_std])
        return X_i @ ridge_beta

    # === Per-Signal Per-Day Analysis ===
    log("Analyzing 6 signals across val + test...")
    signal_names = list(FEATURES.keys()) + ["ridge_5feat"]
    all_results = {}

    for sig_name in signal_names:
        all_results[sig_name] = {"per_day_details": {}}
        if sig_name != "ridge_5feat":
            all_results[sig_name]["feature_index"] = FEATURES[sig_name]

        for split_name, split_data in [("val", val_data), ("test", test_data)]:
            per_day = []
            for day in split_data:
                returns = day["labels"]
                if sig_name == "ridge_5feat":
                    signal_vals = ridge_predict_day(day["features"])
                else:
                    col = feat_col[sig_name]
                    signal_vals = day["features"][:, col]

                result = analyze_signal_per_day(signal_vals, returns)
                if result is not None:
                    result["date"] = day["date"]
                    per_day.append(result)

            # Aggregate
            agg = aggregate_per_day(per_day)
            all_results[sig_name][split_name] = agg
            all_results[sig_name]["per_day_details"][split_name] = per_day

    # === Head-to-Head Comparison ===
    log("Head-to-head comparison...")
    head_to_head = {}
    for split_name in ["val", "test"]:
        # Build per-day IC matrix: [n_days × n_signals]
        n_days_split = len(all_results["spread_bps"]["per_day_details"].get(split_name, []))
        if n_days_split == 0:
            continue
        daily_abs_ics = {}
        for sig_name in signal_names:
            details = all_results[sig_name]["per_day_details"].get(split_name, [])
            daily_abs_ics[sig_name] = [abs(d["ic"]) for d in details]

        # Win fractions
        wins = {s: 0 for s in signal_names}
        for i in range(n_days_split):
            best_sig = max(signal_names, key=lambda s: daily_abs_ics[s][i] if i < len(daily_abs_ics[s]) else -1)
            wins[best_sig] += 1
        win_fracs = {s: c / n_days_split for s, c in wins.items()}
        head_to_head[f"win_fractions_{split_name}"] = win_fracs

        # Pairwise: spread_bps vs ridge
        spread_details = all_results["spread_bps"]["per_day_details"].get(split_name, [])
        ridge_details = all_results["ridge_5feat"]["per_day_details"].get(split_name, [])
        n_compare = min(len(spread_details), len(ridge_details))
        if n_compare > 0:
            spread_wins = sum(1 for i in range(n_compare)
                            if abs(spread_details[i]["ic"]) > abs(ridge_details[i]["ic"]))
            head_to_head[f"spread_vs_ridge_{split_name}"] = spread_wins / n_compare

    # === Ranking Table (by val |IC|) ===
    ranking = []
    for sig_name in signal_names:
        val_data_sig = all_results[sig_name].get("val", {})
        test_data_sig = all_results[sig_name].get("test", {})
        ranking.append({
            "signal": sig_name,
            "val_ic": val_data_sig.get("ic_per_day_mean", 0),
            "val_abs_ic": abs(val_data_sig.get("ic_per_day_mean", 0)),
            "val_t_corrected": val_data_sig.get("ic_per_day_t_corrected", 0),
            "val_frac_pos": val_data_sig.get("ic_per_day_frac_positive", 0),
            "val_q_spread": val_data_sig.get("q_spread_mean_bps", 0),
            "val_grinold_net": val_data_sig.get("grinold_net_of_cost_bps", 0),
            "val_drift": val_data_sig.get("ic_drift_interpretation", "N/A"),
            "test_ic": test_data_sig.get("ic_per_day_mean", 0),
        })
    ranking.sort(key=lambda x: -x["val_abs_ic"])

    # === Decision Gate ===
    spread_val_ic = all_results["spread_bps"].get("val", {}).get("ic_per_day_mean", 0)
    if abs(spread_val_ic) > 0.30:
        verdict = "SPREAD_PRIMARY"
    elif abs(spread_val_ic) > 0.10:
        verdict = "DUAL_TEST"
    else:
        verdict = "RIDGE_ONLY"

    decision = {
        "spread_bps_val_ic": float(spread_val_ic),
        "spread_bps_test_ic": float(all_results["spread_bps"].get("test", {}).get("ic_per_day_mean", 0)),
        "ridge_val_ic": float(all_results["ridge_5feat"].get("val", {}).get("ic_per_day_mean", 0)),
        "ridge_test_ic": float(all_results["ridge_5feat"].get("test", {}).get("ic_per_day_mean", 0)),
        "verdict": verdict,
    }

    # === Assemble Results ===
    results = {
        "schema": "feature_oos_analysis_v1",
        "export_dir": str(export_path),
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "n_train_days": len(train_dates),
        "n_val_days": len(val_dates),
        "n_test_days": len(test_dates),
        "ridge_alpha": RIDGE_ALPHA,
        "signals": all_results,
        "head_to_head": head_to_head,
        "ranking_table": ranking,
        "decision": decision,
    }

    # === Write Outputs ===
    json_path = output_path / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    log(f"JSON: {json_path}")

    report = build_report(results)
    md_path = output_path / "REPORT.md"
    with open(md_path, "w") as f:
        f.write(report)
    log(f"Report: {md_path}")

    elapsed = time.time() - start
    log(f"\nTotal: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # Console summary
    log(f"\n{'='*60}")
    log(f"DECISION: {verdict}")
    log(f"  spread_bps val per-day IC: {spread_val_ic:+.4f}")
    log(f"  spread_bps test per-day IC: {decision['spread_bps_test_ic']:+.4f}")
    log(f"  ridge val per-day IC:      {decision['ridge_val_ic']:+.4f}")
    log(f"  ridge test per-day IC:     {decision['ridge_test_ic']:+.4f}")
    log(f"{'='*60}")
    log("\nRanking (by val |IC|):")
    for i, r in enumerate(ranking, 1):
        log(f"  {i}. {r['signal']:20s} val_IC={r['val_ic']:+.4f}  "
            f"test_IC={r['test_ic']:+.4f}  Q-spread={r['val_q_spread']:+.2f} bps")


if __name__ == "__main__":
    main()
