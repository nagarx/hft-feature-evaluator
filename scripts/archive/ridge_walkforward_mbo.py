#!/usr/bin/env python3
# STATUS: experimental fossil -- NOT a template for new work.
# ARCHIVED: Phase 6 6D (2026-04-17). Preserved for historical reproducibility.
# Per hft-rules §4 (no new ad-hoc scripts for experiments):
#   - NEW experiments MUST be authored as hft-ops manifests under
#     hft-ops/experiments/ OR sweep manifests under hft-ops/experiments/sweeps/
#   - Reusable analysis logic MUST live in library modules
#     (hft_evaluator.experiments.* / lobtrainer.experiments.*)
#   - See scripts/archive/README.md for the replacement for this script.

"""
MBO Ridge Walk-Forward Gate Test (E13 Follow-Up).

Tests whether a standardized Ridge model can capture the MBO point-return signal
(spread_bps IC=0.530 at H=60) with walk-forward stability > 2.0.

9 analyses, 5 decision gates. ~3-5 minutes runtime.

Usage:
    python scripts/ridge_walkforward_mbo.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from hft_evaluator.data.loader import ExportLoader
from hft_metrics import spearman_ic, r_squared, directional_accuracy, profitable_accuracy, ic_ir, EPS


# =============================================================================
# Constants
# =============================================================================

SEED = 42
EXPORT_DIR = "../data/exports/e5_timebased_60s_point_return"
OUTPUT_DIR = "outputs/ridge_walkforward_mbo"
TARGET_HORIZON_IDX = 7  # H=60

SUBSET_A = {
    "spread_bps": 42, "total_ask_volume": 44, "volume_imbalance": 45,
    "true_ofi": 84, "depth_norm_ofi": 85,
}
SUBSET_B = {
    **SUBSET_A, "ask_size_l4": 14, "ask_size_l5": 15, "ask_size_l7": 17,
}
SUBSET_C = {
    **SUBSET_B, "ask_size_l2": 12, "ask_size_l6": 16, "ask_size_l8": 18,
    "ask_size_l9": 19, "bid_size_l0": 30, "avg_fill_ratio": 80,
}
SUBSETS = {"A_5feat": SUBSET_A, "B_8feat": SUBSET_B, "C_14feat": SUBSET_C}

ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
MIN_TRAIN_DAYS = 30
COST_BPS = 1.4


def log(msg: str) -> None:
    print(f"[ridge_wf] {msg}", flush=True)


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
    rho, p = spearman_ic(x, y)
    if rho == 0.0 and p == 1.0:
        return float("nan")
    return float(rho)


# =============================================================================
# Standardized Ridge
# =============================================================================

def _ridge_fit_predict(X_train, y_train, X_test, alpha=1.0):
    """Standardized Ridge regression. Returns (predictions, beta_std, mu, sigma).

    Features are standardized using training statistics before fitting.
    Coefficients (beta_std) are in standardized space — directly comparable.
    """
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
# Analysis 1: Walk-Forward Ridge
# =============================================================================

def walk_forward(day_data, alpha, feature_names):
    """Daily expanding-window walk-forward Ridge."""
    n_days = len(day_data)
    folds = []
    all_betas = []

    for k in range(MIN_TRAIN_DAYS, n_days):
        # Pool training data
        X_train = np.vstack([d["features"] for d in day_data[:k]])
        y_train = np.concatenate([d["labels"] for d in day_data[:k]])
        X_test = day_data[k]["features"]
        y_test = day_data[k]["labels"]

        # Filter non-finite
        tr_valid = np.all(np.isfinite(X_train), axis=1) & np.isfinite(y_train)
        te_valid = np.all(np.isfinite(X_test), axis=1) & np.isfinite(y_test)
        if tr_valid.sum() < 50 or te_valid.sum() < 5:
            continue

        X_tr, y_tr = X_train[tr_valid], y_train[tr_valid]
        X_te, y_te = X_test[te_valid], y_test[te_valid]

        preds, beta, _, _ = _ridge_fit_predict(X_tr, y_tr, X_te, alpha)

        ic = _safe_ic(preds, y_te)
        r2 = r_squared(y_te, preds)
        da = directional_accuracy(y_te, preds)

        # Demeaned DA (remove daily drift)
        daily_mean = np.mean(y_te)
        demeaned_y = y_te - daily_mean
        demeaned_preds = preds - np.mean(preds)
        demeaned_da = directional_accuracy(demeaned_y, demeaned_preds)

        # Profitable accuracy
        prof_da = profitable_accuracy(y_te, preds, breakeven_bps=COST_BPS)

        folds.append({
            "fold": k,
            "date": day_data[k]["date"],
            "n_train": int(tr_valid.sum()),
            "n_test": int(te_valid.sum()),
            "ic": float(ic) if np.isfinite(ic) else 0.0,
            "r_squared": float(r2),
            "da": float(da),
            "demeaned_da": float(demeaned_da),
            "profitable_da": float(prof_da),
        })
        all_betas.append(beta)

    if not folds:
        return {"error": "no_folds"}, []

    fold_ics = np.array([f["ic"] for f in folds])
    fold_das = np.array([f["da"] for f in folds])
    fold_dm_das = np.array([f["demeaned_da"] for f in folds])
    mean_ic = float(np.mean(fold_ics))
    std_ic = float(np.std(fold_ics))
    stability = mean_ic / max(std_ic, EPS)

    # Weekly block stability (group into 5-day blocks)
    n_blocks = len(fold_ics) // 5
    if n_blocks >= 5:
        block_ics = [float(np.mean(fold_ics[i * 5:(i + 1) * 5])) for i in range(n_blocks)]
        block_stability = float(np.mean(block_ics) / max(np.std(block_ics), EPS))
    else:
        block_stability = 0.0

    summary = {
        "n_folds": len(folds),
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "stability": float(stability),
        "block_stability": block_stability,
        "mean_da": float(np.mean(fold_das)),
        "mean_demeaned_da": float(np.mean(fold_dm_das)),
        "mean_r2": float(np.mean([f["r_squared"] for f in folds])),
        "mean_profitable_da": float(np.mean([f["profitable_da"] for f in folds])),
        "regime_shifts": int(np.sum(fold_ics < 0)),
    }
    return summary, all_betas


# =============================================================================
# Analysis 2: Val/Test OOS
# =============================================================================

def evaluate_oos(train_data, split_data, alpha, feature_names):
    """Fit on all train, predict on OOS split."""
    X_train = np.vstack([d["features"] for d in train_data])
    y_train = np.concatenate([d["labels"] for d in train_data])
    tr_valid = np.all(np.isfinite(X_train), axis=1) & np.isfinite(y_train)
    X_tr, y_tr = X_train[tr_valid], y_train[tr_valid]

    results = {}
    for split_name, s_data in split_data.items():
        X_test = np.vstack([d["features"] for d in s_data])
        y_test = np.concatenate([d["labels"] for d in s_data])
        te_valid = np.all(np.isfinite(X_test), axis=1) & np.isfinite(y_test)
        if te_valid.sum() < 10:
            continue
        X_te, y_te = X_test[te_valid], y_test[te_valid]

        preds, beta, mu, sigma = _ridge_fit_predict(X_tr, y_tr, X_te, alpha)
        ic = _safe_ic(preds, y_te)
        n_eff = len(s_data) * (s_data[0]["n"] / 60)  # approx effective n
        ic_se = 1.0 / max(np.sqrt(n_eff - 3), 1) if n_eff > 3 else 1.0
        t_stat = ic / max(ic_se, EPS) if np.isfinite(ic) else 0.0

        # Demeaned DA
        dm_y = y_te - np.mean(y_te)
        dm_p = preds - np.mean(preds)
        dm_da = directional_accuracy(dm_y, dm_p)

        # --- Per-day analysis (dual-offset for correctness) ---
        # preds/y_te are in compressed valid-only space; te_valid is in original space.
        # Must track two offsets to avoid the z-score script's latent indexing bug.
        per_day_details = []
        up_day_ics, down_day_ics = [], []
        orig_offset = 0   # into te_valid (uncompressed, all samples)
        comp_offset = 0   # into preds/y_te (compressed, valid-only)
        for d in s_data:
            n_d = d["n"]
            d_valid_slice = te_valid[orig_offset:orig_offset + n_d]
            n_valid_d = int(d_valid_slice.sum())
            if n_valid_d >= 10:
                d_preds = preds[comp_offset:comp_offset + n_valid_d]
                d_actuals = y_te[comp_offset:comp_offset + n_valid_d]
                d_ic = _safe_ic(d_preds, d_actuals)
                if np.isfinite(d_ic):
                    d_da = directional_accuracy(d_actuals, d_preds)
                    d_prof_da = profitable_accuracy(
                        d_actuals, d_preds, breakeven_bps=COST_BPS
                    )
                    long_mask = d_preds > 0
                    short_mask = d_preds < 0
                    d_long_ret = (
                        float(np.mean(d_actuals[long_mask]))
                        if long_mask.sum() > 0 else 0.0
                    )
                    d_short_ret = (
                        float(np.mean(d_actuals[short_mask]))
                        if short_mask.sum() > 0 else 0.0
                    )
                    per_day_details.append({
                        "date": d["date"],
                        "ic": float(d_ic),
                        "n_valid": n_valid_d,
                        "return_std_bps": float(np.std(d_actuals)),
                        "da": float(d_da),
                        "profitable_da": float(d_prof_da),
                        "mean_long_return_bps": d_long_ret,
                        "mean_short_return_bps": d_short_ret,
                    })
                    # UP/DOWN day classification (valid-only actuals, not raw labels)
                    if np.mean(d_actuals) > 0:
                        up_day_ics.append(float(d_ic))
                    else:
                        down_day_ics.append(float(d_ic))
            comp_offset += n_valid_d
            orig_offset += n_d

        # Invariant asserts (permanent, zero-cost)
        assert comp_offset == int(te_valid.sum()), \
            f"Dual-offset drift: comp={comp_offset} != valid={int(te_valid.sum())}"
        assert orig_offset == len(te_valid), \
            f"Orig-offset drift: orig={orig_offset} != total={len(te_valid)}"

        # Per-day IC distribution statistics
        per_day_ics_arr = (
            np.array([x["ic"] for x in per_day_details])
            if per_day_details else np.array([])
        )
        n_days = len(per_day_ics_arr)
        if n_days >= 2:
            pd_mean = float(np.mean(per_day_ics_arr))
            pd_std = float(np.std(per_day_ics_arr))  # ddof=0, consistent with walk_forward
            pd_t = pd_mean / max(pd_std / np.sqrt(n_days), EPS)
        else:
            pd_mean = float(per_day_ics_arr[0]) if n_days == 1 else 0.0
            pd_std = 0.0
            pd_t = 0.0

        # Grinold's Law: E[r] = IC * sigma_r  (Grinold 1989, lower bound)
        per_day_return_stds = (
            np.array([x["return_std_bps"] for x in per_day_details])
            if per_day_details else np.array([0.0])
        )
        mean_return_std = float(np.mean(per_day_return_stds)) if n_days > 0 else 0.0
        grinold_expected_return_bps = pd_mean * mean_return_std
        grinold_net_of_cost_bps = grinold_expected_return_bps - COST_BPS

        # Calibration: actual = slope * pred + intercept
        if len(preds) > 10:
            cal_slope, cal_intercept = np.polyfit(preds, y_te, 1)
        else:
            cal_slope, cal_intercept = 0.0, 0.0

        # Aggregated per-day trading metrics
        per_day_das = np.array([x["da"] for x in per_day_details]) if per_day_details else np.array([])
        per_day_prof_das = np.array([x["profitable_da"] for x in per_day_details]) if per_day_details else np.array([])
        per_day_long_rets = np.array([x["mean_long_return_bps"] for x in per_day_details]) if per_day_details else np.array([])
        per_day_short_rets = np.array([x["mean_short_return_bps"] for x in per_day_details]) if per_day_details else np.array([])

        results[split_name] = {
            "n": int(te_valid.sum()),
            "n_eff": float(n_eff),
            "n_days": n_days,
            # Pooled IC (renamed from ic/ic_se/ic_t_stat for clarity)
            "ic_pooled": float(ic) if np.isfinite(ic) else 0.0,
            "ic_pooled_se": float(ic_se),
            "ic_pooled_t_stat": float(t_stat),
            # Per-day IC distribution (primary decision metric)
            "ic_per_day_mean": pd_mean,
            "ic_per_day_std": pd_std,
            "ic_per_day_median": float(np.median(per_day_ics_arr)) if n_days > 0 else 0.0,
            "ic_per_day_q25": float(np.percentile(per_day_ics_arr, 25)) if n_days > 0 else 0.0,
            "ic_per_day_q75": float(np.percentile(per_day_ics_arr, 75)) if n_days > 0 else 0.0,
            "ic_per_day_min": float(np.min(per_day_ics_arr)) if n_days > 0 else 0.0,
            "ic_per_day_max": float(np.max(per_day_ics_arr)) if n_days > 0 else 0.0,
            "ic_per_day_frac_positive": float(np.mean(per_day_ics_arr > 0)) if n_days > 0 else 0.0,
            "ic_per_day_t_stat": float(pd_t),
            # Per-day trading metrics
            "da_per_day_mean": float(np.mean(per_day_das)) if n_days > 0 else 0.0,
            "profitable_da_per_day_mean": float(np.mean(per_day_prof_das)) if n_days > 0 else 0.0,
            "mean_long_return_bps": float(np.mean(per_day_long_rets)) if n_days > 0 else 0.0,
            "mean_short_return_bps": float(np.mean(per_day_short_rets)) if n_days > 0 else 0.0,
            "long_short_spread_bps": (
                float(np.mean(per_day_long_rets) - np.mean(per_day_short_rets))
                if n_days > 0 else 0.0
            ),
            # Grinold's Law
            "grinold_expected_return_bps": float(grinold_expected_return_bps),
            "grinold_net_of_cost_bps": float(grinold_net_of_cost_bps),
            "mean_return_std_bps": float(mean_return_std),
            # Calibration
            "calibration_slope": float(cal_slope),
            "calibration_intercept": float(cal_intercept),
            # Standard metrics (unchanged values)
            "r_squared": float(r_squared(y_te, preds)),
            "da": float(directional_accuracy(y_te, preds)),
            "demeaned_da": float(dm_da),
            "profitable_da": float(profitable_accuracy(y_te, preds, breakeven_bps=COST_BPS)),
            "mean_prediction": float(np.mean(preds)),
            "mean_actual": float(np.mean(y_te)),
            "coefficients": {
                name: float(beta[i + 1])
                for i, name in enumerate(feature_names)
            },
            "intercept": float(beta[0]),
            # Per-day detail list (for post-hoc analysis)
            "per_day_details": per_day_details,
        }

        # Conditional IC (positive vs negative returns)
        for label, mask in [("pos_return", y_te > 0), ("neg_return", y_te < 0)]:
            if mask.sum() >= 20:
                cond_ic = _safe_ic(preds[mask], y_te[mask])
                results[split_name][f"ic_{label}"] = float(cond_ic) if np.isfinite(cond_ic) else 0.0
                results[split_name][f"n_{label}"] = int(mask.sum())

        # UP-day vs DOWN-day IC (fixed: uses valid-only actuals, correct dual-offset)
        if len(up_day_ics) >= 3:
            results[split_name]["ic_up_day_mean"] = float(np.mean(up_day_ics))
            results[split_name]["ic_up_day_n_days"] = len(up_day_ics)
        if len(down_day_ics) >= 3:
            results[split_name]["ic_down_day_mean"] = float(np.mean(down_day_ics))
            results[split_name]["ic_down_day_n_days"] = len(down_day_ics)

    return results


# =============================================================================
# Analysis 5: IC Trend
# =============================================================================

def compute_ic_trend(fold_ics):
    """Linear regression of fold ICs on fold number."""
    n = len(fold_ics)
    if n < 10:
        return {"slope": 0.0, "p_value": 1.0, "interpretation": "insufficient_data"}
    x = np.arange(n, dtype=np.float64)
    slope, intercept, r, p, se = scipy_stats.linregress(x, fold_ics)
    if p < 0.05 and slope < 0:
        interp = "degrading"
    elif p < 0.05 and slope > 0:
        interp = "improving"
    else:
        interp = "stable"
    return {"slope": float(slope), "p_value": float(p), "r_squared": float(r ** 2),
            "interpretation": interp}


# =============================================================================
# Analysis 6: Coefficient Stability
# =============================================================================

def compute_coeff_stability(all_betas, feature_names):
    """Track coefficient stability across folds."""
    if not all_betas:
        return {}
    beta_matrix = np.array(all_betas)  # [n_folds, d+1] (intercept + features)
    result = {}
    for i, name in enumerate(feature_names):
        col = beta_matrix[:, i + 1]  # skip intercept
        mean_b = float(np.mean(col))
        std_b = float(np.std(col))
        cv = std_b / max(abs(mean_b), EPS)
        sign_flips = int(np.sum(np.diff(np.sign(col)) != 0))
        result[name] = {
            "mean_beta": mean_b, "std_beta": std_b, "cv": float(cv),
            "sign_flips": sign_flips, "n_folds": len(col),
            "stable": cv < 2.0 and sign_flips < len(col) * 0.3,
        }
    return result


# =============================================================================
# Analysis 8: Walk-Forward Cumulative P&L
# =============================================================================

def compute_walkforward_pnl(folds_data, day_data, alpha, feature_indices):
    """Compute cumulative P&L from walk-forward predictions."""
    monthly_pnl = defaultdict(lambda: {"pnl": 0.0, "trades": 0})
    total_pnl = 0.0
    trade_returns = []

    for k in range(MIN_TRAIN_DAYS, len(day_data)):
        X_train = np.vstack([d["features"] for d in day_data[:k]])
        y_train = np.concatenate([d["labels"] for d in day_data[:k]])
        X_test = day_data[k]["features"]
        y_test = day_data[k]["labels"]

        tr_valid = np.all(np.isfinite(X_train), axis=1) & np.isfinite(y_train)
        te_valid = np.all(np.isfinite(X_test), axis=1) & np.isfinite(y_test)
        if tr_valid.sum() < 50 or te_valid.sum() < 5:
            continue

        preds, _, _, _ = _ridge_fit_predict(
            X_train[tr_valid], y_train[tr_valid], X_test[te_valid], alpha
        )
        actuals = y_test[te_valid]

        # Non-overlapping trades (60-bin spacing)
        last_entry = -60
        for idx in range(len(preds)):
            if idx - last_entry < 60:
                continue
            position = 1.0 if preds[idx] > 0 else -1.0
            trade_ret = position * actuals[idx] - COST_BPS
            trade_returns.append(trade_ret)
            total_pnl += trade_ret
            month = day_data[k]["date"][:6]
            month_key = f"{month[:4]}-{month[4:]}"
            monthly_pnl[month_key]["pnl"] += trade_ret
            monthly_pnl[month_key]["trades"] += 1
            last_entry = idx

    arr = np.array(trade_returns) if trade_returns else np.array([0.0])
    return {
        "total_pnl": float(total_pnl),
        "n_trades": len(trade_returns),
        "mean_trade": float(np.mean(arr)),
        "std_trade": float(np.std(arr)),
        "win_rate": float(np.mean(arr > 0)),
        "monthly": dict(monthly_pnl),
        "always_long_comparison": float(np.mean(np.abs(arr))),  # approximate
    }


# =============================================================================
# Markdown Report
# =============================================================================

def build_report(results):
    lines = []
    w = lines.append

    w("# MBO Ridge Walk-Forward Gate Test Report")
    w(f"\n> **Date**: {results.get('analysis_date', 'N/A')}")
    w(f"> **Export**: {results.get('export_dir', 'N/A')}")
    w(f"> **Train**: {results.get('n_train_days', 0)} days | "
      f"**Val**: {results.get('n_val_days', 0)} | "
      f"**Test**: {results.get('n_test_days', 0)}")

    # Decision Gates
    gates = results.get("decision_gates", {})
    verdict = "GO" if all(g.get("verdict") == "PASS" for g in gates.values()) else "NO-GO"
    w(f"\n> **Verdict**: **{verdict}**\n")

    w("## Decision Gates\n")
    w("| Gate | Metric | Value | Threshold | Verdict |")
    w("|------|--------|-------|-----------|---------|")
    for gname, g in gates.items():
        val = g.get("value")
        val_str = f"{val:.4f}" if isinstance(val, (int, float)) and val is not None else "N/A"
        w(f"| {gname} | {g.get('metric', '')} | {val_str} | {g.get('threshold', '')} | "
          f"**{g.get('verdict', 'N/A')}** |")

    # Walk-forward results
    wf = results.get("walk_forward", {})
    w("\n## Analysis 1: Walk-Forward Ridge\n")
    w("| Subset | Alpha | Folds | Mean IC | Stability | Block Stab | DA (demeaned) | Regime Shifts |")
    w("|--------|-------|-------|---------|-----------|------------|---------------|---------------|")
    for key in sorted(wf.keys()):
        s = wf[key]
        if isinstance(s, dict) and "n_folds" in s:
            w(f"| {key} | | {s['n_folds']} | {s['mean_ic']:.4f} | "
              f"{s['stability']:.3f} | {s.get('block_stability', 0):.3f} | "
              f"{s['mean_demeaned_da']:.4f} | {s['regime_shifts']} |")

    best = results.get("best_config", "N/A")
    w(f"\n**Best configuration**: {best}")

    # OOS
    oos = results.get("val_test", {})
    w("\n## Analysis 2: Val/Test OOS\n")
    for split, s in oos.items():
        if isinstance(s, dict) and "ic_per_day_mean" in s:
            w(f"- **{split}**: n={s['n']} ({s.get('n_days', 0)} days)")
            w(f"  - Per-day IC: **{s['ic_per_day_mean']:.4f}** "
              f"(t={s.get('ic_per_day_t_stat', 0):.2f}, "
              f"std={s.get('ic_per_day_std', 0):.4f}, "
              f"median={s.get('ic_per_day_median', 0):.4f}, "
              f"frac+={s.get('ic_per_day_frac_positive', 0):.0%})")
            w(f"  - Per-day DA: {s.get('da_per_day_mean', 0):.4f}, "
              f"profitable DA: {s.get('profitable_da_per_day_mean', 0):.4f}")
            w(f"  - Long return: {s.get('mean_long_return_bps', 0):+.2f} bps, "
              f"Short return: {s.get('mean_short_return_bps', 0):+.2f} bps, "
              f"Spread: {s.get('long_short_spread_bps', 0):+.2f} bps")
            w(f"  - Grinold E[r]: {s.get('grinold_expected_return_bps', 0):.2f} bps "
              f"(net of {COST_BPS} bps: "
              f"{s.get('grinold_net_of_cost_bps', 0):+.2f} bps)")
            w(f"  - Calibration: slope={s.get('calibration_slope', 0):.3f}")
            w(f"  - Pooled IC: {s.get('ic_pooled', 0):.4f} "
              f"(t={s.get('ic_pooled_t_stat', 0):.2f})")
            w(f"  - UP-day IC: {s.get('ic_up_day_mean', 'N/A')}, "
              f"DOWN-day IC: {s.get('ic_down_day_mean', 'N/A')}")

    # Coefficients
    w("\n## Analysis 6: Coefficient Stability\n")
    cs = results.get("coefficient_stability", {})
    if cs:
        w("| Feature | Mean β | Std β | CV | Sign Flips | Stable? |")
        w("|---------|--------|-------|-----|------------|---------|")
        for name, info in cs.items():
            w(f"| {name} | {info['mean_beta']:+.4f} | {info['std_beta']:.4f} | "
              f"{info['cv']:.2f} | {info['sign_flips']} | "
              f"{'YES' if info['stable'] else 'NO'} |")

    # IC trend
    trend = results.get("ic_trend", {})
    w(f"\n## Analysis 5: IC Trend\n")
    w(f"- Slope: {trend.get('slope', 0):.6f}")
    w(f"- p-value: {trend.get('p_value', 1):.4f}")
    w(f"- Interpretation: {trend.get('interpretation', 'N/A')}")

    # P&L
    pnl = results.get("walkforward_pnl", {})
    w(f"\n## Analysis 8: Walk-Forward P&L\n")
    w(f"- Total P&L: {pnl.get('total_pnl', 0):+.1f} bps over {pnl.get('n_trades', 0)} trades")
    w(f"- Mean trade: {pnl.get('mean_trade', 0):+.2f} bps, Win rate: {pnl.get('win_rate', 0):.3f}")

    # E12 comparison
    w("\n## Analysis 9: E12 Comparison\n")
    w("| Metric | Off-Exchange (E12) | MBO (E13) |")
    w("|--------|--------------------|-----------|")
    best_wf = results.get("walk_forward", {}).get(results.get("best_config", ""), {})
    w(f"| Walk-forward stability | 0.33 | {best_wf.get('stability', 0):.3f} |")
    w(f"| Walk-forward mean IC | 0.075 | {best_wf.get('mean_ic', 0):.4f} |")
    w(f"| Walk-forward mean DA | 0.511 | {best_wf.get('mean_da', 0):.4f} |")
    val_oos = oos.get("val", {})
    w(f"| Val OOS per-day IC | -0.048 | {val_oos.get('ic_per_day_mean', 0):.4f} |")
    w(f"| Val OOS DA | 0.484 | {val_oos.get('da', 0):.4f} |")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MBO Ridge Walk-Forward Gate Test")
    parser.add_argument("--export-dir", default=EXPORT_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    np.random.seed(SEED)
    start = time.time()
    log("MBO Ridge Walk-Forward Gate Test starting...")

    script_dir = Path(__file__).resolve().parent.parent
    export_path = (script_dir / args.export_dir).resolve()
    output_path = (script_dir / args.output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    train_loader = ExportLoader(str(export_path), "train")
    val_loader = ExportLoader(str(export_path), "val")
    test_loader = ExportLoader(str(export_path), "test")

    train_dates = train_loader.list_dates()
    val_dates = val_loader.list_dates()
    test_dates = test_loader.list_dates()

    log(f"Train: {len(train_dates)}, Val: {len(val_dates)}, Test: {len(test_dates)}")

    # Use largest subset indices for data loading (subset selection happens later)
    all_indices = sorted(set(SUBSET_C.values()))
    all_names = [n for n, _ in sorted(SUBSET_C.items(), key=lambda x: x[1])]

    log("Loading train data...")
    train_data_full = load_all_data(train_loader, train_dates, all_indices)
    log("Loading val data...")
    val_data_full = load_all_data(val_loader, val_dates, all_indices)
    log("Loading test data...")
    test_data_full = load_all_data(test_loader, test_dates, all_indices)

    # === Analysis 1: Walk-Forward across all subsets × alphas ===
    log("Analysis 1: Walk-forward Ridge...")
    wf_results = {}
    best_config = None
    best_stability = -999

    for subset_name, subset_dict in SUBSETS.items():
        subset_indices = sorted(subset_dict.values())
        # Map from all_indices to subset columns
        col_map = [all_indices.index(i) for i in subset_indices]
        subset_names = [n for n, _ in sorted(subset_dict.items(), key=lambda x: x[1])]

        # Create subset view of data
        subset_data = [
            {"date": d["date"], "features": d["features"][:, col_map],
             "labels": d["labels"], "n": d["n"]}
            for d in train_data_full
        ]

        for alpha in ALPHAS:
            key = f"{subset_name}_alpha_{alpha}"
            log(f"  {key}...")
            summary, betas = walk_forward(subset_data, alpha, subset_names)
            wf_results[key] = summary

            if isinstance(summary, dict) and "stability" in summary:
                eff_stability = max(summary["stability"], summary.get("block_stability", 0))
                if eff_stability > best_stability:
                    best_stability = eff_stability
                    best_config = key

    log(f"  Best: {best_config} (stability={best_stability:.3f})")

    # Extract best subset and alpha from best_config
    parts = best_config.split("_alpha_")
    best_subset_name = parts[0]
    best_alpha = float(parts[1])
    best_subset = SUBSETS[best_subset_name]
    best_indices = sorted(best_subset.values())
    best_col_map = [all_indices.index(i) for i in best_indices]
    best_feature_names = [n for n, _ in sorted(best_subset.items(), key=lambda x: x[1])]

    best_train_data = [
        {"date": d["date"], "features": d["features"][:, best_col_map],
         "labels": d["labels"], "n": d["n"]}
        for d in train_data_full
    ]

    # === Analysis 2: Val/Test OOS ===
    log("Analysis 2: Val/Test OOS...")
    best_val = [{"date": d["date"], "features": d["features"][:, best_col_map],
                  "labels": d["labels"], "n": d["n"]} for d in val_data_full]
    best_test = [{"date": d["date"], "features": d["features"][:, best_col_map],
                   "labels": d["labels"], "n": d["n"]} for d in test_data_full]
    oos_results = evaluate_oos(best_train_data, {"val": best_val, "test": best_test},
                                best_alpha, best_feature_names)

    # === Analysis 4: Single-Feature Baseline ===
    log("Analysis 4: Single-feature baseline (spread_bps)...")
    spread_col = [all_indices.index(42)]
    sf_data = [{"date": d["date"], "features": d["features"][:, spread_col],
                 "labels": d["labels"], "n": d["n"]} for d in train_data_full]
    sf_summary, _ = walk_forward(sf_data, best_alpha, ["spread_bps"])

    # === Analysis 5: IC Trend (from best config walk-forward) ===
    log("Analysis 5: IC trend...")
    best_wf_summary, best_betas = walk_forward(best_train_data, best_alpha, best_feature_names)
    fold_ics = []
    for k in range(MIN_TRAIN_DAYS, len(best_train_data)):
        X_train = np.vstack([d["features"] for d in best_train_data[:k]])
        y_train = np.concatenate([d["labels"] for d in best_train_data[:k]])
        X_test = best_train_data[k]["features"]
        y_test = best_train_data[k]["labels"]
        tr_v = np.all(np.isfinite(X_train), axis=1) & np.isfinite(y_train)
        te_v = np.all(np.isfinite(X_test), axis=1) & np.isfinite(y_test)
        if tr_v.sum() >= 50 and te_v.sum() >= 5:
            preds, _, _, _ = _ridge_fit_predict(X_train[tr_v], y_train[tr_v], X_test[te_v], best_alpha)
            ic = _safe_ic(preds, y_test[te_v])
            fold_ics.append(float(ic) if np.isfinite(ic) else 0.0)
    ic_trend = compute_ic_trend(np.array(fold_ics))

    # === Analysis 6: Coefficient Stability ===
    log("Analysis 6: Coefficient stability...")
    coeff_stability = compute_coeff_stability(best_betas, best_feature_names)

    # === Analysis 8: Walk-Forward P&L ===
    log("Analysis 8: Walk-forward P&L...")
    pnl = compute_walkforward_pnl(None, best_train_data, best_alpha, best_indices)

    # === Decision Gates ===
    best_wf = wf_results.get(best_config, {})
    val_oos = oos_results.get("val", {})
    test_oos = oos_results.get("test", {})

    eff_stab = max(best_wf.get("stability", 0), best_wf.get("block_stability", 0))
    # Per-day IC for G2 gate (replaces pooled IC — see E13 Phase 5 lesson)
    oos_ic = max(val_oos.get("ic_per_day_mean", 0), test_oos.get("ic_per_day_mean", 0))
    oos_t = max(abs(val_oos.get("ic_per_day_t_stat", 0)), abs(test_oos.get("ic_per_day_t_stat", 0)))
    oos_ic_pooled = max(val_oos.get("ic_pooled", 0), test_oos.get("ic_pooled", 0))
    oos_prof = max(val_oos.get("profitable_da", 0), test_oos.get("profitable_da", 0))

    # Alpha robustness: count alphas with stability > 2.0 for best subset
    alpha_robust_count = 0
    for alpha in ALPHAS:
        key = f"{best_subset_name}_alpha_{alpha}"
        s = wf_results.get(key, {})
        if isinstance(s, dict):
            eff = max(s.get("stability", 0), s.get("block_stability", 0))
            if eff > 2.0:
                alpha_robust_count += 1

    gates = {
        "G1_stability": {
            "metric": "walk_forward_stability (max of per-fold and block)",
            "value": float(eff_stab),
            "threshold": "> 2.0",
            "verdict": "PASS" if eff_stab > 2.0 else "FAIL",
        },
        "G2_oos_ic": {
            "metric": "OOS per-day mean IC AND t > 2.0",
            "value": float(oos_ic),
            "t_stat": float(oos_t),
            "pooled_ic": float(oos_ic_pooled),
            "threshold": "> 0.05 AND t > 2.0",
            "verdict": "PASS" if oos_ic > 0.05 and oos_t > 2.0 else "FAIL",
        },
        "G3_demeaned_da": {
            "metric": "walk_forward_demeaned_DA",
            "value": float(best_wf.get("mean_demeaned_da", 0)),
            "threshold": "> 0.52",
            "verdict": "PASS" if best_wf.get("mean_demeaned_da", 0) > 0.52 else "FAIL",
        },
        "G4_alpha_robust": {
            "metric": "n_alphas_with_stability > 2.0",
            "value": alpha_robust_count,
            "threshold": ">= 2",
            "verdict": "PASS" if alpha_robust_count >= 2 else "FAIL",
        },
        "G5_profitable": {
            "metric": "OOS profitable_accuracy at 1.4 bps",
            "value": float(oos_prof),
            "threshold": "> 0.50",
            "verdict": "PASS" if oos_prof > 0.50 else "FAIL",
        },
    }

    # === Assemble Results ===
    results = {
        "schema": "ridge_walkforward_mbo_v2",
        "export_dir": str(export_path),
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "n_train_days": len(train_dates),
        "n_val_days": len(val_dates),
        "n_test_days": len(test_dates),
        "best_config": best_config,
        "best_alpha": best_alpha,
        "best_subset": best_subset_name,
        "walk_forward": wf_results,
        "val_test": oos_results,
        "single_feature_baseline": sf_summary,
        "ic_trend": ic_trend,
        "coefficient_stability": coeff_stability,
        "walkforward_pnl": pnl,
        "decision_gates": gates,
    }

    # Write outputs
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
    all_pass = all(g["verdict"] == "PASS" for g in gates.values())
    log(f"Verdict: {'GO' if all_pass else 'NO-GO'}")
    for gname, g in gates.items():
        log(f"  {gname}: {g['verdict']} (value={g['value']})")


if __name__ == "__main__":
    main()
