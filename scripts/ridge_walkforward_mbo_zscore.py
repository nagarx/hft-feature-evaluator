#!/usr/bin/env python3
"""
MBO Ridge Walk-Forward with PER-DAY Z-SCORED FEATURES.

Addresses E13 Phase 3 root cause: signal is within-day only. Between-day
feature level shifts (spread mean 0.84→0.58 bps) cause fixed-coefficient
Ridge to fail OOS. Per-day z-scoring removes between-day shift so the model
sees only within-day relative position.

Changes from ridge_walkforward_mbo.py:
  1. Features z-scored within each day at load time
  2. OOS adds per-day mean IC (consistent with walk-forward metric)
  3. Output dir changed to ridge_walkforward_mbo_zscore

Usage:
    python scripts/ridge_walkforward_mbo_zscore.py
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
OUTPUT_DIR = "outputs/ridge_walkforward_mbo_zscore"
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
    """Load all days into per-day feature + label arrays.

    CRITICAL CHANGE: Features are z-scored WITHIN each day before storage.
    This removes between-day distribution shift (e.g., spread_bps mean drifting
    from 0.84 to 0.58 over 8 months). The model sees only within-day relative
    position, not absolute levels.

    Formula: feat_zscore[i] = (feat[i] - day_mean) / day_std
    """
    day_data = []
    for d in dates:
        b = loader.load_day(d)
        feat_raw = np.asarray(b.sequences[:, -1, :][:, feature_indices], dtype=np.float64)
        lab = np.asarray(b.labels[:, TARGET_HORIZON_IDX], dtype=np.float64)

        # Per-day z-score: remove between-day level shift
        day_mu = np.mean(feat_raw, axis=0)
        day_sigma = np.std(feat_raw, axis=0)
        day_sigma[day_sigma < 1e-10] = 1.0  # guard zero-variance
        feat_z = (feat_raw - day_mu) / day_sigma

        day_data.append({
            "date": d, "features": feat_z, "labels": lab, "n": feat_z.shape[0],
            "raw_mu": day_mu, "raw_sigma": day_sigma,
        })
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

        # Per-day mean IC (CONSISTENT metric with walk-forward)
        per_day_ics = []
        offset = 0
        for d in s_data:
            n_d = d["n"]
            d_valid_slice = te_valid[offset:offset + n_d]
            if d_valid_slice.sum() >= 10:
                d_preds = preds[offset:offset + n_d][d_valid_slice[:n_d]]
                d_actuals = y_te[offset:offset + n_d][d_valid_slice[:n_d]]
                if len(d_preds) >= 10:
                    d_ic = _safe_ic(d_preds, d_actuals)
                    if np.isfinite(d_ic):
                        per_day_ics.append(d_ic)
            offset += n_d
        per_day_mean_ic = float(np.mean(per_day_ics)) if per_day_ics else 0.0
        per_day_std_ic = float(np.std(per_day_ics)) if per_day_ics else 1.0
        per_day_stability = per_day_mean_ic / max(per_day_std_ic, EPS)
        per_day_t = per_day_mean_ic / max(per_day_std_ic / np.sqrt(max(len(per_day_ics), 1)), EPS)

        results[split_name] = {
            "n": int(te_valid.sum()),
            "n_eff": float(n_eff),
            "n_days": len(per_day_ics),
            # Pooled IC (across all days combined)
            "ic_pooled": float(ic) if np.isfinite(ic) else 0.0,
            "ic_pooled_se": float(ic_se),
            "ic_pooled_t_stat": float(t_stat),
            # Per-day mean IC (CONSISTENT with walk-forward metric)
            "ic_per_day_mean": per_day_mean_ic,
            "ic_per_day_std": per_day_std_ic,
            "ic_per_day_stability": float(per_day_stability),
            "ic_per_day_t_stat": float(per_day_t),
            # Other metrics
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
        }

        # Conditional IC (positive vs negative returns)
        for label, mask in [("pos_return", y_te > 0), ("neg_return", y_te < 0)]:
            if mask.sum() >= 20:
                cond_ic = _safe_ic(preds[mask], y_te[mask])
                results[split_name][f"ic_{label}"] = float(cond_ic) if np.isfinite(cond_ic) else 0.0
                results[split_name][f"n_{label}"] = int(mask.sum())

        # UP-day vs DOWN-day IC
        for day_label, day_mask_fn in [
            ("up_day", lambda d: np.mean(d["labels"]) > 0),
            ("down_day", lambda d: np.mean(d["labels"]) <= 0),
        ]:
            day_preds_list, day_actuals_list = [], []
            offset = 0
            for d in s_data:
                n_d = d["n"]
                d_valid = te_valid[offset:offset + n_d]
                if d_valid.sum() > 0 and day_mask_fn(d):
                    day_preds_list.append(preds[offset:offset + n_d][d_valid[:te_valid[offset:offset+n_d].shape[0]]])
                    day_actuals_list.append(y_te[offset:offset + n_d][d_valid[:te_valid[offset:offset+n_d].shape[0]]])
                offset += n_d

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
            w(f"  - Per-day mean IC: **{s['ic_per_day_mean']:.4f}** "
              f"(t={s.get('ic_per_day_t_stat', 0):.2f}, stability={s.get('ic_per_day_stability', 0):.3f})")
            w(f"  - Pooled IC: {s.get('ic_pooled', 0):.4f} "
              f"(t={s.get('ic_pooled_t_stat', 0):.2f})")
            w(f"  - DA={s['da']:.4f}, demeaned_DA={s['demeaned_da']:.4f}, "
              f"profitable_DA={s['profitable_da']:.4f}, R²={s['r_squared']:.4f}")

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
    w(f"| Val OOS IC | -0.048 | {val_oos.get('ic', 0):.4f} |")
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
    # Use PER-DAY mean IC for OOS (consistent with walk-forward metric)
    oos_ic = max(val_oos.get("ic_per_day_mean", 0), test_oos.get("ic_per_day_mean", 0))
    oos_t = max(abs(val_oos.get("ic_per_day_t_stat", 0)), abs(test_oos.get("ic_per_day_t_stat", 0)))
    oos_prof = max(val_oos.get("profitable_da", 0), test_oos.get("profitable_da", 0))
    # Also track pooled IC for comparison
    oos_ic_pooled = max(val_oos.get("ic_pooled", 0), test_oos.get("ic_pooled", 0))

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
            "metric": "OOS per-day mean IC (consistent with walk-forward)",
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
        "schema": "ridge_walkforward_mbo_v1",
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
