#!/usr/bin/env python3
"""
Pre-training feature analysis for off-exchange KEEP+ features.

Comprehensive 10-domain analysis matching E5 quality, purpose-built for
GradBoost model preparation on point-return labels.

Usage:
    python scripts/pre_training_analysis.py
    python scripts/pre_training_analysis.py --quick          # 20 train days
    python scripts/pre_training_analysis.py --export-dir /path/to/export

Requires: hft-contracts, hft-metrics, hft-evaluator (data loader), numpy, scipy.
Optional: xgboost (for GradBoost baselines).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

# hft-evaluator data loader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from hft_evaluator.data.loader import ExportLoader

# hft-metrics functions (grouped by domain usage)
from hft_metrics import (
    # IC and cost (Domains 1, 3, 7, 8, 9)
    spearman_ic, ic_ir, windowed_ic,
    expected_return_bps, breakeven_ic, cost_adjusted_ic,
    # Nonlinear signal (Domain 3)
    distance_correlation, ksg_mutual_information, conditional_mi_ksg,
    # Temporal (Domain 4)
    autocorrelation, adf_test, kpss_test, dual_stationarity, arch_test,
    rolling_mean, rolling_slope, rate_of_change,
    # Redundancy (Domain 5)
    correlation_matrix, redundant_pairs, pca, vif, cluster_by_correlation,
    # Regime (Domain 6)
    quantile_buckets,
    # Evaluation (Domains 7, 8)
    r_squared, directional_accuracy, profitable_accuracy,
    # Distribution shift (Domains 1, 2)
    js_divergence,
    # Streaming (Pass 1)
    StreamingColumnStats,
    # Sanitize
    EPS,
)

# Optional: xgboost for GradBoost baselines
try:
    import xgboost as xgb
    # Test that the C library actually loads
    xgb.XGBRegressor()
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


# =============================================================================
# Constants
# =============================================================================

SEED = 42
DEFAULT_EXPORT_DIR = "../data/exports/basic_nvda_60s"
DEFAULT_OUTPUT_DIR = "outputs/pre_training_analysis"
TARGET_HORIZON_IDX = 7  # H=60 (last horizon)
HORIZONS = (1, 2, 3, 5, 10, 20, 30, 60)

# The 14 KEEP+ features from E10 classification
KEEP_FEATURES = {
    "trf_volume": 5,
    "lit_volume": 6,
    "total_volume": 7,
    "subpenny_intensity": 8,
    "odd_lot_ratio": 9,
    "spread_bps": 12,
    "bbo_update_rate": 15,
    "spread_change_rate": 17,
    "block_trade_ratio": 21,
    "trade_count": 22,
    "size_concentration": 23,
    "bin_trade_count": 27,
    "bin_trf_trade_count": 28,
    "session_progress": 31,
}
# Sorted by index for consistent column ordering in extracted arrays
FEATURE_INDICES = sorted(KEEP_FEATURES.values())
IDX_TO_NAME = {v: k for k, v in KEEP_FEATURES.items()}
LOCAL_NAMES = [IDX_TO_NAME[i] for i in FEATURE_INDICES]
N_KEEP = len(FEATURE_INDICES)

# Conditioning variable LOCAL indices (within the 14-feature submatrix)
_COND_SPREAD_LOCAL = FEATURE_INDICES.index(12)
_COND_SESSION_LOCAL = FEATURE_INDICES.index(31)
_COND_ACTIVITY_LOCAL = FEATURE_INDICES.index(27)
CONDITIONING = {
    "spread_bps": _COND_SPREAD_LOCAL,
    "session_progress": _COND_SESSION_LOCAL,
    "bin_trade_count": _COND_ACTIVITY_LOCAL,
}

# Features likely needing log-transform (high skew volume/count)
LOG_CANDIDATE_LOCAL = [
    FEATURE_INDICES.index(i) for i in [5, 6, 7, 15, 22, 27, 28]
]

# Feature pair interaction candidates (local indices)
INTERACTION_PAIRS = [
    ("spread_bps", "bbo_update_rate"),
    ("spread_bps", "trf_volume"),
    ("subpenny_intensity", "bbo_update_rate"),
    ("trf_volume", "trade_count"),
    ("session_progress", "spread_bps"),
    ("session_progress", "bbo_update_rate"),
    ("lit_volume", "trf_volume"),
    ("subpenny_intensity", "odd_lot_ratio"),
    ("size_concentration", "block_trade_ratio"),
    ("spread_change_rate", "spread_bps"),
]

# IBKR 0DTE cost model (bps)
COST_DEEP_ITM_BPS = 1.4
COST_ATM_BPS = 4.9

# Subsampling targets
POOL_SUBSAMPLE = 5000
DCOR_SUBSAMPLE = 3000

# Rolling window sweep
ROLLING_WINDOWS = (3, 5, 10)
ROC_LAGS = (1, 3, 5)


# =============================================================================
# Utilities
# =============================================================================

def log(msg: str) -> None:
    print(f"[pre_training] {msg}", flush=True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _safe_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman IC with degenerate filter (returns NaN if degenerate)."""
    rho, p = spearman_ic(x, y)
    if rho == 0.0 and p == 1.0:
        return float("nan")
    return rho


def _ridge_predict(X_train, y_train, X_test, alpha=1.0):
    """Ridge regression via closed-form solution (no sklearn)."""
    n, d = X_train.shape
    # Add intercept
    X_tr = np.column_stack([np.ones(n), X_train])
    X_te = np.column_stack([np.ones(X_test.shape[0]), X_test])
    # Solve (X^T X + alpha I) beta = X^T y
    I = np.eye(d + 1)
    I[0, 0] = 0.0  # Don't regularize intercept
    try:
        beta = np.linalg.solve(X_tr.T @ X_tr + alpha * I, X_tr.T @ y_train)
        return X_te @ beta
    except np.linalg.LinAlgError:
        return np.full(X_test.shape[0], np.mean(y_train))


def _subsample(pool_features, pool_labels, n, rng):
    """Subsample from pool to n rows."""
    if pool_features.shape[0] <= n:
        return pool_features, pool_labels
    idx = rng.choice(pool_features.shape[0], size=n, replace=False)
    return pool_features[idx], pool_labels[idx]


# =============================================================================
# Pass 1: Streaming Accumulation
# =============================================================================

def run_streaming_pass(loader, dates, quick=False):
    """Single streaming pass over all train days.

    Accumulates:
    - Feature streaming stats (Domain 2)
    - Per-day IC for 14 features x 8 horizons (Domain 3)
    - Per-day rolling IC at H=60 for multiple windows (Domain 4)
    - Per-day log-transform IC at H=60 (Domain 4)
    - Label stats per day per horizon (Domain 1)
    - Monthly mean drift (Domain 2)
    - Reservoir-sampled pool (Domains 3, 5, 6)
    """
    log(f"Pass 1: streaming {len(dates)} train days...")
    t0 = time.time()

    n_horizons = len(HORIZONS)
    feature_stats = StreamingColumnStats(n_columns=N_KEEP)

    # Per-day IC: (local_feature_idx, horizon_idx) -> list[float]
    daily_ics = defaultdict(list)
    # Per-day CF decomposition at H=60: local_idx -> {forward: [], concurrent: []}
    daily_cf = defaultdict(lambda: {"forward": [], "concurrent": []})
    # Per-day rolling IC at H=60: (local_idx, variant_name) -> list[float]
    daily_rolling_ics = defaultdict(list)
    # Per-day log IC at H=60: local_idx -> list[float]
    daily_log_ics = defaultdict(list)
    # Label stats per horizon
    label_pools = defaultdict(list)  # horizon_idx -> list of day arrays
    # Monthly feature means
    monthly_means = defaultdict(lambda: defaultdict(list))

    # Reservoir sampling
    pool_features = np.empty((POOL_SUBSAMPLE, N_KEEP), dtype=np.float64)
    pool_labels = np.empty((POOL_SUBSAMPLE, n_horizons), dtype=np.float64)
    reservoir_count = 0
    rng = np.random.RandomState(SEED)

    for di, bundle in enumerate(loader.iter_days(dates)):
        N = bundle.sequences.shape[0]
        if N < 3:
            continue

        # Extract last-timestep features and labels
        feat_2d = np.asarray(
            bundle.sequences[:, -1, :][:, FEATURE_INDICES], dtype=np.float64
        )
        labels = np.asarray(bundle.labels, dtype=np.float64)
        h60 = labels[:, TARGET_HORIZON_IDX]

        # Feature streaming stats
        feature_stats.update_batch(feat_2d)

        # Monthly means for drift
        month = bundle.date[:7]
        for fi in range(N_KEEP):
            col = feat_2d[:, fi]
            finite = col[np.isfinite(col)]
            if len(finite) > 0:
                monthly_means[month][fi].append(float(np.mean(finite)))

        # Per-day IC for all (feature, horizon) pairs
        for fi in range(N_KEEP):
            col = feat_2d[:, fi]
            for hi in range(n_horizons):
                ic_val = _safe_ic(col, labels[:, hi])
                if np.isfinite(ic_val):
                    daily_ics[(fi, hi)].append(ic_val)

        # Concurrent/Forward IC at H=60 for spread_change_rate
        sc_local = FEATURE_INDICES.index(17)  # spread_change_rate
        for fi in range(N_KEEP):
            col = feat_2d[:, fi]
            # Forward IC: feature[t] vs label[t]
            fwd = _safe_ic(col, h60)
            if np.isfinite(fwd):
                daily_cf[fi]["forward"].append(fwd)
            # Concurrent IC: feature[t] vs label[t-1] (shifted)
            if N > 1:
                conc = _safe_ic(col[1:], h60[:-1])
                if np.isfinite(conc):
                    daily_cf[fi]["concurrent"].append(conc)

        # Rolling feature ICs at H=60 (multi-window sweep)
        for fi in range(N_KEEP):
            global_idx = FEATURE_INDICES[fi]
            seq_j = np.asarray(
                bundle.sequences[:, :, global_idx:global_idx + 1],
                dtype=np.float64,
            )  # [N, T, 1]
            for K in ROLLING_WINDOWS:
                if K >= bundle.sequences.shape[1]:
                    continue
                rm = rolling_mean(seq_j, window=K)[:, -1, 0]
                rs = rolling_slope(seq_j, window=K)[:, -1, 0]
                for name, vals in [
                    (f"rolling_mean_{K}", rm),
                    (f"rolling_slope_{K}", rs),
                ]:
                    valid = np.isfinite(vals) & np.isfinite(h60)
                    if valid.sum() >= 10:
                        ic_val = _safe_ic(vals[valid], h60[valid])
                        if np.isfinite(ic_val):
                            daily_rolling_ics[(fi, name)].append(ic_val)
            for lag in ROC_LAGS:
                if lag >= bundle.sequences.shape[1]:
                    continue
                roc = rate_of_change(seq_j, lag=lag)[:, -1, 0]
                valid = np.isfinite(roc) & np.isfinite(h60)
                if valid.sum() >= 10:
                    ic_val = _safe_ic(roc[valid], h60[valid])
                    if np.isfinite(ic_val):
                        daily_rolling_ics[(fi, f"roc_{lag}")].append(ic_val)

        # Log-transform IC at H=60 for skewed features
        for fi in LOG_CANDIDATE_LOCAL:
            col = feat_2d[:, fi]
            log_col = np.log1p(np.abs(col))
            valid = np.isfinite(log_col) & np.isfinite(h60)
            if valid.sum() >= 10:
                ic_val = _safe_ic(log_col[valid], h60[valid])
                if np.isfinite(ic_val):
                    daily_log_ics[fi].append(ic_val)

        # Label pools (subsample per day to bound memory)
        for hi in range(n_horizons):
            h_labels = labels[:, hi]
            valid_labels = h_labels[np.isfinite(h_labels)]
            if len(valid_labels) > 0:
                label_pools[hi].append(valid_labels)

        # Reservoir sampling
        for row_idx in range(N):
            reservoir_count += 1
            if reservoir_count <= POOL_SUBSAMPLE:
                pool_features[reservoir_count - 1] = feat_2d[row_idx]
                pool_labels[reservoir_count - 1] = labels[row_idx]
            else:
                j = rng.randint(0, reservoir_count)
                if j < POOL_SUBSAMPLE:
                    pool_features[j] = feat_2d[row_idx]
                    pool_labels[j] = labels[row_idx]

        if (di + 1) % 30 == 0:
            log(f"  Pass 1: {di + 1}/{len(dates)} days processed")

    actual_pool = min(reservoir_count, POOL_SUBSAMPLE)
    pool_features = pool_features[:actual_pool]
    pool_labels = pool_labels[:actual_pool]

    elapsed = time.time() - t0
    log(f"Pass 1 complete: {reservoir_count} total samples, "
        f"{actual_pool} pooled, {elapsed:.1f}s")

    return {
        "feature_stats": feature_stats,
        "daily_ics": dict(daily_ics),
        "daily_cf": dict(daily_cf),
        "daily_rolling_ics": dict(daily_rolling_ics),
        "daily_log_ics": dict(daily_log_ics),
        "label_pools": dict(label_pools),
        "monthly_means": dict(monthly_means),
        "pool_features": pool_features,
        "pool_labels": pool_labels,
        "n_total_samples": reservoir_count,
    }


# =============================================================================
# Domain 1: Label/Return Analysis
# =============================================================================

def analyze_labels(pass1, val_loader, val_dates, test_loader, test_dates):
    """Return distribution analysis at all horizons. Gate G2."""
    log("Domain 1: Label/Return analysis...")
    t0 = time.time()
    result = {"per_horizon": {}}

    for hi, h in enumerate(HORIZONS):
        pools = pass1["label_pools"].get(hi, [])
        if not pools:
            continue
        all_returns = np.concatenate(pools)

        stats = {
            "horizon": h,
            "n_samples": len(all_returns),
            "mean": float(np.mean(all_returns)),
            "std": float(np.std(all_returns)),
            "median": float(np.median(all_returns)),
            "skewness": float(scipy_stats.skew(all_returns)),
            "kurtosis": float(scipy_stats.kurtosis(all_returns)),  # excess
            "percentiles": {
                str(p): float(np.percentile(all_returns, p))
                for p in [1, 5, 25, 50, 75, 95, 99]
            },
            "var_1pct": float(np.percentile(all_returns, 1)),
            "cvar_1pct": float(np.mean(
                all_returns[all_returns <= np.percentile(all_returns, 1)]
            )) if len(all_returns) > 100 else None,
            "near_zero_1bps": float(np.mean(np.abs(all_returns) < 1.0)),
            "near_zero_05bps": float(np.mean(np.abs(all_returns) < 0.5)),
        }

        # Huber delta calibration
        iqr = float(np.percentile(all_returns, 75) - np.percentile(all_returns, 25))
        stats["iqr"] = iqr
        stats["huber_delta"] = max(iqr * 0.5, 5.0)

        # Return ACF(1) — measures stride-1 label overlap
        if len(all_returns) > 20:
            acf_vals, hl, dr = autocorrelation(all_returns, max_lag=5)
            stats["acf_1"] = float(acf_vals[1]) if len(acf_vals) > 1 else None
            stats["acf_5"] = float(acf_vals[5]) if len(acf_vals) > 5 else None
        else:
            stats["acf_1"] = None

        result["per_horizon"][f"H{h}"] = stats

    # Cross-horizon correlation matrix
    h_arrays = []
    for hi in range(len(HORIZONS)):
        pools = pass1["label_pools"].get(hi, [])
        if pools:
            arr = np.concatenate(pools)
            h_arrays.append(arr[:min(len(arr), 5000)])
    if len(h_arrays) == len(HORIZONS):
        min_len = min(len(a) for a in h_arrays)
        stacked = np.column_stack([a[:min_len] for a in h_arrays])
        corr = np.corrcoef(stacked.T)
        result["cross_horizon_correlation"] = corr.tolist()

    # Cross-split comparison
    for split_name, split_loader, split_dates in [
        ("val", val_loader, val_dates),
        ("test", test_loader, test_dates),
    ]:
        if not split_dates:
            continue
        split_stats = {}
        for bundle in split_loader.iter_days(split_dates):
            labels = np.asarray(bundle.labels, dtype=np.float64)
            for hi, h in enumerate(HORIZONS):
                h_lab = labels[:, hi]
                valid = h_lab[np.isfinite(h_lab)]
                if len(valid) > 0:
                    split_stats.setdefault(f"H{h}", []).append(valid)
        for key, pools in split_stats.items():
            all_ret = np.concatenate(pools)
            result.setdefault(f"{split_name}_comparison", {})[key] = {
                "mean": float(np.mean(all_ret)),
                "std": float(np.std(all_ret)),
                "n_samples": len(all_ret),
            }

    # G2 gate
    h60_stats = result["per_horizon"].get(f"H{HORIZONS[TARGET_HORIZON_IDX]}", {})
    result["gate_G2"] = {
        "metric": "return_std_bps",
        "value": h60_stats.get("std"),
        "threshold": 5.0,
        "verdict": "PASS" if (h60_stats.get("std") or 0) > 5.0 else "FAIL",
    }

    log(f"  Domain 1 done in {time.time() - t0:.1f}s")
    return result


# =============================================================================
# Domain 2: Feature Distribution Profiling
# =============================================================================

def analyze_feature_distributions(pass1, val_loader, val_dates, test_loader,
                                   test_dates):
    """Per-feature distribution stats, drift, shift, transform flags."""
    log("Domain 2: Feature distribution profiling...")
    t0 = time.time()

    summary = pass1["feature_stats"].get_summary()
    pool = pass1["pool_features"]
    result = {"per_feature": {}, "transform_flags": {}}

    for fi in range(N_KEEP):
        name = LOCAL_NAMES[fi]
        col = pool[:, fi]
        finite_col = col[np.isfinite(col)]

        feat_stats = {
            "index": FEATURE_INDICES[fi],
            "streaming_mean": summary[fi]["mean"] if fi in summary else None,
            "streaming_std": summary[fi]["std"] if fi in summary else None,
        }

        if len(finite_col) > 10:
            feat_stats.update({
                "mean": float(np.mean(finite_col)),
                "std": float(np.std(finite_col)),
                "skewness": float(scipy_stats.skew(finite_col)),
                "kurtosis": float(scipy_stats.kurtosis(finite_col)),
                "percentiles": {
                    str(p): float(np.percentile(finite_col, p))
                    for p in [1, 5, 25, 50, 75, 95, 99]
                },
                "min": float(np.min(finite_col)),
                "max": float(np.max(finite_col)),
                "cv": float(np.std(finite_col) / max(abs(np.mean(finite_col)), EPS)),
                "zero_fraction": float(np.mean(np.abs(finite_col) < EPS)),
            })
        result["per_feature"][name] = feat_stats

        # Transform flags
        flags = {}
        skew = feat_stats.get("skewness", 0)
        kurt = feat_stats.get("kurtosis", 0)
        zf = feat_stats.get("zero_fraction", 0)
        if abs(skew) > 2.0:
            flags["needs_log_transform"] = True
        if kurt > 20.0:
            flags["needs_clipping"] = True
        if zf > 0.50:
            flags["needs_binary_encoding"] = True
        if flags:
            result["transform_flags"][name] = flags

    # Monthly drift
    monthly = pass1["monthly_means"]
    months_sorted = sorted(monthly.keys())
    if len(months_sorted) >= 2:
        first_month = months_sorted[0]
        last_month = months_sorted[-1]
        drift = {}
        for fi in range(N_KEEP):
            first_vals = monthly[first_month].get(fi, [])
            last_vals = monthly[last_month].get(fi, [])
            if first_vals and last_vals:
                fm = np.mean(first_vals)
                lm = np.mean(last_vals)
                drift[LOCAL_NAMES[fi]] = {
                    "first_month": first_month,
                    "first_mean": float(fm),
                    "last_month": last_month,
                    "last_mean": float(lm),
                    "change_pct": float((lm - fm) / max(abs(fm), EPS) * 100),
                }
        result["monthly_drift"] = drift

    # Cross-split JS divergence
    for split_name, split_loader, split_dates in [
        ("val", val_loader, val_dates),
        ("test", test_loader, test_dates),
    ]:
        if not split_dates:
            continue
        split_pool = []
        for bundle in split_loader.iter_days(split_dates[:10]):  # Sample 10 days
            feat_2d = np.asarray(
                bundle.sequences[:, -1, :][:, FEATURE_INDICES], dtype=np.float64
            )
            split_pool.append(feat_2d)
        if not split_pool:
            continue
        split_arr = np.vstack(split_pool)
        js_per_feature = {}
        for fi in range(N_KEEP):
            train_col = pool[:, fi]
            split_col = split_arr[:, fi]
            train_finite = train_col[np.isfinite(train_col)]
            split_finite = split_col[np.isfinite(split_col)]
            if len(train_finite) > 50 and len(split_finite) > 50:
                # Build histograms
                edges = np.histogram_bin_edges(
                    np.concatenate([train_finite, split_finite]), bins=50
                )
                h_train, _ = np.histogram(train_finite, bins=edges, density=True)
                h_split, _ = np.histogram(split_finite, bins=edges, density=True)
                if h_train.sum() > 0 and h_split.sum() > 0:
                    js = js_divergence(h_train, h_split)
                    js_per_feature[LOCAL_NAMES[fi]] = float(js)
        result[f"js_divergence_{split_name}"] = js_per_feature

    log(f"  Domain 2 done in {time.time() - t0:.1f}s")
    return result


# =============================================================================
# Domain 3: Signal Analysis + Nonlinear Shape
# =============================================================================

def analyze_signal(pass1):
    """Full IC matrix, dCor, MI, partial dependence, interactions. Gate G1."""
    log("Domain 3: Signal analysis + nonlinear shape...")
    t0 = time.time()

    pool_f = pass1["pool_features"]
    pool_l = pass1["pool_labels"]
    rng = np.random.RandomState(SEED)
    sub_f, sub_l = _subsample(pool_f, pool_l, DCOR_SUBSAMPLE, rng)

    # --- Full IC matrix (14 x 8) ---
    ic_matrix = {}
    ic_ir_matrix = {}
    pvalue_matrix = {}
    for fi in range(N_KEEP):
        name = LOCAL_NAMES[fi]
        ic_matrix[name] = {}
        ic_ir_matrix[name] = {}
        pvalue_matrix[name] = {}
        for hi, h in enumerate(HORIZONS):
            ics = pass1["daily_ics"].get((fi, hi), [])
            if len(ics) >= 3:
                arr = np.array(ics)
                ic_matrix[name][f"H{h}"] = float(np.mean(arr))
                ic_ir_matrix[name][f"H{h}"] = float(ic_ir(arr))
                _, pv = scipy_stats.ttest_1samp(arr, 0.0)
                pvalue_matrix[name][f"H{h}"] = float(pv)
            else:
                ic_matrix[name][f"H{h}"] = None

    # --- dCor and MI at H=60 ---
    h60_l = sub_l[:, TARGET_HORIZON_IDX]
    dcor_results = {}
    mi_results = {}
    for fi in range(N_KEEP):
        name = LOCAL_NAMES[fi]
        col = sub_f[:, fi]
        valid = np.isfinite(col) & np.isfinite(h60_l)
        if valid.sum() >= 50:
            dcor_results[name] = float(distance_correlation(col[valid], h60_l[valid]))
            mi_results[name] = float(
                ksg_mutual_information(col[valid], h60_l[valid], k=5)
            )

    # --- Best horizon per feature ---
    best_horizons = {}
    for fi in range(N_KEEP):
        name = LOCAL_NAMES[fi]
        ic_row = ic_matrix.get(name, {})
        best_h, best_ic = None, 0.0
        for h_key, ic_val in ic_row.items():
            if ic_val is not None and abs(ic_val) > abs(best_ic):
                best_ic = ic_val
                best_h = h_key
        best_horizons[name] = {"horizon": best_h, "ic": best_ic}

    # --- Partial Dependence Curves at H=60 ---
    partial_dep = {}
    h60_pool = pool_l[:, TARGET_HORIZON_IDX]
    for fi in range(N_KEEP):
        name = LOCAL_NAMES[fi]
        col = pool_f[:, fi]
        valid = np.isfinite(col) & np.isfinite(h60_pool)
        if valid.sum() < 100:
            continue
        col_v, ret_v = col[valid], h60_pool[valid]
        try:
            bins, bin_labels = quantile_buckets(col_v, n_bins=10)
        except Exception:
            continue
        deciles = []
        for d in range(10):
            mask = bins == d
            if mask.sum() >= 10:
                deciles.append({
                    "decile": d,
                    "label": bin_labels.get(d, f"Q{d+1}"),
                    "mean_return": float(np.mean(ret_v[mask])),
                    "std_return": float(np.std(ret_v[mask])),
                    "n_samples": int(mask.sum()),
                    "prob_positive": float(np.mean(ret_v[mask] > 0)),
                    "feature_mean": float(np.mean(col_v[mask])),
                })
        # Check monotonicity
        means = [d["mean_return"] for d in deciles]
        is_monotonic_inc = all(means[i] <= means[i+1] for i in range(len(means)-1))
        is_monotonic_dec = all(means[i] >= means[i+1] for i in range(len(means)-1))
        partial_dep[name] = {
            "deciles": deciles,
            "is_monotonic": is_monotonic_inc or is_monotonic_dec,
            "monotonic_direction": "increasing" if is_monotonic_inc else (
                "decreasing" if is_monotonic_dec else "non_monotonic"
            ),
        }

    # --- Feature Pair Interaction MI ---
    interactions = {}
    h60_sub = sub_l[:, TARGET_HORIZON_IDX]
    for name_a, name_b in INTERACTION_PAIRS:
        li_a = FEATURE_INDICES.index(KEEP_FEATURES[name_a])
        li_b = FEATURE_INDICES.index(KEEP_FEATURES[name_b])
        col_a = sub_f[:, li_a]
        col_b = sub_f[:, li_b]
        valid = np.isfinite(col_a) & np.isfinite(col_b) & np.isfinite(h60_sub)
        if valid.sum() < 50:
            continue
        a, b, y = col_a[valid], col_b[valid], h60_sub[valid]
        mi_a_y = ksg_mutual_information(a, y, k=5)
        cmi_a_y_b = conditional_mi_ksg(a, y, b, k=5)
        synergy = cmi_a_y_b - mi_a_y
        interactions[f"{name_a}_x_{name_b}"] = {
            "mi_a_y": float(mi_a_y),
            "cmi_a_y_given_b": float(cmi_a_y_b),
            "synergy": float(synergy),
            "interpretation": "synergistic" if synergy > 0.01 else (
                "redundant" if synergy < -0.01 else "independent"
            ),
        }

    # --- Forward IC isolation for CF analysis ---
    cf_results = {}
    for fi in range(N_KEEP):
        name = LOCAL_NAMES[fi]
        fwd = pass1["daily_cf"].get(fi, {}).get("forward", [])
        conc = pass1["daily_cf"].get(fi, {}).get("concurrent", [])
        if fwd and conc:
            fwd_mean = float(np.mean(fwd))
            conc_mean = float(np.mean(conc))
            ratio = abs(conc_mean) / max(abs(fwd_mean), EPS)
            cf_results[name] = {
                "forward_ic": fwd_mean,
                "concurrent_ic": conc_mean,
                "cf_ratio": float(ratio),
                "classification": (
                    "contemporaneous" if ratio > 10 else
                    "partially_forward" if ratio > 2 else
                    "forward" if abs(fwd_mean) > 0.02 else
                    "state_variable"
                ),
            }

    # G1 gate
    h60_key = f"H{HORIZONS[TARGET_HORIZON_IDX]}"
    best_ic_h60 = max(
        (abs(ic_matrix[n].get(h60_key, 0) or 0) for n in LOCAL_NAMES),
        default=0,
    )

    result = {
        "ic_matrix": ic_matrix,
        "ic_ir_matrix": ic_ir_matrix,
        "pvalue_matrix": pvalue_matrix,
        "dcor_h60": dcor_results,
        "mi_h60": mi_results,
        "best_horizons": best_horizons,
        "partial_dependence": partial_dep,
        "interactions": interactions,
        "cf_decomposition": cf_results,
        "gate_G1": {
            "metric": "best_feature_abs_IC_at_H60",
            "value": float(best_ic_h60),
            "threshold": 0.05,
            "verdict": "PASS" if best_ic_h60 > 0.05 else "FAIL",
        },
    }
    log(f"  Domain 3 done in {time.time() - t0:.1f}s")
    return result


# =============================================================================
# Domain 4: Temporal Analysis
# =============================================================================

def analyze_temporal(loader, dates, pass1):
    """ACF, timestep IC decay, stationarity, multi-window sweep, directional
    signal verification. Pass 2 streaming."""
    log("Domain 4: Temporal analysis (Pass 2)...")
    t0 = time.time()

    n_horizons = len(HORIZONS)
    T = None  # Will be detected from first bundle

    # Accumulators for timestep IC at H=60
    timestep_ics = defaultdict(lambda: defaultdict(list))  # fi -> t -> [daily_ic]
    # Accumulators for ACF (average across sequences and days)
    acf_accum = defaultdict(lambda: {"sum": None, "count": 0})
    # Level vs change IC
    level_ics = defaultdict(list)
    change_ics = defaultdict(list)
    # Daily mean feature for stationarity
    daily_feature_means = defaultdict(list)

    for di, bundle in enumerate(loader.iter_days(dates)):
        N = bundle.sequences.shape[0]
        if N < 3:
            continue
        if T is None:
            T = bundle.sequences.shape[1]

        labels = np.asarray(bundle.labels, dtype=np.float64)
        h60 = labels[:, TARGET_HORIZON_IDX]

        for fi in range(N_KEEP):
            global_idx = FEATURE_INDICES[fi]
            seq_j = np.asarray(bundle.sequences[:, :, global_idx], dtype=np.float64)

            # Timestep IC at H=60
            for t in range(T):
                col_t = seq_j[:, t]
                valid = np.isfinite(col_t) & np.isfinite(h60)
                if valid.sum() >= 10:
                    ic_val = _safe_ic(col_t[valid], h60[valid])
                    if np.isfinite(ic_val):
                        timestep_ics[fi][t].append(ic_val)

            # ACF within each sequence (average)
            max_lag_acf = min(10, T - 1)
            day_acf_sum = np.zeros(max_lag_acf + 1)
            day_acf_count = 0
            for i in range(min(N, 50)):  # Subsample sequences for speed
                row = seq_j[i]
                if np.all(np.isfinite(row)) and np.std(row) > EPS:
                    acf_vals, _, _ = autocorrelation(row, max_lag=max_lag_acf)
                    day_acf_sum += acf_vals[:max_lag_acf + 1]
                    day_acf_count += 1
            if day_acf_count > 0:
                entry = acf_accum[fi]
                if entry["sum"] is None:
                    entry["sum"] = day_acf_sum / day_acf_count
                else:
                    entry["sum"] += day_acf_sum / day_acf_count
                entry["count"] += 1

            # Level vs change IC at H=60
            last = seq_j[:, -1]
            diff = seq_j[:, -1] - seq_j[:, -2] if T >= 2 else np.zeros(N)
            valid_l = np.isfinite(last) & np.isfinite(h60)
            valid_d = np.isfinite(diff) & np.isfinite(h60)
            if valid_l.sum() >= 10:
                ic_l = _safe_ic(last[valid_l], h60[valid_l])
                if np.isfinite(ic_l):
                    level_ics[fi].append(ic_l)
            if valid_d.sum() >= 10:
                ic_d = _safe_ic(diff[valid_d], h60[valid_d])
                if np.isfinite(ic_d):
                    change_ics[fi].append(ic_d)

            # Daily mean for stationarity
            last_finite = last[np.isfinite(last)]
            if len(last_finite) > 0:
                daily_feature_means[fi].append(float(np.mean(last_finite)))

        if (di + 1) % 30 == 0:
            log(f"  Pass 2: {di + 1}/{len(dates)} days")

    # Compile results
    result = {"acf": {}, "timestep_ic": {}, "stationarity": {},
              "level_vs_change": {}, "multi_window_sweep": {},
              "directional_signal": {}, "log_transform_comparison": {}}

    # ACF
    for fi in range(N_KEEP):
        name = LOCAL_NAMES[fi]
        entry = acf_accum[fi]
        if entry["count"] > 0:
            avg_acf = entry["sum"] / entry["count"]
            hl = None
            for lag in range(1, len(avg_acf)):
                if avg_acf[lag] < 0.5:
                    hl = lag
                    break
            result["acf"][name] = {
                "acf_1": float(avg_acf[1]) if len(avg_acf) > 1 else None,
                "acf_5": float(avg_acf[5]) if len(avg_acf) > 5 else None,
                "acf_10": float(avg_acf[10]) if len(avg_acf) > 10 else None,
                "half_life": hl,
            }

    # Timestep IC curves
    for fi in range(N_KEEP):
        name = LOCAL_NAMES[fi]
        curve = []
        for t in range(T or 20):
            ics = timestep_ics[fi].get(t, [])
            curve.append(float(np.mean(ics)) if ics else None)
        # IC half-life: first timestep from end where |IC| drops below 50% of peak
        finite_curve = [(t, abs(c)) for t, c in enumerate(curve) if c is not None]
        peak_ic = max((v for _, v in finite_curve), default=0)
        ic_hl = None
        if peak_ic > 0:
            for t_idx in range(len(finite_curve) - 1, -1, -1):
                if finite_curve[t_idx][1] < peak_ic * 0.5:
                    ic_hl = (T or 20) - finite_curve[t_idx][0]
                    break
        result["timestep_ic"][name] = {
            "curve": curve,
            "peak_ic": float(peak_ic),
            "ic_half_life_timesteps": ic_hl,
        }

    # Stationarity
    for fi in range(N_KEEP):
        name = LOCAL_NAMES[fi]
        series = np.array(daily_feature_means.get(fi, []))
        if len(series) >= 30:
            ds = dual_stationarity(series)
            has_arch = arch_test(series)
            result["stationarity"][name] = {
                "adf_p": float(ds.adf_p),
                "kpss_p": float(ds.kpss_p),
                "classification": ds.classification,
                "is_stationary": ds.is_stationary,
                "arch_p": float(has_arch[0]),
                "has_arch_effects": bool(has_arch[1]),
            }

    # Level vs change
    for fi in range(N_KEEP):
        name = LOCAL_NAMES[fi]
        l_ics = level_ics.get(fi, [])
        c_ics = change_ics.get(fi, [])
        result["level_vs_change"][name] = {
            "level_ic_mean": float(np.mean(l_ics)) if l_ics else None,
            "change_ic_mean": float(np.mean(c_ics)) if c_ics else None,
            "level_stronger": (
                abs(np.mean(l_ics)) > abs(np.mean(c_ics))
                if l_ics and c_ics else None
            ),
        }

    # Multi-window sweep (from Pass 1 data)
    for fi in range(N_KEEP):
        name = LOCAL_NAMES[fi]
        sweep = {}
        best_variant, best_ic = "raw", abs(
            np.mean(pass1["daily_ics"].get((fi, TARGET_HORIZON_IDX), [0]))
        )
        sweep["raw"] = float(best_ic)
        for key, ics in pass1["daily_rolling_ics"].items():
            if key[0] == fi:
                variant = key[1]
                mean_ic = abs(float(np.mean(ics))) if ics else 0
                sweep[variant] = mean_ic
                if mean_ic > best_ic:
                    best_ic = mean_ic
                    best_variant = variant
        result["multi_window_sweep"][name] = {
            "variants": sweep,
            "best_variant": best_variant,
            "best_ic": float(best_ic),
        }

    # Directional signal from temporal derivatives
    for fi in range(N_KEEP):
        name = LOCAL_NAMES[fi]
        raw_ic = abs(np.mean(pass1["daily_ics"].get((fi, TARGET_HORIZON_IDX), [0])))
        slope_ics = pass1["daily_rolling_ics"].get((fi, "rolling_slope_5"), [])
        roc_ics = pass1["daily_rolling_ics"].get((fi, "roc_1"), [])
        slope_ic = abs(float(np.mean(slope_ics))) if slope_ics else 0
        roc_ic = abs(float(np.mean(roc_ics))) if roc_ics else 0
        result["directional_signal"][name] = {
            "raw_ic": float(raw_ic),
            "slope_5_ic": float(slope_ic),
            "roc_1_ic": float(roc_ic),
            "temporal_adds_signal": slope_ic > raw_ic or roc_ic > raw_ic,
        }

    # Log-transform comparison
    for fi in LOG_CANDIDATE_LOCAL:
        name = LOCAL_NAMES[fi]
        raw_ic = abs(np.mean(pass1["daily_ics"].get((fi, TARGET_HORIZON_IDX), [0])))
        log_ics = pass1["daily_log_ics"].get(fi, [])
        log_ic = abs(float(np.mean(log_ics))) if log_ics else 0
        result["log_transform_comparison"][name] = {
            "raw_ic": float(raw_ic),
            "log_ic": float(log_ic),
            "log_helps": log_ic > raw_ic * 1.05,  # 5% improvement threshold
        }

    log(f"  Domain 4 done in {time.time() - t0:.1f}s")
    return result


# =============================================================================
# Domain 5: Feature Redundancy & Selection
# =============================================================================

def analyze_redundancy(pool_features):
    """Correlation matrix, PCA, VIF, clustering, pruning recommendations."""
    log("Domain 5: Redundancy analysis...")
    t0 = time.time()

    indices = list(range(N_KEEP))

    corr, valid_idx = correlation_matrix(pool_features, indices)
    pairs = redundant_pairs(corr, LOCAL_NAMES, threshold=0.8)
    pca_result = pca(pool_features, indices, LOCAL_NAMES)
    vif_results = vif(pool_features, indices, LOCAL_NAMES)
    clusters = cluster_by_correlation(corr, LOCAL_NAMES, indices, threshold=0.7)

    # Pruning recommendations
    to_drop = []
    for pair in pairs:
        if abs(pair["correlation"]) > 0.95:
            to_drop.append({
                "drop": pair["signal_2"],
                "keep": pair["signal_1"],
                "reason": f"|r| = {abs(pair['correlation']):.3f}",
            })

    result = {
        "correlation_matrix": corr.tolist(),
        "feature_names": LOCAL_NAMES,
        "highly_correlated_pairs": pairs,
        "pca": {
            "n_components_90": pca_result.n_components_90,
            "n_components_95": pca_result.n_components_95,
            "explained_variance_ratio": pca_result.explained_variance_ratio[:5],
            "cumulative_top5": pca_result.cumulative_variance[:5],
        },
        "vif": [
            {
                "feature": v.signal_name,
                "vif": float(v.vif),
                "is_problematic": v.is_problematic,
                "is_severe": v.is_severe,
            }
            for v in vif_results
        ],
        "clusters": [
            {
                "cluster_id": c.cluster_id,
                "signals": c.signals,
                "mean_within_corr": float(c.mean_within_correlation),
            }
            for c in clusters
        ],
        "pruning_recommendations": to_drop,
        "effective_unique_features": N_KEEP - len(to_drop),
    }

    log(f"  Domain 5 done in {time.time() - t0:.1f}s")
    return result


# =============================================================================
# Domain 6: Regime-Conditional Analysis
# =============================================================================

def analyze_regime(pool_features, pool_labels):
    """Regime-conditional IC for 3 conditioning variables."""
    log("Domain 6: Regime analysis...")
    t0 = time.time()

    h60 = pool_labels[:, TARGET_HORIZON_IDX]
    result = {"tercile_ic": {}, "joint_grid": {}, "dependence_scores": {}}

    for cond_name, cond_local in CONDITIONING.items():
        cond_vals = pool_features[:, cond_local]
        valid = np.isfinite(cond_vals) & np.isfinite(h60)
        if valid.sum() < 100:
            continue
        try:
            bins, bin_labels = quantile_buckets(cond_vals[valid], n_bins=3)
        except Exception:
            continue

        per_feature = {}
        for fi in range(N_KEEP):
            name = LOCAL_NAMES[fi]
            feat_vals = pool_features[valid, fi]
            tercile_results = []
            for b in range(3):
                mask = bins == b
                if mask.sum() >= 30:
                    ic_val = _safe_ic(feat_vals[mask], h60[valid][mask])
                    tercile_results.append({
                        "tercile": bin_labels.get(b, f"T{b}"),
                        "ic": float(ic_val) if np.isfinite(ic_val) else 0.0,
                        "n_samples": int(mask.sum()),
                    })
            per_feature[name] = tercile_results

            # Dependence score
            ics = [t["ic"] for t in tercile_results]
            if len(ics) >= 2:
                dep_score = max(ics) - min(ics)
                result["dependence_scores"].setdefault(name, {})[cond_name] = float(dep_score)

        result["tercile_ic"][cond_name] = per_feature

    # Joint grid: session_progress x spread_bps (9 cells)
    sess = pool_features[:, _COND_SESSION_LOCAL]
    spread = pool_features[:, _COND_SPREAD_LOCAL]
    valid = np.isfinite(sess) & np.isfinite(spread) & np.isfinite(h60)
    if valid.sum() >= 200:
        try:
            sess_bins, sess_labels = quantile_buckets(sess[valid], n_bins=3)
            spread_bins, spread_labels = quantile_buckets(spread[valid], n_bins=3)
        except Exception:
            sess_bins = spread_bins = None

        if sess_bins is not None:
            grid = []
            for s in range(3):
                for sp in range(3):
                    mask = (sess_bins == s) & (spread_bins == sp)
                    if mask.sum() >= 20:
                        # Mean IC of top 5 features
                        feature_ics = []
                        for fi in range(N_KEEP):
                            col = pool_features[valid, fi][mask]
                            ret = h60[valid][mask]
                            ic_val = _safe_ic(col, ret)
                            if np.isfinite(ic_val):
                                feature_ics.append(abs(ic_val))
                        feature_ics.sort(reverse=True)
                        top5_mean = float(np.mean(feature_ics[:5])) if feature_ics else 0
                        grid.append({
                            "session": sess_labels.get(s, f"S{s}"),
                            "spread": spread_labels.get(sp, f"SP{sp}"),
                            "n_samples": int(mask.sum()),
                            "top5_mean_ic": top5_mean,
                        })
            result["joint_grid"] = grid

    log(f"  Domain 6 done in {time.time() - t0:.1f}s")
    return result


# =============================================================================
# Domain 7: Walk-Forward Stability
# =============================================================================

def walk_forward_analysis(loader, dates):
    """Daily expanding-window walk-forward with Ridge. Gate G3."""
    log("Domain 7: Walk-forward analysis (Pass 3)...")
    t0 = time.time()

    MIN_TRAIN_DAYS = 30
    if len(dates) <= MIN_TRAIN_DAYS:
        log("  Not enough days for walk-forward")
        return {"error": "insufficient_days"}

    # Load all days' last-timestep features + H=60 labels
    day_data = []
    for bundle in loader.iter_days(dates):
        feat = np.asarray(
            bundle.sequences[:, -1, :][:, FEATURE_INDICES], dtype=np.float64
        )
        lab = np.asarray(bundle.labels[:, TARGET_HORIZON_IDX], dtype=np.float64)
        day_data.append((feat, lab))

    # Daily folds: train on first k days, test on day k
    folds = []
    for k in range(MIN_TRAIN_DAYS, len(day_data)):
        # Pool training data
        train_feats = np.vstack([d[0] for d in day_data[:k]])
        train_labels = np.concatenate([d[1] for d in day_data[:k]])
        test_feats = day_data[k][0]
        test_labels = day_data[k][1]

        # Filter non-finite
        tr_valid = np.all(np.isfinite(train_feats), axis=1) & np.isfinite(train_labels)
        te_valid = np.all(np.isfinite(test_feats), axis=1) & np.isfinite(test_labels)
        if tr_valid.sum() < 50 or te_valid.sum() < 5:
            continue

        X_tr, y_tr = train_feats[tr_valid], train_labels[tr_valid]
        X_te, y_te = test_feats[te_valid], test_labels[te_valid]

        preds = _ridge_predict(X_tr, y_tr, X_te, alpha=1.0)
        ic_val = _safe_ic(preds, y_te)
        r2 = r_squared(y_te, preds)
        da = directional_accuracy(y_te, preds)

        folds.append({
            "fold": k,
            "train_days": k,
            "test_samples": int(te_valid.sum()),
            "ic": float(ic_val) if np.isfinite(ic_val) else 0.0,
            "r_squared": float(r2),
            "da": float(da),
        })

        if (k - MIN_TRAIN_DAYS + 1) % 30 == 0:
            log(f"  Walk-forward: fold {k - MIN_TRAIN_DAYS + 1}/{len(day_data) - MIN_TRAIN_DAYS}")

    # Aggregate
    fold_ics = np.array([f["ic"] for f in folds])
    fold_das = np.array([f["da"] for f in folds])
    mean_ic = float(np.mean(fold_ics)) if len(fold_ics) > 0 else 0
    std_ic = float(np.std(fold_ics)) if len(fold_ics) > 0 else 1
    stability = mean_ic / max(std_ic, EPS)

    # Per-feature daily IC stability (from Pass 1 data not available here,
    # compute from accumulated day_data)
    per_feature_stability = {}
    for fi in range(N_KEEP):
        name = LOCAL_NAMES[fi]
        daily_ics_f = []
        for feat, lab in day_data:
            col = feat[:, fi]
            valid = np.isfinite(col) & np.isfinite(lab)
            if valid.sum() >= 10:
                ic_val = _safe_ic(col[valid], lab[valid])
                if np.isfinite(ic_val):
                    daily_ics_f.append(ic_val)
        if daily_ics_f:
            arr = np.array(daily_ics_f)
            sign_changes = np.sum(np.diff(np.sign(arr)) != 0)
            per_feature_stability[name] = {
                "mean_ic": float(np.mean(arr)),
                "std_ic": float(np.std(arr)),
                "stability_ratio": float(np.mean(arr) / max(np.std(arr), EPS)),
                "sign_flip_rate": float(sign_changes / max(len(arr) - 1, 1)),
            }

    result = {
        "n_folds": len(folds),
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "stability_ratio": float(stability),
        "mean_da": float(np.mean(fold_das)) if len(fold_das) > 0 else 0,
        "regime_shift_count": int(np.sum(fold_ics < 0)),
        "per_feature_stability": per_feature_stability,
        "gate_G3": {
            "metric": "walkforward_stability_ratio",
            "value": float(stability),
            "threshold": 2.0,
            "verdict": "PASS" if stability > 2.0 else "FAIL",
        },
    }

    log(f"  Domain 7 done: {len(folds)} folds, stability={stability:.2f}, "
        f"{time.time() - t0:.1f}s")
    return result


# =============================================================================
# Domain 8: Baseline Model Performance
# =============================================================================

def baseline_models(loader, dates, val_loader, val_dates, test_loader,
                    test_dates, d4_result):
    """Persistence, single-feature, Ridge, GradBoost baselines. Gate G4."""
    log("Domain 8: Baseline models...")
    t0 = time.time()

    # Load all train data
    all_feats, all_labels = [], []
    for bundle in loader.iter_days(dates):
        feat = np.asarray(
            bundle.sequences[:, -1, :][:, FEATURE_INDICES], dtype=np.float64
        )
        lab = np.asarray(bundle.labels[:, TARGET_HORIZON_IDX], dtype=np.float64)
        all_feats.append(feat)
        all_labels.append(lab)
    X_train = np.vstack(all_feats)
    y_train = np.concatenate(all_labels)
    valid = np.all(np.isfinite(X_train), axis=1) & np.isfinite(y_train)
    X_train, y_train = X_train[valid], y_train[valid]

    baselines = {}

    # --- Persistence baseline ---
    if len(y_train) > 1:
        pers_pred = np.roll(y_train, 1)
        pers_pred[0] = 0.0
        baselines["persistence"] = {
            "r_squared": float(r_squared(y_train[1:], pers_pred[1:])),
            "ic": float(_safe_ic(pers_pred[1:], y_train[1:])),
            "da": float(directional_accuracy(y_train[1:], pers_pred[1:])),
        }

    # --- Best single feature (spread_bps) ---
    spread_local = FEATURE_INDICES.index(12)
    preds_sf = _ridge_predict(
        X_train[:, spread_local:spread_local + 1], y_train,
        X_train[:, spread_local:spread_local + 1], alpha=1.0,
    )
    baselines["single_feature_spread_bps"] = {
        "r_squared": float(r_squared(y_train, preds_sf)),
        "ic": float(_safe_ic(preds_sf, y_train)),
        "da": float(directional_accuracy(y_train, preds_sf)),
    }

    # --- Ridge on raw 14 ---
    preds_ridge = _ridge_predict(X_train, y_train, X_train, alpha=1.0)
    baselines["ridge_raw_14"] = {
        "r_squared": float(r_squared(y_train, preds_ridge)),
        "ic": float(_safe_ic(preds_ridge, y_train)),
        "da": float(directional_accuracy(y_train, preds_ridge)),
    }

    # --- Ridge on temporal features ---
    # Build temporal features from sequences
    temporal_feats_train = []
    for bundle in loader.iter_days(dates):
        N = bundle.sequences.shape[0]
        feat_last = np.asarray(
            bundle.sequences[:, -1, :][:, FEATURE_INDICES], dtype=np.float64
        )
        temporal_block = [feat_last]
        for fi in range(N_KEEP):
            global_idx = FEATURE_INDICES[fi]
            # Use best window from Domain 4 or default K=5
            best = d4_result.get("multi_window_sweep", {}).get(
                LOCAL_NAMES[fi], {}
            ).get("best_variant", "rolling_mean_5")
            K = 5
            if "3" in best:
                K = 3
            elif "10" in best:
                K = 10
            seq_j = np.asarray(
                bundle.sequences[:, :, global_idx:global_idx + 1], dtype=np.float64
            )
            rm = rolling_mean(seq_j, window=K)[:, -1, 0]
            rs = rolling_slope(seq_j, window=K)[:, -1, 0]
            temporal_block.append(rm.reshape(-1, 1))
            temporal_block.append(rs.reshape(-1, 1))
        temporal_feats_train.append(np.hstack(temporal_block))

    X_temp = np.vstack(temporal_feats_train)
    valid_t = np.all(np.isfinite(X_temp), axis=1) & np.isfinite(y_train)
    X_temp_v, y_temp_v = X_temp[valid_t], y_train[valid_t]
    if len(X_temp_v) > 100:
        preds_ridge_t = _ridge_predict(X_temp_v, y_temp_v, X_temp_v, alpha=1.0)
        baselines["ridge_temporal"] = {
            "n_features": X_temp_v.shape[1],
            "r_squared": float(r_squared(y_temp_v, preds_ridge_t)),
            "ic": float(_safe_ic(preds_ridge_t, y_temp_v)),
            "da": float(directional_accuracy(y_temp_v, preds_ridge_t)),
        }

    # --- GradBoost with mini grid search ---
    best_gb_da = 0.0
    if HAS_XGBOOST:
        log("  Running GradBoost grid search (xgboost available)...")
        gb_results = []
        for max_d in [3, 5, 7]:
            for lr in [0.01, 0.05, 0.1]:
                model = xgb.XGBRegressor(
                    max_depth=max_d, n_estimators=300, learning_rate=lr,
                    subsample=0.8, min_child_weight=50,
                    random_state=SEED, n_jobs=1, verbosity=0,
                )
                # Simple train/evaluate (use last 20% as holdout)
                split_idx = int(len(X_train) * 0.8)
                model.fit(X_train[:split_idx], y_train[:split_idx])
                preds_gb = model.predict(X_train[split_idx:])
                y_test_gb = y_train[split_idx:]
                ic_gb = _safe_ic(preds_gb, y_test_gb)
                da_gb = directional_accuracy(y_test_gb, preds_gb)
                gb_results.append({
                    "max_depth": max_d,
                    "learning_rate": lr,
                    "ic": float(ic_gb) if np.isfinite(ic_gb) else 0.0,
                    "da": float(da_gb),
                    "r_squared": float(r_squared(y_test_gb, preds_gb)),
                })
                if da_gb > best_gb_da:
                    best_gb_da = da_gb

        gb_results.sort(key=lambda x: -x["ic"])
        baselines["gradboost_grid"] = gb_results
        baselines["gradboost_best"] = gb_results[0] if gb_results else None

        # Feature importance from best config
        best_cfg = gb_results[0] if gb_results else None
        if best_cfg:
            model = xgb.XGBRegressor(
                max_depth=best_cfg["max_depth"],
                n_estimators=300,
                learning_rate=best_cfg["learning_rate"],
                subsample=0.8, min_child_weight=50,
                random_state=SEED, n_jobs=1, verbosity=0,
            )
            model.fit(X_train, y_train)
            importances = model.feature_importances_
            baselines["feature_importance"] = {
                LOCAL_NAMES[fi]: float(importances[fi])
                for fi in range(N_KEEP)
            }
    else:
        log("  xgboost not available, skipping GradBoost baselines")
        baselines["gradboost_note"] = "xgboost not installed"

    # Val/Test evaluation
    for split_name, split_loader, split_dates in [
        ("val", val_loader, val_dates),
        ("test", test_loader, test_dates),
    ]:
        if not split_dates:
            continue
        split_feats, split_labels = [], []
        for bundle in split_loader.iter_days(split_dates):
            feat = np.asarray(
                bundle.sequences[:, -1, :][:, FEATURE_INDICES], dtype=np.float64
            )
            lab = np.asarray(bundle.labels[:, TARGET_HORIZON_IDX], dtype=np.float64)
            split_feats.append(feat)
            split_labels.append(lab)
        if split_feats:
            X_s = np.vstack(split_feats)
            y_s = np.concatenate(split_labels)
            valid_s = np.all(np.isfinite(X_s), axis=1) & np.isfinite(y_s)
            if valid_s.sum() > 10:
                preds_s = _ridge_predict(X_train, y_train, X_s[valid_s])
                baselines[f"ridge_raw_{split_name}"] = {
                    "r_squared": float(r_squared(y_s[valid_s], preds_s)),
                    "ic": float(_safe_ic(preds_s, y_s[valid_s])),
                    "da": float(directional_accuracy(y_s[valid_s], preds_s)),
                }

    # G4 gate
    all_das = [
        baselines.get("ridge_raw_14", {}).get("da", 0),
        baselines.get("ridge_temporal", {}).get("da", 0),
        best_gb_da,
    ]
    best_da = max(all_das) if all_das else 0
    baselines["gate_G4"] = {
        "metric": "best_baseline_DA",
        "value": float(best_da),
        "threshold": 0.52,
        "verdict": "PASS" if best_da > 0.52 else "FAIL",
    }

    log(f"  Domain 8 done in {time.time() - t0:.1f}s")
    return baselines


# =============================================================================
# Domain 9: Cost-Adjusted Tradability
# =============================================================================

def analyze_tradability(d1, d3, d7):
    """Per-horizon cost analysis, tradability verdicts."""
    log("Domain 9: Tradability analysis...")
    result = {"per_horizon": {}}
    stability = d7.get("stability_ratio", 0)

    for hi, h in enumerate(HORIZONS):
        h_key = f"H{h}"
        h_stats = d1.get("per_horizon", {}).get(h_key, {})
        ret_std = h_stats.get("std", 0)
        if ret_std <= 0:
            continue

        # Best raw IC at this horizon
        best_ic = 0
        best_feat = None
        for name in LOCAL_NAMES:
            ic_val = d3.get("ic_matrix", {}).get(name, {}).get(h_key)
            if ic_val is not None and abs(ic_val) > abs(best_ic):
                best_ic = ic_val
                best_feat = name

        be_deep = breakeven_ic(COST_DEEP_ITM_BPS, ret_std)
        be_atm = breakeven_ic(COST_ATM_BPS, ret_std)
        net_deep = cost_adjusted_ic(abs(best_ic), COST_DEEP_ITM_BPS, ret_std)
        net_atm = cost_adjusted_ic(abs(best_ic), COST_ATM_BPS, ret_std)
        e_ret = expected_return_bps(abs(best_ic), ret_std)

        tradeable = net_deep > 0 and stability > 2.0
        result["per_horizon"][h_key] = {
            "return_std": float(ret_std),
            "best_ic": float(best_ic),
            "best_feature": best_feat,
            "breakeven_ic_deep_itm": float(be_deep),
            "breakeven_ic_atm": float(be_atm),
            "net_ic_deep_itm": float(net_deep),
            "net_ic_atm": float(net_atm),
            "expected_return_bps": float(e_ret),
            "tradeable_deep_itm": tradeable,
        }

    # Recommend best horizon
    best_h = None
    best_score = -999
    for h_key, info in result["per_horizon"].items():
        score = info["net_ic_deep_itm"] * stability
        if score > best_score:
            best_score = score
            best_h = h_key
    result["recommended_horizon"] = best_h
    result["recommended_score"] = float(best_score) if best_h else 0

    log("  Domain 9 done")
    return result


# =============================================================================
# Domain 10: Synthesis & Recommendations
# =============================================================================

def synthesize_recommendations(d1, d2, d3, d4, d5, d6, d7, d8, d9):
    """GO/NO-GO verdict, training configuration recommendations."""
    log("Domain 10: Synthesis...")

    gates = {
        "G1": d3.get("gate_G1", {}),
        "G2": d1.get("gate_G2", {}),
        "G3": d7.get("gate_G3", {}),
        "G4": d8.get("gate_G4", {}),
    }
    all_pass = all(g.get("verdict") == "PASS" for g in gates.values())

    # Recommended features after pruning
    pruned = set()
    for rec in d5.get("pruning_recommendations", []):
        pruned.add(rec.get("drop"))
    # Drop spread_change_rate if forward IC < 0.02
    cf = d3.get("cf_decomposition", {}).get("spread_change_rate", {})
    if cf and abs(cf.get("forward_ic", 0)) < 0.02:
        pruned.add("spread_change_rate")
    recommended_features = [n for n in LOCAL_NAMES if n not in pruned]

    # Recommended rolling window per feature
    rolling_config = {}
    for name in recommended_features:
        sweep = d4.get("multi_window_sweep", {}).get(name, {})
        best = sweep.get("best_variant", "rolling_mean_5")
        rolling_config[name] = best

    # Monotone constraints from partial dependence
    monotone = {}
    for name in recommended_features:
        pd = d3.get("partial_dependence", {}).get(name, {})
        if pd.get("is_monotonic"):
            direction = pd.get("monotonic_direction", "none")
            if direction == "increasing":
                monotone[name] = 1
            elif direction == "decreasing":
                monotone[name] = -1

    # Huber delta
    h60_stats = d1.get("per_horizon", {}).get(
        f"H{HORIZONS[TARGET_HORIZON_IDX]}", {}
    )
    huber_delta = h60_stats.get("huber_delta", 10.0)

    # GradBoost config from grid search
    gb_best = d8.get("gradboost_best", {})

    result = {
        "decision_gates": gates,
        "all_gates_pass": all_pass,
        "verdict": "GO" if all_pass else "NO-GO",
        "recommended_features": recommended_features,
        "n_recommended_features": len(recommended_features),
        "pruned_features": list(pruned),
        "recommended_horizon": d9.get("recommended_horizon"),
        "huber_delta": float(huber_delta),
        "rolling_config": rolling_config,
        "monotone_constraints": monotone,
        "gradboost_config": {
            "max_depth": gb_best.get("max_depth", 5),
            "learning_rate": gb_best.get("learning_rate", 0.05),
            "n_estimators": 300,
            "subsample": 0.8,
            "min_child_weight": 50,
        } if gb_best else None,
        "expected_baseline_da": d8.get("gate_G4", {}).get("value"),
        "risk_factors": [],
    }

    # Risk factors
    drift = d2.get("monthly_drift", {})
    for name, info in drift.items():
        if abs(info.get("change_pct", 0)) > 50:
            result["risk_factors"].append(
                f"{name}: {info['change_pct']:.0f}% drift ({info['first_month']} to {info['last_month']})"
            )
    if "spread_change_rate" not in pruned:
        cf_scr = d3.get("cf_decomposition", {}).get("spread_change_rate", {})
        if cf_scr.get("cf_ratio", 0) > 10:
            result["risk_factors"].append(
                f"spread_change_rate: CF={cf_scr['cf_ratio']:.1f} (overwhelmingly contemporaneous)"
            )

    log("  Domain 10 done")
    return result


# =============================================================================
# Output Generation
# =============================================================================

def build_markdown_report(results):
    """Build a comprehensive Markdown report."""
    lines = []
    w = lines.append

    d10 = results.get("domain_10_recommendations", {})
    gates = d10.get("decision_gates", {})
    verdict = d10.get("verdict", "UNKNOWN")

    w("# Pre-Training Feature Analysis Report")
    w(f"\n> **Date**: {results.get('analysis_date', 'N/A')}")
    w(f"> **Export**: {results.get('export_dir', 'N/A')}")
    w(f"> **Train**: {results.get('n_train_days', 'N/A')} days | "
      f"**Val**: {results.get('n_val_days', 'N/A')} | "
      f"**Test**: {results.get('n_test_days', 'N/A')}")
    w(f"> **Verdict**: **{verdict}**")

    # Decision Gates
    w("\n## Decision Gates\n")
    w("| Gate | Metric | Value | Threshold | Verdict |")
    w("|------|--------|-------|-----------|---------|")
    for gname, ginfo in gates.items():
        val = ginfo.get("value")
        val_str = f"{val:.4f}" if isinstance(val, (int, float)) and val is not None else "N/A"
        w(f"| {gname} | {ginfo.get('metric', '')} | {val_str} | "
          f"{ginfo.get('threshold', '')} | **{ginfo.get('verdict', 'N/A')}** |")

    # Domain 1: Returns
    d1 = results.get("domain_1_returns", {})
    w("\n## Domain 1: Label/Return Distribution\n")
    w("| Horizon | Mean | Std | Skew | Kurt | Near-Zero (<1bps) | Huber δ |")
    w("|---------|------|-----|------|------|-------------------|---------|")
    for h in HORIZONS:
        h_key = f"H{h}"
        s = d1.get("per_horizon", {}).get(h_key, {})
        if s:
            w(f"| H={h} | {s.get('mean', 0):.3f} | {s.get('std', 0):.2f} | "
              f"{s.get('skewness', 0):.2f} | {s.get('kurtosis', 0):.1f} | "
              f"{s.get('near_zero_1bps', 0):.1%} | {s.get('huber_delta', 0):.1f} |")

    # Domain 3: Signal
    d3 = results.get("domain_3_signal", {})
    w("\n## Domain 3: Feature-Return Signal\n")
    w("### IC Matrix at Key Horizons\n")
    w("| Feature | H=1 | H=5 | H=10 | H=30 | H=60 | dCor(H60) | CF |")
    w("|---------|-----|-----|------|------|------|-----------|-----|")
    for name in LOCAL_NAMES:
        ic_row = d3.get("ic_matrix", {}).get(name, {})
        dcor = d3.get("dcor_h60", {}).get(name, 0)
        cf_info = d3.get("cf_decomposition", {}).get(name, {})
        cf_ratio = cf_info.get("cf_ratio", "N/A")
        vals = []
        for h in [1, 5, 10, 30, 60]:
            v = ic_row.get(f"H{h}")
            vals.append(f"{v:.4f}" if v is not None else "—")
        cf_str = f"{cf_ratio:.1f}" if isinstance(cf_ratio, (int, float)) else cf_ratio
        w(f"| {name} | {' | '.join(vals)} | {dcor:.3f} | {cf_str} |")

    # Partial dependence summary
    w("\n### Partial Dependence (H=60)\n")
    for name in LOCAL_NAMES[:5]:  # Top 5 features
        pd = d3.get("partial_dependence", {}).get(name, {})
        if pd:
            shape = pd.get("monotonic_direction", "unknown")
            w(f"- **{name}**: {shape}")

    # Domain 5: Redundancy
    d5 = results.get("domain_5_redundancy", {})
    w("\n## Domain 5: Redundancy\n")
    w(f"- PCA: {d5.get('pca', {}).get('n_components_90', 'N/A')} components for 90% variance")
    pairs = d5.get("highly_correlated_pairs", [])
    if pairs:
        w("\n**Highly correlated pairs (|r| > 0.8)**:\n")
        for p in pairs[:5]:
            w(f"- {p.get('signal_1', '')} ↔ {p.get('signal_2', '')}: "
              f"r = {p.get('correlation', 0):.3f}")

    # Domain 7: Walk-forward
    d7 = results.get("domain_7_walkforward", {})
    w("\n## Domain 7: Walk-Forward Stability\n")
    w(f"- Folds: {d7.get('n_folds', 'N/A')}")
    w(f"- Mean IC: {d7.get('mean_ic', 0):.4f}")
    w(f"- Stability ratio: {d7.get('stability_ratio', 0):.2f}")
    w(f"- Mean DA: {d7.get('mean_da', 0):.3f}")
    w(f"- Regime shifts: {d7.get('regime_shift_count', 'N/A')}")

    # Domain 8: Baselines
    d8 = results.get("domain_8_baselines", {})
    w("\n## Domain 8: Baselines\n")
    w("| Model | R² | IC | DA |")
    w("|-------|-----|-----|-----|")
    for model_key in ["persistence", "single_feature_spread_bps",
                       "ridge_raw_14", "ridge_temporal"]:
        info = d8.get(model_key, {})
        if info and "r_squared" in info:
            w(f"| {model_key} | {info['r_squared']:.4f} | "
              f"{info.get('ic', 0):.4f} | {info.get('da', 0):.3f} |")

    # Recommendations
    w("\n## Recommendations\n")
    w(f"- **Features**: {d10.get('n_recommended_features', 'N/A')} "
      f"(pruned: {', '.join(d10.get('pruned_features', []))})")
    w(f"- **Horizon**: {d10.get('recommended_horizon', 'N/A')}")
    w(f"- **Huber delta**: {d10.get('huber_delta', 'N/A'):.1f} bps")
    if d10.get("risk_factors"):
        w("\n**Risk factors**:")
        for rf in d10["risk_factors"]:
            w(f"- {rf}")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pre-training feature analysis for off-exchange KEEP+ features"
    )
    parser.add_argument("--export-dir", default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (20 train days, skip GradBoost grid)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)

    start = time.time()
    log(f"Pre-training analysis starting (seed={seed}, quick={args.quick})")
    log(f"Export: {args.export_dir}")
    log(f"xgboost available: {HAS_XGBOOST}")

    # Resolve paths relative to script location
    script_dir = Path(__file__).resolve().parent.parent
    export_path = (script_dir / args.export_dir).resolve()
    output_path = (script_dir / args.output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # Load splits
    train_loader = ExportLoader(str(export_path), "train")
    val_loader = ExportLoader(str(export_path), "val")
    test_loader = ExportLoader(str(export_path), "test")

    train_dates = train_loader.list_dates()
    val_dates = val_loader.list_dates()
    test_dates = test_loader.list_dates()

    if args.quick:
        train_dates = train_dates[:20]
        val_dates = val_dates[:5]
        test_dates = test_dates[:5]

    log(f"Train: {len(train_dates)} days, Val: {len(val_dates)}, "
        f"Test: {len(test_dates)}")
    log(f"Schema: {train_loader.schema.contract_version}, "
        f"Features: {train_loader.schema.n_features}, "
        f"Horizons: {train_loader.schema.horizons}")

    # === Pass 1: Streaming accumulation ===
    pass1 = run_streaming_pass(train_loader, train_dates, args.quick)

    # === Domain 1: Label analysis ===
    d1 = analyze_labels(pass1, val_loader, val_dates, test_loader, test_dates)

    # === Domain 2: Feature distributions ===
    d2 = analyze_feature_distributions(
        pass1, val_loader, val_dates, test_loader, test_dates
    )

    # === Domain 3: Signal analysis ===
    d3 = analyze_signal(pass1)

    # === Domain 4: Temporal analysis (Pass 2) ===
    d4 = analyze_temporal(train_loader, train_dates, pass1)

    # === Domain 5: Redundancy ===
    d5 = analyze_redundancy(pass1["pool_features"])

    # === Domain 6: Regime ===
    d6 = analyze_regime(pass1["pool_features"], pass1["pool_labels"])

    # === Domain 7: Walk-forward (Pass 3) ===
    d7 = walk_forward_analysis(train_loader, train_dates)

    # === Domain 8: Baselines ===
    d8 = baseline_models(
        train_loader, train_dates, val_loader, val_dates,
        test_loader, test_dates, d4,
    )

    # === Domain 9: Tradability ===
    d9 = analyze_tradability(d1, d3, d7)

    # === Domain 10: Synthesis ===
    d10 = synthesize_recommendations(d1, d2, d3, d4, d5, d6, d7, d8, d9)

    # === Write outputs ===
    results = {
        "schema": "pre_training_analysis_v1",
        "export_dir": str(export_path),
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "seed": seed,
        "n_train_days": len(train_dates),
        "n_val_days": len(val_dates),
        "n_test_days": len(test_dates),
        "target_horizon_idx": TARGET_HORIZON_IDX,
        "target_horizon_bins": HORIZONS[TARGET_HORIZON_IDX],
        "feature_set": KEEP_FEATURES,
        "domain_1_returns": d1,
        "domain_2_distributions": d2,
        "domain_3_signal": d3,
        "domain_4_temporal": d4,
        "domain_5_redundancy": d5,
        "domain_6_regime": d6,
        "domain_7_walkforward": d7,
        "domain_8_baselines": d8,
        "domain_9_tradability": d9,
        "domain_10_recommendations": d10,
    }

    json_path = output_path / "analysis.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    log(f"JSON written to {json_path}")

    md_report = build_markdown_report(results)
    md_path = output_path / "PRE_TRAINING_ANALYSIS_REPORT.md"
    with open(md_path, "w") as f:
        f.write(md_report)
    log(f"Report written to {md_path}")

    elapsed = time.time() - start
    log(f"\nTotal runtime: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    log(f"Verdict: {d10.get('verdict', 'UNKNOWN')}")


if __name__ == "__main__":
    main()
