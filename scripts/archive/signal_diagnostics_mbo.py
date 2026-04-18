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
Signal diagnostics for MBO spread_bps threshold strategy with POINT-RETURN labels.

E13 variant: MBO 98-feature export with point returns at 8 horizons.
Key difference from off-exchange: feature indices are MBO-specific (spread_bps=42),
session_progress derived from sequence position (no explicit feature).

Usage:
    python scripts/signal_diagnostics_mbo.py
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

from hft_metrics import spearman_ic, EPS
from hft_metrics.discretization import quantile_buckets


# =============================================================================
# Constants
# =============================================================================

SEED = 42
DEFAULT_EXPORT_DIR = "../data/exports/e5_timebased_60s_point_return"
DEFAULT_OUTPUT_DIR = "outputs/signal_diagnostics_mbo"
HORIZONS = (1, 2, 3, 5, 10, 20, 30, 60)
TARGET_HORIZONS = [3, 4, 5, 6, 7]  # indices for H=5,10,20,30,60
SPREAD_BPS_IDX = 42        # MBO spread_bps index (was 12 for off-exchange)
VOLUME_IMBALANCE_IDX = 45  # MBO volume_imbalance (CF=0.01, pure forward)
TRUE_OFI_IDX = 84          # MBO true_ofi
# MBO has no continuous session_progress feature — derived from sequence position
COST_DEEP_ITM_BPS = 1.4
THRESHOLDS = [70, 75, 80, 85, 90, 95]
WARMUP_DAYS = 20


def log(msg: str) -> None:
    print(f"[diagnostics] {msg}", flush=True)


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


# =============================================================================
# Data Loading
# =============================================================================

def load_split_data(loader, dates):
    """Load per-day spread_bps, derived session_progress, volume_imbalance, and labels."""
    data = {}
    for d in dates:
        b = loader.load_day(d)
        n = b.sequences.shape[0]
        data[d] = {
            "spread": np.asarray(b.sequences[:, -1, SPREAD_BPS_IDX], dtype=np.float64),
            # MBO has no continuous session_progress feature — derive from sequence position
            "session": np.linspace(0.0, 1.0, n, dtype=np.float64),
            "bbo_rate": np.asarray(b.sequences[:, -1, VOLUME_IMBALANCE_IDX], dtype=np.float64),
            "labels": np.asarray(b.labels, dtype=np.float64),
            "n": n,
        }
    return data


# =============================================================================
# Diagnostic 1: Multi-Horizon Threshold Backtest
# =============================================================================

def run_threshold_backtest(daily_data, dates, h_idx, horizon_bins, pct_threshold,
                           warmup=WARMUP_DAYS, cost_bps=COST_DEEP_ITM_BPS,
                           session_filter=None):
    """Walk-forward threshold backtest with non-overlapping trades.

    Args:
        daily_data: dict of per-day data from load_split_data
        dates: ordered date list
        h_idx: label column index
        horizon_bins: number of bins for the horizon (for non-overlap spacing)
        pct_threshold: percentile threshold (e.g., 85)
        warmup: trailing days for threshold computation
        cost_bps: round-trip cost in bps
        session_filter: optional (min, max) tuple for session_progress filtering

    Returns:
        dict with trade-level results
    """
    trades = []

    for di in range(warmup, len(dates)):
        # Trailing window threshold
        trailing_spreads = np.concatenate([
            daily_data[dates[j]]["spread"]
            for j in range(max(0, di - warmup), di)
        ])
        threshold = np.percentile(trailing_spreads, pct_threshold)

        today = daily_data[dates[di]]
        spread = today["spread"]
        labels = today["labels"][:, h_idx]
        session = today["session"]
        n = today["n"]

        # Scan for non-overlapping trade entries
        last_entry = -horizon_bins
        for idx in range(n):
            if idx - last_entry < horizon_bins:
                continue
            if not np.isfinite(spread[idx]) or not np.isfinite(labels[idx]):
                continue
            if spread[idx] <= threshold:
                continue
            if session_filter is not None:
                sp = session[idx]
                if not (session_filter[0] <= sp <= session_filter[1]):
                    continue

            trades.append({
                "date": dates[di],
                "bin_idx": int(idx),
                "spread_bps": float(spread[idx]),
                "session_progress": float(session[idx]),
                "gross_return": float(labels[idx]),
                "net_return": float(labels[idx]) - cost_bps,
                "threshold": float(threshold),
            })
            last_entry = idx

    return trades


def run_unconditional_backtest(daily_data, dates, h_idx, horizon_bins,
                                warmup=WARMUP_DAYS, cost_bps=COST_DEEP_ITM_BPS):
    """Unconditional long every horizon_bins bins (baseline comparison)."""
    trades = []
    for di in range(warmup, len(dates)):
        today = daily_data[dates[di]]
        labels = today["labels"][:, h_idx]
        n = today["n"]
        last_entry = -horizon_bins
        for idx in range(n):
            if idx - last_entry < horizon_bins:
                continue
            if not np.isfinite(labels[idx]):
                continue
            trades.append({
                "date": dates[di],
                "gross_return": float(labels[idx]),
                "net_return": float(labels[idx]) - cost_bps,
            })
            last_entry = idx
    return trades


def summarize_trades(trades, label=""):
    """Compute summary statistics for a list of trades."""
    if not trades:
        return {"n": 0, "label": label}

    net = np.array([t["net_return"] for t in trades])
    gross = np.array([t["gross_return"] for t in trades])
    n = len(net)

    # t-test
    t_stat, p_val = scipy_stats.ttest_1samp(net, 0.0) if n > 2 else (0.0, 1.0)

    # Cumulative P&L
    cum_pnl = np.cumsum(net)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - peak
    max_dd = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

    # Consecutive losses
    is_loss = net < 0
    max_consec_loss = 0
    current_streak = 0
    for loss in is_loss:
        if loss:
            current_streak += 1
            max_consec_loss = max(max_consec_loss, current_streak)
        else:
            current_streak = 0

    return {
        "label": label,
        "n": n,
        "mean_gross": float(np.mean(gross)),
        "mean_net": float(np.mean(net)),
        "std_net": float(np.std(net)),
        "median_net": float(np.median(net)),
        "win_rate": float(np.mean(net > 0)),
        "win_rate_gross": float(np.mean(gross > 0)),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "skewness": float(scipy_stats.skew(net)),
        "kurtosis": float(scipy_stats.kurtosis(net)),
        "max_net": float(np.max(net)),
        "min_net": float(np.min(net)),
        "pct_5": float(np.percentile(net, 5)),
        "pct_95": float(np.percentile(net, 95)),
        "total_pnl": float(np.sum(net)),
        "max_drawdown_bps": max_dd,
        "max_consecutive_losses": max_consec_loss,
        "cum_pnl_final": float(cum_pnl[-1]) if len(cum_pnl) > 0 else 0.0,
    }


def diagnostic_1_multi_horizon(train_data, train_dates):
    """Multi-horizon threshold backtest at P85 and P90."""
    log("Diagnostic 1: Multi-horizon threshold backtest...")
    t0 = time.time()
    results = {}

    for h_idx in range(len(HORIZONS)):
        h = HORIZONS[h_idx]
        horizon_bins = h  # Non-overlap spacing = horizon

        for pct in [80, 85, 90]:
            trades = run_threshold_backtest(
                train_data, train_dates, h_idx, horizon_bins, pct
            )
            summary = summarize_trades(trades, f"H{h}_P{pct}")

            # Also run unconditional for this horizon
            if pct == 85:  # Only once per horizon
                uncond = run_unconditional_backtest(
                    train_data, train_dates, h_idx, horizon_bins
                )
                uncond_summary = summarize_trades(uncond, f"H{h}_unconditional")
                results[f"H{h}_unconditional"] = uncond_summary

            results[f"H{h}_P{pct}"] = summary

    # Find best horizon × threshold combination
    best_key = None
    best_t = -999
    for key, info in results.items():
        if "unconditional" in key:
            continue
        if info.get("t_stat", 0) > best_t and info.get("n", 0) >= 50:
            best_t = info["t_stat"]
            best_key = key

    results["best_combination"] = best_key
    results["best_t_stat"] = float(best_t) if best_key else 0.0

    log(f"  Diagnostic 1 done in {time.time() - t0:.1f}s. Best: {best_key} (t={best_t:.3f})")
    return results


# =============================================================================
# Diagnostic 2: Val/Test OOS Threshold Backtest
# =============================================================================

def diagnostic_2_oos(train_data, train_dates, val_data, val_dates,
                     test_data, test_dates, best_h_idx, best_pct):
    """OOS threshold backtest using threshold from trailing train days."""
    log("Diagnostic 2: Val/Test OOS backtest...")
    t0 = time.time()
    results = {}

    h = HORIZONS[best_h_idx]
    horizon_bins = h

    # Strategy A: Fixed threshold from last 20 train days
    last_20_spreads = np.concatenate([
        train_data[d]["spread"] for d in train_dates[-20:]
    ])
    fixed_threshold = np.percentile(last_20_spreads, best_pct)
    results["fixed_threshold"] = float(fixed_threshold)

    for split_name, split_data, split_dates in [
        ("val", val_data, val_dates),
        ("test", test_data, test_dates),
    ]:
        if not split_dates:
            continue

        # Fixed threshold approach
        fixed_trades = []
        uncond_trades = []
        for d in split_dates:
            today = split_data[d]
            spread = today["spread"]
            labels = today["labels"][:, best_h_idx]
            last_entry = -horizon_bins
            last_entry_u = -horizon_bins
            for idx in range(today["n"]):
                if np.isfinite(labels[idx]):
                    # Unconditional
                    if idx - last_entry_u >= horizon_bins:
                        uncond_trades.append({
                            "date": d,
                            "gross_return": float(labels[idx]),
                            "net_return": float(labels[idx]) - COST_DEEP_ITM_BPS,
                        })
                        last_entry_u = idx
                    # Conditional
                    if (np.isfinite(spread[idx]) and spread[idx] > fixed_threshold
                            and idx - last_entry >= horizon_bins):
                        fixed_trades.append({
                            "date": d,
                            "gross_return": float(labels[idx]),
                            "net_return": float(labels[idx]) - COST_DEEP_ITM_BPS,
                        })
                        last_entry = idx

        results[f"{split_name}_fixed"] = summarize_trades(
            fixed_trades, f"{split_name}_fixed_P{best_pct}"
        )
        results[f"{split_name}_unconditional"] = summarize_trades(
            uncond_trades, f"{split_name}_unconditional"
        )

        # Strategy B: Rolling threshold (update every day with trailing 20)
        all_dates_for_rolling = train_dates[-20:] + split_dates
        all_data_for_rolling = {**train_data, **split_data}
        rolling_trades = run_threshold_backtest(
            all_data_for_rolling, all_dates_for_rolling, best_h_idx,
            horizon_bins, best_pct, warmup=20,
        )
        # Only keep trades from split dates
        split_date_set = set(split_dates)
        rolling_trades = [t for t in rolling_trades if t["date"] in split_date_set]
        results[f"{split_name}_rolling"] = summarize_trades(
            rolling_trades, f"{split_name}_rolling_P{best_pct}"
        )

    log(f"  Diagnostic 2 done in {time.time() - t0:.1f}s")
    return results


# =============================================================================
# Diagnostic 3: Regime Robustness
# =============================================================================

def diagnostic_3_regime(train_data, train_dates):
    """IC robustness across market regimes."""
    log("Diagnostic 3: Regime robustness...")
    t0 = time.time()
    results = {}

    # Compute per-day IC and daily properties
    daily_records = []
    for d in train_dates:
        today = train_data[d]
        spread = today["spread"]
        for h_idx in range(len(HORIZONS)):
            h = HORIZONS[h_idx]
            labels = today["labels"][:, h_idx]
            valid = np.isfinite(spread) & np.isfinite(labels)
            if valid.sum() < 20:
                continue
            rho, p = spearman_ic(spread[valid], labels[valid])
            if rho == 0.0 and p == 1.0:
                continue

            if h_idx == 7:  # H=60
                h1_labels = today["labels"][:, 0]
                h1_valid = np.isfinite(h1_labels)
                daily_mean_ret = float(np.mean(labels[valid]))
                daily_std_h1 = float(np.std(h1_labels[h1_valid])) if h1_valid.sum() > 0 else 0
                daily_mean_spread = float(np.mean(spread[valid]))

                # Demeaned IC
                demeaned = labels[valid] - np.mean(labels[valid])
                rho_dm, _ = spearman_ic(spread[valid], demeaned)

                daily_records.append({
                    "date": d,
                    "ic_signed": float(rho),
                    "ic_demeaned": float(rho_dm) if rho_dm != 0.0 or True else 0.0,
                    "daily_mean_ret": daily_mean_ret,
                    "daily_std_h1": daily_std_h1,
                    "daily_mean_spread": daily_mean_spread,
                    "is_up_day": daily_mean_ret > 0,
                    "month": f"{d[:4]}-{d[4:6]}",
                })

    records = daily_records
    ics = np.array([r["ic_signed"] for r in records])
    ics_dm = np.array([r["ic_demeaned"] for r in records])

    # A1: Up-day vs Down-day
    up_ics = np.array([r["ic_signed"] for r in records if r["is_up_day"]])
    down_ics = np.array([r["ic_signed"] for r in records if not r["is_up_day"]])
    t_down, p_down = scipy_stats.ttest_1samp(down_ics, 0.0) if len(down_ics) > 2 else (0, 1)
    results["up_vs_down"] = {
        "up_days": {"n": len(up_ics), "mean_ic": float(np.mean(up_ics)),
                     "std_ic": float(np.std(up_ics)),
                     "sign_flip": float(np.mean(up_ics < 0))},
        "down_days": {"n": len(down_ics), "mean_ic": float(np.mean(down_ics)),
                       "std_ic": float(np.std(down_ics)),
                       "sign_flip": float(np.mean(down_ics < 0)),
                       "t_stat": float(t_down), "p_value": float(p_down)},
    }

    # A2: High-vol vs Low-vol (using H=1 std as volatility proxy)
    vols = np.array([r["daily_std_h1"] for r in records])
    med_vol = np.median(vols)
    hivol_ics = np.array([r["ic_signed"] for r in records if r["daily_std_h1"] > med_vol])
    lovol_ics = np.array([r["ic_signed"] for r in records if r["daily_std_h1"] <= med_vol])
    results["high_vs_low_vol"] = {
        "high_vol": {"n": len(hivol_ics), "mean_ic": float(np.mean(hivol_ics))},
        "low_vol": {"n": len(lovol_ics), "mean_ic": float(np.mean(lovol_ics))},
    }

    # A3: Temporal halves
    mid = len(records) // 2
    first_ics = ics[:mid]
    second_ics = ics[mid:]
    results["temporal_halves"] = {
        "first_half": {"n": len(first_ics), "mean_ic": float(np.mean(first_ics)),
                        "dates": f"{records[0]['date']} to {records[mid-1]['date']}"},
        "second_half": {"n": len(second_ics), "mean_ic": float(np.mean(second_ics)),
                         "dates": f"{records[mid]['date']} to {records[-1]['date']}"},
    }

    # A4: Monthly breakdown
    monthly = defaultdict(list)
    for r in records:
        monthly[r["month"]].append(r["ic_signed"])
    results["monthly"] = {
        m: {"n": len(v), "mean_ic": float(np.mean(v)), "std_ic": float(np.std(v))}
        for m, v in sorted(monthly.items())
    }

    # A5: Spread regime (daily mean spread above/below median)
    med_spread = np.median([r["daily_mean_spread"] for r in records])
    wide_ics = np.array([r["ic_signed"] for r in records if r["daily_mean_spread"] > med_spread])
    tight_ics = np.array([r["ic_signed"] for r in records if r["daily_mean_spread"] <= med_spread])
    results["spread_regime"] = {
        "wide_spread": {"n": len(wide_ics), "mean_ic": float(np.mean(wide_ics))},
        "tight_spread": {"n": len(tight_ics), "mean_ic": float(np.mean(tight_ics))},
    }

    # A6: Drift-adjusted IC
    results["drift_adjusted"] = {
        "raw_ic_mean": float(np.mean(ics)),
        "demeaned_ic_mean": float(np.mean(ics_dm)),
        "ic_loss_pct": float((1 - abs(np.mean(ics_dm)) / max(abs(np.mean(ics)), EPS)) * 100),
    }

    # Multi-horizon one-sidedness
    one_sided = {}
    for h_idx, h in enumerate(HORIZONS):
        if h_idx not in [3, 4, 5, 6, 7]:  # H=5,10,20,30,60
            continue
        signed_ics, abs_ics, pos_ics, neg_ics = [], [], [], []
        for d in train_dates:
            today = train_data[d]
            spread = today["spread"]
            labels = today["labels"][:, h_idx]
            valid = np.isfinite(spread) & np.isfinite(labels)
            if valid.sum() < 20:
                continue
            s, r = spread[valid], labels[valid]

            rho_s, _ = spearman_ic(s, r)
            if not (rho_s == 0.0):
                signed_ics.append(rho_s)

            rho_a, _ = spearman_ic(s, np.abs(r))
            if not (rho_a == 0.0):
                abs_ics.append(rho_a)

            pos_mask = r > 0
            if pos_mask.sum() >= 20:
                rho_p, _ = spearman_ic(s[pos_mask], r[pos_mask])
                pos_ics.append(rho_p)

            neg_mask = r < 0
            if neg_mask.sum() >= 20:
                rho_n, _ = spearman_ic(s[neg_mask], r[neg_mask])
                neg_ics.append(rho_n)

        one_sided[f"H{h}"] = {
            "ic_signed": float(np.mean(signed_ics)) if signed_ics else 0,
            "ic_abs": float(np.mean(abs_ics)) if abs_ics else 0,
            "ic_pos": float(np.mean(pos_ics)) if pos_ics else 0,
            "ic_neg": float(np.mean(neg_ics)) if neg_ics else 0,
        }

    results["one_sidedness"] = one_sided

    log(f"  Diagnostic 3 done in {time.time() - t0:.1f}s")
    return results


# =============================================================================
# Diagnostic 4: Trade Return Distribution
# =============================================================================

def diagnostic_4_distribution(trades):
    """Detailed distribution analysis of trade returns."""
    log("Diagnostic 4: Trade distribution analysis...")
    if not trades:
        return {"error": "no trades"}

    net = np.array([t["net_return"] for t in trades])
    gross = np.array([t["gross_return"] for t in trades])

    # Histogram bins
    edges = np.arange(-300, 301, 20)
    hist, _ = np.histogram(net, bins=edges)

    # Outlier analysis
    p1, p99 = np.percentile(net, [1, 99])
    outlier_mask = (net < p1) | (net > p99)
    n_outliers = int(outlier_mask.sum())
    mean_without_outliers = float(np.mean(net[~outlier_mask])) if (~outlier_mask).sum() > 0 else 0

    # Win/loss size asymmetry
    wins = net[net > 0]
    losses = net[net < 0]
    mean_win = float(np.mean(wins)) if len(wins) > 0 else 0
    mean_loss = float(np.mean(losses)) if len(losses) > 0 else 0

    # Time-of-day breakdown
    tod_buckets = {"morning": [], "midday": [], "afternoon": []}
    for t in trades:
        sp = t.get("session_progress", 0.5)
        if sp < 0.33:
            tod_buckets["morning"].append(t["net_return"])
        elif sp < 0.67:
            tod_buckets["midday"].append(t["net_return"])
        else:
            tod_buckets["afternoon"].append(t["net_return"])

    tod_results = {}
    for period, rets in tod_buckets.items():
        if rets:
            arr = np.array(rets)
            tod_results[period] = {
                "n": len(arr),
                "mean_net": float(np.mean(arr)),
                "win_rate": float(np.mean(arr > 0)),
            }

    return {
        "histogram_edges": edges.tolist(),
        "histogram_counts": hist.tolist(),
        "n_outliers_1pct": n_outliers,
        "mean_without_outliers": mean_without_outliers,
        "mean_win": mean_win,
        "mean_loss": mean_loss,
        "win_loss_ratio": abs(mean_win / min(mean_loss, -EPS)),
        "time_of_day": tod_results,
    }


# =============================================================================
# Diagnostic 5: Cumulative P&L + Drawdown
# =============================================================================

def diagnostic_5_pnl_curve(trades):
    """Cumulative P&L and drawdown analysis."""
    log("Diagnostic 5: P&L curve analysis...")
    if not trades:
        return {"error": "no trades"}

    net = np.array([t["net_return"] for t in trades])
    dates = [t["date"] for t in trades]

    cum_pnl = np.cumsum(net)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - peak

    # Monthly P&L
    monthly_pnl = defaultdict(float)
    monthly_trades = defaultdict(int)
    for t in trades:
        m = t["date"][:7]
        monthly_pnl[m] += t["net_return"]
        monthly_trades[m] += 1

    # Recovery analysis
    in_drawdown = drawdown < 0
    dd_start = None
    max_recovery_trades = 0
    current_dd_trades = 0
    for i in range(len(drawdown)):
        if drawdown[i] < 0:
            current_dd_trades += 1
        else:
            max_recovery_trades = max(max_recovery_trades, current_dd_trades)
            current_dd_trades = 0

    return {
        "cum_pnl_final": float(cum_pnl[-1]),
        "max_drawdown_bps": float(np.min(drawdown)),
        "max_peak_bps": float(np.max(cum_pnl)),
        "max_recovery_trades": max_recovery_trades,
        "monthly_pnl": {m: {"pnl": float(v), "trades": monthly_trades[m]}
                         for m, v in sorted(monthly_pnl.items())},
        "positive_months": sum(1 for v in monthly_pnl.values() if v > 0),
        "total_months": len(monthly_pnl),
    }


# =============================================================================
# Diagnostic 6: Statistical Power
# =============================================================================

def diagnostic_6_power(summary, horizons_results):
    """Power analysis and required sample size."""
    log("Diagnostic 6: Statistical power analysis...")

    mean_net = summary.get("mean_net", 0)
    std_net = summary.get("std_net", 1)
    n = summary.get("n", 0)

    # How many trades needed for significance at current effect size?
    if abs(mean_net) > EPS and std_net > EPS:
        # n needed for 80% power at alpha=0.05 (two-sided)
        z_alpha = 1.96
        z_beta = 0.84
        n_needed = ((z_alpha + z_beta) * std_net / mean_net) ** 2
        trades_per_day = n / max(146, 1)  # approximate
        days_needed = n_needed / max(trades_per_day, 0.1)
    else:
        n_needed = float("inf")
        days_needed = float("inf")

    # Minimum detectable effect at current n
    min_detectable = 1.96 * std_net / max(np.sqrt(n), 1)

    # Projected Sharpe (annualized)
    if std_net > EPS and n > 0:
        sharpe_per_trade = mean_net / std_net
        trades_per_year = (n / 146) * 252  # extrapolate
        annual_sharpe = sharpe_per_trade * np.sqrt(trades_per_year)
    else:
        sharpe_per_trade = 0
        annual_sharpe = 0

    # Multi-horizon power comparison
    best_power_h = None
    best_power_score = 0
    for key, info in horizons_results.items():
        if "unconditional" in key or not isinstance(info, dict):
            continue
        n_h = info.get("n", 0)
        mean_h = info.get("mean_net", 0)
        std_h = info.get("std_net", 1)
        if n_h > 20 and std_h > EPS:
            t_h = abs(mean_h) / std_h * np.sqrt(n_h)
            if t_h > best_power_score:
                best_power_score = t_h
                best_power_h = key

    return {
        "current_n": n,
        "current_mean_net": float(mean_net),
        "current_std_net": float(std_net),
        "current_t_stat": float(mean_net / std_net * np.sqrt(n)) if std_net > EPS and n > 0 else 0,
        "min_detectable_bps": float(min_detectable),
        "n_needed_for_significance": float(n_needed),
        "days_needed": float(days_needed),
        "sharpe_per_trade": float(sharpe_per_trade),
        "projected_annual_sharpe": float(annual_sharpe),
        "best_power_horizon": best_power_h,
        "best_power_t_stat": float(best_power_score),
    }


# =============================================================================
# Diagnostic 7: Feature Conditioning (session_progress)
# =============================================================================

def diagnostic_7_conditioning(train_data, train_dates, best_h_idx, best_pct):
    """Test whether adding session_progress filter improves results."""
    log("Diagnostic 7: Feature conditioning...")
    t0 = time.time()

    h = HORIZONS[best_h_idx]
    horizon_bins = h
    results = {}

    # Baseline: spread-only threshold
    baseline_trades = run_threshold_backtest(
        train_data, train_dates, best_h_idx, horizon_bins, best_pct
    )
    results["baseline"] = summarize_trades(baseline_trades, f"spread_only_P{best_pct}")

    # Test session_progress filters
    session_filters = [
        ("morning", (0.0, 0.33)),
        ("midday", (0.33, 0.67)),
        ("afternoon", (0.67, 1.0)),
        ("not_close", (0.0, 0.85)),
        ("open_30min", (0.0, 0.08)),
    ]

    for filter_name, (lo, hi) in session_filters:
        trades = run_threshold_backtest(
            train_data, train_dates, best_h_idx, horizon_bins, best_pct,
            session_filter=(lo, hi),
        )
        results[f"session_{filter_name}"] = summarize_trades(
            trades, f"spread_P{best_pct}+{filter_name}"
        )

    log(f"  Diagnostic 7 done in {time.time() - t0:.1f}s")
    return results


# =============================================================================
# Markdown Report
# =============================================================================

def build_report(results):
    lines = []
    w = lines.append

    w("# Signal Diagnostics Report")
    w(f"\n> **Date**: {results.get('analysis_date', 'N/A')}")
    w(f"> **Export**: {results.get('export_dir', 'N/A')}")
    w(f"> **Train**: {results.get('n_train_days', 0)} days | "
      f"**Val**: {results.get('n_val_days', 0)} | "
      f"**Test**: {results.get('n_test_days', 0)}")

    # Diagnostic 1: Multi-horizon
    d1 = results.get("d1_multi_horizon", {})
    w("\n## Diagnostic 1: Multi-Horizon Threshold Backtest\n")
    w("| Horizon | Threshold | Trades | Mean Net | Std | Win% | t-stat | p |")
    w("|---------|-----------|--------|----------|-----|------|--------|---|")
    for h in HORIZONS:
        for pct in [80, 85, 90]:
            key = f"H{h}_P{pct}"
            s = d1.get(key, {})
            if s and s.get("n", 0) > 0:
                w(f"| H={h} | P{pct} | {s['n']} | {s['mean_net']:+.2f} | "
                  f"{s['std_net']:.1f} | {s['win_rate']:.3f} | "
                  f"{s['t_stat']:+.3f} | {s['p_value']:.3f} |")
        # Unconditional
        ukey = f"H{h}_unconditional"
        u = d1.get(ukey, {})
        if u and u.get("n", 0) > 0:
            w(f"| H={h} | uncond | {u['n']} | {u['mean_net']:+.2f} | "
              f"{u['std_net']:.1f} | {u['win_rate']:.3f} | "
              f"{u['t_stat']:+.3f} | {u['p_value']:.3f} |")

    best = d1.get("best_combination", "N/A")
    w(f"\n**Best combination**: {best} (t={d1.get('best_t_stat', 0):.3f})")

    # Diagnostic 2: OOS
    d2 = results.get("d2_oos", {})
    w("\n## Diagnostic 2: Val/Test Out-of-Sample\n")
    for split in ["val", "test"]:
        for variant in ["fixed", "rolling", "unconditional"]:
            key = f"{split}_{variant}"
            s = d2.get(key, {})
            if s and s.get("n", 0) > 0:
                w(f"- **{split} {variant}**: n={s['n']}, mean_net={s['mean_net']:+.2f}, "
                  f"win={s['win_rate']:.3f}, t={s['t_stat']:+.3f}")

    # Diagnostic 3: Regime
    d3 = results.get("d3_regime", {})
    w("\n## Diagnostic 3: Regime Robustness\n")

    ud = d3.get("up_vs_down", {})
    up = ud.get("up_days", {})
    down = ud.get("down_days", {})
    w(f"- **UP days**: n={up.get('n', 0)}, IC={up.get('mean_ic', 0):+.4f}, "
      f"sign_flip={up.get('sign_flip', 0):.3f}")
    w(f"- **DOWN days**: n={down.get('n', 0)}, IC={down.get('mean_ic', 0):+.4f}, "
      f"sign_flip={down.get('sign_flip', 0):.3f}, p={down.get('p_value', 1):.4f}")

    da = d3.get("drift_adjusted", {})
    w(f"- **Drift-adjusted IC**: {da.get('demeaned_ic_mean', 0):+.4f} "
      f"(loss: {da.get('ic_loss_pct', 0):.1f}%)")

    # Monthly
    monthly = d3.get("monthly", {})
    if monthly:
        w("\n**Monthly IC**:\n")
        w("| Month | Days | IC | Std |")
        w("|-------|------|----|-----|")
        for m, info in monthly.items():
            w(f"| {m} | {info['n']} | {info['mean_ic']:+.4f} | {info['std_ic']:.4f} |")

    # One-sidedness
    os = d3.get("one_sidedness", {})
    if os:
        w("\n**One-Sidedness Across Horizons**:\n")
        w("| Horizon | IC(signed) | IC(|ret|) | IC(ret>0) | IC(ret<0) |")
        w("|---------|-----------|----------|----------|----------|")
        for h_key in sorted(os.keys()):
            info = os[h_key]
            w(f"| {h_key} | {info['ic_signed']:+.4f} | {info['ic_abs']:+.4f} | "
              f"{info['ic_pos']:+.4f} | {info['ic_neg']:+.4f} |")

    # Diagnostic 5: P&L
    d5 = results.get("d5_pnl", {})
    w("\n## Diagnostic 5: Cumulative P&L\n")
    w(f"- Final cum P&L: {d5.get('cum_pnl_final', 0):+.1f} bps")
    w(f"- Max drawdown: {d5.get('max_drawdown_bps', 0):.1f} bps")
    w(f"- Positive months: {d5.get('positive_months', 0)}/{d5.get('total_months', 0)}")
    monthly_pnl = d5.get("monthly_pnl", {})
    if monthly_pnl:
        w("\n| Month | P&L (bps) | Trades |")
        w("|-------|-----------|--------|")
        for m, info in monthly_pnl.items():
            w(f"| {m} | {info['pnl']:+.1f} | {info['trades']} |")

    # Diagnostic 6: Power
    d6 = results.get("d6_power", {})
    w("\n## Diagnostic 6: Statistical Power\n")
    w(f"- Current t-stat: {d6.get('current_t_stat', 0):.3f}")
    w(f"- Min detectable effect: {d6.get('min_detectable_bps', 0):.1f} bps")
    w(f"- Trades needed for significance: {d6.get('n_needed_for_significance', 0):.0f}")
    w(f"- Days needed: {d6.get('days_needed', 0):.0f}")
    w(f"- Projected annual Sharpe: {d6.get('projected_annual_sharpe', 0):.2f}")
    w(f"- Best power horizon: {d6.get('best_power_horizon', 'N/A')} "
      f"(t={d6.get('best_power_t_stat', 0):.3f})")

    # Diagnostic 7: Conditioning
    d7 = results.get("d7_conditioning", {})
    w("\n## Diagnostic 7: Session Conditioning\n")
    w("| Filter | Trades | Mean Net | Win% | t-stat |")
    w("|--------|--------|----------|------|--------|")
    for key, info in sorted(d7.items()):
        if isinstance(info, dict) and info.get("n", 0) > 0:
            w(f"| {info.get('label', key)} | {info['n']} | "
              f"{info['mean_net']:+.2f} | {info['win_rate']:.3f} | "
              f"{info['t_stat']:+.3f} |")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Signal diagnostics")
    parser.add_argument("--export-dir", default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    np.random.seed(SEED)
    start = time.time()

    script_dir = Path(__file__).resolve().parent.parent
    export_path = (script_dir / args.export_dir).resolve()
    output_path = (script_dir / args.output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    log("Signal diagnostics starting...")
    log(f"Export: {export_path}")

    # Load all splits
    train_loader = ExportLoader(str(export_path), "train")
    val_loader = ExportLoader(str(export_path), "val")
    test_loader = ExportLoader(str(export_path), "test")

    train_dates = train_loader.list_dates()
    val_dates = val_loader.list_dates()
    test_dates = test_loader.list_dates()

    log(f"Train: {len(train_dates)}, Val: {len(val_dates)}, Test: {len(test_dates)}")

    # Load data
    log("Loading train data...")
    train_data = load_split_data(train_loader, train_dates)
    log("Loading val data...")
    val_data = load_split_data(val_loader, val_dates)
    log("Loading test data...")
    test_data = load_split_data(test_loader, test_dates)

    # === Diagnostic 1: Multi-horizon threshold backtest ===
    d1 = diagnostic_1_multi_horizon(train_data, train_dates)

    # Determine best horizon and threshold for subsequent diagnostics
    best_key = d1.get("best_combination", "H60_P85")
    # Parse H and P from key like "H20_P85"
    parts = best_key.split("_") if best_key else ["H60", "P85"]
    best_h = int(parts[0][1:]) if len(parts) >= 1 else 60
    best_pct = int(parts[1][1:]) if len(parts) >= 2 else 85
    best_h_idx = list(HORIZONS).index(best_h) if best_h in HORIZONS else 7
    log(f"Best combination: H={best_h}, P{best_pct}")

    # === Diagnostic 2: Val/Test OOS ===
    d2 = diagnostic_2_oos(train_data, train_dates, val_data, val_dates,
                           test_data, test_dates, best_h_idx, best_pct)

    # === Diagnostic 3: Regime robustness ===
    d3 = diagnostic_3_regime(train_data, train_dates)

    # === Get the trades for best combination for D4-D6 ===
    best_trades = run_threshold_backtest(
        train_data, train_dates, best_h_idx, best_h, best_pct
    )
    best_summary = summarize_trades(best_trades, f"H{best_h}_P{best_pct}")

    # === Diagnostic 4: Trade distribution ===
    d4 = diagnostic_4_distribution(best_trades)

    # === Diagnostic 5: P&L curve ===
    d5 = diagnostic_5_pnl_curve(best_trades)

    # === Diagnostic 6: Power analysis ===
    d6 = diagnostic_6_power(best_summary, d1)

    # === Diagnostic 7: Session conditioning ===
    d7 = diagnostic_7_conditioning(train_data, train_dates, best_h_idx, best_pct)

    # === Assemble results ===
    results = {
        "schema": "signal_diagnostics_v1",
        "export_dir": str(export_path),
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "n_train_days": len(train_dates),
        "n_val_days": len(val_dates),
        "n_test_days": len(test_dates),
        "best_horizon": best_h,
        "best_threshold": best_pct,
        "best_summary": best_summary,
        "d1_multi_horizon": d1,
        "d2_oos": d2,
        "d3_regime": d3,
        "d4_distribution": d4,
        "d5_pnl": d5,
        "d6_power": d6,
        "d7_conditioning": d7,
    }

    # Write outputs
    json_path = output_path / "diagnostics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    log(f"JSON: {json_path}")

    report = build_report(results)
    md_path = output_path / "SIGNAL_DIAGNOSTICS_REPORT.md"
    with open(md_path, "w") as f:
        f.write(report)
    log(f"Report: {md_path}")

    elapsed = time.time() - start
    log(f"\nTotal: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log(f"Best: H={best_h}, P{best_pct}, mean_net={best_summary.get('mean_net', 0):+.2f} bps, "
        f"t={best_summary.get('t_stat', 0):.3f}, p={best_summary.get('p_value', 1):.3f}")


if __name__ == "__main__":
    main()
