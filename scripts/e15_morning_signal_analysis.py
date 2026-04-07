#!/usr/bin/env python3
"""
E15 Approach 1: Morning Signal -> Afternoon Return Analysis.

Tests whether morning microstructure features predict afternoon returns
across 233 trading days.  Between-day temporal signal (1 observation/day).

Data: e5_timebased_60s export (MBO, 98 features, 60s bins)
Forward prices: (N, 306), column 5 = base price, 300 future bins (5 hours)

Key discovery (pre-analysis): Afternoon returns have ACF(1) = -0.29 at H=240.
A naive "bet against yesterday" has IC~0.29 -- this is the BASELINE, not zero.
Morning features must demonstrate incremental value beyond this baseline.

Usage:
    cd hft-feature-evaluator
    .venv/bin/python scripts/e15_morning_signal_analysis.py
    .venv/bin/python scripts/e15_morning_signal_analysis.py --phase 1
    .venv/bin/python scripts/e15_morning_signal_analysis.py --morning-windows 30 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sp_stats
from sklearn.linear_model import Ridge, RidgeCV

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "hft-feature-evaluator" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "hft-metrics" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "hft-contracts" / "src"))

from hft_evaluator.data.loader import ExportLoader  # noqa: E402
from hft_metrics import spearman_ic, EPS  # noqa: E402
from hft_metrics.ic import expected_return_bps, breakeven_ic as compute_breakeven_ic  # noqa: E402
from hft_metrics.bootstrap import block_bootstrap_ci  # noqa: E402
from hft_metrics.regression import directional_accuracy  # noqa: E402
from hft_metrics.testing import benjamini_hochberg  # noqa: E402
from hft_contracts import FeatureIndex  # noqa: E402

# ---------------------------------------------------------------------------
# Constants  (from hft-contracts FeatureIndex, verified at import)
# ---------------------------------------------------------------------------
SEED = 42
SMOOTHING_WINDOW_OFFSET = 5  # k=5, column 5 = base price in forward_prices
EXPORT_WINDOW_SIZE = 20       # sequence window_size from e5_timebased_60s.toml

IDX_SPREAD_BPS = int(FeatureIndex.SPREAD_BPS)                # 42
IDX_VOLUME_IMBALANCE = int(FeatureIndex.VOLUME_IMBALANCE)    # 45
IDX_TRUE_OFI = int(FeatureIndex.TRUE_OFI)                    # 84
IDX_DEPTH_NORM_OFI = int(FeatureIndex.DEPTH_NORM_OFI)        # 85
IDX_TOTAL_BID_VOLUME = int(FeatureIndex.TOTAL_BID_VOLUME)    # 43
IDX_TOTAL_ASK_VOLUME = int(FeatureIndex.TOTAL_ASK_VOLUME)    # 44
IDX_NET_ORDER_FLOW = int(FeatureIndex.NET_ORDER_FLOW)        # 54
IDX_FLOW_REGIME = int(FeatureIndex.FLOW_REGIME_INDICATOR)    # 59

FEATURE_NAMES = [
    "spread_bps_mean", "spread_bps_std", "spread_bps_max", "spread_bps_range",
    "volume_imbalance_mean", "true_ofi_sum", "true_ofi_abs_mean",
    "depth_norm_ofi_sum", "total_volume_mean", "net_order_flow_sum",
    "flow_regime_mean", "morning_return", "morning_range",
]
N_FEATURES = len(FEATURE_NAMES)  # 13


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[e15] {msg}", flush=True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _safe_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman IC with NaN guard."""
    rho, p = spearman_ic(x, y)
    if rho == 0.0 and p == 1.0:
        return float("nan")
    return float(rho)


def _safe_ic_with_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman IC + p-value with NaN guard."""
    rho, p = spearman_ic(x, y)
    if rho == 0.0 and p == 1.0:
        return float("nan"), 1.0
    return float(rho), float(p)


def _compute_rt_cost_bps(
    price: float, shares: int, spread_bps: float, commission_per_trade: float,
) -> float:
    """Round-trip cost in bps from actual entry price."""
    notional = price * shares
    if notional < EPS:
        return 0.0
    spread_cost_per_side = notional * spread_bps / 10_000
    total_per_side = spread_cost_per_side + commission_per_trade
    rt_cost = total_per_side * 2
    return rt_cost / notional * 10_000


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    log(f"  Saved {path.name}")


def _morning_seq_count(window_minutes: int) -> int:
    """Number of usable export sequences for a morning window.

    First usable sequence ends at raw bin (EXPORT_WINDOW_SIZE - 1) = bin 19.
    Window ends at raw bin = window_minutes.
    M = window_minutes - (EXPORT_WINDOW_SIZE - 1) + 1
    """
    first_usable_bin = EXPORT_WINDOW_SIZE - 1  # 19
    return window_minutes - first_usable_bin + 1


# ---------------------------------------------------------------------------
# Phase 1: Daily Aggregation
# ---------------------------------------------------------------------------
def aggregate_morning_features(
    sequences: np.ndarray,      # (N, T, F) float32
    forward_prices: np.ndarray, # (N, 306) float64
    M: int,
) -> np.ndarray:
    """Aggregate M morning sequences into 13 daily features.

    Returns: (13,) float64 array.
    """
    k = SMOOTHING_WINDOW_OFFSET
    morning = np.asarray(sequences[:M, -1, :], dtype=np.float64)  # (M, F)
    morning_prices = np.asarray(forward_prices[:M, k], dtype=np.float64)  # (M,)

    feat = np.empty(N_FEATURES, dtype=np.float64)

    # Spread features
    spread = morning[:, IDX_SPREAD_BPS]
    feat[0] = np.mean(spread)                                 # spread_bps_mean
    feat[1] = np.std(spread, ddof=1) if M > 1 else 0.0       # spread_bps_std
    feat[2] = np.max(spread)                                  # spread_bps_max
    feat[3] = np.max(spread) - np.min(spread)                 # spread_bps_range

    # Order flow features
    feat[4] = np.mean(morning[:, IDX_VOLUME_IMBALANCE])       # volume_imbalance_mean
    feat[5] = np.sum(morning[:, IDX_TRUE_OFI])                # true_ofi_sum
    feat[6] = np.mean(np.abs(morning[:, IDX_TRUE_OFI]))       # true_ofi_abs_mean
    feat[7] = np.sum(morning[:, IDX_DEPTH_NORM_OFI])          # depth_norm_ofi_sum

    # Volume and regime
    total_vol = morning[:, IDX_TOTAL_BID_VOLUME] + morning[:, IDX_TOTAL_ASK_VOLUME]
    feat[8] = np.mean(total_vol)                              # total_volume_mean
    feat[9] = np.sum(morning[:, IDX_NET_ORDER_FLOW])          # net_order_flow_sum
    feat[10] = np.mean(morning[:, IDX_FLOW_REGIME])           # flow_regime_mean

    # Derived from forward_prices
    if morning_prices[0] > EPS:
        feat[11] = (morning_prices[-1] - morning_prices[0]) / morning_prices[0] * 10_000
    else:
        feat[11] = 0.0                                        # morning_return (bps)

    mean_price = np.mean(morning_prices)
    if mean_price > EPS:
        feat[12] = (np.max(morning_prices) - np.min(morning_prices)) / mean_price * 10_000
    else:
        feat[12] = 0.0                                        # morning_range (bps)

    return feat


def compute_afternoon_returns(
    forward_prices: np.ndarray,  # (N, 306) float64
    endpoint_idx: int,
    horizons: list[int],
) -> np.ndarray:
    """Point returns from morning endpoint to each horizon.

    Returns: (len(horizons),) float64 with NaN for invalid.
    """
    k = SMOOTHING_WINDOW_OFFSET
    base_price = forward_prices[endpoint_idx, k]
    returns = np.full(len(horizons), np.nan, dtype=np.float64)
    for i, h in enumerate(horizons):
        col = k + h
        if col >= forward_prices.shape[1]:
            continue
        future_price = forward_prices[endpoint_idx, col]
        if np.isfinite(future_price) and base_price > EPS:
            returns[i] = (future_price - base_price) / base_price * 10_000
    return returns


def run_phase1(
    export_path: Path,
    output_dir: Path,
    morning_windows: list[int],
    afternoon_horizons: list[int],
) -> dict:
    """Phase 1: Load all 233 days, aggregate morning features, compute returns."""
    log("=== Phase 1: Daily Aggregation ===")
    t0 = time.time()

    # Load all days across splits, sorted chronologically
    all_days: list[dict] = []
    for split in ["train", "val", "test"]:
        loader = ExportLoader(str(export_path), split)
        for date in sorted(loader.list_dates()):
            bundle = loader.load_day(date)
            fp_path = export_path / split / f"{date}_forward_prices.npy"
            fp = np.load(str(fp_path))
            assert fp.shape[0] == bundle.sequences.shape[0], (
                f"{date}: fp.shape[0]={fp.shape[0]} != seq.shape[0]={bundle.sequences.shape[0]}"
            )
            all_days.append({
                "date": date, "split": split, "n": bundle.sequences.shape[0],
                "sequences": bundle.sequences, "fp": fp,
            })

    # Sort by date (chronological)
    all_days.sort(key=lambda d: d["date"])
    n_days = len(all_days)
    log(f"  Loaded {n_days} days ({sum(d['n'] for d in all_days)} total sequences)")

    # Results per morning window
    results = {}
    for w_min in morning_windows:
        M = _morning_seq_count(w_min)
        log(f"  Window {w_min}min: M={M} sequences/day")

        features_all = np.full((n_days, N_FEATURES), np.nan, dtype=np.float64)
        returns_all = np.full((n_days, len(afternoon_horizons)), np.nan, dtype=np.float64)
        entry_prices = np.full(n_days, np.nan, dtype=np.float64)
        valid_mask = np.zeros(n_days, dtype=bool)
        dates = []
        splits = []
        day_of_week = np.full(n_days, -1, dtype=np.int32)

        for i, d in enumerate(all_days):
            dates.append(d["date"])
            splits.append(d["split"])

            # Parse day-of-week (0=Mon..4=Fri)
            dt = datetime.strptime(d["date"].replace("-", ""), "%Y%m%d")
            day_of_week[i] = dt.weekday()

            if d["n"] < M:
                log(f"    SKIP {d['date']} (N={d['n']} < M={M})")
                continue

            features_all[i] = aggregate_morning_features(d["sequences"], d["fp"], M)
            returns_all[i] = compute_afternoon_returns(d["fp"], M - 1, afternoon_horizons)
            entry_prices[i] = float(d["fp"][M - 1, SMOOTHING_WINDOW_OFFSET])

            # Valid if at least H=120 return is finite
            if np.isfinite(returns_all[i, 0]):
                valid_mask[i] = True

        # Compute prev_day_return (per horizon)
        prev_day_returns = np.full((n_days, len(afternoon_horizons)), np.nan, dtype=np.float64)
        for i in range(1, n_days):
            prev_day_returns[i] = returns_all[i - 1]

        # Day-of-week one-hot (5 dummies)
        dow_dummies = np.zeros((n_days, 5), dtype=np.float64)
        for i in range(n_days):
            if 0 <= day_of_week[i] <= 4:
                dow_dummies[i, day_of_week[i]] = 1.0

        results[w_min] = {
            "features": features_all,
            "returns": returns_all,
            "entry_prices": entry_prices,
            "valid_mask": valid_mask,
            "dates": dates,
            "splits": splits,
            "day_of_week": day_of_week,
            "prev_day_returns": prev_day_returns,
            "dow_dummies": dow_dummies,
            "M": M,
        }

        n_valid = int(valid_mask.sum())
        log(f"    Valid days: {n_valid}/{n_days}")

        # Per-horizon valid counts
        for j, h in enumerate(afternoon_horizons):
            n_h_valid = int(np.isfinite(returns_all[:, j]).sum())
            log(f"    H={h}: {n_h_valid} days with finite return")

    # Free raw data (after all windows processed)
    for d in all_days:
        d.pop("sequences", None)
        d.pop("fp", None)

    # --- Diagnostics ---
    diag = {"timestamp": datetime.utcnow().isoformat(), "n_days": n_days}
    # Use the first window for return diagnostics (returns are window-independent
    # except for endpoint, but differences are negligible for diagnostics)
    w0 = morning_windows[0]
    r0 = results[w0]

    for j, h in enumerate(afternoon_horizons):
        col = r0["returns"][:, j]
        finite = col[np.isfinite(col)]
        if len(finite) < 5:
            continue
        ret_key = f"H{h}"
        diag[ret_key] = {
            "n_valid": len(finite),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite, ddof=1)),
            "median": float(np.median(finite)),
            "skewness": float(sp_stats.skew(finite)),
            "kurtosis": float(sp_stats.kurtosis(finite)),
            "frac_positive": float(np.mean(finite > 0)),
            "t_stat_mean": float(np.mean(finite) / (np.std(finite, ddof=1) / np.sqrt(len(finite)))),
        }

        # Lag-1 autocorrelation (critical confound check)
        valid_idx = np.where(np.isfinite(col))[0]
        consec_pairs = [(valid_idx[k], valid_idx[k + 1])
                        for k in range(len(valid_idx) - 1)
                        if valid_idx[k + 1] == valid_idx[k] + 1]
        if len(consec_pairs) > 10:
            x_lag = np.array([col[a] for a, _ in consec_pairs])
            y_lag = np.array([col[b] for _, b in consec_pairs])
            acf1, acf1_p = sp_stats.pearsonr(x_lag, y_lag)
            diag[ret_key]["acf1"] = float(acf1)
            diag[ret_key]["acf1_p"] = float(acf1_p)

        # Day-of-week ANOVA
        dow = r0["day_of_week"]
        groups = [finite[dow[np.isfinite(col)] == d] for d in range(5)]
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) >= 2:
            f_stat, anova_p = sp_stats.f_oneway(*groups)
            dow_means = {f"dow_{d}": float(np.mean(g)) for d, g in enumerate(groups)}
            diag[ret_key]["anova_f"] = float(f_stat)
            diag[ret_key]["anova_p"] = float(anova_p)
            diag[ret_key]["dow_means"] = dow_means

    # Excluded days audit
    for w_min in morning_windows:
        M = _morning_seq_count(w_min)
        excluded = [all_days[i]["date"] for i in range(n_days) if all_days[i]["n"] < M]
        if excluded:
            diag[f"excluded_w{w_min}"] = excluded

    # Feature distributions
    for w_min in morning_windows:
        feat = results[w_min]["features"]
        valid = results[w_min]["valid_mask"]
        f_diag = {}
        for fi, fname in enumerate(FEATURE_NAMES):
            vals = feat[valid, fi]
            if len(vals) < 5:
                continue
            f_diag[fname] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)),
                "cv": float(np.std(vals, ddof=1) / max(abs(np.mean(vals)), EPS)),
            }
        diag[f"features_w{w_min}"] = f_diag

    # Save artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(output_dir / "phase1_diagnostics.json", diag)

    # Save NPY per window
    for w_min in morning_windows:
        r = results[w_min]
        np.save(str(output_dir / f"phase1_features_w{w_min}.npy"), r["features"])
    # Returns and confounds (use first window; endpoint difference is minimal)
    np.save(str(output_dir / "phase1_returns.npy"), results[morning_windows[0]]["returns"])
    np.save(str(output_dir / "phase1_prev_day_returns.npy"),
            results[morning_windows[0]]["prev_day_returns"])
    np.save(str(output_dir / "phase1_dow_dummies.npy"),
            results[morning_windows[0]]["dow_dummies"])

    # Metadata
    meta = {
        "dates": results[morning_windows[0]]["dates"],
        "splits": results[morning_windows[0]]["splits"],
        "feature_names": FEATURE_NAMES,
        "morning_windows": morning_windows,
        "afternoon_horizons": afternoon_horizons,
        "n_days": n_days,
        "export_path": str(export_path),
        "timestamp": datetime.utcnow().isoformat(),
    }
    _save_json(output_dir / "phase1_metadata.json", meta)

    elapsed = time.time() - t0
    log(f"  Phase 1 complete in {elapsed:.1f}s")
    return results


# ---------------------------------------------------------------------------
# Phase 2A: Baseline Quantification
# ---------------------------------------------------------------------------
def run_phase2a(
    results: dict, afternoon_horizons: list[int], output_dir: Path,
    cost_spread_bps: float, cost_commission: float, shares: int,
    n_bootstraps: int,
) -> dict:
    """Quantify what is predictable WITHOUT morning features."""
    log("=== Phase 2A: Baseline Quantification ===")

    # Use first available window (baselines are window-independent)
    w_min = list(results.keys())[0]
    r = results[w_min]
    baseline_results = {}

    for j, h in enumerate(afternoon_horizons):
        ret = r["returns"][:, j]
        prev = r["prev_day_returns"][:, j]

        # Valid: both current and prev finite, skip day 0
        valid = np.isfinite(ret) & np.isfinite(prev)
        n_valid = int(valid.sum())
        if n_valid < 20:
            log(f"  H={h}: only {n_valid} valid days, skipping")
            continue

        ret_v = ret[valid]
        prev_v = prev[valid]
        neg_prev = -prev_v  # "bet against yesterday"

        # Baseline 1: return mean-reversion
        ic_rev, p_rev = _safe_ic_with_p(neg_prev, ret_v)
        _, ci_lo, ci_hi = block_bootstrap_ci(
            lambda x, y: spearman_ic(x, y)[0], neg_prev, ret_v,
            n_bootstraps=n_bootstraps, seed=SEED,
        )
        da_rev = directional_accuracy(ret_v, neg_prev)
        return_std = float(np.std(ret_v, ddof=1))
        grinold = expected_return_bps(ic_rev, return_std) if np.isfinite(ic_rev) else 0.0

        # P&L of "bet against yesterday" (simple: sign(prediction) * return - cost)
        entry_p = r["entry_prices"][valid]
        pnl_per_day = np.zeros(n_valid)
        for t in range(n_valid):
            direction = 1.0 if neg_prev[t] > 0 else (-1.0 if neg_prev[t] < 0 else 0.0)
            gross = direction * ret_v[t]
            rt_cost = _compute_rt_cost_bps(entry_p[t], shares, cost_spread_bps, cost_commission)
            pnl_per_day[t] = gross - rt_cost if direction != 0.0 else 0.0

        cum_pnl = float(np.sum(pnl_per_day))
        frac_profitable = float(np.mean(pnl_per_day > 0)) if n_valid > 0 else 0.0

        baseline_results[f"H{h}"] = {
            "n_valid": n_valid,
            "return_std_bps": return_std,
            "breakeven_ic": float(compute_breakeven_ic(
                _compute_rt_cost_bps(np.nanmean(entry_p), shares, cost_spread_bps, cost_commission),
                return_std,
            )),
            "baseline_ic": ic_rev,
            "baseline_p": p_rev,
            "baseline_ci": [ci_lo, ci_hi],
            "baseline_da": da_rev,
            "baseline_grinold_bps": grinold,
            "baseline_cum_pnl_bps": cum_pnl,
            "baseline_frac_profitable": frac_profitable,
            "baseline_mean_pnl_bps": float(np.mean(pnl_per_day)),
        }
        log(f"  H={h}: baseline IC={ic_rev:.3f} CI=[{ci_lo:.3f},{ci_hi:.3f}] "
            f"DA={da_rev:.1%} P&L={cum_pnl:.1f} bps ({n_valid} days)")

    _save_json(output_dir / "phase2a_baseline.json", baseline_results)
    return baseline_results


# ---------------------------------------------------------------------------
# Phase 2B: Unconditional IC
# ---------------------------------------------------------------------------
def run_phase2b(
    results: dict, morning_windows: list[int], afternoon_horizons: list[int],
    output_dir: Path, n_bootstraps: int,
) -> dict:
    """Compute raw IC for each feature x window x horizon."""
    log("=== Phase 2B: Unconditional IC ===")

    all_tests = []
    for w_min in morning_windows:
        r = results[w_min]
        for j, h in enumerate(afternoon_horizons):
            ret = r["returns"][:, j]
            for fi, fname in enumerate(FEATURE_NAMES):
                feat = r["features"][:, fi]
                valid = np.isfinite(feat) & np.isfinite(ret) & r["valid_mask"]
                n_valid = int(valid.sum())
                if n_valid < 20:
                    continue

                feat_v, ret_v = feat[valid], ret[valid]
                ic, p = _safe_ic_with_p(feat_v, ret_v)
                _, ci_lo, ci_hi = block_bootstrap_ci(
                    lambda x, y: spearman_ic(x, y)[0], feat_v, ret_v,
                    n_bootstraps=n_bootstraps, seed=SEED,
                )

                # Quintile analysis
                q_returns = [np.nan] * 5
                try:
                    quintiles = np.percentile(feat_v, [20, 40, 60, 80])
                    bins = np.digitize(feat_v, quintiles)
                    q_returns = [float(np.mean(ret_v[bins == q])) if np.sum(bins == q) > 0
                                 else np.nan for q in range(5)]
                except Exception:
                    pass

                # Per-split IC
                split_ics = {}
                for split_name in ["train", "val", "test"]:
                    split_mask = valid & np.array([s == split_name for s in r["splits"]])
                    if split_mask.sum() >= 10:
                        split_ics[split_name] = _safe_ic(feat[split_mask], ret[split_mask])

                all_tests.append({
                    "feature": fname, "window": w_min, "horizon": h,
                    "ic": ic, "p": p, "ci_lo": ci_lo, "ci_hi": ci_hi,
                    "n": n_valid,
                    "quintile_returns": q_returns,
                    "quintile_spread": q_returns[4] - q_returns[0] if all(np.isfinite(q_returns)) else np.nan,
                    "ci_crosses_zero": (ci_lo <= 0 <= ci_hi),
                    "split_ics": split_ics,
                })

    # BH correction
    p_values = np.array([t["p"] for t in all_tests])
    bh_mask = benjamini_hochberg(p_values, q=0.10)
    for i, t in enumerate(all_tests):
        t["bh_significant"] = bool(bh_mask[i])

    # Sort by |IC| descending
    all_tests.sort(key=lambda t: abs(t["ic"]) if np.isfinite(t["ic"]) else 0, reverse=True)

    # Log top 10
    log("  Top 10 by |IC|:")
    for t in all_tests[:10]:
        log(f"    {t['feature']:25s} w={t['window']} H={t['horizon']} "
            f"IC={t['ic']:+.3f} CI=[{t['ci_lo']:.3f},{t['ci_hi']:.3f}] "
            f"BH={'*' if t['bh_significant'] else ' '} Q-spread={t['quintile_spread']:+.1f}")

    result = {"tests": all_tests, "n_tests": len(all_tests)}
    _save_json(output_dir / "phase2b_unconditional_ic.json", result)
    return result


# ---------------------------------------------------------------------------
# Phase 2C: Confound Investigation
# ---------------------------------------------------------------------------
def run_phase2c(
    results: dict, morning_windows: list[int], afternoon_horizons: list[int],
    output_dir: Path, n_bootstraps: int,
) -> dict:
    """Partial IC after controlling for yesterday's return + day-of-week."""
    log("=== Phase 2C: Confound Investigation ===")

    all_partial = []
    for w_min in morning_windows:
        r = results[w_min]
        for j, h in enumerate(afternoon_horizons):
            ret = r["returns"][:, j]
            prev = r["prev_day_returns"][:, j]

            # Confound matrix: [-prev_day_return, dow_dummies]
            confounds = np.column_stack([-prev, r["dow_dummies"]])  # (n_days, 6)

            for fi, fname in enumerate(FEATURE_NAMES):
                feat = r["features"][:, fi]
                valid = (np.isfinite(feat) & np.isfinite(ret)
                         & np.isfinite(prev) & r["valid_mask"])
                n_valid = int(valid.sum())
                if n_valid < 30:
                    continue

                feat_v = feat[valid]
                ret_v = ret[valid]
                conf_v = confounds[valid]

                # Feature-confound correlation (is feature driven by yesterday?)
                r_confound = _safe_ic(feat_v, -prev[valid])

                # Partial correlation: residualize both against confounds
                ridge_feat = Ridge(alpha=100.0, fit_intercept=True)
                ridge_ret = Ridge(alpha=100.0, fit_intercept=True)
                ridge_feat.fit(conf_v, feat_v)
                ridge_ret.fit(conf_v, ret_v)
                feat_resid = feat_v - ridge_feat.predict(conf_v)
                ret_resid = ret_v - ridge_ret.predict(conf_v)

                partial_ic, partial_p = _safe_ic_with_p(feat_resid, ret_resid)
                _, pci_lo, pci_hi = block_bootstrap_ci(
                    lambda x, y: spearman_ic(x, y)[0], feat_resid, ret_resid,
                    n_bootstraps=n_bootstraps, seed=SEED,
                )

                # Classification
                uncond_ic = _safe_ic(feat_v, ret_v)
                if abs(uncond_ic) < 0.02 if np.isfinite(uncond_ic) else True:
                    classification = "NO_SIGNAL"
                elif abs(partial_ic) < 0.02 if np.isfinite(partial_ic) else True:
                    classification = "CONFOUNDED"
                else:
                    classification = "GENUINE"

                all_partial.append({
                    "feature": fname, "window": w_min, "horizon": h,
                    "unconditional_ic": uncond_ic,
                    "partial_ic": partial_ic, "partial_p": partial_p,
                    "partial_ci_lo": pci_lo, "partial_ci_hi": pci_hi,
                    "partial_ci_crosses_zero": (pci_lo <= 0 <= pci_hi),
                    "r_confound_prev_day": r_confound,
                    "classification": classification,
                    "n": n_valid,
                })

    # Sort by |partial IC|
    all_partial.sort(key=lambda t: abs(t["partial_ic"]) if np.isfinite(t["partial_ic"]) else 0,
                     reverse=True)

    # Log top results
    log("  Top 10 by |partial IC|:")
    for t in all_partial[:10]:
        log(f"    {t['feature']:25s} w={t['window']} H={t['horizon']} "
            f"uncond={t['unconditional_ic']:+.3f} partial={t['partial_ic']:+.3f} "
            f"CI=[{t['partial_ci_lo']:.3f},{t['partial_ci_hi']:.3f}] "
            f"r_confound={t['r_confound_prev_day']:+.3f} -> {t['classification']}")

    # Summary counts
    class_counts = {}
    for c in ["GENUINE", "CONFOUNDED", "NO_SIGNAL"]:
        class_counts[c] = sum(1 for t in all_partial if t["classification"] == c)
    log(f"  Classification: {class_counts}")

    result = {"partial_tests": all_partial, "classification_counts": class_counts}
    _save_json(output_dir / "phase2c_confound_analysis.json", result)
    return result


# ---------------------------------------------------------------------------
# Phase 2D: Conditional & Interaction Analysis
# ---------------------------------------------------------------------------
def run_phase2d(
    results: dict, morning_windows: list[int], afternoon_horizons: list[int],
    output_dir: Path,
) -> dict:
    """Interaction terms, conditional IC by morning return direction."""
    log("=== Phase 2D: Conditional & Interaction Analysis ===")

    cond_results = []
    for w_min in morning_windows:
        r = results[w_min]
        for j, h in enumerate(afternoon_horizons):
            ret = r["returns"][:, j]
            feat = r["features"]
            valid = np.isfinite(ret) & r["valid_mask"]
            n_valid = int(valid.sum())
            if n_valid < 30:
                continue

            ret_v = ret[valid]
            morning_ret = feat[valid, FEATURE_NAMES.index("morning_return")]
            spread_mean = feat[valid, FEATURE_NAMES.index("spread_bps_mean")]

            # Interaction: spread x sign(morning_return)
            direction = np.sign(morning_ret)
            interaction = spread_mean * direction
            ic_interaction, _ = _safe_ic_with_p(interaction, ret_v)
            ic_spread_alone, _ = _safe_ic_with_p(spread_mean, ret_v)

            # Conditional IC by morning return direction
            up_mask = morning_ret > 0
            down_mask = morning_ret < 0
            ic_spread_up = _safe_ic(spread_mean[up_mask], ret_v[up_mask]) if up_mask.sum() >= 10 else np.nan
            ic_spread_down = _safe_ic(spread_mean[down_mask], ret_v[down_mask]) if down_mask.sum() >= 10 else np.nan
            ic_ofi_up = _safe_ic(feat[valid, FEATURE_NAMES.index("true_ofi_sum")][up_mask],
                                 ret_v[up_mask]) if up_mask.sum() >= 10 else np.nan
            ic_ofi_down = _safe_ic(feat[valid, FEATURE_NAMES.index("true_ofi_sum")][down_mask],
                                   ret_v[down_mask]) if down_mask.sum() >= 10 else np.nan

            # Rolling IC stability (30-day window)
            ic_rolling = []
            feat_all_valid = feat[valid]
            for start in range(0, n_valid - 30, 10):
                end = start + 30
                for fi, fname in enumerate(FEATURE_NAMES[:4]):  # spread features
                    w_ic = _safe_ic(feat_all_valid[start:end, fi], ret_v[start:end])
                    ic_rolling.append({"start": start, "feature": fname, "ic": w_ic})

            # Feature correlation matrix
            feat_valid = feat[valid]
            corr_matrix = np.corrcoef(feat_valid.T) if feat_valid.shape[0] > 5 else None

            cond_results.append({
                "window": w_min, "horizon": h, "n": n_valid,
                "ic_spread_alone": ic_spread_alone,
                "ic_interaction_spread_x_dir": ic_interaction,
                "interaction_stronger": (abs(ic_interaction) > abs(ic_spread_alone)
                                         if all(np.isfinite([ic_interaction, ic_spread_alone])) else False),
                "ic_spread_given_up": ic_spread_up,
                "ic_spread_given_down": ic_spread_down,
                "ic_ofi_given_up": ic_ofi_up,
                "ic_ofi_given_down": ic_ofi_down,
                "n_up_days": int(up_mask.sum()),
                "n_down_days": int(down_mask.sum()),
                "ic_rolling_sample": ic_rolling[:20],
                "feature_corr_matrix": corr_matrix.tolist() if corr_matrix is not None else None,
            })

    _save_json(output_dir / "phase2d_conditional_ic.json", {"conditional_tests": cond_results})
    log(f"  Computed {len(cond_results)} conditional analyses")
    return {"conditional_tests": cond_results}


# ---------------------------------------------------------------------------
# Phase 2: Gate Decision
# ---------------------------------------------------------------------------
def phase2_gate(
    phase2c: dict, baseline: dict, breakeven_threshold: float,
) -> tuple[bool, list[dict]]:
    """Apply screening gate: partial IC > breakeven AND CI not crossing zero AND GENUINE."""
    passing = []
    for t in phase2c["partial_tests"]:
        if t["classification"] != "GENUINE":
            continue
        pic = t["partial_ic"]
        if not np.isfinite(pic):
            continue
        h_key = f"H{t['horizon']}"
        if h_key in baseline:
            be_ic = baseline[h_key].get("breakeven_ic", breakeven_threshold)
        else:
            be_ic = breakeven_threshold
        if abs(pic) > be_ic and not t["partial_ci_crosses_zero"]:
            passing.append(t)

    passing.sort(key=lambda t: abs(t["partial_ic"]), reverse=True)
    gate_pass = len(passing) > 0
    return gate_pass, passing


# ---------------------------------------------------------------------------
# Phase 3: Walk-Forward Ridge
# ---------------------------------------------------------------------------
def run_phase3(
    results: dict, passing_features: list[dict], afternoon_horizons: list[int],
    baseline: dict, output_dir: Path,
    min_train_days: int, alpha_grid: list[float], max_features: int,
    cost_spread_bps: float, cost_commission: float, shares: int,
) -> dict:
    """Walk-forward Ridge: Model A (baseline), Model B (morning), Model C (combined)."""
    log("=== Phase 3: Walk-Forward Ridge ===")

    if not passing_features:
        log("  No passing features, skipping Phase 3")
        return {}

    # Determine best window and horizon from passing features
    best = passing_features[0]
    best_w = best["window"]
    best_h = best["horizon"]
    h_idx = afternoon_horizons.index(best_h)
    log(f"  Best: {best['feature']} w={best_w} H={best_h} partial_ic={best['partial_ic']:.3f}")

    # Collect top-K feature indices for the best window/horizon
    top_features = []
    seen = set()
    for t in passing_features:
        if t["window"] == best_w and t["horizon"] == best_h and t["feature"] not in seen:
            top_features.append(FEATURE_NAMES.index(t["feature"]))
            seen.add(t["feature"])
            if len(top_features) >= max_features:
                break
    log(f"  Using {len(top_features)} morning features: {[FEATURE_NAMES[i] for i in top_features]}")

    r = results[best_w]
    ret = r["returns"][:, h_idx]
    feat = r["features"]
    prev = r["prev_day_returns"][:, h_idx]
    dow = r["dow_dummies"]
    entry_p = r["entry_prices"]
    n_days = len(ret)

    # Valid mask: need current return, prev return, and features all finite
    valid_full = (np.isfinite(ret) & np.isfinite(prev) & r["valid_mask"]
                  & np.all(np.isfinite(feat[:, top_features]), axis=1))

    # Walk-forward
    model_results = {"A": [], "B": [], "C": []}
    alphas = np.array(alpha_grid)

    for d in range(min_train_days, n_days):
        if not valid_full[d]:
            continue

        # Training indices: all valid days before d
        train_mask = valid_full.copy()
        train_mask[d:] = False
        if train_mask.sum() < 20:
            continue

        # Build features
        # Model A: baseline (-prev_return, dow)
        X_A_train = np.column_stack([-prev[train_mask], dow[train_mask]])
        X_A_test = np.column_stack([[-prev[d]], [dow[d]]])

        # Model B: morning features only
        X_B_train = feat[train_mask][:, top_features]
        X_B_test = feat[d:d + 1, top_features]

        # Model C: combined
        X_C_train = np.column_stack([X_A_train, X_B_train])
        X_C_test = np.column_stack([X_A_test, X_B_test])

        y_train = ret[train_mask]

        # Standardize within training fold
        for X_train, X_test, model_name in [
            (X_A_train, X_A_test, "A"),
            (X_B_train, X_B_test, "B"),
            (X_C_train, X_C_test, "C"),
        ]:
            mu = np.mean(X_train, axis=0)
            sigma = np.std(X_train, axis=0, ddof=1)
            sigma[sigma < EPS] = 1.0
            X_tr_std = (X_train - mu) / sigma
            X_te_std = (X_test - mu) / sigma

            ridge = RidgeCV(alphas=alphas, fit_intercept=True, scoring="neg_mean_squared_error")
            ridge.fit(X_tr_std, y_train)
            pred = float(ridge.predict(X_te_std)[0])
            actual = float(ret[d])

            # Cost
            rt_cost = _compute_rt_cost_bps(entry_p[d], shares, cost_spread_bps, cost_commission)
            direction = 1.0 if pred > 0 else (-1.0 if pred < 0 else 0.0)
            pnl = direction * actual - rt_cost if direction != 0.0 else 0.0

            model_results[model_name].append({
                "day_idx": d, "date": r["dates"][d],
                "pred": pred, "actual": actual,
                "direction": direction, "pnl_bps": pnl,
                "alpha": float(ridge.alpha_),
            })

    # Compute OOS metrics per model
    summary = {}
    for model_name in ["A", "B", "C"]:
        preds = np.array([r["pred"] for r in model_results[model_name]])
        actuals = np.array([r["actual"] for r in model_results[model_name]])
        pnls = np.array([r["pnl_bps"] for r in model_results[model_name]])
        alphas_used = np.array([r["alpha"] for r in model_results[model_name]])

        if len(preds) < 20:
            summary[model_name] = {"n_oos": len(preds), "insufficient": True}
            continue

        oos_ic, oos_p = _safe_ic_with_p(preds, actuals)
        _, ic_ci_lo, ic_ci_hi = block_bootstrap_ci(
            lambda x, y: spearman_ic(x, y)[0], preds, actuals,
            n_bootstraps=1000, seed=SEED,
        )
        da = directional_accuracy(actuals, preds)

        # IC stability (20-day blocks)
        block_ics = []
        for start in range(0, len(preds) - 20, 20):
            blk_ic = _safe_ic(preds[start:start + 20], actuals[start:start + 20])
            if np.isfinite(blk_ic):
                block_ics.append(blk_ic)
        ic_stability = (abs(np.mean(block_ics)) / max(np.std(block_ics, ddof=1), EPS)
                        if len(block_ics) > 1 else 0.0)

        h_key = f"H{best_h}"
        return_std = baseline.get(h_key, {}).get("return_std_bps", 44.0)

        summary[model_name] = {
            "n_oos": len(preds),
            "oos_ic": oos_ic, "oos_p": oos_p,
            "oos_ic_ci": [ic_ci_lo, ic_ci_hi],
            "da": da,
            "grinold_bps": expected_return_bps(oos_ic, return_std) if np.isfinite(oos_ic) else 0.0,
            "cum_pnl_bps": float(np.sum(pnls)),
            "mean_pnl_bps": float(np.mean(pnls)),
            "frac_profitable": float(np.mean(pnls > 0)),
            "max_drawdown_bps": float(np.min(np.cumsum(pnls) - np.maximum.accumulate(np.cumsum(pnls)))),
            "ic_stability": ic_stability,
            "median_alpha": float(np.median(alphas_used)),
        }
        log(f"  Model {model_name}: IC={oos_ic:.3f} CI=[{ic_ci_lo:.3f},{ic_ci_hi:.3f}] "
            f"DA={da:.1%} P&L={np.sum(pnls):.1f} bps stability={ic_stability:.2f}")

    # Decision matrix
    a_profitable = summary.get("A", {}).get("cum_pnl_bps", 0) > 0
    ic_a = summary.get("A", {}).get("oos_ic", 0)
    ic_c = summary.get("C", {}).get("oos_ic", 0)
    c_better = (ic_c - ic_a) > 0.02 if all(np.isfinite([ic_a, ic_c])) else False

    if a_profitable and c_better:
        decision = "BEST_CASE: Morning features + baseline -> live strategy"
    elif a_profitable and not c_better:
        decision = "Morning features add nothing. Use baseline alone."
    elif not a_profitable and c_better:
        decision = "Morning features help but baseline drags. Test Model B standalone."
    else:
        decision = "E15 FAILS. Document lessons, consider Approaches 2-5."

    summary["decision"] = decision
    summary["best_feature"] = best["feature"]
    summary["best_window"] = best_w
    summary["best_horizon"] = best_h
    summary["top_features"] = [FEATURE_NAMES[i] for i in top_features]

    log(f"  Decision: {decision}")

    _save_json(output_dir / "phase3_walkforward.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------
def build_report(
    output_dir: Path, baseline: dict, phase2b: dict, phase2c: dict,
    phase2d: dict, phase3: dict, gate_pass: bool, passing: list[dict],
) -> None:
    """Generate human-readable REPORT.md."""
    lines = [
        "# E15 Approach 1: Morning Signal -> Afternoon Return",
        f"\nGenerated: {datetime.utcnow().isoformat()}Z",
        "\n## Phase 2A: Baseline (\"Bet Against Yesterday\")\n",
    ]
    for h_key, b in sorted(baseline.items()):
        lines.append(f"| {h_key} | IC={b['baseline_ic']:.3f} "
                     f"CI=[{b['baseline_ci'][0]:.3f},{b['baseline_ci'][1]:.3f}] "
                     f"DA={b['baseline_da']:.1%} P&L={b['baseline_cum_pnl_bps']:.1f} bps "
                     f"return_std={b['return_std_bps']:.1f} breakeven_ic={b['breakeven_ic']:.3f} |")

    lines.append("\n## Phase 2B: Top Unconditional IC (sorted by |IC|)\n")
    lines.append("| Feature | Window | Horizon | IC | CI | Q-spread | BH |")
    lines.append("|---------|--------|---------|----|----|----------|-----|")
    for t in (phase2b.get("tests") or [])[:20]:
        lines.append(f"| {t['feature']} | {t['window']} | {t['horizon']} | "
                     f"{t['ic']:+.3f} | [{t['ci_lo']:.3f},{t['ci_hi']:.3f}] | "
                     f"{t['quintile_spread']:+.1f} | {'*' if t['bh_significant'] else ''} |")

    lines.append("\n## Phase 2C: Confound Analysis (sorted by |partial IC|)\n")
    lines.append("| Feature | W | H | Uncond IC | Partial IC | r_confound | Class |")
    lines.append("|---------|---|---|-----------|------------|------------|-------|")
    for t in (phase2c.get("partial_tests") or [])[:20]:
        lines.append(f"| {t['feature']} | {t['window']} | {t['horizon']} | "
                     f"{t['unconditional_ic']:+.3f} | {t['partial_ic']:+.3f} | "
                     f"{t['r_confound_prev_day']:+.3f} | {t['classification']} |")

    lines.append(f"\n## Phase 2 Gate: {'PASS' if gate_pass else 'FAIL'}\n")
    if passing:
        lines.append(f"Passing features ({len(passing)}):")
        for t in passing[:10]:
            lines.append(f"  - {t['feature']} w={t['window']} H={t['horizon']} "
                         f"partial_ic={t['partial_ic']:.3f}")
    else:
        lines.append("No features passed G1+G2+G3.")
        if any(b.get("baseline_cum_pnl_bps", 0) > 0 for b in baseline.values()):
            lines.append("However, baseline ('bet against yesterday') is profitable at some horizons.")

    if phase3:
        lines.append("\n## Phase 3: Walk-Forward Ridge\n")
        lines.append("| Model | IC | CI | DA | P&L (bps) | Stability |")
        lines.append("|-------|----|----|-----|-----------|-----------|")
        for m in ["A", "B", "C"]:
            s = phase3.get(m, {})
            if s.get("insufficient"):
                lines.append(f"| {m} | insufficient data | | | | |")
                continue
            lines.append(f"| {m} | {s.get('oos_ic', 0):.3f} | "
                         f"[{s.get('oos_ic_ci', [0, 0])[0]:.3f},"
                         f"{s.get('oos_ic_ci', [0, 0])[1]:.3f}] | "
                         f"{s.get('da', 0):.1%} | "
                         f"{s.get('cum_pnl_bps', 0):.1f} | "
                         f"{s.get('ic_stability', 0):.2f} |")
        lines.append(f"\n**Decision**: {phase3.get('decision', 'N/A')}")

    report_path = output_dir / "REPORT.md"
    report_path.write_text("\n".join(lines))
    log(f"  Report saved to {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E15: Morning Signal -> Afternoon Return")
    p.add_argument("--export-dir", type=str,
                   default=str(PROJECT_ROOT / "data" / "exports" / "e5_timebased_60s"))
    p.add_argument("--output-dir", type=str,
                   default=str(PROJECT_ROOT / "hft-feature-evaluator" / "outputs" / "e15_morning_signal"))
    p.add_argument("--morning-windows", type=int, nargs="+", default=[30, 45, 60],
                   help="Minutes from market open (default: 30 45 60)")
    p.add_argument("--afternoon-horizons", type=int, nargs="+", default=[120, 180, 240, 300],
                   help="Bins forward from morning endpoint (default: 120 180 240 300)")
    p.add_argument("--cost-spread-bps", type=float, default=0.28)
    p.add_argument("--cost-commission", type=float, default=0.35)
    p.add_argument("--shares", type=int, default=100)
    p.add_argument("--min-train-days", type=int, default=100)
    p.add_argument("--alpha-grid", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0, 1000.0])
    p.add_argument("--max-features", type=int, default=5)
    p.add_argument("--n-bootstraps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--phase", type=str, default="all",
                   choices=["all", "1", "2", "3"],
                   help="Run specific phase (default: all)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    export_path = Path(args.export_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    global SEED
    SEED = args.seed
    np.random.seed(SEED)

    log(f"Export: {export_path}")
    log(f"Output: {output_dir}")
    log(f"Morning windows: {args.morning_windows} min")
    log(f"Afternoon horizons: {args.afternoon_horizons} bins")

    # Phase 1
    results = run_phase1(export_path, output_dir, args.morning_windows, args.afternoon_horizons)

    if args.phase == "1":
        log("Phase 1 complete. Exiting (--phase 1).")
        return

    # Phase 2A: Baseline
    baseline = run_phase2a(
        results, args.afternoon_horizons, output_dir,
        args.cost_spread_bps, args.cost_commission, args.shares,
        args.n_bootstraps,
    )

    # Phase 2B: Unconditional IC
    phase2b = run_phase2b(
        results, args.morning_windows, args.afternoon_horizons,
        output_dir, args.n_bootstraps,
    )

    # Phase 2C: Confound investigation
    phase2c = run_phase2c(
        results, args.morning_windows, args.afternoon_horizons,
        output_dir, args.n_bootstraps,
    )

    # Phase 2D: Conditional analysis
    phase2d = run_phase2d(
        results, args.morning_windows, args.afternoon_horizons, output_dir,
    )

    # Gate decision
    # Use the most conservative breakeven IC from baseline
    be_threshold = max(
        (b.get("breakeven_ic", 0.025) for b in baseline.values()),
        default=0.025,
    )
    gate_pass, passing = phase2_gate(phase2c, baseline, be_threshold)
    log(f"Phase 2 Gate: {'PASS' if gate_pass else 'FAIL'} "
        f"({len(passing)} features pass, threshold={be_threshold:.4f})")

    if args.phase == "2":
        build_report(output_dir, baseline, phase2b, phase2c, phase2d, {}, gate_pass, passing)
        log("Phase 2 complete. Exiting (--phase 2).")
        return

    # Phase 3 (conditional on gate pass)
    phase3 = {}
    if gate_pass:
        phase3 = run_phase3(
            results, passing, args.afternoon_horizons,
            baseline, output_dir,
            args.min_train_days, args.alpha_grid, args.max_features,
            args.cost_spread_bps, args.cost_commission, args.shares,
        )
    else:
        log("Phase 2 gate FAILED. Skipping Phase 3.")
        # Check if baseline alone is worth reporting
        for h_key, b in baseline.items():
            if b.get("baseline_cum_pnl_bps", 0) > 0:
                log(f"  NOTE: Baseline profitable at {h_key}: "
                    f"P&L={b['baseline_cum_pnl_bps']:.1f} bps")

    # Generate report
    build_report(output_dir, baseline, phase2b, phase2c, phase2d, phase3, gate_pass, passing)

    # Summary table to stdout
    log("\n=== Phase 2 Summary Table ===")
    log(f"{'Feature':25s} {'W':>3s} {'H':>4s} {'Uncond IC':>10s} {'Partial IC':>10s} "
        f"{'r_confound':>10s} {'Classification':>15s}")
    log("-" * 85)
    for t in (phase2c.get("partial_tests") or [])[:30]:
        log(f"{t['feature']:25s} {t['window']:3d} {t['horizon']:4d} "
            f"{t['unconditional_ic']:+10.3f} {t['partial_ic']:+10.3f} "
            f"{t['r_confound_prev_day']:+10.3f} {t['classification']:>15s}")

    log("\nE15 Approach 1 complete.")


if __name__ == "__main__":
    main()
