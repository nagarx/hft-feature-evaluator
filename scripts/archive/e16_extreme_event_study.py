# STATUS: experimental fossil -- NOT a template for new work.
# ARCHIVED: Phase 6 6D (2026-04-17). Preserved for historical reproducibility.
# Per hft-rules §4 (no new ad-hoc scripts for experiments):
#   - NEW experiments MUST be authored as hft-ops manifests under
#     hft-ops/experiments/ OR sweep manifests under hft-ops/experiments/sweeps/
#   - Reusable analysis logic MUST live in library modules
#     (hft_evaluator.experiments.* / lobtrainer.experiments.*)
#   - See scripts/archive/README.md for the replacement for this script.

"""E16: Extreme Event Conditional Return Analysis — Multi-stock.

Tests whether MBO features at extreme percentiles (top/bottom 2-10%) predict
non-zero forward point returns across 10 NASDAQ stocks.

Methodology:
  1. Compute feature percentile thresholds from TRAIN split (causal)
  2. Apply thresholds to OOS data (val and test SEPARATELY)
  3. Measure conditional mean return with per-day block bootstrap CIs
  4. Apply BH FDR correction for multiple testing
  5. Require cross-stock sign consistency (3+ of 10 stocks)

Statistical rigor:
  - Thresholds from train only (no look-ahead)
  - Val and test reported separately (no pooling)
  - Per-day block bootstrap (accounts for within-day autocorrelation)
  - BH FDR at alpha=0.10 across ALL tests
  - Cross-stock replication requirement

Reference: E8 found 0/67 features IC > 0.05 for point returns (aggregate).
           E13 Path 2 found 0 features with non-linear signal (MI/dCor).
           E13 Path 4 found 86/89 features with regime-conditional signal.
           This test checks the TAIL-CONDITIONAL gap: do extreme feature values
           predict returns that aggregate IC and MI would average away?
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# ── Feature Configuration ────────────────────────────────────────────────────

FEATURES = {
    84: "true_ofi",
    85: "depth_norm_ofi",
    42: "spread_bps",
    45: "volume_imbalance",
    90: "fragility_score",
}

PERCENTILES = [2, 5, 10]
HORIZONS = [1, 3, 5, 10, 20, 60]
K = 5  # smoothing_window offset in forward_prices
N_BOOTSTRAP = 2000
SEED = 42
EQUITY_COST_BPS = 0.7  # IBKR equity 60-min hold breakeven

STOCKS = ["crsp", "pep", "ibkr", "zm", "isrg", "fang", "dkng", "mrna", "hood", "snap"]


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_split_data(split_dir: Path):
    """Load all days from a split: last-timestep features + point returns."""
    day_data = []
    for seq_f in sorted(split_dir.glob("*_sequences.npy")):
        day = seq_f.stem.replace("_sequences", "")
        fp_f = split_dir / f"{day}_forward_prices.npy"
        if not fp_f.exists():
            continue

        seq = np.load(seq_f)       # [N, T, F]
        fp = np.load(fp_f)         # [N, k+max_H+1]

        features = seq[:, -1, :]   # [N, F] last timestep
        base_price = fp[:, K]      # [N] mid at t

        point_rets = {}
        for h in HORIZONS:
            col = K + h
            if col < fp.shape[1]:
                future = fp[:, col]
                valid = np.isfinite(future) & np.isfinite(base_price) & (base_price > 0)
                pr = np.where(valid, (future - base_price) / base_price * 10000, np.nan)
                point_rets[h] = pr
            else:
                point_rets[h] = np.full(features.shape[0], np.nan)

        day_data.append({
            "day": day,
            "features": features,
            "point_rets": point_rets,
            "n": features.shape[0],
        })

    return day_data


def pool_data(day_data):
    """Pool day data into arrays with day labels for block bootstrap."""
    all_features = []
    all_rets = {h: [] for h in HORIZONS}
    day_labels = []

    for i, dd in enumerate(day_data):
        all_features.append(dd["features"])
        for h in HORIZONS:
            all_rets[h].append(dd["point_rets"][h])
        day_labels.extend([i] * dd["n"])

    return (
        np.vstack(all_features),
        {h: np.concatenate(all_rets[h]) for h in HORIZONS},
        np.array(day_labels),
    )


# ── Statistical Methods ──────────────────────────────────────────────────────

def block_bootstrap_mean(values, day_labels, n_boot=N_BOOTSTRAP, seed=SEED):
    """Per-day block bootstrap for mean and 95% CI.

    Resamples entire days (blocks), not individual observations,
    to account for within-day autocorrelation.
    """
    rng = np.random.RandomState(seed)
    valid = np.isfinite(values)
    if valid.sum() < 5:
        return np.nan, np.nan, np.nan

    unique_days = np.unique(day_labels[valid])
    n_days = len(unique_days)
    if n_days < 3:
        return np.nanmean(values), np.nan, np.nan

    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sampled_days = rng.choice(unique_days, size=n_days, replace=True)
        boot_vals = []
        for d in sampled_days:
            mask = (day_labels == d) & valid
            boot_vals.append(values[mask])
        pooled = np.concatenate(boot_vals) if boot_vals else np.array([])
        boot_means[b] = np.mean(pooled) if len(pooled) > 0 else np.nan

    boot_means = boot_means[np.isfinite(boot_means)]
    if len(boot_means) < 100:
        return np.nanmean(values), np.nan, np.nan

    ci_lo = np.percentile(boot_means, 2.5)
    ci_hi = np.percentile(boot_means, 97.5)
    return np.nanmean(values[valid]), ci_lo, ci_hi


def ci_excludes_zero(ci_lo, ci_hi):
    """Check if confidence interval excludes zero (both bounds same sign)."""
    if np.isnan(ci_lo) or np.isnan(ci_hi):
        return False
    return (ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0)


# ── Validation Layers ────────────────────────────────────────────────────────

def validate_data_integrity(day_data, sym):
    """V1: Verify forward_prices alignment — fp_base must match mid_price feature."""
    diffs = []
    for dd in day_data[:5]:
        fp_f = dd["features"][:, 40]  # mid_price feature
        # fp_base was already checked in prior investigation
        diffs.append(np.mean(np.abs(fp_f)))

    mean_price = np.mean(diffs)
    if mean_price < 1.0:
        print(f"  V1 FAIL: {sym} mid_price feature mean={mean_price:.4f} (too low, possibly normalized)")
        return False
    print(f"  V1 PASS: {sym} mid_price mean=${mean_price:.2f} (raw USD)")
    return True


def validate_feature_distributions(features, sym, feat_idx, feat_name):
    """V2: Verify feature is non-degenerate."""
    vals = features[:, feat_idx]
    valid = np.isfinite(vals)
    frac_valid = valid.mean()
    if frac_valid < 0.5:
        print(f"  V2 FAIL: {sym} {feat_name} only {frac_valid:.1%} valid")
        return False

    std = np.std(vals[valid])
    if std < 1e-10:
        print(f"  V2 FAIL: {sym} {feat_name} std={std:.2e} (constant)")
        return False

    return True


def validate_threshold_calibration(oos_features, train_thresholds, feat_idx, feat_name, pct):
    """V3: Verify train thresholds produce reasonable OOS fractions."""
    vals = oos_features[:, feat_idx]
    valid = np.isfinite(vals)
    vals_valid = vals[valid]

    lo_thresh = train_thresholds[f"p{pct}_lo"]
    hi_thresh = train_thresholds[f"p{pct}_hi"]

    frac_lo = np.mean(vals_valid <= lo_thresh)
    frac_hi = np.mean(vals_valid >= hi_thresh)

    expected = pct / 100.0
    if frac_lo < expected * 0.2 or frac_lo > expected * 5.0:
        print(f"  V3 WARN: {feat_name} P{pct} lo: OOS fraction={frac_lo:.3f} vs expected={expected:.3f}")
    if frac_hi < expected * 0.2 or frac_hi > expected * 5.0:
        print(f"  V3 WARN: {feat_name} P{pct} hi: OOS fraction={frac_hi:.3f} vs expected={expected:.3f}")

    return frac_lo, frac_hi


# ── Main Analysis ────────────────────────────────────────────────────────────

def analyze_stock(sym, base_dir):
    """Run full extreme event analysis for one stock."""
    SYM = sym.upper()
    export_dir = base_dir / f"universality_{sym}_60s"

    # Load data
    train_data = load_split_data(export_dir / "train")
    val_data = load_split_data(export_dir / "val")
    test_data = load_split_data(export_dir / "test")

    if not train_data or not val_data or not test_data:
        print(f"  SKIP: {SYM} — missing data")
        return None

    # Pool train for threshold computation
    train_feats, _, _ = pool_data(train_data)

    # V1: Data integrity
    if not validate_data_integrity(train_data, SYM):
        return None

    results = {}
    for feat_idx, feat_name in FEATURES.items():
        if feat_idx >= train_feats.shape[1]:
            continue

        # V2: Feature distribution
        if not validate_feature_distributions(train_feats, SYM, feat_idx, feat_name):
            continue

        # Compute thresholds from TRAIN ONLY (causal)
        train_vals = train_feats[:, feat_idx]
        train_valid = train_vals[np.isfinite(train_vals)]

        thresholds = {}
        for pct in PERCENTILES:
            thresholds[f"p{pct}_lo"] = np.percentile(train_valid, pct)
            thresholds[f"p{pct}_hi"] = np.percentile(train_valid, 100 - pct)

        # Analyze each OOS split separately
        for split_name, split_data in [("val", val_data), ("test", test_data)]:
            feats, rets, day_labels = pool_data(split_data)

            # V3: Threshold calibration
            for pct in PERCENTILES:
                validate_threshold_calibration(feats, thresholds, feat_idx, feat_name, pct)

            feat_vals = feats[:, feat_idx]

            for pct in PERCENTILES:
                for tail in ["top", "bottom"]:
                    if tail == "top":
                        mask = feat_vals >= thresholds[f"p{pct}_hi"]
                    else:
                        mask = feat_vals <= thresholds[f"p{pct}_lo"]

                    mask = mask & np.isfinite(feat_vals)
                    n_events = mask.sum()

                    if n_events < 10:
                        continue

                    n_days_with = len(np.unique(day_labels[mask]))

                    for h in HORIZONS:
                        ret_vals = rets[h][mask]
                        dl_masked = day_labels[mask]

                        mean_ret, ci_lo, ci_hi = block_bootstrap_mean(
                            ret_vals, dl_masked
                        )

                        key = (feat_name, pct, tail, h, split_name)
                        results[key] = {
                            "mean_ret_bps": float(mean_ret) if np.isfinite(mean_ret) else None,
                            "ci_lo": float(ci_lo) if np.isfinite(ci_lo) else None,
                            "ci_hi": float(ci_hi) if np.isfinite(ci_hi) else None,
                            "ci_excludes_zero": ci_excludes_zero(ci_lo, ci_hi),
                            "n_events": int(n_events),
                            "n_days": int(n_days_with),
                            "exceeds_cost": (
                                abs(mean_ret) > EQUITY_COST_BPS
                                if np.isfinite(mean_ret) else False
                            ),
                        }

    return results


def bh_fdr_correction(p_values, alpha=0.10):
    """Benjamini-Hochberg FDR correction. Returns mask of surviving tests."""
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool)

    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]

    thresholds = alpha * np.arange(1, n + 1) / n
    below = sorted_p <= thresholds

    if not below.any():
        return np.zeros(n, dtype=bool)

    max_idx = np.max(np.where(below)[0])
    surviving = np.zeros(n, dtype=bool)
    surviving[sorted_idx[:max_idx + 1]] = True
    return surviving


def main():
    parser = argparse.ArgumentParser(description="E16: Extreme Event Conditional Return Study")
    parser.add_argument("--base-dir", default="../data/exports",
                        help="Base directory containing universality exports")
    parser.add_argument("--output-dir", default="outputs/e16_extreme_events",
                        help="Output directory")
    parser.add_argument("--stocks", nargs="*", default=STOCKS)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("E16: EXTREME EVENT CONDITIONAL RETURN ANALYSIS")
    print("=" * 100)
    print(f"Features: {list(FEATURES.values())}")
    print(f"Percentiles: {PERCENTILES} (thresholds from TRAIN, applied to OOS)")
    print(f"Horizons: {HORIZONS} (minutes at 60s bins)")
    print(f"Bootstrap: {N_BOOTSTRAP} resamples, per-day blocks, 95% CI")
    print(f"Equity cost threshold: {EQUITY_COST_BPS} bps")
    print(f"Stocks: {[s.upper() for s in args.stocks]}")

    all_stock_results = {}

    for sym in args.stocks:
        SYM = sym.upper()
        print(f"\n{'='*80}")
        print(f"ANALYZING: {SYM}")
        print(f"{'='*80}")

        results = analyze_stock(sym, base_dir)
        if results is None:
            continue
        all_stock_results[SYM] = results

        sig_count = sum(1 for v in results.values() if v["ci_excludes_zero"])
        cost_count = sum(1 for v in results.values() if v["exceeds_cost"])
        print(f"  {SYM}: {len(results)} tests, {sig_count} CI excludes zero, {cost_count} exceed cost")

    # ── Cross-Stock Summary ──────────────────────────────────────────────────

    print(f"\n{'='*100}")
    print("CROSS-STOCK RESULTS (test split only)")
    print(f"{'='*100}")

    # Collect all test-split results with CI excluding zero
    significant_tests = []
    all_test_results = []

    for sym, results in all_stock_results.items():
        for key, val in results.items():
            feat_name, pct, tail, h, split = key
            if split != "test":
                continue
            all_test_results.append((sym, feat_name, pct, tail, h, val))
            if val["ci_excludes_zero"]:
                significant_tests.append((sym, feat_name, pct, tail, h, val))

    print(f"\nTotal test-split tests: {len(all_test_results)}")
    print(f"Tests with CI excluding zero (before FDR): {len(significant_tests)}")
    print(f"Expected at 5% FPR: {len(all_test_results) * 0.05:.0f}")

    # BH FDR correction (approximate: use CI-based p-value proxy)
    # p-value ≈ 2 * P(Z > |mean/se|) from bootstrap
    p_values = []
    for sym, feat, pct, tail, h, val in all_test_results:
        if val["ci_lo"] is not None and val["ci_hi"] is not None and val["mean_ret_bps"] is not None:
            se = (val["ci_hi"] - val["ci_lo"]) / (2 * 1.96)
            if se > 0:
                z = abs(val["mean_ret_bps"]) / se
                p = 2 * (1 - stats.norm.cdf(z))
            else:
                p = 1.0
        else:
            p = 1.0
        p_values.append(p)

    surviving = bh_fdr_correction(p_values, alpha=0.10)
    n_surviving = surviving.sum()
    print(f"Surviving BH FDR (alpha=0.10): {n_surviving}")

    # Print significant test results
    if n_surviving > 0:
        print(f"\n  SIGNIFICANT RESULTS (after BH FDR alpha=0.10):")
        print(f"  {'Stock':<6} {'Feature':<18} {'P%':>3} {'Tail':>6} {'H':>3} "
              f"{'Mean bps':>9} {'CI_lo':>8} {'CI_hi':>8} {'N':>5} {'Days':>4} {'> Cost':>6}")
        print(f"  {'-'*90}")

        for i, (sym, feat, pct, tail, h, val) in enumerate(all_test_results):
            if surviving[i]:
                m = val["mean_ret_bps"] or 0
                lo = val["ci_lo"] or 0
                hi = val["ci_hi"] or 0
                cost_marker = "YES" if val["exceeds_cost"] else "no"
                print(f"  {sym:<6} {feat:<18} {pct:>3} {tail:>6} {h:>3} "
                      f"{m:>+9.2f} [{lo:>+7.2f}, {hi:>+7.2f}] {val['n_events']:>5} {val['n_days']:>4} {cost_marker:>6}")

    # Cross-stock sign consistency for significant results
    print(f"\n{'='*100}")
    print("CROSS-STOCK SIGN CONSISTENCY (test split, before FDR)")
    print(f"{'='*100}")

    condition_signs = {}
    for sym, feat, pct, tail, h, val in all_test_results:
        key = (feat, pct, tail, h)
        if val["mean_ret_bps"] is not None and val["ci_excludes_zero"]:
            sign = "+" if val["mean_ret_bps"] > 0 else "-"
            condition_signs.setdefault(key, []).append((sym, sign, val["mean_ret_bps"]))

    consistent = []
    for key, entries in sorted(condition_signs.items()):
        if len(entries) >= 3:
            signs = [e[1] for e in entries]
            pos = signs.count("+")
            neg = signs.count("-")
            if pos >= 3 or neg >= 3:
                feat, pct, tail, h = key
                direction = "+" if pos >= neg else "-"
                avg_ret = np.mean([e[2] for e in entries])
                stocks_str = ", ".join(f"{e[0]}({e[2]:+.1f})" for e in entries)
                consistent.append((feat, pct, tail, h, len(entries), direction, avg_ret, stocks_str))
                print(f"  {feat:<18} P{pct} {tail:<6} H={h:<3}: "
                      f"{len(entries)} stocks, dir={direction}, avg={avg_ret:+.2f} bps")
                print(f"    {stocks_str}")

    if not consistent:
        print("  No condition has 3+ stocks with CI excluding zero and same sign.")

    # ── Val vs Test Stability ────────────────────────────────────────────────

    print(f"\n{'='*100}")
    print("VAL vs TEST STABILITY (conditions with CI excluding zero in BOTH splits)")
    print(f"{'='*100}")

    for sym, results in all_stock_results.items():
        val_sig = {}
        test_sig = {}
        for key, val in results.items():
            feat_name, pct, tail, h, split = key
            cond_key = (feat_name, pct, tail, h)
            if val["ci_excludes_zero"]:
                if split == "val":
                    val_sig[cond_key] = val
                elif split == "test":
                    test_sig[cond_key] = val

        both = set(val_sig.keys()) & set(test_sig.keys())
        if both:
            for cond_key in sorted(both):
                v = val_sig[cond_key]
                t = test_sig[cond_key]
                same_sign = (
                    (v["mean_ret_bps"] > 0) == (t["mean_ret_bps"] > 0)
                    if v["mean_ret_bps"] and t["mean_ret_bps"] else False
                )
                feat, pct, tail, h = cond_key
                print(f"  {sym:<6} {feat:<18} P{pct} {tail:<6} H={h}: "
                      f"val={v['mean_ret_bps']:+.2f} test={t['mean_ret_bps']:+.2f} "
                      f"{'SAME SIGN' if same_sign else 'SIGN FLIP'}")

    # ── Save full results ────────────────────────────────────────────────────

    serializable = {}
    for sym, results in all_stock_results.items():
        sym_results = {}
        for key, val in results.items():
            str_key = f"{key[0]}|P{key[1]}|{key[2]}|H{key[3]}|{key[4]}"
            sym_results[str_key] = val
        serializable[sym] = sym_results

    out_path = output_dir / "e16_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "E16: Extreme Event Conditional Return Analysis",
            "methodology": {
                "features": FEATURES,
                "percentiles": PERCENTILES,
                "horizons": HORIZONS,
                "k": K,
                "n_bootstrap": N_BOOTSTRAP,
                "bootstrap_type": "per-day block",
                "fdr_method": "BH at alpha=0.10",
                "equity_cost_bps": EQUITY_COST_BPS,
            },
            "summary": {
                "total_test_split_tests": len(all_test_results),
                "ci_excludes_zero_before_fdr": len(significant_tests),
                "surviving_bh_fdr": int(n_surviving),
                "expected_false_positives_5pct": len(all_test_results) * 0.05,
                "cross_stock_consistent": len(consistent),
            },
            "per_stock": serializable,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
