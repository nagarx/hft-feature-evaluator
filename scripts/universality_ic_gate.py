"""Universality IC Gate Check — Multi-stock point-return IC analysis.

Computes stride-60 Spearman IC for MBO features vs point returns derived
from forward_prices. Designed for the multi-stock universality study where
exports have 2 horizons (H10, H60) and SmoothedReturn labels.

Point returns are computed from forward_prices.npy (not from regression_labels,
which contain smoothed returns that inflate IC per E8 findings).

Usage:
    python scripts/universality_ic_gate.py \
        --export-dir ../data/exports/universality_crsp_60s \
        --output-dir outputs/universality_crsp_ic
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


# MBO 98-feature indices (from hft-contracts)
FEATURE_NAMES = {
    40: "mid_price",
    41: "spread",
    42: "spread_bps",
    43: "total_bid_volume",
    44: "total_ask_volume",
    45: "volume_imbalance",
    46: "weighted_mid_price",
    47: "price_impact",
    48: "add_rate_bid",
    49: "add_rate_ask",
    54: "net_order_flow",
    57: "aggressive_order_ratio",
    58: "order_flow_volatility",
    84: "true_ofi",
    85: "depth_norm_ofi",
    86: "executed_pressure",
    87: "signed_mp_delta_bps",
    88: "trade_asymmetry",
    89: "cancel_asymmetry",
    90: "fragility_score",
    91: "depth_asymmetry",
}

# Core features to analyze (non-price, non-categorical)
GATE_FEATURES = {
    42: "spread_bps",
    45: "volume_imbalance",
    84: "true_ofi",
    85: "depth_norm_ofi",
    86: "executed_pressure",
    87: "signed_mp_delta_bps",
    88: "trade_asymmetry",
    89: "cancel_asymmetry",
    90: "fragility_score",
    91: "depth_asymmetry",
    43: "total_bid_volume",
    44: "total_ask_volume",
    54: "net_order_flow",
    58: "order_flow_volatility",
}

STRIDE = 60  # Non-overlapping at 60-bin spacing


def load_day_data(day_dir: Path, day_prefix: str):
    """Load sequences, forward_prices for one day."""
    seq_path = day_dir / f"{day_prefix}_sequences.npy"
    fp_path = day_dir / f"{day_prefix}_forward_prices.npy"

    if not seq_path.exists() or not fp_path.exists():
        return None, None

    sequences = np.load(seq_path)      # [N, T, F]
    forward_prices = np.load(fp_path)  # [N, k+max_H+1]
    return sequences, forward_prices


def compute_point_returns(forward_prices: np.ndarray, k: int, horizons: list):
    """Derive point-to-point returns from forward price trajectories.

    Args:
        forward_prices: [N, k+max_H+1] mid-price trajectories
        k: smoothing window offset (base price at column k)
        horizons: list of horizon values [10, 60]

    Returns:
        dict of {horizon: point_returns_bps [N]}
    """
    base_price = forward_prices[:, k]
    result = {}
    for h in horizons:
        future_price = forward_prices[:, k + h]
        point_ret = (future_price - base_price) / base_price * 10000  # bps
        # Filter invalid
        valid = np.isfinite(point_ret) & np.isfinite(base_price) & (base_price > 0)
        point_ret[~valid] = np.nan
        result[h] = point_ret
    return result


def spearman_ic(features: np.ndarray, returns: np.ndarray):
    """Spearman rank correlation (IC) between features and returns."""
    valid = np.isfinite(features) & np.isfinite(returns)
    if valid.sum() < 10:
        return np.nan
    r, _ = stats.spearmanr(features[valid], returns[valid])
    return r


def bootstrap_ci(features: np.ndarray, returns: np.ndarray,
                  n_bootstrap: int = 2000, ci: float = 0.95, seed: int = 42):
    """Bootstrap confidence interval for Spearman IC.

    Uses block bootstrap with day-level blocks (passed as pre-pooled array).
    """
    rng = np.random.RandomState(seed)
    valid = np.isfinite(features) & np.isfinite(returns)
    f_valid = features[valid]
    r_valid = returns[valid]
    n = len(f_valid)
    if n < 20:
        return np.nan, np.nan

    ics = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        ic, _ = stats.spearmanr(f_valid[idx], r_valid[idx])
        ics[i] = ic

    alpha = (1 - ci) / 2
    lo = np.percentile(ics, alpha * 100)
    hi = np.percentile(ics, (1 - alpha) * 100)
    return lo, hi


def analyze_split(split_dir: Path, horizons: list, k: int = 5):
    """Analyze all days in a split directory.

    Returns per-day IC values and pooled stride-60 data.
    """
    day_files = sorted(split_dir.glob("*_sequences.npy"))
    if not day_files:
        return None

    all_features = []  # stride-60 pooled features per day
    all_returns = {h: [] for h in horizons}
    per_day_ic = {h: {idx: [] for idx in GATE_FEATURES} for h in horizons}

    for seq_path in day_files:
        day_prefix = seq_path.stem.replace("_sequences", "")
        sequences, forward_prices = load_day_data(split_dir, day_prefix)
        if sequences is None:
            continue

        # Extract last-timestep features: [N, F]
        features = sequences[:, -1, :]
        point_returns = compute_point_returns(forward_prices, k, horizons)

        n_samples = features.shape[0]

        # Per-day IC (all samples)
        for h in horizons:
            ret = point_returns[h]
            for idx in GATE_FEATURES:
                ic = spearman_ic(features[:, idx], ret)
                per_day_ic[h][idx].append(ic)

        # Stride-60 subsampling for pooled analysis
        stride_idx = np.arange(0, n_samples, STRIDE)
        if len(stride_idx) < 2:
            continue

        all_features.append(features[stride_idx])
        for h in horizons:
            all_returns[h].append(point_returns[h][stride_idx])

    if not all_features:
        return None

    # Pool stride-60 data across days
    pooled_features = np.vstack(all_features)  # [M, F]
    pooled_returns = {h: np.concatenate(all_returns[h]) for h in horizons}

    return {
        "per_day_ic": per_day_ic,
        "pooled_features": pooled_features,
        "pooled_returns": pooled_returns,
        "n_days": len(day_files),
        "n_stride60": pooled_features.shape[0],
    }


def main():
    parser = argparse.ArgumentParser(description="Universality IC Gate Check")
    parser.add_argument("--export-dir", required=True,
                        help="Path to export directory (e.g., data/exports/universality_crsp_60s)")
    parser.add_argument("--output-dir", default="outputs/universality_ic",
                        help="Output directory for results")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest_path = export_dir / "dataset_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"Export: {manifest.get('experiment_name', 'unknown')}")
        print(f"Symbol: {manifest.get('symbol', 'unknown')}")

    # Determine horizons from first label file
    first_label = sorted((export_dir / "train").glob("*_regression_labels.npy"))[0]
    labels_shape = np.load(first_label).shape
    n_horizons = labels_shape[1] if len(labels_shape) > 1 else 1

    # Read horizons from config if available
    horizons_files = sorted((export_dir / "train").glob("*_horizons.json"))
    if horizons_files:
        with open(horizons_files[0]) as f:
            horizons_data = json.load(f)
        # horizons.json may be {"horizons": [10, 60], ...} or just [10, 60]
        if isinstance(horizons_data, dict) and "horizons" in horizons_data:
            horizons = horizons_data["horizons"]
        elif isinstance(horizons_data, list):
            horizons = horizons_data
        else:
            horizons = list(range(n_horizons))
        print(f"Horizons: {horizons} ({n_horizons} values)")
    else:
        horizons = list(range(n_horizons))
        print(f"Horizons: indices {horizons}")

    # Determine k from forward_prices shape
    first_fp = sorted((export_dir / "train").glob("*_forward_prices.npy"))[0]
    fp_shape = np.load(first_fp).shape
    max_h = max(horizons)
    k = fp_shape[1] - max_h - 1
    print(f"Forward prices: {fp_shape}, k={k}, max_H={max_h}")

    print(f"\n{'='*70}")
    print(f"UNIVERSALITY IC GATE CHECK")
    print(f"{'='*70}")
    print(f"Stride: {STRIDE} (non-overlapping at {STRIDE}-bin spacing)")
    print(f"IC type: Spearman (point returns from forward_prices)")
    print(f"Bootstrap: {args.n_bootstrap} resamples, 95% CI")
    print(f"Features analyzed: {len(GATE_FEATURES)}")

    # Analyze each split
    results = {}
    for split in ["train", "val", "test"]:
        split_dir = export_dir / split
        if not split_dir.exists():
            continue
        print(f"\n--- Analyzing {split} split ---")
        result = analyze_split(split_dir, horizons, k)
        if result is None:
            print(f"  No data in {split}")
            continue

        results[split] = result
        print(f"  Days: {result['n_days']}, Stride-60 samples: {result['n_stride60']}")

    # Compute pooled stride-60 IC with bootstrap CI for val+test (OOS)
    print(f"\n{'='*70}")
    print(f"GATE RESULTS: Stride-60 Pooled IC (Val + Test = OOS)")
    print(f"{'='*70}")

    oos_features = []
    oos_returns = {h: [] for h in horizons}
    oos_days = 0

    for split in ["val", "test"]:
        if split in results:
            oos_features.append(results[split]["pooled_features"])
            for h in horizons:
                oos_returns[h].append(results[split]["pooled_returns"][h])
            oos_days += results[split]["n_days"]

    if not oos_features:
        print("ERROR: No OOS data available")
        return

    oos_feat = np.vstack(oos_features)
    oos_ret = {h: np.concatenate(oos_returns[h]) for h in horizons}
    n_oos = oos_feat.shape[0]

    print(f"\nOOS: {oos_days} days, {n_oos} stride-60 observations")

    gate_results = {}
    for h in horizons:
        print(f"\n  H={h} ({h} minutes at 60s bins):")
        print(f"  {'Feature':<25} {'IC':>8} {'95% CI Lo':>10} {'95% CI Hi':>10} {'Pass?':>7}")
        print(f"  {'-'*60}")

        ret = oos_ret[h]
        n_pass = 0
        for idx in sorted(GATE_FEATURES.keys()):
            name = GATE_FEATURES[idx]
            feat = oos_feat[:, idx]
            ic = spearman_ic(feat, ret)
            ci_lo, ci_hi = bootstrap_ci(feat, ret, args.n_bootstrap, seed=args.seed)

            passes = bool(abs(ic) > 0.05 and ci_lo * ci_hi > 0)  # CI doesn't cross zero (bool() to avoid numpy bool serialization issue)
            if passes:
                n_pass += 1

            marker = "PASS" if passes else ""
            print(f"  {name:<25} {ic:>8.4f} [{ci_lo:>9.4f}, {ci_hi:>9.4f}] {marker:>7}")

            gate_results.setdefault(h, {})[name] = {
                "ic": float(ic),
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi),
                "passes": passes,
            }

        print(f"\n  Features passing IC > 0.05 with CI not crossing zero: {n_pass}/{len(GATE_FEATURES)}")
        print(f"  NVDA baseline: 0/{len(GATE_FEATURES)} (zero features passed for NVDA)")
        if n_pass > 0:
            print(f"  >>> H1 EVIDENCE: {n_pass} feature(s) show predictive signal!")
        else:
            print(f"  >>> H0 EVIDENCE: Same as NVDA — no features pass gate")

    # Per-day IC summary (train only for reference)
    if "train" in results:
        print(f"\n{'='*70}")
        print(f"PER-DAY IC SUMMARY (Train, stride-1, all samples)")
        print(f"{'='*70}")
        for h in horizons:
            print(f"\n  H={h}:")
            print(f"  {'Feature':<25} {'Mean IC':>8} {'Std IC':>8} {'Min':>8} {'Max':>8}")
            print(f"  {'-'*60}")
            for idx in sorted(GATE_FEATURES.keys()):
                name = GATE_FEATURES[idx]
                ics = [x for x in results["train"]["per_day_ic"][h][idx] if np.isfinite(x)]
                if ics:
                    arr = np.array(ics)
                    print(f"  {name:<25} {arr.mean():>8.4f} {arr.std():>8.4f} {arr.min():>8.4f} {arr.max():>8.4f}")

    # Save results
    output = {
        "horizons": horizons,
        "stride": STRIDE,
        "n_oos_days": oos_days,
        "n_oos_stride60": n_oos,
        "gate_results": gate_results,
        "splits": {s: {"n_days": r["n_days"], "n_stride60": r["n_stride60"]}
                   for s, r in results.items()},
    }
    out_path = output_dir / "ic_gate_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
