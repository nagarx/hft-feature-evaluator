#!/usr/bin/env python3
"""
Batch convert all 10 universality SmoothedReturn exports to point-return exports.

For each stock:
1. Reads forward_prices.npy from universality_{symbol}_60s/
2. Computes point-to-point returns at 8 horizons [1,2,3,5,10,20,30,60]
3. Saves as _labels.npy (takes priority over _regression_labels.npy in ExportLoader)
4. Symlinks sequences and normalization files
5. Updates metadata

Usage:
    python scripts/prepare_universality_point_returns.py
    python scripts/prepare_universality_point_returns.py --stock crsp  # single stock
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


STOCKS = ["crsp", "snap", "hood", "mrna", "dkng", "fang", "isrg", "zm", "ibkr", "pep"]
K = 5  # smoothing window offset (base price at column k)
HORIZONS = [1, 2, 3, 5, 10, 20, 30, 60]
SPLITS = ["train", "val", "test"]


def log(msg: str) -> None:
    print(f"[prepare] {msg}", flush=True)


def compute_point_returns(forward_prices: np.ndarray, k: int,
                          horizons: list) -> np.ndarray:
    """Point-to-point returns from forward_prices.

    Formula: (fp[:, k+H] - fp[:, k]) / fp[:, k] * 10000 bps
    """
    n_samples, n_cols = forward_prices.shape
    labels = np.empty((n_samples, len(horizons)), dtype=np.float64)
    base_price = forward_prices[:, k]

    for h_idx, h in enumerate(horizons):
        col = k + h
        if col >= n_cols:
            raise ValueError(
                f"Horizon H={h} requires column {col} but forward_prices "
                f"has only {n_cols} columns"
            )
        future_price = forward_prices[:, col]
        valid = np.isfinite(base_price) & (np.abs(base_price) > 1e-10) & np.isfinite(future_price)
        pr = np.full(n_samples, np.nan, dtype=np.float64)
        pr[valid] = (future_price[valid] - base_price[valid]) / base_price[valid] * 10000.0
        labels[:, h_idx] = pr

    return labels


def process_split(source_dir: Path, target_dir: Path) -> dict:
    """Process one split: symlink sequences, compute point-return labels."""
    target_dir.mkdir(parents=True, exist_ok=True)
    stats = {"days": 0, "total_sequences": 0, "nan_labels": 0}

    meta_files = sorted(source_dir.glob("*_metadata.json"))
    for meta_path in meta_files:
        date_str = meta_path.name.replace("_metadata.json", "")

        fp_path = source_dir / f"{date_str}_forward_prices.npy"
        if not fp_path.exists():
            continue

        fp = np.load(str(fp_path))
        n_samples = fp.shape[0]
        labels = compute_point_returns(fp, K, HORIZONS)

        # Verify sequence count matches
        seq_path = source_dir / f"{date_str}_sequences.npy"
        if seq_path.exists():
            seq_shape = np.load(str(seq_path), mmap_mode="r").shape
            assert labels.shape[0] == seq_shape[0], (
                f"Label count {labels.shape[0]} != sequence count {seq_shape[0]}"
            )

        # Save point-return labels as _labels.npy (ExportLoader priority)
        np.save(str(target_dir / f"{date_str}_labels.npy"), labels)

        # Update metadata
        with open(meta_path) as f:
            meta = json.load(f)
        meta["label_strategy"] = "regression"
        meta["labeling"] = {
            "horizons": HORIZONS,
            "label_encoding": {
                "description": "PointReturn forward return in bps",
                "dtype": "float64",
                "format": "continuous_bps",
                "unit": "basis_points",
            },
            "label_mode": "regression",
            "num_horizons": len(HORIZONS),
            "return_type": "PointReturn",
        }
        meta["point_return_derivation"] = {
            "source": "forward_prices",
            "formula": "(fp[:, k+H] - fp[:, k]) / fp[:, k] * 10000",
            "k": K,
            "horizons": HORIZONS,
        }
        with open(target_dir / f"{date_str}_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Symlink sequences and normalization (unchanged data)
        for suffix in ["_sequences.npy", "_normalization.json"]:
            src = source_dir / f"{date_str}{suffix}"
            dst = target_dir / f"{date_str}{suffix}"
            if src.exists() and not dst.exists():
                os.symlink(str(src.resolve()), str(dst))

        stats["days"] += 1
        stats["total_sequences"] += n_samples
        stats["nan_labels"] += int(np.sum(~np.isfinite(labels)))

    return stats


def process_stock(symbol: str, base_dir: Path) -> None:
    """Convert one stock's export to point returns."""
    source = base_dir / f"universality_{symbol}_60s"
    target = base_dir / f"universality_{symbol}_60s_point_return"

    if not source.exists():
        log(f"  {symbol.upper()}: source not found at {source}, SKIPPING")
        return

    log(f"\n{'='*60}")
    log(f"  {symbol.upper()}: {source.name} → {target.name}")
    log(f"  Horizons: {HORIZONS}, k={K}")

    target.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        source_dir = source / split
        target_dir = target / split
        if not source_dir.exists():
            continue
        stats = process_split(source_dir, target_dir)
        log(f"  {split}: {stats['days']} days, {stats['total_sequences']} sequences, "
            f"{stats['nan_labels']} NaN labels")

    # Copy and update manifest
    manifest_src = source / "dataset_manifest.json"
    if manifest_src.exists():
        with open(manifest_src) as f:
            manifest = json.load(f)
        manifest["label_info"] = {
            "strategy": "regression",
            "return_type": "PointReturn",
            "horizons": HORIZONS,
            "units": "basis_points",
        }
        with open(target / "dataset_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Batch convert universality exports to point returns")
    parser.add_argument("--stock", default=None, help="Single stock to process (default: all 10)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent.parent / "data" / "exports"
    log(f"Base directory: {base_dir}")
    log(f"Horizons: {HORIZONS}")

    stocks = [args.stock] if args.stock else STOCKS
    for symbol in stocks:
        process_stock(symbol, base_dir)

    log(f"\n{'='*60}")
    log("All stocks processed. Point-return exports ready for full pipeline evaluation.")


if __name__ == "__main__":
    main()
