#!/usr/bin/env python3
"""
Prepare MBO point-return export from existing e5_timebased_60s forward_prices.

Creates a new export directory with:
- Symlinked sequences and normalization files (unchanged)
- New point-return labels at 8 horizons [1,2,3,5,10,20,30,60]
- Updated metadata reflecting point_return label type

No Rust re-export needed — computes labels directly from forward_prices.

Usage:
    python scripts/prepare_mbo_point_return.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np


# =============================================================================
# Configuration
# =============================================================================

SOURCE_EXPORT = "../data/exports/e5_timebased_60s"
TARGET_EXPORT = "../data/exports/e5_timebased_60s_point_return"
SMOOTHING_WINDOW_OFFSET = 5  # k=5, base price at column 5
HORIZONS = [1, 2, 3, 5, 10, 20, 30, 60]
SPLITS = ["train", "val", "test"]


def log(msg: str) -> None:
    print(f"[prepare] {msg}", flush=True)


def compute_point_returns(forward_prices: np.ndarray, k: int,
                           horizons: list[int]) -> np.ndarray:
    """Compute point-return labels from forward_prices.

    Formula: point_return(H) = (fp[:, k+H] - fp[:, k]) / fp[:, k] * 10000

    Args:
        forward_prices: [N, n_cols] float64 mid-prices in USD.
        k: Smoothing window offset (base price column index).
        horizons: List of horizon values (bins forward).

    Returns:
        labels: [N, len(horizons)] float64 point returns in basis points.
    """
    n_samples, n_cols = forward_prices.shape
    n_horizons = len(horizons)
    labels = np.empty((n_samples, n_horizons), dtype=np.float64)

    base_price = forward_prices[:, k]

    for h_idx, h in enumerate(horizons):
        col = k + h
        if col >= n_cols:
            raise ValueError(
                f"Horizon H={h} requires column {col} but forward_prices "
                f"has only {n_cols} columns"
            )
        future_price = forward_prices[:, col]

        # Guard: NaN for zero or non-finite base prices
        valid = np.isfinite(base_price) & (np.abs(base_price) > 1e-10)
        valid &= np.isfinite(future_price)

        pr = np.full(n_samples, np.nan, dtype=np.float64)
        pr[valid] = (
            (future_price[valid] - base_price[valid])
            / base_price[valid]
            * 10000.0
        )
        labels[:, h_idx] = pr

    return labels


def update_metadata(original_meta: dict, horizons: list[int],
                     n_sequences: int, labels: np.ndarray) -> dict:
    """Create updated metadata for point-return export."""
    meta = json.loads(json.dumps(original_meta))  # deep copy

    # Update labeling section
    meta["label_strategy"] = "regression"  # keep as regression (contract requirement)
    meta["labeling"] = {
        "horizons": horizons,
        "label_encoding": {
            "description": "PointReturn forward return in bps at each horizon",
            "dtype": "float64",
            "format": "continuous_bps",
            "unit": "basis_points",
        },
        "label_mode": "regression",
        "num_horizons": len(horizons),
        "return_type": "PointReturn",
    }

    # Update label distribution (based on first horizon)
    finite_labels = labels[:, 0]
    finite_labels = finite_labels[np.isfinite(finite_labels)]
    meta["label_distribution"] = {
        "positive": int(np.sum(finite_labels > 0)),
        "negative": int(np.sum(finite_labels < 0)),
        "zero": int(np.sum(finite_labels == 0)),
    }

    # Add provenance note
    meta["point_return_derivation"] = {
        "source": "forward_prices",
        "formula": "(fp[:, k+H] - fp[:, k]) / fp[:, k] * 10000",
        "k": SMOOTHING_WINDOW_OFFSET,
        "horizons": horizons,
        "units": "basis_points",
    }

    return meta


def process_split(source_dir: Path, target_dir: Path) -> dict:
    """Process one split: create symlinks + compute labels."""
    target_dir.mkdir(parents=True, exist_ok=True)

    # Find all days by listing metadata files
    meta_files = sorted(source_dir.glob("*_metadata.json"))
    stats = {"days": 0, "total_sequences": 0, "nan_labels": 0}

    for meta_path in meta_files:
        date_str = meta_path.name.replace("_metadata.json", "")

        # Read original metadata
        with open(meta_path) as f:
            original_meta = json.load(f)

        # Load forward prices
        fp_path = source_dir / f"{date_str}_forward_prices.npy"
        if not fp_path.exists():
            log(f"  WARNING: {fp_path.name} missing, skipping {date_str}")
            continue

        fp = np.load(str(fp_path))  # [N, 306] float64
        n_samples = fp.shape[0]

        # Compute point-return labels
        labels = compute_point_returns(fp, SMOOTHING_WINDOW_OFFSET, HORIZONS)
        n_nan = int(np.sum(~np.isfinite(labels)))

        # Verify shape consistency with sequences
        seq_path = source_dir / f"{date_str}_sequences.npy"
        if seq_path.exists():
            seq_shape = np.load(str(seq_path), mmap_mode="r").shape
            assert labels.shape[0] == seq_shape[0], (
                f"Label count {labels.shape[0]} != sequence count {seq_shape[0]} "
                f"for {date_str}"
            )

        # Save new labels as _labels.npy (takes priority in ExportLoader)
        label_path = target_dir / f"{date_str}_labels.npy"
        np.save(str(label_path), labels)

        # Update and save metadata
        updated_meta = update_metadata(original_meta, HORIZONS, n_samples, labels)
        meta_out = target_dir / f"{date_str}_metadata.json"
        with open(meta_out, "w") as f:
            json.dump(updated_meta, f, indent=2)

        # Symlink sequences and normalization (unchanged)
        for suffix in ["_sequences.npy", "_normalization.json"]:
            src = source_dir / f"{date_str}{suffix}"
            dst = target_dir / f"{date_str}{suffix}"
            if src.exists() and not dst.exists():
                os.symlink(str(src.resolve()), str(dst))

        stats["days"] += 1
        stats["total_sequences"] += n_samples
        stats["nan_labels"] += n_nan

    return stats


def main():
    script_dir = Path(__file__).resolve().parent.parent
    source_root = (script_dir / SOURCE_EXPORT).resolve()
    target_root = (script_dir / TARGET_EXPORT).resolve()

    log(f"Source: {source_root}")
    log(f"Target: {target_root}")
    log(f"Horizons: {HORIZONS}")
    log(f"Smoothing window offset (k): {SMOOTHING_WINDOW_OFFSET}")

    if not source_root.exists():
        log(f"ERROR: source export not found at {source_root}")
        sys.exit(1)

    target_root.mkdir(parents=True, exist_ok=True)

    # Process each split
    for split in SPLITS:
        source_dir = source_root / split
        target_dir = target_root / split

        if not source_dir.exists():
            log(f"  Split '{split}' not found in source, skipping")
            continue

        log(f"Processing {split}...")
        stats = process_split(source_dir, target_dir)
        log(f"  {split}: {stats['days']} days, {stats['total_sequences']} sequences, "
            f"{stats['nan_labels']} NaN labels")

    # Copy dataset_manifest.json if it exists (update horizons)
    manifest_src = source_root / "dataset_manifest.json"
    if manifest_src.exists():
        with open(manifest_src) as f:
            manifest = json.load(f)
        manifest["label_info"] = {
            "strategy": "regression",
            "return_type": "PointReturn",
            "horizons": HORIZONS,
            "units": "basis_points",
        }
        with open(target_root / "dataset_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        log("Updated dataset_manifest.json")

    log("Done.")

    # Final verification: load one day and check
    log("\n=== VERIFICATION ===")
    test_split = target_root / "train"
    first_meta = sorted(test_split.glob("*_metadata.json"))[0]
    date_str = first_meta.name.replace("_metadata.json", "")

    labels = np.load(str(test_split / f"{date_str}_labels.npy"))
    seq = np.load(str(test_split / f"{date_str}_sequences.npy"), mmap_mode="r")
    with open(first_meta) as f:
        meta = json.load(f)

    log(f"Day: {date_str}")
    log(f"Sequences: {seq.shape}, Labels: {labels.shape}")
    log(f"Label strategy: {meta['label_strategy']}")
    log(f"Return type: {meta['labeling']['return_type']}")
    log(f"Horizons: {meta['labeling']['horizons']}")
    log(f"Schema version: {meta['schema_version']}")
    log(f"Label sample (first row): {labels[0]}")
    log(f"Any NaN: {np.any(np.isnan(labels))}")
    log(f"Label stats H=10 (col 4): mean={np.mean(labels[:, 4]):.2f}, "
        f"std={np.std(labels[:, 4]):.2f} bps")


if __name__ == "__main__":
    main()
