#!/usr/bin/env python3
"""
Run full 5-path evaluation pipeline on all 10 universality stock point-return exports.

Executes: IC + dCor+MI + Temporal IC + Transfer Entropy + Regime IC + JMI
        + Stability selection + Holdout validation + 4-tier classification

Usage:
    python scripts/run_universality_full_pipeline.py
    python scripts/run_universality_full_pipeline.py --stock crsp  # single stock
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hft_evaluator.config import EvaluationConfig
from hft_evaluator.pipeline import EvaluationPipeline


STOCKS = ["crsp", "snap", "hood", "mrna", "dkng", "fang", "isrg", "zm", "ibkr", "pep"]


def run_stock(stock: str, config_dir: Path, output_base: Path) -> dict:
    """Run full pipeline on one stock and save results."""
    config_path = config_dir / f"universality_{stock}_point_return.yaml"
    output_dir = output_base / f"universality_{stock}_full_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  {stock.upper()}: Full 5-path evaluation")
    print(f"  Config: {config_path.name}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}")

    config = EvaluationConfig.from_yaml(str(config_path))

    pipeline = EvaluationPipeline(config)
    t0 = time.time()
    result = pipeline.run()
    elapsed = time.time() - t0

    # Save classification table
    table_path = output_dir / "classification_table.json"
    # FeatureClassification is a dataclass — serialize manually
    table_data = {}
    for feat_name, feat_tier in result.per_feature.items():
        table_data[feat_name] = {
            "tier": feat_tier.tier.name if hasattr(feat_tier.tier, 'name') else str(feat_tier.tier),
            "passing_paths": feat_tier.passing_paths,
            "stability_pct": feat_tier.stability_pct,
            "best_p": feat_tier.best_p,
            "holdout_confirmed": getattr(feat_tier, "holdout_confirmed", False),
        }
    with open(table_path, "w") as f:
        json.dump(table_data, f, indent=2, default=str)

    # Extract summary
    tier_counts = {}
    for ft in result.per_feature.values():
        tier_name = ft.tier.name if hasattr(ft.tier, 'name') else str(ft.tier)
        tier_counts[tier_name] = tier_counts.get(tier_name, 0) + 1
    n_evaluated = len(result.per_feature)
    n_excluded = len(result.excluded_features) if result.excluded_features else 0

    # Identify non-price STRONG-KEEP features (the ones that matter)
    price_features = {f"ask_price_l{i}" for i in range(1, 11)} | {f"bid_price_l{i}" for i in range(1, 11)} | {"mid_price", "weighted_mid_price"}
    non_price_strong_keep = [
        name for name, ft in result.per_feature.items()
        if (ft.tier.name if hasattr(ft.tier, 'name') else str(ft.tier)) == "STRONG_KEEP"
        and name not in price_features
    ]

    summary = {
        "stock": stock.upper(),
        "elapsed_seconds": round(elapsed, 1),
        "n_features_evaluated": n_evaluated,
        "n_excluded": n_excluded,
        "strong_keep": tier_counts.get("STRONG_KEEP", 0),
        "keep": tier_counts.get("KEEP", 0),
        "investigate": tier_counts.get("INVESTIGATE", 0),
        "discard": tier_counts.get("DISCARD", 0),
        "non_price_strong_keep": non_price_strong_keep,
        "non_price_strong_keep_count": len(non_price_strong_keep),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results: STRONG-KEEP={summary['strong_keep']}, KEEP={summary['keep']}, "
          f"INVESTIGATE={summary['investigate']}, DISCARD={summary['discard']}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Full 5-path universality evaluation")
    parser.add_argument("--stock", default=None, help="Single stock (default: all 10)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config_dir = Path(__file__).resolve().parent.parent / "configs"
    output_base = Path(__file__).resolve().parent.parent / "outputs"

    stocks = [args.stock] if args.stock else STOCKS
    all_summaries = []

    for stock in stocks:
        try:
            summary = run_stock(stock, config_dir, output_base)
            all_summaries.append(summary)
        except Exception as e:
            print(f"\n  ERROR on {stock.upper()}: {e}")
            import traceback
            traceback.print_exc()
            all_summaries.append({"stock": stock.upper(), "error": str(e)})

    # Save consolidated summary
    consolidated_path = output_base / "universality_full_eval_summary.json"
    with open(consolidated_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n{'='*70}")
    print("CROSS-STOCK SUMMARY (Full 5-Path Pipeline)")
    print(f"{'='*70}")
    print(f"{'Stock':>6s} | {'S-KEEP':>6s} | {'KEEP':>6s} | {'INVEST':>6s} | {'DISCARD':>7s} | {'Time':>6s}")
    print("-" * 55)
    for s in all_summaries:
        if "error" in s:
            print(f"{s['stock']:>6s} | ERROR: {s['error'][:40]}")
        else:
            print(f"{s['stock']:>6s} | {s['strong_keep']:>6d} | {s['keep']:>6d} | "
                  f"{s['investigate']:>6d} | {s['discard']:>7d} | {s['elapsed_seconds']:>5.0f}s")

    print(f"\nConsolidated: {consolidated_path}")


if __name__ == "__main__":
    main()
