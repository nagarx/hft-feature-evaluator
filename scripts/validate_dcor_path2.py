#!/usr/bin/env python3
"""Validate Path 2 (dCor+MI) with corrected permutation counts.

Runs ONLY the dCor+MI screening on CRSP point returns with 500 dCor perms
and 200 MI perms (vs the buggy 100/50 that was structurally incompatible
with BH correction on 712 tests).

Usage:
    python scripts/validate_dcor_path2.py --stock crsp
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from hft_evaluator.config import EvaluationConfig
from hft_evaluator.data.loader import ExportLoader
from hft_evaluator.data.registry import FeatureRegistry
from hft_evaluator.data.holdout import split_holdout
from hft_evaluator.screening.dcor_screening import screen_dcor


STOCKS = ["crsp", "snap", "hood", "mrna", "dkng", "fang", "isrg", "zm", "ibkr", "pep"]

# Feature names for reporting
FEATURE_NAMES = {
    0: "ask_price_l1", 40: "mid_price", 41: "spread", 42: "spread_bps",
    43: "total_bid_vol", 44: "total_ask_vol", 45: "vol_imbalance",
    54: "net_order_flow", 84: "true_ofi", 85: "depth_norm_ofi",
    86: "exec_pressure", 87: "signed_mp_delta", 90: "fragility",
    91: "depth_asymmetry",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", default="crsp")
    args = parser.parse_args()

    stock = args.stock
    config_path = f"configs/universality_{stock}_point_return.yaml"
    print(f"Loading config: {config_path}")

    config = EvaluationConfig.from_yaml(config_path)
    print(f"dCor permutations: {config.screening.dcor_permutations}")
    print(f"MI permutations: {config.screening.mi_permutations}")
    print(f"BH FDR level: {config.screening.bh_fdr_level}")

    loader = ExportLoader(config.export_dir, config.split)
    registry = FeatureRegistry(loader.schema)
    horizons = list(config.screening.horizons)

    all_dates = loader.list_dates()
    eval_dates, holdout_dates = split_holdout(all_dates, config.holdout_days)
    print(f"Eval dates: {len(eval_dates)}, Holdout: {len(holdout_dates)}")

    # Pre-screen
    evaluable = registry.evaluable_indices()
    print(f"Evaluable features: {len(evaluable)}")
    print(f"Total tests: {len(evaluable)} x {len(horizons)} = {len(evaluable) * len(horizons)}")

    print(f"\nRunning dCor+MI screening (Path 2) with {config.screening.dcor_permutations} dCor perms, "
          f"{config.screening.mi_permutations} MI perms...")
    t0 = time.time()

    dcor_results_dict = screen_dcor(
        loader=loader,
        evaluation_dates=eval_dates,
        evaluable_indices=evaluable,
        config=config,
    )
    # Flatten dict[feature_name][horizon] -> list[DcorScreeningResult]
    dcor_results = []
    for fname, h_dict in dcor_results_dict.items():
        for h, result in h_dict.items():
            dcor_results.append(result)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    # Count passes
    n_pass = sum(1 for r in dcor_results if r.passes_path2)
    print(f"\nPath 2 passes: {n_pass}/{len(dcor_results)}")

    # Report all results with raw values
    print(f"\n{'Feature':<20s} {'H':>3s} {'dCor':>8s} {'dCor_p':>8s} {'MI':>8s} {'MI_p':>8s} {'Pass':>5s}")
    print("-" * 60)

    # Sort by dCor descending
    sorted_results = sorted(dcor_results, key=lambda r: r.dcor_value, reverse=True)
    for r in sorted_results[:30]:  # top 30
        fname = FEATURE_NAMES.get(r.feature_index, r.feature_name)
        pass_str = "PASS" if r.passes_path2 else ""
        print(f"{fname:<20s} {r.horizon:>3d} {r.dcor_value:>8.4f} {r.dcor_p:>8.4f} "
              f"{r.mi_value:>8.4f} {r.mi_p:>8.4f} {pass_str:>5s}")

    # Save results
    output = {
        "stock": stock.upper(),
        "dcor_permutations": config.screening.dcor_permutations,
        "mi_permutations": config.screening.mi_permutations,
        "n_tests": len(dcor_results),
        "n_pass": n_pass,
        "elapsed_seconds": round(elapsed, 1),
        "top_results": [
            {
                "feature": FEATURE_NAMES.get(r.feature_index, r.feature_name),
                "feature_index": r.feature_index,
                "horizon": r.horizon,
                "dcor": round(r.dcor_value, 6),
                "dcor_p_bh": round(r.dcor_p, 6),
                "mi": round(r.mi_value, 6),
                "mi_p_bh": round(r.mi_p, 6),
                "passes": r.passes_path2,
            }
            for r in sorted_results[:50]
        ],
    }

    out_path = f"outputs/dcor_validation_{stock}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
