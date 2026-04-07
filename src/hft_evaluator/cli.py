"""CLI entry point for hft-feature-evaluator.

Usage:
    evaluate --config configs/offexchange_34feat_lean.yaml
    evaluate --config configs/mbo_98feat_lean.yaml --output results.json
"""

import argparse
import sys

from hft_evaluator.config import EvaluationConfig
from hft_evaluator.pipeline import EvaluationPipeline


def main() -> int:
    """Run feature evaluation from the command line.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description="Evaluate features using the 5-path framework",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: classification_table.json or feature_profiles.json)",
    )
    parser.add_argument(
        "--v2", action="store_true",
        help="Use profile-based v2 pipeline (produces FeatureProfile output)",
    )

    args = parser.parse_args()

    try:
        config = EvaluationConfig.from_yaml(args.config)
        pipeline = EvaluationPipeline(config)

        if args.v2:
            profiles = pipeline.run_v2()
            output_path = args.output or "feature_profiles.json"
            pipeline.to_json_v2(profiles, output_path)
        else:
            result = pipeline.run()
            output_path = args.output or "classification_table.json"
            pipeline.to_json(result, output_path)

        print(f"[evaluate] Output written to {output_path}")
        return 0
    except Exception as exc:
        print(f"[evaluate] Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
