#!/usr/bin/env python3
"""DEPRECATED shim — use `hft_evaluator.experiments.run_offexchange_gate`.

Phase 5 Preview (2026-04-16): the logic moved to
``hft_evaluator/experiments/offexchange_gate.py``. This file is preserved as
a thin shim so:

1. The historical E14 experiment record
   (``lob-model-trainer/EXPERIMENT_INDEX.md:1618``) remains reproducible by
   running ``python scripts/offexchange_gate_check.py`` exactly as before.
2. Any external reference (prior `git log`, bookmarked paths) still works.

New code should import the library directly::

    from hft_evaluator.experiments import (
        GateCheckConfig,
        run_offexchange_gate,
    )

    config = GateCheckConfig(
        export_dir="/path/to/basic_nvda_60s",
        output_dir="outputs/offexchange_gate_check",
    )
    result = run_offexchange_gate(config)
    if result.verdict == "PASS":
        ...

The library version can be composed in sweep manifests
(``hft-ops/experiments/sweeps/*.yaml``); this shim cannot.
"""

from __future__ import annotations

import warnings


def main() -> None:
    warnings.warn(
        "scripts/offexchange_gate_check.py is deprecated (Phase 5 Preview, "
        "2026-04-16). Import `hft_evaluator.experiments.run_offexchange_gate` "
        "directly; the script is preserved only for E14 reproducibility.",
        DeprecationWarning,
        stacklevel=2,
    )
    from hft_evaluator.experiments import (
        GateCheckConfig,
        run_offexchange_gate,
    )

    # Preserve legacy defaults byte-for-byte. See
    # hft_evaluator/experiments/offexchange_gate.py for the full Config.
    config = GateCheckConfig(
        export_dir="../data/exports/basic_nvda_60s",
        output_dir="outputs/offexchange_gate_check",
    )
    result = run_offexchange_gate(config)
    print(f"[gate] verdict: {result.verdict} ({len(result.passes)} pass / {len(result.findings)} total)")


if __name__ == "__main__":
    main()
