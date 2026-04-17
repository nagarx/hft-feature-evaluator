"""hft_evaluator.experiments — runnable experiment library (Phase 5).

Each module in this package exposes a ``run(config) -> result`` entry point
that can be called directly from Python, from a notebook, or from a sweep
manifest. Previously these lived as ``scripts/*.py`` files that baked in
constants and bypassed the ledger.

Migration pattern (Phase 5 Preview, 2026-04-16):

1. Move script LOGIC into a module here (sweep-orchestratable, testable).
2. Replace the original `scripts/*.py` with a thin deprecation shim that
   calls `run()` so prior reports (EXPERIMENT_INDEX.md, etc.) remain
   byte-for-byte reproducible.
3. Archive the original logic is NOT required — the shim preserves the
   script path and invokes the new library module.

Currently shipped:
- `offexchange_gate` — 3-gate filter for off-exchange signal features (E14).

See hft-rules §4 for the "no new ad-hoc scripts" policy that governs
future additions.
"""

from __future__ import annotations

from hft_evaluator.experiments.offexchange_gate import (
    GateCheckConfig,
    GateCheckFinding,
    GateCheckResult,
    run as run_offexchange_gate,
)

__all__ = [
    "GateCheckConfig",
    "GateCheckFinding",
    "GateCheckResult",
    "run_offexchange_gate",
]
