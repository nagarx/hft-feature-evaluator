# Evaluator Scripts — Archived Fossils (Phase 6 6D, 2026-04-17)

**STATUS**: Experimental fossils — NOT templates for new work.

Per [hft-rules §4](/.claude/rules/hft-rules.md#4-modularity--genericity),
new feature-evaluation work MUST be authored as:

1. **Library modules** under `hft_evaluator.experiments.*` with a
   `run(config) -> result` entry point (see `offexchange_gate.py` for the
   reference pattern shipped in Phase 5 Preview), OR
2. **hft-ops manifests** at `hft-ops/experiments/<name>.yaml` that
   dispatch to the `evaluate` stage + library modules.

New files under `scripts/` are reserved for:
- Production infra (`export_signals.py`, `run_universality_full_pipeline.py`).
- Data-prep utilities (must begin with the header
  `# DATA PREP UTILITY — not an experiment`).

## Archived Files (this directory)

| File | Original Purpose | Retired In | Replacement |
|---|---|---|---|
| `ridge_walkforward_mbo.py` (838 LOC) | Walk-forward Ridge on MBO features @ H10 | Phase 6 6D | Library module — author as `hft_evaluator.experiments.ridge_walkforward` with cross-sectional variant routing |
| `ridge_walkforward_mbo_trailing.py` (761 LOC) | Trailing-window Ridge variant | Phase 6 6D | Same — library module with `mode: trailing` parameter |
| `ridge_walkforward_mbo_zscore.py` (763 LOC) | Z-score-normalized Ridge variant | Phase 6 6D | Same — library module with `normalization: zscore` parameter |
| `signal_diagnostics.py` (974 LOC) | Full diagnostics report for event-based signals | Phase 6 6D | `hft_evaluator.run_v2()` + `FeatureProfile` surface (shipped in Phase 4) |
| `signal_diagnostics_mbo.py` (971 LOC) | Full diagnostics report for MBO 60s signals | Phase 6 6D | Same — use `run_v2()` with config `mbo_98feat_lean.yaml` |
| `e15_morning_signal_analysis.py` (1140 LOC) | E15 morning-hours ACF study (failed; see `project_e15_finding.md`) | Phase 6 6D | Library module if repeated; else archive permanently |
| `e16_extreme_event_study.py` (522 LOC) | E16 extreme-events signal study (failed; see `project_multi_asset_plan.md`) | Phase 6 6D | Library module if repeated; else archive permanently |

**Total archived**: ~5,969 LOC.

## Why Archived Rather Than Deleted?

1. **Reproducibility**: each script reproduces a historical experiment
   (E15 morning-signal, E16 extreme events, Ridge walk-forward series).
2. **Historical analysis methods**: the diagnostics scripts codify
   analytical methods (ACF + IC gate + bootstrap + percentile tables)
   that may be cited in future architecture plans.
3. **Preventing imitation**: LLM coders surveying the repo for templates
   will see the fossil header (`# STATUS: experimental fossil — NOT a
   template for new work.`) at the top of each script and be steered
   away from reviving the ad-hoc-script pattern.

## Restoration

If a genuine need arises to re-run an archived script:

```bash
# Check out the script at its archived path:
python hft-feature-evaluator/scripts/archive/<name>.py --help
```

They are NOT deleted — just relocated. For NEW evaluator work:
1. Author a library module under `hft_evaluator.experiments.*`.
2. Wire it into an hft-ops manifest that invokes the `evaluate` stage.
3. Commit + update `EXPERIMENT_INDEX.md` + run IC gate.
