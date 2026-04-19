# hft-feature-evaluator

Profile-based feature characterization engine for HFT microstructure data. Evaluates features across 5 statistical paths with stability selection, holdout validation, and configurable selection criteria.

**Version**: 0.1.0 | **Tests**: 298 (across 23 test files) | **Last Updated**: 2026-04-20 (Phase 7 Stage 7.4 Round 4)

## Quick Links

- **Pre-training IC gate library** (mandatory per hft-rules §13): `from hft_evaluator.fast_gate import run_fast_gate, GateThresholds`. Used by `hft-ops::ValidationRunner` as a direct library import (not subprocess).
- **FeatureSet registry producer** (Phase 4 Batch 4a): `hft-ops evaluate --config <evaluator.yaml> --criteria <criteria.yaml> --save-feature-set <name>_v1 --applies-to-assets NVDA --applies-to-horizons 10` → writes `contracts/feature_sets/<name>_v1.json`.
- **Off-exchange gate** (Phase 5 Preview library port): `from hft_evaluator.experiments.offexchange_gate import run_offexchange_gate, GateCheckConfig, GateCheckResult`.
- **Archived fossils** (Phase 6 6D): 7 experimental scripts under `scripts/archive/` — NOT templates. See `scripts/archive/README.md` for replacement paths per hft-rules §4.

## Architecture

Two pipeline modes:

- **v1** (`run()`): 4-tier classification table (STRONG-KEEP / KEEP / INVESTIGATE / DISCARD)
- **v2** (`run_v2()`): Rich `FeatureProfile` output with per-path stability, redundancy analysis, and `SelectionCriteria`-based feature selection

### v2 Data Flow

```
NPY export (train/) --> loader (auto-detect MBO/off-exchange schema)
    --> build_cache() (2-pass: pre-screen + main computation)
    --> DataCache (daily IC cubes + pooled data, ~126 MB)
    --> 5 paths from cache:
        Path 1: Forward IC + IC_IR + BH correction
        Path 2: dCor + MI + BH correction
        Path 3a: Temporal rolling IC (rolling_mean, slope, RoC)
        Path 3b: Transfer entropy (informational only)
        Path 4: Regime-conditional IC (spread/time/activity terciles)
        Path 5: JMI forward selection (Brown et al. 2012)
    --> Per-path stability (date-bootstrap, no disk I/O)
    --> Redundancy (Spearman correlation, clustering, VIF)
    --> FeatureProfile per feature
    --> Holdout validation
    --> feature_profiles.json
```

## Install

```bash
# Requires sibling packages: hft-contracts, hft-metrics
pip install -e ".[dev]"
```

**Dependencies**: `hft-contracts` and `hft-metrics` are internal sibling packages in the HFT pipeline monorepo. They must be installed before this package.

## Usage

### Python API

```python
from hft_evaluator.config import EvaluationConfig
from hft_evaluator.pipeline import EvaluationPipeline

config = EvaluationConfig.from_yaml("configs/offexchange_34feat_lean.yaml")
pipeline = EvaluationPipeline(config)

# v2: Profile-based evaluation (recommended)
profiles = pipeline.run_v2()
pipeline.to_json_v2(profiles, "feature_profiles.json")

# v1: 4-tier classification (backward-compatible)
result = pipeline.run()
pipeline.to_json(result, "classification_table.json")
```

### Feature Selection from Profiles

```python
from hft_evaluator.criteria import SelectionCriteria, select_features

criteria = SelectionCriteria(
    min_combined_stability=0.6,
    max_cf_ratio=10.0,           # Exclude contemporaneous features
    max_vif=10.0,                # Exclude highly multicollinear
)
selected = select_features(profiles, criteria)
```

### CLI

```bash
# v1 output
evaluate --config configs/offexchange_34feat_lean.yaml

# v2 output
evaluate --config configs/offexchange_34feat_lean.yaml --v2
```

## Tests

```bash
pytest tests/ -v          # Full suite (298 tests, ~40 min due to permutation tests)
pytest tests/ -v -k "not pipeline and not stability"  # Fast subset (~7 sec)
```

## Documentation

| Document | Purpose |
|----------|---------|
| `CLAUDE.md` | LLM coding guide: architecture, module structure, design rules, known limitations |
| `CODEBASE.md` | Deep technical reference: every module, data flow, algorithms, testing strategy |
| `EVALUATION_REPORT.md` | Off-exchange 34-feature evaluation results (E10) |
| `MBO_EVALUATION_REPORT.md` | MBO 98-feature evaluation results (E11) |

## Module Structure

```
src/hft_evaluator/
    data/
        loader.py          # Schema-aware NPY loading (MBO + off-exchange)
        registry.py        # Feature group registry from hft-contracts
        holdout.py         # Holdout day reservation
        cache.py           # DataCache + build_cache() (v2 data layer)
    screening/
        ic_screening.py    # Path 1: Forward IC + IC_IR + BH
        dcor_screening.py  # Path 2: dCor + MI + BH
    selection/
        jmi_selection.py          # Path 5: JMI forward selection
        concurrent_forward.py     # Concurrent vs forward IC decomposition
    temporal/
        temporal_ic.py     # Path 3a: Temporal rolling IC
        transfer_entropy.py # Path 3b: TE screening (informational)
    regime/
        regime_ic.py       # Path 4: Regime-conditional IC
    stability/
        stability_selection.py  # v1: Paths 1+2 bootstrap. v2: per-path stability
    pipeline.py            # Orchestrates v1 run() + v2 run_v2()
    decision.py            # 5-path -> 4-tier classification (v1)
    profile.py             # FeatureProfile, PathEvidence, StabilityDetail (v2)
    criteria.py            # SelectionCriteria + select_features() (v2)
    feedback.py            # Model feedback protocol stub (STG, LOCO, IG)
    config.py              # YAML config parsing
    cli.py                 # CLI entry point
```
