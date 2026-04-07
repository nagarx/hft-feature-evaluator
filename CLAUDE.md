# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Build Commands

```bash
# Install (editable, with dev dependencies)
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run a single test category
pytest tests/test_decision/ -v
pytest tests/test_screening/ -v

# Run the evaluator CLI
evaluate --config configs/offexchange_34feat.yaml

# Run with coverage
pytest tests/ --cov=hft_evaluator --cov-report=term-missing
```

## Architecture Overview

Feature evaluation implementing the 5-path decision framework from `feature_evaluation_research/FEATURE_EVALUATION_FRAMEWORK.md`. Two pipeline modes:

- **v1** (`run()`): 4-tier classification table (backward-compatible)
- **v2** (`run_v2()`): Rich FeatureProfile output with per-path stability, redundancy, ACF, and SelectionCriteria-based feature selection

### v2 Data Flow (preferred)

```
NPY export (train/) → loader (auto-detect schema) → holdout split
    → build_cache() (2-pass: pre-screen + main computation)
    → DataCache (daily IC cubes + pooled data)
    → 5 paths from cache (IC, dCor+MI, temporal, TE, regime, JMI)
    → per-path stability from cache
    → redundancy (Spearman corr matrix, clustering, VIF)
    → ACF of daily IC series
    → FeatureProfile per feature
    → holdout validation
    → feature_profiles.json (v2 schema, superset of v1)
    → select_features(profiles, criteria) → feature set for training
```

### v1 Data Flow (backward-compatible)

```
NPY export (train/) → loader → holdout split
    → 5 paths (IC, dCor+MI, temporal, regime, JMI)
    → stability selection (bootstraps × Paths 1+2)
    → decision (classify_feature → 4-tier)
    → holdout validation
    → classification_table.json
```

### Module Structure

| Module | Purpose | Framework Reference |
|--------|---------|-------------------|
| `data/loader.py` | Schema-aware NPY loading | — |
| `data/registry.py` | Feature group registry from hft-contracts | — |
| `data/holdout.py` | Holdout day reservation | §6.5 |
| `data/cache.py` | DataCache + build_cache() (v2 data layer) | — |
| `screening/ic_screening.py` | Forward IC + IC_IR + BH (+ `_from_cache`) | §2.1, §6.2 Path 1 |
| `screening/dcor_screening.py` | dCor + MI + BH (+ `_from_cache`) | §2.1, §6.2 Path 2 |
| `selection/jmi_selection.py` | JMI forward selection (+ `_from_cache`) | §2.2, §6.2 Path 5 |
| `selection/concurrent_forward.py` | Concurrent vs forward IC (+ `_from_cache`) | §4 |
| `temporal/temporal_ic.py` | Temporal IC + rolling features (+ `_from_cache`) | §2.3, §6.2 Path 3 |
| `temporal/transfer_entropy.py` | TE screening (informational, + `_from_cache`) | §2.3, §6.2 Path 3 |
| `regime/regime_ic.py` | Regime-conditional IC (+ `_from_cache`) | §6.2 Path 4 |
| `stability/stability_selection.py` | v1: Paths 1+2 bootstrap. v2: per-path stability from cache | §6.3 |
| `pipeline.py` | Orchestrates v1 `run()` + v2 `run_v2()` | §6 |
| `decision.py` | 5-path → 4-tier classification | §6.6 |
| `profile.py` | FeatureProfile, PathEvidence, StabilityDetail, compute_tier | — |
| `criteria.py` | SelectionCriteria + select_features() | — |
| `feedback.py` | Model feedback protocol stub (STG, LOCO, IG) | — |
| `config.py` | YAML config parsing | — |
| `cli.py` | CLI entry point (`evaluate --config ... [--v2]`) | — |

### Dependencies

```toml
hft-contracts     # Feature names, categorical indices, off-exchange schema
hft-metrics       # IC, dCor, MI, bootstrap, BH, quantile_buckets, transfer_entropy
numpy>=1.24       # Array operations
pyyaml>=6.0       # Config parsing
```

### Key Design Rules

1. **Framework is authoritative**: All mathematical definitions, formulas, and decision rules are in FEATURE_EVALUATION_FRAMEWORK.md. This package implements them. Do not invent new paths or change thresholds without updating the framework first.
2. **v2 uses DataCache**: `build_cache()` loads data in 2 passes (pre-screen + main). All paths read from cache — no path touches the loader directly. This reduces I/O from 49-109 disk passes to 2.
3. **v2 stability is per-path**: `combined_stability = max(path1, path2, path3a)`. Features passing only Paths 3-5 are no longer auto-DISCARDed (the critical bug fix from v1).
4. **BH correction per test family**: Each screening module applies BH within its own test family.
5. **Holdout is for STRONG-KEEP only**: STRONG-KEEP candidates that fail holdout downgrade to KEEP.
6. **TE is informational**: Transfer entropy results are stored but NOT used in classification (structurally cannot survive BH with permutation counts).
7. **Per-test seeds**: `_test_seed(base, j, h_idx, lag_idx)` ensures each permutation test gets a unique deterministic seed. No shared seed across (feature, horizon) pairs.
8. **Profiles over verdicts**: v2 produces FeatureProfile (rich characterization) not FeatureTier (4-tier verdict). SelectionCriteria matches against profiles for per-experiment feature selection.

### Decision Algorithm

```python
def classify_feature(passing_paths, stability_pct, best_p, holdout_confirmed, config):
    if len(passing_paths) == 0 or stability_pct < config.investigate_threshold:
        return "DISCARD"
    if stability_pct < config.stable_threshold:
        return "INVESTIGATE"
    if best_p < config.strong_keep_p and holdout_confirmed:
        return "STRONG-KEEP"
    return "KEEP"
```

### External Dependencies

- `hft-contracts` — OffExchangeFeatureIndex, FeatureIndex, categorical indices
- `hft-metrics` — all statistical computations
- NPY export at `data/exports/` — input data

### Known Limitations & Safeguards

**This evaluator is a pre-training model-free tool.** It measures statistical relationships between individual features and returns. It CANNOT detect features whose value only emerges within a trained neural network.

**Feature types the evaluator CANNOT detect:**

| Blind Spot | Example | Why Invisible | Resolution |
|---|---|---|---|
| Context/conditioning features | dark_share, odd_lot_ratio, time_regime | Zero unconditional IC; value is as transformer attention context | STG/IG+TSR via feedback.py |
| Interaction-only features | Feature A useful only when combined with B | Zero individual signal; XOR-like patterns | STG/LOCO-Group |
| Early-timestep features | Signal at t=5 within 20-step window, zero at t=20 | All paths evaluate last timestep only | Per-timestep IC curve, IG+TSR |
| Architecture-specific features | Features the transformer's attention exploits | No model in the evaluation loop | STG, LassoNet |

**Mandatory safeguards when using evaluation results:**

1. **NEVER use DISCARD as a hard gate for model training.** DISCARD means "low pre-training signal" — NOT "useless." Use KEEP+ for initial training; add INVESTIGATE features in a second round. Features with zero IC may have high interaction or context value.
2. **Always include known context features** (dark_share, session_progress, time_regime, odd_lot_ratio) in model training regardless of tier — they have documented value as conditioning variables (Framework §3.2).
3. **After training, compare model attribution with evaluation tiers.** If the model assigns high importance to a DISCARDed feature, the evaluator missed an interaction or temporal pattern.
4. **Use `include_names` in SelectionCriteria** to force-include features known to be valuable from prior experiments or domain knowledge, bypassing pre-training statistical gates.

**Path to closing these gaps:** The `feedback.py` module defines `ModelFeedbackProvider` — a protocol for lob-model-trainer to export STG gate values, LOCO importance, and IG+TSR attributions back into feature profiles. Once implemented, `SelectionCriteria` can filter on model-derived importance, enabling the evaluator to rescue features that fail pre-training screens but are valuable to the trained model.

### What Does NOT Belong Here

- Model training or model architectures (use lob-model-trainer, lob-models)
- Backtesting or P&L computation (use lob-backtester)
- Diagnostic analysis (use lob-dataset-analyzer)
- Feature extraction from raw data (use feature-extractor or basic-quote-processor)
- STG, LassoNet, IG+TSR (implemented in lob-model-trainer; feedback protocol in `feedback.py`)
