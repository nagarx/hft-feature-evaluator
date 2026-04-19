# hft-feature-evaluator: Codebase Technical Reference

> **Version**: 0.1.0 | **Tests**: 298 (across 23 test files) | **Last Updated**: 2026-04-20 (Phase 7 Stage 7.4 Round 4)
>
> **Phase 7 state (cumulative)**: (a) Phase 2b `fast_gate.py` library — `run_fast_gate()` entry point + `GateReport`/`GateThresholds`; consumed by `hft_ops.stages.validation::ValidationRunner` as library import. (b) Phase 4 Batch 4a FeatureSet producer primitives — `EvaluationPipeline.last_profile_hash` + module-level `compute_profile_hash(profiles)` + `SelectionCriteria.from_yaml/from_dict` + `criteria_schema_version` field + `_KNOWN_CRITERIA_KEYS` strict-rejection frozenset. Consumed by `hft_ops.feature_sets.producer::produce_feature_set` (Phase 4 Batch 4b lives hft-ops-side). (c) Phase 5 Preview `experiments/offexchange_gate.py` library port with schema `offexchange_gate_check_v2` + thin deprecation shim at `scripts/offexchange_gate_check.py`. (d) Phase 6 6D — 7 experimental-fossil scripts archived to `scripts/archive/` with fossil headers + migration map per hft-rules §4 (~5,969 LOC moved). (e) Phase 7 Stage 7.4 Round 4 — `GateReport` docstring now references `ExperimentRecord.gate_reports["validation"]` (Phase 7 Round 4 Option C generic `gate_reports` field); `verdict` field unchanged for back-compat, hft-ops validation adapter injects lowercased `status` at captured_metrics boundary (Phase 7 Round 5).
>
> **Purpose**: Feature evaluation implementing the 5-path decision framework
> **Framework**: `feature_evaluation_research/FEATURE_EVALUATION_FRAMEWORK.md` (741 lines, authoritative reference)
> **Dependencies**: hft-contracts, hft-metrics, numpy, pyyaml, scipy
> **Output (v1)**: 4-tier classification table (STRONG-KEEP / KEEP / INVESTIGATE / DISCARD)
> **Output (v2)**: FeatureProfile per feature + SelectionCriteria-based feature selection

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Package Structure](#2-package-structure)
3. [Data Layer](#3-data-layer) (loader, registry, holdout, cache)
4. [Screening Layer (Framework Layer 1)](#4-screening-layer) (IC, dCor+MI)
5. [Selection Layer (Framework Layer 2)](#5-selection-layer) (JMI, concurrent/forward)
6. [Temporal Layer (Framework Layer 3)](#6-temporal-layer) (temporal IC, transfer entropy)
7. [Regime Layer (Framework Path 4)](#7-regime-layer)
8. [Stability Layer (Framework Section 6.3)](#8-stability-layer)
9. [Pipeline Orchestration](#9-pipeline-orchestration) (v1 `run()` and v2 `run_v2()`)
10. [Decision Logic](#10-decision-logic) (v1: FeatureTier, v2: FeatureProfile + compute_tier)
11. [Selection Criteria](#11-selection-criteria) (v2: SelectionCriteria + select_features)
12. [Model Feedback Protocol](#12-model-feedback-protocol)
13. [Known Limitations and Safeguards](#13-known-limitations-and-safeguards)
14. [Configuration](#14-configuration)
15. [Output Format](#15-output-format) (v1 and v2 JSON schemas)
16. [CLI](#16-cli)
17. [Memory Strategy](#17-memory-strategy)
18. [Testing Strategy](#18-testing-strategy)
19. [Integration with Pipeline](#19-integration-with-pipeline)
20. [Deferred Work](#20-deferred-work)

---

## 1. Architecture Overview

### Purpose

Reads ANY NPY export (MBO or off-exchange), runs 5 evaluation paths with per-path stability selection, and produces per-feature characterizations. Two pipeline modes:

- **v1** (`run()`): 4-tier classification table. Backward-compatible with v0.1.0 consumers.
- **v2** (`run_v2()`): Rich FeatureProfile output with per-path stability, redundancy analysis (Spearman corr, clustering, VIF), ACF of daily IC series, and declarative feature selection via SelectionCriteria. Preferred for all new work.

### Design Boundary

| Concern | hft-feature-evaluator | lob-dataset-analyzer |
|---------|----------------------|---------------------|
| Question answered | "Should we KEEP this feature?" | "What does this feature look like?" |
| Statistical rigor | BH correction, stability selection, holdout validation | Exploratory, no correction |
| Output | FeatureProfile (v2) or classification table (v1) | JSON diagnostic reports |
| Multi-path | 5 paths aggregated | Single-metric per analyzer |

### v2 Data Flow (preferred)

```
NPY export (train/) --> loader (auto-detect schema) --> holdout split
    --> build_cache() (2-pass: pre-screen + main computation)
    --> DataCache (daily IC cubes + pooled data)
    --> 5 paths from cache (IC, dCor+MI, temporal, TE, regime, JMI)
    --> per-path stability from cache
    --> redundancy (Spearman corr matrix, clustering, VIF)
    --> ACF of daily IC series
    --> FeatureProfile per feature
    --> holdout validation
    --> feature_profiles.json (v2 schema, superset of v1)
    --> select_features(profiles, criteria) --> feature set for training
```

### v1 Data Flow (backward-compatible)

```
NPY export (train/) --> loader --> holdout split
    --> 5 paths (IC, dCor+MI, temporal, regime, JMI)
    --> stability selection (50 bootstraps x Paths 1+2)
    --> decision (classify_feature --> 4-tier)
    --> holdout validation
    --> classification_table.json
```

**Critical difference**: v1 stability bootstraps only Paths 1+2 (IC and dCor+MI). Features passing only Paths 3-5 get stability_pct=0.0 and are auto-DISCARDed. v2 fixes this bug with per-path stability (combined = max(path1, path2, path3a)). v1 does 49-109 disk passes across all paths; v2 does exactly 2 via DataCache.

### Dependency Graph

```
hft-contracts (feature names, categorical indices, off-exchange schema)
    |
hft-metrics (IC, dCor, MI, bootstrap, BH, quantile_buckets, transfer_entropy,
             correlation_matrix, cluster_by_correlation, vif, autocorrelation,
             welford, temporal rolling features)
    |
hft-feature-evaluator (THIS PACKAGE -- orchestration + domain logic)
```

---

## 2. Package Structure

```
hft-feature-evaluator/
  src/hft_evaluator/
    __init__.py

    # === Data Layer ===
    data/
      __init__.py
      loader.py              Schema-aware NPY loader (auto-detects MBO vs off-exchange)
      registry.py            Feature group registry (from hft-contracts)
      holdout.py             Train+val holdout reservation logic
      cache.py               DataCache + build_cache() (v2 data layer)

    # === Screening Layer (Framework Layer 1) ===
    screening/
      __init__.py            _test_seed(), bh_adjusted_pvalues()
      dcor_screening.py      dCor + MI independence screening with BH correction
                             (screen_dcor + screen_dcor_from_cache)
      ic_screening.py        Forward IC with bootstrap CI and IC_IR
                             (screen_ic + screen_ic_from_cache)

    # === Selection Layer (Framework Layer 2) ===
    selection/
      __init__.py
      jmi_selection.py       JMI forward selection (Brown et al. 2012)
                             (jmi_forward_selection + jmi_from_cache)
      concurrent_forward.py  Concurrent vs forward IC decomposition
                             (decompose_concurrent_forward + compute_cf_from_cache)

    # === Temporal Layer (Framework Layer 3) ===
    temporal/
      __init__.py
      temporal_ic.py         IC at each of T timesteps + rolling feature IC
                             (compute_temporal_ic + compute_temporal_ic_from_cache)
      transfer_entropy.py    Transfer entropy screening with BH correction
                             (screen_transfer_entropy + screen_te_from_cache)

    # === Regime Layer (Framework Path 4) ===
    regime/
      __init__.py
      regime_ic.py           Regime-conditional IC (spread/time/activity terciles)
                             (compute_regime_ic + compute_regime_ic_from_cache)

    # === Stability Layer (Framework Section 6.3) ===
    stability/
      __init__.py
      stability_selection.py v1: stability_selection() -- Paths 1+2 bootstrap
                             v2: compute_stability_from_cache() -- per-path stability

    # === Orchestration ===
    pipeline.py              EvaluationPipeline: v1 run() + v2 run_v2()
    decision.py              v1: 5-path aggregation --> 4-tier classification
    profile.py               v2: FeatureProfile, PathEvidence, StabilityDetail, compute_tier()
    criteria.py              v2: SelectionCriteria + select_features()
    feedback.py              Model feedback protocol stub (STG, LOCO, IG+TSR)
    config.py                EvaluationConfig (YAML-driven, immutable, validated)
    cli.py                   CLI entry point: evaluate --config ... [--v2]

  configs/
    offexchange_34feat.yaml               Off-exchange evaluation config (full)
    offexchange_34feat_lean.yaml          Off-exchange lean config
    mbo_98feat_lean.yaml                  MBO stable features lean config
    mbo_98feat_point_return_lean.yaml     MBO point-return lean config
    universality_*_point_return.yaml      Per-asset universality study configs (10 assets)

  tests/
    test_cache.py                         DataCache and build_cache() tests
    test_config.py                        EvaluationConfig parsing and validation tests
    test_concurrent_forward.py            CF decomposition tests
    test_criteria.py                      SelectionCriteria + select_features() tests
    test_dcor_screening.py                dCor+MI screening tests
    test_decision.py                      classify_feature golden tests (all 4 tiers)
    test_feedback.py                      ModelFeedbackProvider protocol tests
    test_holdout.py                       split_holdout reservation tests
    test_ic_screening.py                  IC screening + BH correction tests
    test_jmi_selection.py                 JMI forward selection tests
    test_loader.py                        ExportLoader auto-detection tests
    test_pipeline.py                      End-to-end pipeline tests (v1 + v2)
    test_profile.py                       FeatureProfile + compute_tier() tests
    test_regime_ic.py                     Regime-conditional IC tests
    test_registry.py                      FeatureRegistry tests
    test_stability.py                     Stability selection tests (v1 + v2)
    test_temporal_ic.py                   Temporal IC tests
    test_transfer_entropy_eval.py         Transfer entropy screening tests

  pyproject.toml
```

---

## 3. Data Layer

### 3.1 `data/loader.py` -- Schema-Aware NPY Loader

```python
class ExportLoader:
    """Loads ANY pipeline export by auto-detecting schema from metadata JSON."""

    def __init__(self, export_dir: str, split: str = "train"):
        """Reads first metadata file in split/ directory.
        Detects schema via 'contract_version' field:
            'off_exchange_1.0' --> off-exchange (34 features, 20 timesteps)
            absent or 'mbo_*'  --> MBO (98-148 features, 100 timesteps)
        """

    def load_day(self, date: str) -> DayBundle:
        """Load sequences [N,T,F], labels [N,H], metadata for one day."""

    def iter_days(self, dates: list[str] | None = None) -> Iterator[DayBundle]:
        """Streaming day-by-day iteration. Sorted by date.
        If dates is provided, iterates only over those dates.
        """

    def list_dates(self) -> list[str]:
        """All available dates in the split directory."""

    @property
    def schema(self) -> ExportSchema:
        """Detected schema properties."""

class ExportSchema:
    schema_version: str        # "2.2" or "1.0"
    contract_version: str      # "off_exchange_1.0" or None
    n_features: int            # 98, 34, etc.
    window_size: int           # 100 or 20
    horizons: list[int]        # [10,20,50,100,200] or [1,2,3,5,10,20,30,60]
    bin_size_seconds: int | None  # 60 for time-based, None for event-based
    feature_names: dict[int, str]  # From hft-contracts
    categorical_indices: set   # Non-evaluable features (from contract)

class DayBundle:
    date: str
    sequences: np.ndarray      # [N, T, F] float32
    labels: np.ndarray         # [N, H] float64
    metadata: dict
```

**Auto-detection verified against actual export metadata**:
```json
{
  "contract_version": "off_exchange_1.0",
  "bin_size_seconds": 60,
  "n_features": 34,
  "window_size": 20,
  "horizons": [1, 2, 3, 5, 10, 20, 30, 60]
}
```

### 3.2 `data/registry.py` -- Feature Group Registry

```python
class FeatureRegistry:
    """Maps feature indices to groups and evaluation properties.

    Sources from hft-contracts:
        Off-exchange: OffExchangeFeatureIndex (indices 0-33, 10 groups)
        MBO: FeatureIndex (indices 0-97) + ExperimentalFeatureIndex (98-147)

    Properties per feature:
        - group name (e.g., 'signed_flow', 'bbo_dynamics')
        - evaluable (True if not categorical/constant/disabled)
        - sign convention (signed vs unsigned)

    def conditioning_indices(self) -> dict[str, int]:
        \"\"\"Auto-detect conditioning variables for regime analysis.
        Off-exchange: {spread_bps: 12, ...}
        MBO: {spread_bps: 42, time_regime: 93, ...}
        \"\"\"
    """
```

### 3.3 `data/holdout.py` -- Holdout Reservation

```python
def split_holdout(dates: list[str], holdout_days: int
                  ) -> tuple[list[str], list[str]]:
    """Reserve the LAST holdout_days dates as meta-holdout.

    Returns: (evaluation_dates, holdout_dates)
    The holdout dates are the chronologically latest days.
    holdout_days=0 --> no holdout, all dates used for evaluation.
    """
```

### 3.4 `data/cache.py` -- DataCache (v2 Data Layer)

The DataCache eliminates redundant disk I/O by loading all data in two streaming passes, then serving all evaluation paths from memory.

```python
@dataclass
class DataCache:
    """Shared data cache populated in two streaming passes.

    All evaluation paths read from this cache instead of the loader.

    Cube layout:
        daily_ic_cube[d, f, h] = Spearman IC for day d, evaluable feature f, horizon h.
        NaN where IC could not be computed (degenerate day/feature).
        Feature axis maps to evaluable_indices[f].
        Day axis maps to evaluation_dates[d].

    Pooled layout:
        pooled_features[n, :] = last-timestep features for sample n (ALL features).
        pooled_labels[n, :] = labels for sample n.
        Each _from_cache path subsamples independently for its own needs.
    """

    schema: ExportSchema
    evaluation_dates: tuple[str, ...]       # D dates
    evaluable_indices: tuple[int, ...]      # F_eval feature indices (sorted)
    excluded_features: dict[str, str]       # {feature_name: exclusion_reason}
    horizons: tuple[int, ...]               # H horizons from config
    seed: int                               # Global random seed

    # Per-day IC cubes -- NaN where invalid
    daily_ic_cube: np.ndarray               # [D, F_eval, H] float64
    daily_temporal_cubes: dict[str, np.ndarray]  # metric_name -> [D, F_eval, H]
    daily_forward_ic_cube: np.ndarray       # [D, F_eval, H] float64
    daily_concurrent_ic_cube: np.ndarray    # [D, F_eval, H] float64

    # Pooled data
    pooled_features: np.ndarray             # [N_total, F_all] float64
    pooled_labels: np.ndarray               # [N_total, H] float64
    pooled_date_indices: np.ndarray         # [N_total] int32 -- day index per sample
    n_total_samples: int
```

**`build_cache()` algorithm:**

```python
def build_cache(
    loader: ExportLoader,
    evaluation_dates: list[str],
    config: EvaluationConfig,
) -> DataCache:
    """Build unified data cache in two streaming passes.

    Pass 1 (pre-screen): Stream all days, compute column variance via
        StreamingColumnStats (Welford's algorithm). Determine evaluable
        indices from variance + categorical exclusion.
        Lightweight: O(N x F) per day, no Spearman.

    Pass 2 (main): Stream all days again. For each day:
        - Compute per-day Spearman IC for all (evaluable_feature, horizon).
        - Compute per-day temporal rolling ICs
          (rolling_mean, rolling_slope, rate_of_change -- vectorized 3D).
        - Compute per-day forward/concurrent ICs (CF decomposition).
        - Accumulate last-timestep features + labels for pooling.
        With mmap, Pass 2 finds pages cached from Pass 1 --> effective I/O ~ 1 pass.

    Returns: DataCache with all per-day cubes and pooled data.
    """
```

**Pass 2 temporal rolling computation**: For each day, the full 3D sequence tensor `[N, T, F]` is passed to vectorized rolling functions from `hft_metrics.temporal`:

```python
rm_3d = rolling_mean(seq_3d, window=rolling_window)     # [N, T, F]
rs_3d = rolling_slope(seq_3d, window=rolling_window)     # [N, T, F]
roc_3d = rate_of_change(seq_3d, lag=rolling_window)      # [N, T, F]

# Extract last timestep for IC computation
rm_last = rm_3d[:, -1, :]   # [N, F]
```

Each rolling feature at each evaluable index gets a per-day Spearman IC against each horizon's labels, stored in `daily_temporal_cubes[metric_name][day, pos, h_idx]`.

**Temporal metrics computed in cache**: `("rolling_mean", "rolling_slope", "rate_of_change")` -- matching the three rolling transforms from `hft_metrics.temporal`.

### 3.5 Pre-Screening (Framework Section 6.1)

Pre-screening runs as the first phase of both v1 and v2 pipelines. In v1, it is a separate `_pre_screen()` method. In v2, it is Pass 1 of `build_cache()`.

Exclusion criteria (applied identically in both modes):

1. **Categorical features**: `j in schema.categorical_indices` --> excluded with reason `"categorical"`.
2. **No data**: `summary[j]["n"] == 0` --> excluded with reason `"no_data"`.
3. **Zero variance**: `summary[j]["std"]**2 < 1e-10` --> excluded with reason `"zero_variance"`.

Pre-screening uses `StreamingColumnStats` from hft-metrics (Welford's online algorithm) to compute per-feature variance in a single streaming pass.

---

## 4. Screening Layer (Framework Layer 1)

> **Authoritative reference**: FEATURE_EVALUATION_FRAMEWORK.md Sections 2.1, 6.2 (Paths 1-2)

**Feature extraction convention** (used by ALL screening/selection/regime modules):
```python
# For each day, extract per-sample feature values and labels:
features_2d = day_bundle.sequences[:, -1, :]   # [N, F] -- last timestep only
labels_2d = day_bundle.labels                    # [N, H] -- all horizons

# For a specific feature j at horizon h:
feature_j = features_2d[:, j]                    # [N] -- one value per sample
label_h = labels_2d[:, h_idx]                    # [N] -- point return at horizon h
```
This extracts the last timestep because with stride=1, `sequences[:, -1, :]` gives one unique bin-level value per sample. The temporal dimension (T=20 timesteps within each window) is only used by the Temporal Layer (Section 6).

**Deterministic per-test seeding** (from `screening/__init__.py`):
```python
def _test_seed(base_seed: int, feature_idx: int, horizon_idx: int,
               lag_idx: int = 0) -> int:
    """Deterministic per-test seed for permutation tests.
    Order-independent: changing iteration order of features/horizons
    does not change the seed for any (feature, horizon, lag) triple.
    Returns: Unique deterministic seed in [0, 2^31 - 2].
    """
    return (base_seed + feature_idx * 10007
            + horizon_idx * 31 + lag_idx * 7) % (2**31 - 1)
```

### 4.1 `screening/ic_screening.py` -- Path 1: Linear Signal

**v1 entry point**:
```python
def screen_ic(loader: ExportLoader, evaluation_dates: list[str],
              evaluable_indices: list[int], config: EvaluationConfig
              ) -> dict[str, dict[int, ICResult]]:
    """Forward IC screening for all features at all horizons.

    For each feature j and horizon h:
        1. Stream day-by-day, extract feature[t] = last timestep of each sequence
        2. Compute IC_w = spearman_ic(feature[t], label[t,h]) per day
        3. Aggregate: IC_mean, IC_std, IC_IR across days
        4. Bootstrap 95% CI via block_bootstrap_ci
        5. BH correction across all (feature, horizon) pairs

    Path 1 pass criteria (Framework Section 6.2):
        |Forward IC| > 0.05 at any horizon
        AND bootstrap 95% CI excludes zero
        AND IC_IR > config.ic_ir_threshold (default 0.5)
        AND BH-adjusted p < bh_fdr_level (default 0.05)

    Returns: dict[feature_name -> dict[horizon -> ICResult]]
    """
```

**v2 entry point** (reads from DataCache instead of streaming from disk):
```python
def screen_ic_from_cache(cache: DataCache, config: EvaluationConfig
                         ) -> dict[str, dict[int, ICResult]]:
    """IC screening from pre-computed daily IC cube.

    Reads cache.daily_ic_cube[d, f, h] directly. For each (feature, horizon):
        IC_mean = nanmean(cube[:, pos, h_idx])
        t-test on the per-day IC series
        Bootstrap CI on the per-day IC series
        BH correction across all (feature, horizon) pairs.
    Same pass criteria as screen_ic(). Zero disk I/O.
    """
```

### 4.2 `screening/dcor_screening.py` -- Path 2: Non-Linear Signal

**v1 entry point**:
```python
def screen_dcor(loader: ExportLoader, evaluation_dates: list[str],
                evaluable_indices: list[int], config: EvaluationConfig
                ) -> dict[str, dict[int, DCorResult]]:
    """dCor + MI independence screening with BH correction.

    Computational strategy: Per-day dCor + t-test (consistent with IC screening).
    dCor is O(n^2) -- infeasible on pooled ~45K samples. Per-day computation
    on ~308 samples is fast. Daily dCor values are aggregated with t-test
    p-value for BH.

    Path 2 pass criteria (Framework Section 6.2):
        dCor BH-adjusted p < 0.05 at any horizon
        AND MI BH-adjusted p < 0.05 at any horizon.

    Input: last timestep x_{w,T}^(j) of each sequence (NOT full [T,F] tensor).
    """
```

**v2 entry point**:
```python
def screen_dcor_from_cache(cache: DataCache, config: EvaluationConfig
                           ) -> dict[str, dict[int, DCorResult]]:
    """dCor + MI screening from pooled cache data.

    Reads cache.pooled_features and cache.pooled_labels. Subsamples to
    config.screening.dcor_subsample (default 3000) for bias mitigation
    (dCor is sensitive to sample size). Uses _test_seed for per-pair
    deterministic seeding. Same pass criteria as screen_dcor().
    """
```

---

## 5. Selection Layer (Framework Layer 2)

> **Authoritative reference**: FEATURE_EVALUATION_FRAMEWORK.md Sections 2.2, 6.2 (Path 5)

### 5.1 `selection/jmi_selection.py` -- Path 5: Interaction Value

**v1 entry point**:
```python
def jmi_forward_selection(loader: ExportLoader, evaluation_dates: list[str],
                           evaluable_indices: list[int], horizon: int,
                           config: EvaluationConfig,
                           ) -> list[tuple[str, float]]:
    """JMI greedy forward selection (Brown et al. 2012, Eq. 17-18).

    At each step, add feature X_k maximizing:
        J_JMI(X_k) = I(X_k;Y) - (1/|S|) sum I(X_k;X_j) + (1/|S|) sum I(X_k;X_j|Y)

    Three-term decomposition:
        - I(X_k;Y): relevancy (KSG MI)
        - (1/|S|) sum I(X_k;X_j): redundancy (KSG MI)
        - (1/|S|) sum I(X_k;X_j|Y): conditional redundancy (conditional_mi_ksg)

    JMI best_horizon: selected by pipeline -- horizon with most Path 1
    passing features (ties broken by max mean |IC|).

    Elbow detection: stop when gain_k / gain_1 < config.selection.jmi_elbow_threshold.
    Returns: ordered list of (feature_name, jmi_score), best first.
    """
```

**v2 entry point**:
```python
def jmi_from_cache(cache: DataCache, horizon: int,
                   config: EvaluationConfig
                   ) -> list[tuple[str, float]]:
    """JMI forward selection from pooled cache data.
    Reads cache.pooled_features and cache.pooled_labels.
    Same algorithm as jmi_forward_selection(). Zero disk I/O.
    """
```

### 5.2 `selection/concurrent_forward.py` -- Concurrent vs Forward IC

**v1 entry point**:
```python
def decompose_concurrent_forward(
    loader: ExportLoader, evaluation_dates: list[str],
    evaluable_indices: list[int], horizons: list[int],
) -> dict[str, dict[int, ConcurrentForwardResult]]:
    """Concurrent vs forward IC decomposition for all features.

    Methodology (Framework Section 4.2):
        Forward IC:    spearman_ic(feature[t], label[t])
        Concurrent IC: spearman_ic(feature[t], label[t-1])

    Classification (Framework Section 4.3):
        ratio > 10  --> purely contemporaneous
        2-10        --> partially forward
        < 2         --> purely forward
        concurrent ~ 0, forward > 0 --> state variable

    Returns: dict[feature_name -> dict[horizon -> ConcurrentForwardResult]]
    """

@dataclass
class ConcurrentForwardResult:
    forward_ic: float
    concurrent_ic: float
    ratio: float                # concurrent / max(forward, EPS)
    classification: str         # "contemporaneous" | "partially_forward" |
                                # "forward" | "state_variable"
```

**v2 entry point**:
```python
def compute_cf_from_cache(
    cache: DataCache, horizons: list[int],
) -> dict[str, dict[int, ConcurrentForwardResult]]:
    """CF decomposition from pre-computed daily cubes.
    Reads cache.daily_forward_ic_cube and cache.daily_concurrent_ic_cube.
    Aggregates per-day values into mean forward/concurrent ICs.
    """
```

---

## 6. Temporal Layer (Framework Layer 3)

> **Authoritative reference**: FEATURE_EVALUATION_FRAMEWORK.md Section 2.3

### 6.1 `temporal/temporal_ic.py` -- IC at Each Timestep + Rolling Feature IC

**v1 entry point**:
```python
def compute_temporal_ic(loader: ExportLoader, evaluation_dates: list[str],
                         evaluable_indices: list[int], horizons: list[int],
                         config: EvaluationConfig
                         ) -> dict[str, TemporalICResult]:
    """IC at each of T timesteps within the sequence window.

    For each feature j, horizon h, timestep t (t=1..T):
        IC(j, h, t) = spearman_ic(x_{w,t}^(j), r_{w+h})

    Also computes rolling feature IC:
        rolling_mean_K, rolling_slope_K, rate_of_change_K
        using hft_metrics.temporal functions on each feature's T-dim trajectory.

    Path 3 pass (Framework Section 6.2):
        Temporal IC of rolling features > 0.05 at any horizon
        AND bootstrap 95% CI excludes zero.

    Returns: dict[feature_name -> TemporalICResult]
    """

@dataclass
class TemporalICResult:
    timestep_ic: list[float]           # IC at each of T timesteps
    rolling_mean_ic: float             # IC of rolling_mean_K
    rolling_slope_ic: float            # IC of rolling_slope_K
    rate_of_change_ic: float           # IC of rate_of_change_K
    best_temporal_ic: float            # max of the rolling ICs
    best_temporal_metric: str          # which rolling metric was best
    best_temporal_p: float             # p-value of best rolling metric
    best_horizon: int                  # horizon of best rolling metric
    passes_path3: bool
```

**v2 entry point**:
```python
def compute_temporal_ic_from_cache(cache: DataCache, config: EvaluationConfig
                                   ) -> dict[str, TemporalICResult]:
    """Temporal IC from pre-computed daily temporal cubes.
    Reads cache.daily_temporal_cubes["rolling_mean"], etc.
    Same pass criteria. Zero disk I/O.
    """
```

### 6.2 `temporal/transfer_entropy.py` -- Transfer Entropy Screening

**v1 entry point**:
```python
def screen_transfer_entropy(loader: ExportLoader,
                             evaluation_dates: list[str],
                             evaluable_indices: list[int],
                             horizons: list[int], config: EvaluationConfig
                             ) -> dict[str, dict[int, TEResult]]:
    """Transfer entropy screening with BH correction.

    Computes on raw BIN-LEVEL time series (one value per 60s bin),
    NOT on windowed sequences. Per-day computation, averaged across days.

    TE horizon parameter: Use horizon=1 in transfer_entropy call (predicts
    NEXT bin's label). Labels are already forward-looking at horizon h --
    using horizon=h in the TE call would double-count the lookahead.

    BIN-LEVEL EXTRACTION:
        feature_bins = day_bundle.sequences[:, -1, j]   # [N] bin-level feature
        return_bins = day_bundle.labels[:, h_idx]        # [N] bin-level returns

    For each feature j, horizon h, lag L in {1, 2, 3}:
        TE, p = transfer_entropy_test(feature_bins, return_bins, lag=L, horizon=1)

    BH correction across all (feature, horizon, lag) triplets.
    """
```

**v2 entry point**:
```python
def screen_te_from_cache(cache: DataCache, horizons: list[int],
                         config: EvaluationConfig
                         ) -> dict[str, dict[int, TEResult]]:
    """TE screening from pooled cache data.
    Uses cache.pooled_features and cache.pooled_labels.
    """
```

**IMPORTANT**: In v2, TE results are stored with `is_informational=True` in PathEvidence. They are NOT counted in `passing_paths` and do NOT affect tier classification. TE structurally cannot survive BH with feasible permutation counts (see Section 20).

---

## 7. Regime Layer (Framework Path 4)

> **Authoritative reference**: FEATURE_EVALUATION_FRAMEWORK.md Section 6.2 (Path 4)

**v1 entry point**:
```python
def compute_regime_ic(loader: ExportLoader, evaluation_dates: list[str],
                       evaluable_indices: list[int], horizons: list[int],
                       conditioning: dict[str, int], config: EvaluationConfig
                       ) -> dict[str, dict[int, list[RegimeICResult]]]:
    """Regime-conditional IC (spread/time/activity terciles).

    For each conditioning variable (spread_bps, time_of_day, activity):
        1. Use quantile_buckets(values, n_bins=3) to assign terciles
        2. Compute IC within each tercile
        3. Bootstrap CI per tercile

    Path 4 pass (Framework Section 6.2):
        Bootstrap 95% CI of regime-conditional IC excludes zero
        in ANY regime cell.

    Sample budget (Framework Section 9):
        Single-variable tercile: ~22,962/cell (adequate, 7.3x required n=3,138)
        Two-variable cross (9 cells): ~7,654/cell (adequate, 2.4x)
        Three-variable cross: INFEASIBLE (0.81x required n). DO NOT USE.

    Returns: dict[feature_name -> dict[horizon -> list[RegimeICResult]]]
    """

@dataclass
class RegimeICResult:
    conditioning_variable: str
    best_tercile_ic: float
    passes_path4: bool
```

**v2 entry point**:
```python
def compute_regime_ic_from_cache(
    cache: DataCache, horizons: list[int],
    conditioning: dict[str, int], config: EvaluationConfig,
) -> dict[str, dict[int, list[RegimeICResult]]]:
    """Regime IC from pooled cache data.
    Uses cache.pooled_features, cache.pooled_labels.
    Same tercile bucketing and pass criteria.
    """
```

---

## 8. Stability Layer (Framework Section 6.3)

### 8.1 v1: `stability_selection()` -- Paths 1+2 Bootstrap

```python
def stability_selection(
    loader: ExportLoader, evaluation_dates: list[str],
    evaluable_indices: list[int], horizons: list[int],
    config: EvaluationConfig,
) -> dict[str, float]:
    """Bootstrap stability selection (Meinshausen & Buhlmann 2010).

    Runs Layer 1 screening (IC + dCor+MI) on 50 bootstrap subsamples:
        1. Sample 80% of evaluation_dates (without replacement)
        2. Run screen_ic() + screen_dcor() on the subsample
        3. A feature passes if it passes Path 1 OR Path 2
        4. Repeat 50 times

    Stability = fraction of subsamples where feature passes (0.0 to 1.0).

    ONLY Layer 1 is bootstrapped (per Framework Section 6.3).
    Paths 3-5 run once on full data, NOT per bootstrap.

    BUG: Features passing only Paths 3-5 always get stability=0.0
    and are auto-DISCARDed in v1. Fixed in v2 with per-path stability.

    Returns: dict[feature_name -> stability_fraction]
    """
```

### 8.2 v2: `compute_stability_from_cache()` -- Per-Path Stability

```python
def compute_stability_from_cache(
    cache: DataCache, config: EvaluationConfig,
) -> dict[str, StabilityDetail]:
    """Per-path stability from cached data.

    Fixes the critical bug where features passing only Paths 3-5 were
    auto-DISCARDed.

    Path 1 (IC): Date-bootstrap on daily_ic_cube. For each iteration,
        sample 80% of day indices, re-aggregate ICs (mean, t-test, BH),
        record pass/fail. Pure array operations -- no disk I/O.

    Path 2 (dCor+MI): Date-bootstrap on pooled data. For each iteration,
        select pooled samples from bootstrapped days (via pooled_date_indices),
        subsample to dcor_subsample, run dcor_test + ksg_mi_test, BH correct.
        Expensive but necessary.

    Path 3a (Temporal IC): Date-bootstrap on daily_temporal_cubes.
        Same approach as Path 1. Passes if ANY temporal metric passes.

    Path 4 (Regime): CI coverage from the main regime IC pass.
        n_pass / n_total triplets. No re-bootstrap needed -- windowed_ic
        already computes bootstrap CIs internally.

    Path 5 (JMI): 1.0 if JMI selected on main run, 0.0 if not.
        JMI is a greedy heuristic; sub-sample bootstrapping adds noise
        without meaningful signal.

    Combined stability = max(path1, path2, path3a).
    A feature is stable if stable on ANY screening path.
    Used as the primary stability gate in compute_tier().

    Returns: dict[feature_name -> StabilityDetail]
    """
```

**StabilityDetail dataclass** (from `profile.py`):

```python
@dataclass(frozen=True)
class StabilityDetail:
    path1_stability: float       # IC screening stability (date bootstrap) [0.0, 1.0]
    path2_stability: float       # dCor+MI stability (subsample bootstrap) [0.0, 1.0]
    path3a_stability: float      # Temporal IC stability (date bootstrap) [0.0, 1.0]
    combined_stability: float    # max(path1, path2, path3a) [0.0, 1.0]
    path4_ci_coverage: float     # Regime IC CI coverage fraction [0.0, 1.0]
    path5_jmi_stability: float   # JMI selection stability [0.0 or 1.0]
```

**Internal helpers** in `stability_selection.py`:

```python
def _ic_bootstrap_pass(boot_cube, ic_threshold, ic_ir_threshold,
                       bh_fdr_level, seed) -> np.ndarray:
    """Check which features pass IC screening on a bootstrapped cube.
    Returns: Boolean array [F_eval] -- True if feature passes at any horizon.
    Uses scipy.stats.ttest_1samp + block_bootstrap_ci + benjamini_hochberg.
    """

def _dcor_bootstrap_pass(features, labels, evaluable_indices,
                         horizons, config) -> np.ndarray:
    """Check which features pass dCor+MI on bootstrapped pooled data.
    Returns: Boolean array [F_eval] -- True if feature passes at any horizon.
    BOTH dCor AND MI must pass (conjunction) at any horizon.
    """
```

---

## 9. Pipeline Orchestration

### 9.1 `pipeline.py` -- EvaluationPipeline

```python
class EvaluationPipeline:
    """Full 5-path evaluation orchestrator with stability selection.

    Runs all paths independently, aggregates results, applies stability
    selection, classifies features, and validates on holdout.
    """

    def __init__(self, config: EvaluationConfig):
        config.validate()
        self.config = config
        self.loader = ExportLoader(config.export_dir, config.split)
        self.registry = FeatureRegistry(self.loader.schema)
        # Phase 4: populated by run_v2(); read via last_profile_hash.
        self._last_profile_hash: str | None = None

    @property
    def last_profile_hash(self) -> str | None:
        """Phase 4: 64-char hex SHA-256 of most recent run_v2() profiles.

        None before any run, after v1 run(), or after a crashed run_v2().
        Both run() and run_v2() reset to None at entry — no stale value
        leaks across call-mode interleave.
        """

    def run(self) -> FeatureClassification:
        """v1 pipeline: 4-tier classification table."""

    def run_v2(self) -> dict[str, FeatureProfile]:
        """v2 pipeline: rich FeatureProfile output.

        Populates self._last_profile_hash = compute_profile_hash(profiles)
        before return (Phase 4).
        """

    def to_json(self, result: FeatureClassification, output_path: str) -> None:
        """Write v1 classification_table.json."""

    def to_json_v2(self, profiles: dict[str, FeatureProfile],
                   output_path: str) -> None:
        """Write v2 feature_profiles.json."""
```

### 9.1.1 Phase 4: `compute_profile_hash` Module-Level Helper

```python
def compute_profile_hash(profiles: dict[str, FeatureProfile]) -> str:
    """Deterministic content hash over evaluator-produced profiles.

    Hashes asdict(profile) for each profile sorted by feature name.
    Downstream FeatureSet producers record this as
    produced_by.source_profile_hash.

    Canonical form matches hft-ops convention (dedup.py:391,
    lineage.py:153): json.dumps(obj, sort_keys=True, default=str) →
    SHA-256 hex. NaN/Inf floats are sanitized to None via
    _sanitize_for_hash (strict-JSON safe + semantically correct —
    NaN p-value = "no hypothesis test run").

    Returns 64-char lowercase hex SHA-256 digest. No `sha256:` prefix
    (matches ExperimentRecord.fingerprint; prefix reserved for
    external identifiers like databento manifests).

    Portability: stable across CPython versions + platforms, but NOT
    byte-portable to other languages' default JSON serializers
    (e.g., Rust serde_json uses `","` where Python default uses `", "`).
    Consumers needing polyglot reproducibility must mirror Python's
    whitespace convention.
    """
```

`_sanitize_for_hash` is a private recursive helper that walks dict/list/tuple and maps non-finite floats to `None`. Tuples canonicalize to lists (JSON-identical representation).

### 9.2 v1 `run()` Step-by-Step

```
pipeline.run():
    |
    +-- 1. dates = loader.list_dates()
    +-- 2. eval_dates, holdout_dates = split_holdout(dates, config.holdout_days)
    +-- 3. evaluable, excluded = _pre_screen(eval_dates)
    |
    +-- 4. ic_results = screen_ic(loader, eval_dates, evaluable, config)
    +-- 5. cf_results = decompose_concurrent_forward(loader, eval_dates, evaluable, horizons)
    +-- 6. dcor_results = screen_dcor(loader, eval_dates, evaluable, config)
    +-- 7. temporal_results = compute_temporal_ic(loader, eval_dates, evaluable, horizons, config)
    +-- 8. te_results = screen_transfer_entropy(loader, eval_dates, evaluable, horizons, config)
    +-- 9. regime_results = compute_regime_ic(loader, eval_dates, evaluable, horizons,
    |                                         conditioning, config)
    +--10. jmi_results = jmi_forward_selection(loader, eval_dates, evaluable,
    |                                          best_horizon, config)
    |
    +--11. stability = stability_selection(loader, eval_dates, evaluable, horizons, config)
    |      --> dict[feature_name -> stability_fraction]
    |
    +--12. For each evaluable feature:
    |      |  Collect passing_paths from steps 4-10
    |      |  best_p = min p-value across all passing PathResults
    |      |  tier = classify_feature(passing_paths, stability_pct, best_p,
    |      |                           holdout_confirmed=False, config)
    |      +-- If tier == "STRONG-KEEP": mark as holdout candidate
    |
    +--13. holdout_confirmed = _validate_holdout(loader, holdout_dates, candidates)
    |      --> Re-runs Layer 1 screening on holdout days (no BH correction)
    |      --> Candidates that fail: downgrade STRONG-KEEP --> KEEP
    |
    +--14. Return FeatureClassification --> classification_table.json
```

### 9.3 v2 `run_v2()` Step-by-Step

```
pipeline.run_v2():
    |
    +-- 1. eval_dates, holdout_dates = split_holdout(dates, config.holdout_days)
    |
    +-- 2. cache = build_cache(loader, eval_dates, config)
    |      --> 2-pass streaming: pre-screen + main computation
    |      --> DataCache with daily IC cubes + pooled data
    |
    +-- 3. ic_results = screen_ic_from_cache(cache, config)
    +-- 4. dcor_results = screen_dcor_from_cache(cache, config)
    +-- 5. temporal_results = compute_temporal_ic_from_cache(cache, config)
    +-- 6. te_results = screen_te_from_cache(cache, horizons, config)
    +-- 7. regime_results = compute_regime_ic_from_cache(cache, horizons,
    |                                                     conditioning, config)
    +-- 8. jmi_results = jmi_from_cache(cache, best_horizon, config)
    +-- 9. cf_results = compute_cf_from_cache(cache, horizons)
    |
    +--10. stability_map = compute_stability_from_cache(cache, config)
    |      --> dict[feature_name -> StabilityDetail]
    |      --> Update Path 5 JMI stability (1.0 if selected, 0.0 if not)
    |
    +--11. Redundancy analysis (from pooled data):
    |      - Rank-transform pooled_features via scipy.stats.rankdata (Spearman)
    |      - corr_matrix = correlation_matrix(ranked, eval_list)
    |      - clusters = cluster_by_correlation(corr_matrix, names, threshold=0.7)
    |      - max pairwise correlation per feature
    |      - vif = compute_vif(ranked, eval_list, names)
    |
    +--12. ACF of daily IC series per feature:
    |      - For each feature, find horizon with highest mean |IC|
    |      - Compute autocorrelation(ic_valid, max_lag=50) from hft_metrics.acf
    |      - Record half_life (lag at which ACF drops below 0.5)
    |
    +--13. Build FeatureProfile per feature:
    |      - Collect all PathEvidence (TE marked is_informational=True)
    |      - passing_paths = non-informational evidence that passes
    |      - best_p = min finite p across passing non-informational evidence
    |      - Set redundancy fields (cluster_id, max_corr, vif)
    |      - Set ic_acf_half_life
    |      - Update Path 4 CI coverage in StabilityDetail
    |
    +--14. Holdout validation for STRONG-KEEP candidates:
    |      - Criteria: passing > 0 AND best_p < strong_keep_p
    |        AND combined_stability >= stable_threshold
    |      - Re-run Layer 1 on holdout (no BH correction)
    |      - Confirmed candidates get holdout_confirmed=True via dataclasses.replace()
    |
    +--15. Return dict[feature_name -> FeatureProfile]
```

**Key principles**:
- Each evaluation module receives DataCache, not the loader. No module touches disk.
- No module calls another module. The pipeline aggregates results.
- `best_p`: minimum finite p-value across passing non-informational PathEvidence. Since each path applies BH within its own test family, this is conservative (Bonferroni-like cross-path combination).

---

## 10. Decision Logic

### 10.1 v1: `decision.py`

```python
class Tier(str, Enum):
    """4-tier feature classification."""
    STRONG_KEEP = "STRONG-KEEP"
    KEEP = "KEEP"
    INVESTIGATE = "INVESTIGATE"
    DISCARD = "DISCARD"

@dataclass(frozen=True)
class PathResult:
    """Result of one evaluation path for one feature at one horizon."""
    path_name: str         # "linear_signal", "nonlinear_signal", "temporal_value",
                           # "regime_conditional", "interaction_value"
    horizon: int
    metric_name: str       # "forward_ic", "dcor", "temporal_ic_slope", etc.
    metric_value: float
    p_value: float         # BH-adjusted p-value (within this path's test family)
    ci_lower: float
    ci_upper: float
    passes: bool

@dataclass(frozen=True)
class FeatureTier:
    """Per-feature 4-tier classification with full provenance."""
    tier: Tier
    passing_paths: tuple[str, ...]
    best_horizon: int
    best_metric: str
    best_value: float
    best_p: float
    stability_pct: float | None      # None before stability phase
    concurrent_forward_ratio: float | None
    all_path_results: tuple[PathResult, ...]

@dataclass(frozen=True)
class HoldoutReport:
    holdout_dates: tuple[str, ...]
    n_holdout_days: int
    candidates_tested: int
    candidates_confirmed: int
    per_feature: dict[str, bool]     # feature_name -> holdout_confirmed

@dataclass
class FeatureClassification:
    per_feature: dict[str, FeatureTier]
    config: EvaluationConfig
    excluded_features: dict[str, str]  # feature_name -> exclusion reason
    schema: ExportSchema
    holdout: HoldoutReport | None = None

def classify_feature(
    passing_paths: list[str], stability_pct: float | None,
    best_p: float, holdout_confirmed: bool,
    config: EvaluationConfig,
) -> Tier:
    """4-tier classification per Framework Section 6.6.

    Algorithm:
        if len(passing_paths) == 0:
            return DISCARD
        if stability_pct is not None and stability_pct < investigate_threshold:
            return DISCARD
        if stability_pct is not None and stability_pct < stable_threshold:
            return INVESTIGATE
        if best_p < strong_keep_p and holdout_confirmed:
            return STRONG_KEEP
        return KEEP
    """

def compute_best_p(path_results: list[PathResult]) -> float:
    """Minimum BH-adjusted p-value across all passing PathResults.
    Returns 1.0 if no paths pass.
    """
```

### 10.2 v2: `profile.py` -- FeatureProfile + PathEvidence + compute_tier()

```python
@dataclass(frozen=True)
class PathEvidence:
    """Evidence from a single evaluation path at one horizon.

    Replaces PathResult with an explicit is_informational flag for
    paths that store results but don't participate in classification
    (e.g., transfer entropy).
    """
    path_name: str          # "linear_signal", "nonlinear_signal", "temporal_value",
                            # "regime_conditional", "interaction_value", "transfer_entropy"
    horizon: int
    metric_name: str        # "forward_ic", "dcor", "rolling_slope", "te_L2", etc.
    metric_value: float
    p_value: float          # NaN for paths without hypothesis tests (regime, JMI)
    ci_lower: float         # NaN if not available
    ci_upper: float         # NaN if not available
    passes: bool
    is_informational: bool  # True = stored but NOT counted in passing_paths

@dataclass(frozen=True)
class FeatureProfile:
    """Complete characterization of a single feature.

    Not a tier -- a rich profile that selection criteria match against.
    The tier property is derived on demand via compute_tier() for
    backward compatibility with v1 consumers.

    Frozen after construction. Use dataclasses.replace() to update
    holdout_confirmed after holdout validation.
    """
    feature_name: str
    feature_index: int

    # Signal characterization
    best_horizon: int
    best_metric: str
    best_value: float
    best_p: float               # NaN if no hypothesis-tested path passes

    # Passing paths (only non-informational paths that survived their gate)
    passing_paths: tuple[str, ...]

    # Stability
    stability: StabilityDetail

    # Concurrent/forward decomposition
    concurrent_forward_ratio: float | None
    cf_classification: str | None   # "forward", "partially_forward",
                                    # "contemporaneous", "state_variable"

    # Redundancy (from hft-metrics correlation analysis)
    redundancy_cluster_id: int | None
    max_pairwise_correlation: float | None
    vif: float | None

    # Temporal dynamics (ACF of daily IC series)
    ic_acf_half_life: int | None

    # Holdout validation
    holdout_confirmed: bool = False

    # All evidence (for traceability, includes informational)
    all_evidence: tuple[PathEvidence, ...] = ()

def compute_tier(
    profile: FeatureProfile,
    stable_threshold: float = 0.6,
    investigate_threshold: float = 0.4,
    strong_keep_p: float = 0.01,
) -> str:
    """Backward-compatible tier derivation from profile.

    Uses combined_stability (max across Paths 1/2/3a) as the stability
    gate -- NOT path1-only, which was the v1 bug.

    Algorithm:
        if len(passing_paths) == 0: return "DISCARD"
        if combined_stability < investigate_threshold: return "DISCARD"
        if combined_stability < stable_threshold: return "INVESTIGATE"
        if isfinite(best_p) and best_p < strong_keep_p and holdout_confirmed:
            return "STRONG-KEEP"
        return "KEEP"
    """
```

---

## 11. Selection Criteria

> **v2 only**. Replaces the hard-coded classify_feature() verdict with configurable criteria matching.

### 11.1 `criteria.py` -- SelectionCriteria

```python
@dataclass(frozen=True)
class SelectionCriteria:
    """Declarative criteria for selecting features from profiles.

    Frozen dataclass, YAML-serializable. All fields are optional filters.
    A feature must satisfy ALL specified criteria to be selected.
    None/unset fields are not checked.
    """

    name: str = "default"

    # Schema version (Phase 4, 2026-04-15). Bumped when adding hash-
    # affecting fields. Downstream FeatureSet producers include this in
    # `produced_by` provenance for criteria-schema compatibility.
    criteria_schema_version: str = "1.0"

    # Path requirements
    min_passing_paths: int = 1
    required_paths: tuple[str, ...] = ()

    # Stability
    min_combined_stability: float = 0.6

    # Signal strength
    min_abs_metric: float | None = None        # Minimum |best_value|
    max_p_value: float | None = None           # Maximum best_p

    # CF decomposition gating
    max_cf_ratio: float | None = None          # Max concurrent/forward ratio
    allowed_cf_classes: tuple[str, ...] | None = None

    # Redundancy
    max_vif: float | None = None               # Max VIF
    max_pairwise_corr: float | None = None     # Max pairwise Spearman |rho|

    # Horizon constraints
    allowed_horizons: tuple[int, ...] | None = None

    # Holdout verification (Phase 4). When True, only features whose
    # FeatureProfile.holdout_confirmed is True survive selection (the
    # STRONG-KEEP criterion that passed out-of-sample). Composes
    # AND-wise with other gates; `include_names` still bypasses.
    require_holdout_confirmed: bool = False

    # Explicit inclusion/exclusion (override all other criteria)
    include_names: tuple[str, ...] | None = None   # Force-include these names
    exclude_names: tuple[str, ...] | None = None   # Force-exclude these names
```

**Phase 4 YAML loaders** — `SelectionCriteria.from_yaml(path)` and `from_dict(d)` classmethods mirror `EvaluationConfig.from_yaml`:
- Accept optional single-key `{criteria: {...}}` wrapper for composability with multi-section configs.
- Strict unknown-key rejection via `_KNOWN_CRITERIA_KEYS`.
- Tuple-field coercion (`required_paths`, `allowed_cf_classes`, `allowed_horizons`, `include_names`, `exclude_names`) with a string-guard that rejects `str`/`bytes` values to prevent silent per-char tuplification.
- `None` values preserved for Optional-tuple fields (not coerced to empty tuple).

### 11.2 `select_features()` -- Matching Function

```python
def select_features(
    profiles: dict[str, FeatureProfile],
    criteria: SelectionCriteria,
) -> list[str]:
    """Apply selection criteria to profiles.

    Returns feature names that satisfy ALL specified criteria,
    sorted alphabetically for determinism.
    """
```

**Matching logic** (from internal `_matches()` function):

1. If `exclude_names` is set and feature name is in it, reject.
2. If `include_names` is set, accept only if feature name is in the set (overrides ALL other criteria including the Phase 4 holdout gate).
3. Check `min_passing_paths`: reject if fewer paths pass.
4. Check `required_paths`: reject if any required path is missing.
5. Check `min_combined_stability`: reject if combined_stability is below threshold.
6. Check `min_abs_metric`: reject if `|best_value|` is below threshold.
7. Check `max_p_value`: reject if `best_p` exceeds threshold (only if `best_p` is finite).
8. Check `max_cf_ratio`: reject if `concurrent_forward_ratio` exceeds threshold.
9. Check `allowed_cf_classes`: reject if `cf_classification` is not in allowed set.
10. Check `max_vif`: reject if VIF exceeds threshold.
11. Check `max_pairwise_corr`: reject if `max_pairwise_correlation` exceeds threshold.
12. Check `allowed_horizons`: reject if `best_horizon` is not in allowed set.
13. **Phase 4** — Check `require_holdout_confirmed`: if True, reject features whose `holdout_confirmed` is False.

**Example YAML usage**:
```yaml
criteria:
    name: "momentum_hft"
    min_passing_paths: 1
    min_combined_stability: 0.6
    exclude_contemporaneous: true
    allowed_cf_classes: ["forward", "partially_forward", "state_variable"]
    include_names: ["dark_share", "session_progress", "time_regime"]
```

**Design principle**: The same profiles serve multiple experiments without re-evaluation. Each experiment defines its own SelectionCriteria. This decouples evaluation (expensive, run once) from selection (cheap, run many times).

---

## 12. Model Feedback Protocol

### 12.1 `feedback.py` -- Post-Training Feature Importance

**Status: NOT YET IMPLEMENTED (protocol definition only).**

This module defines the artifact protocol that `lob-model-trainer` will implement to feed importance scores back into the evaluator for iterative refinement.

```python
@dataclass(frozen=True)
class FeatureImportance:
    """Post-training importance score for one feature.

    Produced by lob-model-trainer after a training run.
    Consumed by the evaluator to enrich FeatureProfiles.
    """
    feature_name: str
    importance_score: float
    importance_method: str      # "stg_gate", "loco_group", "ig_tsr", "permutation"
    model_id: str               # Experiment name
    horizon: int                # Prediction horizon the model was trained for

@runtime_checkable
class ModelFeedbackProvider(Protocol):
    """Protocol for models to report feature importance back to evaluator."""

    def get_feature_importances(self, model_id: str) -> list[FeatureImportance]:
        """Return importance scores for all features used by this model."""
        ...

    def get_training_metrics(self, model_id: str) -> dict[str, float]:
        """Return training/validation metrics (loss, IC, R2, DA, etc.)."""
        ...

def merge_feedback_into_profiles(
    profiles: dict, importances: list[FeatureImportance],
) -> dict:
    """Merge model feedback into feature profiles.

    Not yet implemented. When implemented, this will:
    1. Match importances to profiles by feature_name
    2. Return new profiles with model_importance fields populated
    3. Enable criteria like min_model_importance in SelectionCriteria

    Raises: NotImplementedError (always, stub).
    """
```

**Importance methods**:

| Method | Source | What It Measures |
|--------|--------|-----------------|
| `stg_gate` | STG (Stochastic Gates, Yamada et al. 2020) | Gate opening probability per feature |
| `loco_group` | Leave-One-Cluster-Out | Group-level importance via ablation |
| `ig_tsr` | Integrated Gradients + Temporal Saliency Redistribution | Per-timestep attribution |
| `permutation` | Permutation importance | Drop in metric when feature is shuffled |

---

## 13. Known Limitations and Safeguards

**This evaluator is a pre-training model-free tool.** It measures statistical relationships between individual features and returns. It CANNOT detect features whose value only emerges within a trained neural network.

### Blind Spots

| Blind Spot | Example | Why Invisible | Resolution |
|---|---|---|---|
| Context/conditioning features | dark_share, odd_lot_ratio, time_regime | Zero unconditional IC; value is as transformer attention context | STG/IG+TSR via feedback.py |
| Interaction-only features | Feature A useful only when combined with B | Zero individual signal; XOR-like patterns | STG/LOCO-Group |
| Early-timestep features | Signal at t=5 within 20-step window, zero at t=20 | All paths evaluate last timestep only | Per-timestep IC curve, IG+TSR |
| Architecture-specific features | Features the transformer's attention exploits | No model in the evaluation loop | STG, LassoNet |

### Safeguards

1. **NEVER use DISCARD as a hard gate for model training.** DISCARD means "low pre-training signal" -- NOT "useless." Use KEEP+ for initial training; add INVESTIGATE features in a second round. Features with zero IC may have high interaction or context value.
2. **Always include known context features** (dark_share, session_progress, time_regime, odd_lot_ratio) in model training regardless of tier -- they have documented value as conditioning variables (Framework Section 3.2).
3. **After training, compare model attribution with evaluation tiers.** If the model assigns high importance to a DISCARDed feature, the evaluator missed an interaction or temporal pattern.
4. **Use `include_names` in SelectionCriteria** to force-include features known to be valuable from prior experiments or domain knowledge, bypassing pre-training statistical gates.

**Path to closing these gaps**: The `feedback.py` module defines `ModelFeedbackProvider` -- a protocol for `lob-model-trainer` to export STG gate values, LOCO importance, and IG+TSR attributions back into feature profiles. Once implemented, `SelectionCriteria` can filter on model-derived importance, enabling the evaluator to rescue features that fail pre-training screens but are valuable to the trained model.

---

## 14. Configuration

### 14.1 `config.py` -- EvaluationConfig

All configuration is parsed from YAML, validated at construction time, and frozen (immutable).

```python
@dataclass(frozen=True)
class EvaluationConfig:
    """Top-level evaluation configuration. Immutable after construction."""

    export_dir: str                            # Path to NPY export directory
    split: str = "train"                       # "train", "val", or "test"
    holdout_days: int = 20                     # Last N days reserved for holdout
    seed: int = 42                             # Global random seed
    screening: ScreeningConfig = ...
    stability: StabilityConfig = ...
    classification: ClassificationConfig = ...
    regime: RegimeConfig = ...
    temporal: TemporalConfig = ...
    selection: SelectionConfig = ...
    verbose: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "EvaluationConfig": ...

    @classmethod
    def from_dict(cls, d: dict) -> "EvaluationConfig": ...

    def validate(self) -> None:
        """Validate all parameters. Raises ValueError on violation."""
```

### 14.2 Nested Config Sections

#### ScreeningConfig (Paths 1-2)

| Field | Type | Default | Valid Range | Description |
|-------|------|---------|-------------|-------------|
| `horizons` | `tuple[int, ...]` | `(1,2,3,5,10,20,30,60)` | all > 0, sorted ascending | Horizons to evaluate |
| `bh_fdr_level` | `float` | `0.05` | (0, 1) | Benjamini-Hochberg FDR level |
| `ic_threshold` | `float` | `0.05` | (0, 1) | Forward IC pass threshold |
| `dcor_permutations` | `int` | `500` | [100, 10000] | Permutations for dCor test |
| `dcor_subsample` | `int` | `3000` | [500, 10000] | Subsample size for pooled dCor/MI |
| `mi_permutations` | `int` | `200` | [50, 5000] | Permutations for ksg_mi_test |
| `mi_k` | `int` | `5` | [1, 20] | k for KSG MI estimator |

#### StabilityConfig (Section 6.3)

| Field | Type | Default | Valid Range | Description |
|-------|------|---------|-------------|-------------|
| `n_bootstraps` | `int` | `50` | [10, 500] | Number of bootstrap subsamples |
| `subsample_fraction` | `float` | `0.8` | (0.5, 0.99) | Fraction of training days per subsample |
| `stable_threshold` | `float` | `0.6` | (0, 1) | >= this = KEEP/STRONG-KEEP eligible |
| `investigate_threshold` | `float` | `0.4` | (0, 1) | >= this but < stable = INVESTIGATE |

**Constraint**: `stable_threshold > investigate_threshold` (validated).

#### ClassificationConfig (Section 6.6)

| Field | Type | Default | Valid Range | Description |
|-------|------|---------|-------------|-------------|
| `strong_keep_p` | `float` | `0.01` | (0, 1) | Best BH-adjusted p < this for STRONG-KEEP |
| `ic_ir_threshold` | `float` | `0.5` | (0, 10) | IC_IR > this for Path 1 |

**Constraint**: `strong_keep_p < bh_fdr_level` (validated).

#### RegimeConfig (Path 4)

| Field | Type | Default | Valid Range | Description |
|-------|------|---------|-------------|-------------|
| `n_bins` | `int` | `3` | [2, 5] | Number of conditioning bins (terciles) |
| `min_samples_per_bin` | `int` | `30` | >= 10 | Minimum samples per regime cell |
| `conditioning_indices` | `dict[str, int] \| None` | `None` | -- | None = auto-detect from schema |

#### TemporalConfig (Path 3)

| Field | Type | Default | Valid Range | Description |
|-------|------|---------|-------------|-------------|
| `rolling_window` | `int` | `5` | [2, 10] | K for rolling_mean_K, rolling_slope_K |
| `te_lags` | `tuple[int, ...]` | `(1, 2, 3)` | all >= 1, sorted ascending | Lag range L for transfer entropy |

#### SelectionConfig (Path 5)

| Field | Type | Default | Valid Range | Description |
|-------|------|---------|-------------|-------------|
| `jmi_max_features` | `int \| None` | `None` | >= 1 or None | None = use elbow detection |
| `jmi_elbow_threshold` | `float` | `0.05` | (0, 1) | Relative gain threshold for stopping |

### 14.3 Example Config

```yaml
# configs/offexchange_34feat_lean.yaml
export_dir: "../data/exports/basic_nvda_60s"
split: "train"
holdout_days: 20
seed: 42

screening:
  horizons: [1, 2, 3, 5, 10, 20, 30, 60]
  bh_fdr_level: 0.05
  ic_threshold: 0.05
  dcor_permutations: 500
  dcor_subsample: 3000
  mi_permutations: 200
  mi_k: 5

stability:
  n_bootstraps: 50
  subsample_fraction: 0.8
  stable_threshold: 0.6
  investigate_threshold: 0.4

classification:
  strong_keep_p: 0.01
  ic_ir_threshold: 0.5

regime:
  n_bins: 3
  min_samples_per_bin: 30

temporal:
  rolling_window: 5
  te_lags: [1, 2, 3]

selection:
  jmi_elbow_threshold: 0.05

verbose: true
```

---

## 15. Output Format

### 15.1 v1: `classification_table.json`

```json
{
  "schema": "feature_evaluation_v1",
  "export_dir": "../data/exports/basic_nvda_60s",
  "export_schema": "off_exchange_1.0",
  "n_features_evaluated": 27,
  "n_features_excluded": 7,
  "evaluation_date": "2026-03-26",
  "seed": 42,
  "holdout_days": 20,
  "n_bootstraps": 50,
  "tier_summary": {
    "STRONG-KEEP": 3,
    "KEEP": 5,
    "INVESTIGATE": 8,
    "DISCARD": 11
  },
  "features": {
    "spread_bps": {
      "tier": "STRONG-KEEP",
      "passing_paths": ["linear_signal", "regime_conditional"],
      "best_horizon": 60,
      "best_metric": "forward_ic",
      "best_value": 0.163,
      "best_p": 0.0003,
      "stability_pct": 94.0,
      "concurrent_forward_ratio": 1.8
    }
  },
  "excluded_features": {
    "trf_vpin": "zero_variance",
    "bin_valid": "categorical",
    "schema_version": "categorical"
  }
}
```

### 15.2 v2: `feature_profiles.json`

```json
{
  "schema": "feature_evaluation_v2",
  "export_dir": "../data/exports/basic_nvda_60s",
  "export_schema": "off_exchange_1.0",
  "n_features_evaluated": 27,
  "n_features_excluded": 7,
  "evaluation_date": "2026-04-07",
  "seed": 42,
  "holdout_days": 20,
  "n_bootstraps": 50,
  "tier_summary": {
    "STRONG-KEEP": 3,
    "KEEP": 5,
    "INVESTIGATE": 8,
    "DISCARD": 11
  },
  "features": {
    "spread_bps": {
      "tier": "STRONG-KEEP",
      "passing_paths": ["linear_signal", "regime_conditional"],
      "best_horizon": 60,
      "best_metric": "forward_ic",
      "best_value": 0.163,
      "best_p": 0.0003,
      "stability_pct": 94.0,
      "concurrent_forward_ratio": 1.8,
      "stability_detail": {
        "path1": 0.94,
        "path2": 0.72,
        "path3a": 0.56,
        "combined": 0.94,
        "path4_ci_coverage": 0.667,
        "path5_jmi": 1.0
      },
      "cf_classification": "forward",
      "redundancy_cluster_id": 2,
      "max_pairwise_correlation": 0.631,
      "vif": 3.42,
      "ic_acf_half_life": 12,
      "holdout_confirmed": true
    }
  },
  "excluded_features": {
    "trf_vpin": "zero_variance",
    "bin_valid": "categorical"
  }
}
```

The v2 schema is a strict superset of v1. All v1 fields are present at the same JSON paths. New fields are additive. `best_p` is `null` (not present) when no hypothesis-tested path passes (instead of a numeric value).

---

## 16. CLI

### 16.1 `cli.py` -- Entry Point

```python
def main() -> int:
    """Run feature evaluation from the command line.

    Usage:
        evaluate --config configs/offexchange_34feat_lean.yaml
        evaluate --config configs/mbo_98feat_lean.yaml --output results.json
        evaluate --config configs/offexchange_34feat_lean.yaml --v2

    Args (via argparse):
        --config    Path to YAML config file (required)
        --output    Output JSON path (default: classification_table.json or
                    feature_profiles.json depending on mode)
        --v2        Use profile-based v2 pipeline (produces FeatureProfile output)

    Returns: Exit code: 0 on success, 1 on error.
    """
```

Entry point registered in `pyproject.toml`: `evaluate = "hft_evaluator.cli:main"`

**Examples**:
```bash
# v1: 4-tier classification table
evaluate --config configs/offexchange_34feat_lean.yaml

# v2: rich FeatureProfile output
evaluate --config configs/offexchange_34feat_lean.yaml --v2

# v2 with custom output path
evaluate --config configs/mbo_98feat_lean.yaml --v2 --output mbo_profiles.json

# v1 MBO evaluation
evaluate --config configs/mbo_98feat_lean.yaml --output mbo_classification.json
```

---

## 17. Memory Strategy

### v1: Day-Level Streaming

- **Per-path streaming**: each path streams day-by-day, accumulating statistics
- **Stability bootstraps**: 50 x stream through subsampled day list
- **Peak memory**: ~1 day at a time (~1 MB for off-exchange [308, 20, 34] float32)
- **Total disk passes**: 49-109 (each path + each stability bootstrap streams independently)

### v2: DataCache Approach

- **2-pass loading**: Pass 1 (pre-screen) + Pass 2 (main computation + pooling)
- **Effective disk I/O**: ~1 pass (OS mmap caches pages from Pass 1 for Pass 2)
- **Daily IC cubes**: `[D, F_eval, H]` float64 in memory. For off-exchange: `[213, 27, 8]` = ~37 KB
- **Pooled data**: `[N_total, F_all]` float64 + `[N_total, H]` float64. For off-exchange: `[~65K, 34]` + `[~65K, 8]` = ~22 MB
- **Stability bootstraps**: Pure array operations on cubes. No disk I/O.
- **Redundancy**: Correlation matrix `[F_eval, F_eval]` = ~6 KB. VIF uses same matrix.
- **Peak memory**: ~25 MB for off-exchange. ~150 MB for MBO (100K samples, 128 features).

---

## 18. Testing Strategy

**Test count**: 298 tests across 23 test files (verified via `pytest --collect-only -q` at HEAD `a1a2cef`). Phase 2b/4/5 added test files: `test_fast_gate.py` (Phase 2b IC gate library), `test_profile_hash.py` (Phase 4 Batch 4a content-hash primitives), `test_experiments_offexchange_gate.py` (Phase 5 Preview library port), `test_criteria.py` (Phase 4 SelectionCriteria YAML parser + strict unknown-key rejection).

| Test File | What It Validates |
|---|---|
| `test_cache.py` | DataCache construction, build_cache() two-pass algorithm, cube shapes, evaluable indices |
| `test_config.py` | EvaluationConfig parsing, validation, unknown key rejection, range checks |
| `test_concurrent_forward.py` | CF decomposition on simulated data, ratio classification |
| `test_criteria.py` | SelectionCriteria matching, select_features(), include/exclude overrides, all filter types |
| `test_dcor_screening.py` | dCor+MI screening, BH correction, per-day + t-test aggregation |
| `test_decision.py` | classify_feature with known inputs -> known tier (all 4 tiers). Holdout downgrade. compute_best_p. |
| `test_feedback.py` | ModelFeedbackProvider protocol checks, FeatureImportance construction, merge_feedback stub |
| `test_holdout.py` | split_holdout reservation, edge cases (0 days, all days) |
| `test_ic_screening.py` | IC screening, bootstrap CI, IC_IR, BH correction on known p-values |
| `test_jmi_selection.py` | JMI on 3 features with known MI values -> correct ranking. Elbow detection. |
| `test_loader.py` | ExportLoader auto-detection (MBO vs off-exchange), DayBundle loading |
| `test_pipeline.py` | End-to-end pipeline tests (v1 run + v2 run_v2), holdout validation |
| `test_profile.py` | FeatureProfile construction, compute_tier() with all tier outcomes, StabilityDetail |
| `test_regime_ic.py` | Regime IC with known tercile splits, sample budget validation |
| `test_registry.py` | FeatureRegistry group mapping, conditioning_indices auto-detection |
| `test_stability.py` | v1 stability_selection + v2 compute_stability_from_cache, per-path breakdown |
| `test_temporal_ic.py` | Temporal IC on synthetic AR(1) features, rolling feature IC |
| `test_transfer_entropy_eval.py` | TE screening on known causal pairs, BH correction |

---

## 19. Integration with Pipeline

### Upstream

```
feature-extractor-MBO-LOB --> NPY exports (data/exports/nvda_*/train/)
basic-quote-processor     --> NPY exports (data/exports/basic_nvda_60s/train/)
```

### Downstream

```
feature_profiles.json (v2) --> lob-model-trainer (feature selection for training)
                            --> select_features(profiles, criteria)
classification_table.json (v1) --> lob-model-trainer (backward compat)
                               --> lob-model-trainer/feature_selection/ (STG/LassoNet)
```

### Contract

The classification table output format is registered in `pipeline_contract.toml` under `[evaluation]`.

### What Does NOT Belong Here

- Model training or model architectures (use lob-model-trainer, lob-models)
- Backtesting or P&L computation (use lob-backtester)
- Diagnostic analysis (use lob-dataset-analyzer)
- Feature extraction from raw data (use feature-extractor or basic-quote-processor)
- STG, LassoNet, LOCO-Group, IG+TSR (implemented in lob-model-trainer; feedback protocol in `feedback.py`)

---

## 20. Deferred Work

### `data/alignment.py` -- Cross-Pipeline Fusion

**Status**: DEFERRED until both pipelines independently pass signal quality gates.

**Why**: FEATURE_EVALUATION_FRAMEWORK.md Section 8.1 requires sample-level pairwise correlation between off-exchange and MBO features (3,332 pairs). This CANNOT be done without temporal alignment: MBO exports cover pre/post-market (569 bins/day) vs off-exchange RTH only (387 bins/day). MBO time-based exports lack `bin_size_seconds` metadata. IC-profile-similarity is NOT a valid proxy for feature redundancy.

**Prerequisite**: Add `bin_size_seconds` to MBO time-based export metadata, then implement bin-level temporal alignment per `off-exchange-approach/06_INTEGRATION_POINTS.md`.

### Phase 5: Model-Embedded Selection

STG, LassoNet, LOCO-Group, IG+TSR -- implemented in `lob-model-trainer`, not here. Requires GPU. The `feedback.py` module defines the protocol for feeding model-derived importance scores back into profiles. Deferred until model-free evaluation (this package) produces actionable results and `lob-model-trainer` exports the required artifacts.

### Per-Timestep IC Curve

Currently all paths evaluate features at the last timestep (`sequences[:, -1, :]`). A per-timestep IC curve (IC at each of T=20 timesteps) would detect features with signal at early timesteps that decays to zero by the last timestep. The `TemporalICResult.timestep_ic` field exists but is only populated in v1 mode. v2 could compute this from the DataCache by extending Pass 2 to store per-timestep features, at the cost of ~20x memory for pooled data.

### Level-vs-Change Signal Decomposition

Some features may have predictive signal in their rate-of-change but not their level (or vice versa). The current temporal IC path partially captures this via rolling_slope and rate_of_change, but a formal decomposition (level IC vs delta IC) would make this explicit. Related to the per-timestep IC gap.

### Transfer Entropy BH Feasibility

TE is marked `is_informational=True` in v2 because it structurally cannot survive BH with feasible permutation counts. With `dcor_permutations=500`, the minimum achievable p-value is `1/501 = 0.002`. For 34 features x 8 horizons x 3 lags = 816 tests, BH requires `p < 0.05 * rank / 816`. Even the smallest possible p (0.002) only survives if `rank >= 33`. This means at most ~33 features can pass -- which is adequate for dCor but marginal for TE where fewer features have genuine signal. Increasing permutations to 5000+ would help but makes each bootstrap iteration ~10x slower.

---

*Last updated: April 7, 2026*
