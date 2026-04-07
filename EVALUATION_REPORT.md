# Off-Exchange Feature Evaluation Report

> **Date**: 2026-03-27
> **Export**: basic_nvda_60s (233 days NVDA, XNAS.BASIC CMBP-1)
> **Config**: offexchange_34feat_lean.yaml
> **Runtime**: 138 minutes (lean config)

---

## Executive Summary

28 features evaluated across 146 training days (20 holdout) using the 5-path decision framework with 20-bootstrap stability selection.

| Tier | Count | Description |
|------|-------|-------------|
| **STRONG-KEEP** | 2 | Passes multiple paths, stability=100%, holdout confirmed |
| **KEEP** | 12 | Passes at least one path, stability >= 60% |
| **INVESTIGATE** | 4 | Passes at least one path, stability 40-55% |
| **DISCARD** | 10 | Unstable signal (stability < 40%) |
| **Excluded** | 6 | 4 categorical + 2 zero-variance (VPIN disabled) |

**Key findings:**
1. **spread_bps** and **bbo_update_rate** are the only features with robust, holdout-confirmed signal across multiple evaluation paths.
2. Nonlinear dependence dominates: Path 2 (dCor+MI) detects 100 significant pairs vs Path 1 (Spearman IC) detecting only 6. Off-exchange features have predominantly nonlinear relationships with returns.
3. Transfer entropy found zero signal — consistent with E8's finding that these features describe current market state, not future returns.
4. **trf_signed_imbalance is DISCARD** (stability=10%) despite E9's IC=0.103. The signal is unstable across market conditions and bootstrap subsamples.

---

## Methodology

### 5-Path Framework

| Path | Name | Method | P-value Strategy |
|------|------|--------|-----------------|
| 1 | Linear Signal | Per-day Spearman IC + t-test + BH | Unbiased statistic, t-test valid |
| 2 | Non-Linear Signal | Subsampled pooled dcor_test + ksg_mi_test + BH | Bias-corrected: dCor has +0.102 bias at n=308 |
| 3a | Temporal Value | Per-day rolling feature IC + t-test + BH | Spearman on rolling_mean/slope, unbiased |
| 3b | Transfer Entropy | Subsampled pooled transfer_entropy_test + BH | Bias-corrected: TE has +0.051 bias at n=308 |
| 4 | Regime-Conditional | Pooled windowed_ic with quantile terciles | CI-based pass (proxy p=0.049) |
| 5 | Interaction Value | Pooled+subsampled JMI greedy forward selection | No p-value (nominal p=0.049) |

### Configuration (Lean)

| Parameter | Value | Statistical Adequacy |
|-----------|-------|---------------------|
| dCor permutations | 100 | p-resolution = 0.0099 (< BH q=0.05) |
| dCor subsample | 1000 | Sufficient for dCor power |
| MI permutations | 50 | p-resolution = 0.0196 (< BH q=0.05) |
| MI k (KSG neighbors) | 5 | Standard KSG Algorithm 1 |
| Stability bootstraps | 20 | Resolution = 5% per step |
| Subsample fraction | 80% | ~117 days per bootstrap |
| BH FDR level | 0.05 | Standard multiple testing correction |
| IC threshold | 0.05 | Framework Section 6.2 |
| IC_IR threshold | 0.5 | Framework Section 2.1 |
| Strong-keep p | 0.01 | Cross-path conservative |
| Stable threshold | 0.6 (60%) | Framework Section 6.3 |
| Investigate threshold | 0.4 (40%) | Framework Section 6.3 |
| Rolling window (K) | 5 | For rolling_mean_5, rolling_slope_5 |
| TE lags | [1, 2, 3] | Framework Section 2.3 |
| Holdout days | 20 | ~14% of training data |

### Critical Bias Corrections

Validated empirically on independent data (n=308, 50 simulations):
- **dCor**: mean bias = +0.102. T-testing daily dCor against 0 gives 100% false positive rate. **Fixed**: subsampled pooled permutation test.
- **KSG MI**: mean bias = -0.003. Approximately unbiased. Per-day t-test is valid.
- **Transfer Entropy**: mean bias = +0.051. **Fixed**: subsampled pooled permutation test.
- **Spearman rho**: unbiased by construction. Per-day t-test is valid.

---

## Classification Table

| Feature | Tier | Stability | best_p | best_IC | CF Ratio | Best Horizon | Passing Paths |
|---------|------|-----------|--------|---------|----------|-------------|---------------|
| spread_bps | **STRONG-KEEP** | 100% | 0.0000 | +0.302 | 0.6 | H=2 | linear, nonlinear, temporal, regime, interaction |
| bbo_update_rate | **STRONG-KEEP** | 100% | 0.0010 | +0.326 | 0.4 | H=3 | nonlinear, temporal, regime, interaction |
| trf_volume | KEEP | 100% | 0.0173 | +0.305 | 6.7 | H=1 | nonlinear, regime, interaction |
| total_volume | KEEP | 100% | 0.0173 | +0.308 | 5.6 | H=1 | nonlinear, regime, interaction |
| trade_count | KEEP | 100% | 0.0173 | +0.304 | 0.8 | H=3 | nonlinear, regime, interaction |
| bin_trade_count | KEEP | 100% | 0.0173 | +0.304 | 0.8 | H=3 | nonlinear, regime |
| bin_trf_trade_count | KEEP | 100% | 0.0173 | +0.304 | 1.1 | H=1 | nonlinear, regime, interaction |
| lit_volume | KEEP | 100% | 0.0173 | +0.295 | 1.1 | H=2 | nonlinear, regime, interaction |
| subpenny_intensity | KEEP | 100% | 0.0173 | +0.243 | 0.3 | H=5 | nonlinear, regime, interaction |
| size_concentration | KEEP | 100% | 0.0173 | +0.200 | 1.6 | H=3 | nonlinear, regime, interaction |
| spread_change_rate | KEEP | 100% | 0.0173 | +0.160 | 29.3 | H=5 | nonlinear, interaction |
| block_trade_ratio | KEEP | 80% | 0.0173 | +0.117 | 1.7 | H=1 | nonlinear, regime |
| session_progress | KEEP | 80% | 0.0173 | +0.172 | 1.0 | H=3 | nonlinear, regime, interaction |
| odd_lot_ratio | KEEP | 60% | 0.0490 | +0.048 | 0.5 | H=60 | regime, interaction |
| mean_trade_size | INVESTIGATE | 50% | 0.0147 | -0.068 | 0.6 | H=60 | temporal, regime, interaction |
| time_since_burst | INVESTIGATE | 40% | 0.0310 | +0.130 | 1.6 | H=3 | nonlinear, regime |
| retail_volume_fraction | INVESTIGATE | 45% | 0.0490 | +0.045 | 0.4 | H=60 | regime, interaction |
| trf_lit_volume_ratio | INVESTIGATE | 55% | 0.0490 | +0.056 | 0.6 | H=60 | regime |
| dark_share | DISCARD | 35% | 0.0173 | +0.096 | 0.6 | H=30 | nonlinear, regime, interaction |
| mroib | DISCARD | 10% | 0.0310 | +0.090 | 4.1 | H=20 | nonlinear, regime |
| inv_inst_direction | DISCARD | 10% | 0.0310 | +0.090 | 4.1 | H=20 | nonlinear, regime, interaction |
| trf_signed_imbalance | DISCARD | 10% | 0.0490 | +0.058 | 2.9 | H=60 | regime, interaction |
| bvc_imbalance | DISCARD | 30% | 0.0490 | -0.034 | 0.4 | H=60 | regime, interaction |
| retail_trade_rate | DISCARD | 25% | 0.0490 | +0.066 | 1.2 | H=30 | regime, interaction |
| trf_burst_intensity | DISCARD | 15% | 0.0490 | +0.024 | 14.5 | H=30 | regime, interaction |
| bid_pressure | DISCARD | 0% | 0.0490 | +0.023 | 1.0 | H=1 | regime, interaction |
| ask_pressure | DISCARD | 0% | 0.0490 | -0.027 | 2.4 | H=1 | regime, interaction |
| quote_imbalance | DISCARD | 5% | 0.0490 | +0.031 | 0.1 | H=1 | regime |

**Excluded** (6): bin_valid (categorical), bbo_valid (categorical), time_bucket (categorical), schema_version (categorical), trf_vpin (zero_variance), lit_vpin (zero_variance).

---

## Per-Path Analysis

### Path 1: Linear Signal (IC Screening)

6 (feature, horizon) pairs pass BH correction at q=0.05. Only **spread_bps** passes the full Path 1 criteria (|IC| > 0.05, bootstrap CI excludes zero, IC_IR > 0.5, BH-rejected).

IC screening is the strictest path: it requires consistent linear correlation across days (IC_IR measures day-to-day stability of the IC signal). Most off-exchange features have IC_IR well below 0.5.

### Path 2: Non-Linear Signal (dCor+MI Screening)

**100 (feature, horizon) pairs pass** — 17x more than Path 1. This is the dominant detection path for off-exchange features. Key implication: the relationship between these features and returns is overwhelmingly nonlinear. Models that assume linear relationships (Ridge, linear regression) will miss most of the signal.

Features detected by Path 2 but NOT Path 1: bbo_update_rate, trade_count, bin_trade_count, all volume features, subpenny_intensity, size_concentration.

### Path 3: Temporal Value

**3a (Temporal IC)**: 3 features pass — mean_trade_size, spread_bps, bbo_update_rate. The rolling features (rolling_mean_5, rolling_slope_5) capture trajectory-based signal not present in instantaneous values.

**3b (Transfer Entropy)**: 0 features pass. No Granger-causal structure detected in any feature at any horizon or lag. This confirms E8's finding: off-exchange features describe the current market state rather than predict future returns.

### Path 4: Regime-Conditional IC

372 (feature, horizon, conditioning) triplets pass. Nearly universal — most features show statistically significant IC within at least one regime tercile. The three conditioning variables (spread_bps, session_progress, bin_trade_count) create market regimes where even weak features show localized signal.

Regime IC uses a proxy p=0.049 (CI-based), so it cannot alone qualify a feature for STRONG-KEEP (requires best_p < 0.01).

### Path 5: JMI Selection

22 of 28 features selected at best_horizon=60 (selected from IC screening: the horizon with the most Path 1 passing features). The elbow detection stopped at 22 features (relative gain dropped below 5% of initial). JMI penalizes redundant features but with 28 features and many correlated, most pass the elbow threshold.

JMI uses nominal p=0.049 — cannot alone qualify for STRONG-KEEP.

---

## Stability Analysis

Stability selection (20 bootstraps, 80% subsampling without replacement) is the critical discriminator in this evaluation.

| Stability Range | Count | Tier Implications |
|-----------------|-------|-------------------|
| 100% | 12 | Consistently detectable across market conditions |
| 60-95% | 2 | Detectable in most but not all conditions |
| 40-55% | 4 | Borderline — signal present but fragile |
| 0-35% | 10 | Unreliable — signal appears and disappears |

The 10 DISCARD features all have stability < 40%, meaning their signal fails to reproduce in more than 60% of bootstrap subsamples. This is the clearest evidence of overfitting to specific market conditions.

---

## Concurrent/Forward Decomposition

| Classification | CF Ratio | Count | Examples |
|---------------|----------|-------|----------|
| Forward (< 2) | Low | 17 | spread_bps (0.6), bbo_update_rate (0.4), subpenny_intensity (0.3) |
| Partially forward (2-10) | Medium | 6 | trf_signed_imbalance (2.9), mroib (4.1), total_volume (5.6) |
| Contemporaneous (> 10) | High | 3 | spread_change_rate (29.3), trf_burst_intensity (14.5) |
| State variable | ~0 fwd, ~0 conc | 2 | quote_imbalance (0.1) |

STRONG-KEEP features both have low CF ratios (spread_bps: 0.6, bbo_update_rate: 0.4), meaning they have genuine forward predictive power rather than just describing current state.

---

## E9 Cross-Validation

| Feature | E9 Finding | This Evaluation | Reconciliation |
|---------|------------|-----------------|----------------|
| trf_signed_imbalance | IC=0.103 at H=1 | DISCARD, stability=10% | E9 likely measured concurrent IC (our concurrent IC=0.150). Forward IC=0.010. Signal is unstable across days. |
| subpenny_intensity | IC=0.104 at H=60 | KEEP, stability=100% | Confirmed via nonlinear path (dCor). The signal is nonlinear — Spearman IC is 0.243 (dCor value). |

---

## Implications for Downstream

### For Model Training

1. **Use spread_bps and bbo_update_rate** as primary features in any model. Both have robust, holdout-confirmed signal.
2. **Include the 12 KEEP features** for nonlinear models (tree-based, neural network) that can capture nonlinear dependence. Linear models will not benefit.
3. **Exclude the 10 DISCARD features** — their signal is not reproducible.
4. **Investigate the 4 borderline features** further: mean_trade_size, time_since_burst, retail_volume_fraction, trf_lit_volume_ratio.

### For Feature Engineering

The dominance of Path 2 (nonlinear) over Path 1 (linear) suggests:
- Nonlinear feature transformations (log, polynomial, interaction terms) may unlock additional signal.
- Models should be evaluated with nonlinear capacity (not just Ridge/linear).

### For Trading Strategy

- spread_bps has CF ratio 0.6 → genuine forward predictive power (not just contemporaneous).
- bbo_update_rate has CF ratio 0.4 → same property.
- Most volume-related features have high CF ratios (5-7) → partially contemporaneous, less useful for forward prediction.

---

## Technical Notes

- **Pipeline**: hft-feature-evaluator v0.1.0, 14 source modules, 162 tests
- **Dependencies**: hft-metrics v0.1.0 (248 tests), hft-contracts (151 tests)
- **Data**: 166 training days (2025-02-03 to 2025-09-30), 20 holdout days
- **Sequences**: [N, 20, 34] float32, stride=1, bin_size=60s
- **Labels**: point_return (continuous bps) at 8 horizons [1,2,3,5,10,20,30,60]
- **Feature extraction**: sequences[:, -1, j] (last timestep = unique bin per sample)
- **Runtime**: 138 min on Apple Silicon (lean config)
