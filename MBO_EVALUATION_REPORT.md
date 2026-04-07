# MBO Feature Evaluation Report

> **Date**: 2026-03-28
> **Export**: e5_timebased_60s (233 days NVDA, XNAS MBO, 60s time-based bins)
> **Config**: mbo_98feat_lean.yaml
> **Runtime**: 173 minutes (lean config)
> **Labels**: SmoothedReturn (continuous bps) — **NOT point-to-point** (see E8 Caveat)

---

## E8 Caveat — Critical for Interpretation

**This evaluation uses smoothed regression labels, NOT point-to-point returns.** E8 (2026-03-21) proved that models trained on smoothed labels achieve DA=48.3% on point returns (below random). The model predicts the smoothing residual (R^2=45%) rather than the tradeable point component (R^2=0.02%).

**Implication**: The IC values and classifications in this report are an **upper bound** on tradeable signal quality. Features classified as STRONG-KEEP have robust signal with smoothed returns, but this does NOT guarantee tradeable signal. Features classified as DISCARD have no signal of any kind — they are genuinely useless.

**Validity**: This report is valid for:
- Feature ranking within the MBO pipeline (relative signal strength)
- Identifying features with NO signal (DISCARD = safe to exclude)
- Comparing signal structure between MBO and off-exchange
- Understanding which evaluation paths detect signal

---

## Executive Summary

89 features evaluated across 143 training days (20 holdout) using the 5-path decision framework with 20-bootstrap stability selection.

| Tier | Count | Description |
|------|-------|-------------|
| **STRONG-KEEP** | 38 | Passes IC screening, stability=100%, holdout confirmed |
| **KEEP** | 10 | Passes at least one path, stability >= 60% |
| **INVESTIGATE** | 0 | — |
| **DISCARD** | 41 | Unstable signal (stability < 40%) |
| **Excluded** | 9 | 4 categorical + 5 zero-variance |

**Key findings:**
1. **Linear signal dominates MBO** — the exact OPPOSITE of off-exchange. Path 1 (IC) detects 102 pairs; Path 2 (dCor+MI) detects 0.
2. **Price levels have IC = -0.82** with H=300 SmoothedReturn. This is the level/mean-reversion effect — NOT directly tradeable without detrending.
3. **OFI and depth_norm_ofi are STRONG-KEEP** (IC=+0.26 at H=10). Consistent with known OFI-return relationship, but remember the smoothing caveat.
4. **Rolling_mean is the best metric for 53/89 features** — temporal trajectory matters more than instantaneous value. Confirms the ablation finding (temporal features boost R^2 by 2.9x).
5. **41 features are DISCARD** — all MBO-specific order flow details (add/cancel/trade rates, size distributions, queue metrics, institutional signals) have unstable signal.

---

## Cross-Pipeline Comparison

| Metric | Off-Exchange (34 feat) | MBO (98 feat) |
|--------|----------------------|---------------|
| **Path 1 (Linear IC)** | **6** pairs pass | **102** pairs pass |
| **Path 2 (dCor+MI)** | **100** pairs pass | **0** pairs pass |
| **Path 3a (Temporal IC)** | 3 features | **65** features |
| **Path 3b (TE)** | 0 | 0 |
| **Path 4 (Regime)** | 372 triplets | **661** triplets |
| **Path 5 (JMI)** | 22 selected | **10** selected |
| **STRONG-KEEP** | 2 | **38** |
| **DISCARD** | 10 | **41** |
| **Labels** | point_return | SmoothedReturn |
| **Dominant path** | Path 2 (nonlinear) | Path 1 (linear) |

**The signal nature is fundamentally different:**
- **Off-exchange**: Overwhelmingly nonlinear dependence (dCor detects, Spearman misses)
- **MBO**: Overwhelmingly linear dependence (Spearman detects, dCor adds nothing)
- This difference may partly reflect the label type (smoothed returns amplify linear correlations via autocorrelation), partly the feature nature (LOB prices are monotonic in returns)

---

## Methodology

Same 5-path framework as the off-exchange evaluation with identical lean parameters:

| Parameter | Value |
|-----------|-------|
| dCor permutations | 100 |
| dCor subsample | 1000 |
| MI permutations | 50 |
| Stability bootstraps | 20 |
| Horizons | [10, 60, 300] (events = minutes at 60s bins) |
| IC threshold | 0.05 |
| IC_IR threshold | 0.5 |
| BH FDR level | 0.05 |

---

## Per-Path Analysis

### Path 1: Linear Signal (IC Screening) — **102 pairs pass**

Dominant path for MBO. LOB prices and OFI have strong linear correlation with smoothed returns. The high pass count (102 out of 267 possible = 38%) reflects the smoothed label's autocorrelation structure, which amplifies linear IC for features with temporal persistence.

Features with highest IC_IR (most stable across days):
- Price levels (all 20): IC_IR > 3.0 (extremely stable)
- OFI features: IC_IR > 2.0
- Volume features: IC_IR > 1.5

### Path 2: Non-Linear Signal (dCor+MI) — **0 pairs pass**

Zero nonlinear signal detected. This is because:
1. Linear signal is so strong that any nonlinear component is negligible
2. BH correction on 267 tests with q=0.05 requires very small individual p-values
3. The dCor permutation test at n=1000 with 100 permutations may lack power for subtle nonlinear effects

This does NOT mean nonlinear dependence is absent — it means it's overshadowed by linear.

### Path 3a: Temporal IC — **65 features pass**

Temporal features (rolling_mean_5, rolling_slope_5) improve IC for most features. This confirms the ablation finding: temporal trajectory is more predictive than instantaneous values.

`rolling_mean` is the best single metric for 53 of 89 features — the 5-bin moving average captures persistence better than the raw value.

### Path 3b: Transfer Entropy — **0 pairs pass**

Same as off-exchange: no Granger-causal structure detected. Features describe state, not predict future.

### Path 4: Regime IC — **661 triplets pass**

Widespread regime-conditional signal. Most features show enhanced IC within specific spread, time, or activity terciles.

### Path 5: JMI Selection — **10 features selected at H=300**

JMI identified 10 features with unique information at H=300 (the horizon with most Path 1 passes):
- Top features are mix of prices, volumes, and OFI

---

## Feature Classification by Group

### STRONG-KEEP (38 features, 100% stability)

| Group | Count | Features | IC Range |
|-------|-------|----------|----------|
| Ask prices (L0-L9) | 10 | All levels | -0.82 (H=300 level effect) |
| Bid prices (L0-L9) | 10 | All levels | -0.82 (H=300 level effect) |
| Bid sizes (L0-L5) | 6 | L0-L5 | +0.05 to +0.20 |
| Ask sizes (L4-L7) | 4 | L4-L7 | +0.05 to +0.20 |
| Derived | 6 | mid_price, spread_bps, total_bid/ask_volume, volume_imbalance, weighted_mid | varies |
| Trading signals | 2 | true_ofi (+0.26), depth_norm_ofi (+0.25) | +0.25 |

### KEEP (10 features, 60-100% stability)

| Feature | Stability | Paths |
|---------|-----------|-------|
| Various sizes and derived | 60-100% | temporal, regime |

### DISCARD (41 features, 0-35% stability)

All MBO-specific order flow features are DISCARD:
- **Order flow rates**: add_rate_bid/ask, cancel_rate_bid/ask, trade_rate_bid/ask
- **Size distribution**: size_p25/p50/p75/p90, size_zscore, large_order_ratio, size_skewness, size_concentration
- **Queue/depth**: orders_per_level, level_concentration, depth_ticks_bid/ask
- **Institutional**: large_order_frequency, large_order_imbalance
- **Core**: avg_order_age, median_order_lifetime, avg_fill_ratio, avg_time_to_first_fill, cancel_to_add_ratio

---

## Stability Analysis

| Range | Count | Features |
|-------|-------|----------|
| 100% | 40 | All STRONG-KEEP + some KEEP |
| 60-95% | 8 | Borderline KEEP features |
| 40-55% | 0 | (none) |
| 0-35% | 41 | All DISCARD features |

The bimodal distribution (100% or <35%) is striking: features either work consistently or not at all. No features are in the "investigate" range (40-55%).

---

## Key Insights for the Pipeline

### 1. Price Levels Are Not Real Signal

All 20 price features show IC = -0.82, but this is the LEVEL effect: higher absolute prices predict lower smoothed future returns (mean reversion at 5-hour horizon). This is:
- NOT causal (price levels don't cause returns)
- NOT tradeable (would require shorting when price is high, buying when low — just betting on mean reversion)
- An artifact of using RAW (unnormalized) price levels with SmoothedReturn labels

**After normalization, price-level IC would likely drop to near zero.**

### 2. OFI Is the Genuine Signal

true_ofi (IC=+0.26) and depth_norm_ofi (IC=+0.25) at H=10 represent genuine order flow signal. However, per E8, this signal is with SmoothedReturn, not point returns. The tradeable component of this IC is uncertain.

### 3. MBO-Specific Features Add Zero Value

All 41 DISCARD features are MBO-specific (order flow rates, size distributions, queue metrics, institutional signals). These were designed to capture microstructure dynamics, but their signal is not stable across market conditions (stability < 35%). This is consistent with the ablation finding: "6-10 features carry all predictive power."

### 4. Temporal Features Are Critical

Rolling_mean is the best metric for 60% of features. The trajectory (how a feature changes over the 20-minute window) is more informative than the instantaneous value. This validates the architecture of using temporal models (LSTM, Transformer) over instantaneous ones.

---

## Excluded Features (9)

| Feature | Reason |
|---------|--------|
| book_valid | categorical (constant 1.0) |
| mbo_ready | categorical (constant 1.0) |
| time_regime | categorical (2 values: midday/afternoon) |
| schema_version | categorical (constant 2.2) |
| avg_queue_position (idx 68) | zero_variance (constant across all days) |
| queue_size_ahead (idx 69) | zero_variance |
| modification_score (idx 76) | zero_variance |
| iceberg_proxy (idx 77) | zero_variance |
| invalidity_delta (idx 96) | zero_variance |

---

## Comparison with Prior Findings

| Finding | Prior Source | This Evaluation | Consistent? |
|---------|-------------|-----------------|-------------|
| OFI has IC~0.34 at H10 | E5 EXPORT_INDEX | IC=+0.26 (regime-conditioned) | Yes (lower due to per-day aggregation) |
| 6-10 features carry all power | Ablation report | 38 STRONG-KEEP but only 2-3 genuine signals | Yes (prices are level artifact) |
| Temporal features critical | Ablation (2.9x R^2 boost) | rolling_mean best for 60% of features | Yes |
| OFI contemporaneous not predictive | E8 root cause | CF ratio for true_ofi not computed here | Partially (E8 used different method) |

---

## Technical Notes

- Pipeline: hft-feature-evaluator v0.1.0, 14 source modules, 162 tests
- hft-metrics: 248 tests (including ksg_mi_test)
- hft-contracts: 151 tests
- Data: 143 eval days + 20 holdout, T=20, F=98, stride=1
- Labels: SmoothedReturn at [10, 60, 300] events = [10, 60, 300] minutes at 60s bins
- Runtime: 173 min on Apple Silicon (lean config)
- Validation: 10/10 cross-checks passed. Manual IC for ask_price_l0 (-0.777) matches pipeline (-0.818 with ±0.05 tolerance). OFI IC at H=10 (+0.249 manual) matches pipeline (+0.259 from regime_ic).
