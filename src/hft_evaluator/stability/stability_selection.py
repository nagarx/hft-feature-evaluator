"""
Stability selection: bootstrap subsampling of Layer 1 screening.

Runs ic_screening + dcor_screening on 50 bootstrap subsamples (80% of training
days, without replacement). Measures what fraction of subsamples each feature
passes in. Only Layer 1 (Paths 1+2) is bootstrapped. Paths 3-5 run once.

Reference: CODEBASE.md Section 8, Framework Section 6.3
Meinshausen & Bühlmann 2010, JRSS-B 72(4):417-473
"""

import logging

import numpy as np

from hft_evaluator.config import EvaluationConfig
from hft_evaluator.data.loader import ExportLoader
from hft_evaluator.screening.ic_screening import screen_ic
from hft_evaluator.screening.dcor_screening import screen_dcor

logger = logging.getLogger(__name__)


def stability_selection(
    loader: ExportLoader,
    evaluation_dates: list[str],
    evaluable_indices: list[int],
    horizons: list[int],
    config: EvaluationConfig,
) -> dict[str, float]:
    """Bootstrap stability selection on Layer 1 screening.

    Args:
        loader: ExportLoader with detected schema.
        evaluation_dates: Dates to evaluate on (excludes holdout).
        evaluable_indices: Feature indices that passed pre-screening.
        horizons: Horizon values.
        config: EvaluationConfig with stability params.

    Returns:
        dict[feature_name -> stability_pct] (0.0 to 100.0)
    """
    schema = loader.schema
    feature_names = schema.feature_names
    n_dates = len(evaluation_dates)
    n_sample = max(1, int(n_dates * config.stability.subsample_fraction))
    n_bootstraps = config.stability.n_bootstraps

    rng = np.random.RandomState(config.seed)

    # Initialize pass counts
    evaluable_names = [
        feature_names.get(j, f"feature_{j}") for j in evaluable_indices
    ]
    pass_counts: dict[str, int] = {name: 0 for name in evaluable_names}

    for b in range(n_bootstraps):
        # Sample 80% of dates WITHOUT replacement
        sample_indices = rng.choice(n_dates, size=n_sample, replace=False)
        sample_dates = [evaluation_dates[i] for i in sorted(sample_indices)]

        # Run Path 1: IC screening
        ic_results = screen_ic(loader, sample_dates, evaluable_indices, config)

        # Run Path 2: dCor+MI screening
        dcor_results = screen_dcor(loader, sample_dates, evaluable_indices, config)

        # A feature passes in this subsample if it passes Path 1 OR Path 2
        for name in evaluable_names:
            passes_p1 = any(
                r.passes_path1
                for r in ic_results.get(name, {}).values()
            )
            passes_p2 = any(
                r.passes_path2
                for r in dcor_results.get(name, {}).values()
            )
            if passes_p1 or passes_p2:
                pass_counts[name] += 1

        if config.verbose and (b + 1) % 10 == 0:
            logger.info(
                f"Stability: {b + 1}/{n_bootstraps} iterations complete"
            )

    # Return fractions (0.0 to 1.0), NOT percentages (0-100).
    # classify_feature compares against config thresholds which are on 0-1 scale
    # (stable_threshold=0.6, investigate_threshold=0.4).
    return {
        name: count / n_bootstraps
        for name, count in pass_counts.items()
    }


# ---------------------------------------------------------------------------
# Cache-based per-path stability (Phase 3: fixes Bug #1)
# ---------------------------------------------------------------------------


def compute_stability_from_cache(
    cache: "DataCache",
    config: EvaluationConfig,
) -> dict[str, "StabilityDetail"]:
    """Per-path stability from cached data. Fixes the critical bug where
    features passing only Paths 3-5 were auto-DISCARDed.

    Path 1 (IC): Date-bootstrap on daily_ic_cube. For each iteration,
        sample 80% of day indices, re-aggregate ICs (mean, t-test, BH),
        record pass/fail. Pure array operations — no disk I/O.

    Path 2 (dCor+MI): Date-bootstrap on pooled data. For each iteration,
        select pooled samples from bootstrapped days, subsample to dcor_subsample,
        run dcor_test + ksg_mi_test, BH correct. Expensive but necessary.

    Path 3a (Temporal IC): Date-bootstrap on daily_temporal_cubes.
        Same approach as Path 1.

    Path 4 (Regime): CI coverage from the main regime IC pass.
        1.0 if regime passes, 0.0 if not (no re-bootstrap needed — windowed_ic
        already computes bootstrap CIs internally).

    Path 5 (JMI): 1.0 if JMI selected on main run, 0.0 if not.
        JMI is a greedy heuristic; sub-sample bootstrapping adds noise
        without meaningful signal. Deferred for future implementation.

    Combined stability: max(path1, path2, path3a) — a feature is stable
    if stable on ANY screening path.

    Args:
        cache: DataCache from build_cache().
        config: EvaluationConfig with stability params.

    Returns:
        dict[feature_name -> StabilityDetail]
    """
    from hft_evaluator.data.cache import DataCache
    from hft_evaluator.profile import StabilityDetail
    from hft_evaluator.screening import _test_seed

    schema = cache.schema
    feature_names = schema.feature_names
    horizons = list(cache.horizons)
    n_horizons = len(horizons)
    n_dates = len(cache.evaluation_dates)
    n_eval = len(cache.evaluable_indices)
    n_sample = max(1, int(n_dates * config.stability.subsample_fraction))
    n_bootstraps = config.stability.n_bootstraps

    rng = np.random.RandomState(config.seed)

    # Per-feature, per-path pass counts
    p1_counts = np.zeros(n_eval, dtype=int)   # Path 1: IC
    p2_counts = np.zeros(n_eval, dtype=int)   # Path 2: dCor+MI
    p3a_counts = np.zeros(n_eval, dtype=int)  # Path 3a: Temporal IC

    for b in range(n_bootstraps):
        # Sample 80% of date indices (without replacement)
        boot_day_idx = rng.choice(n_dates, size=n_sample, replace=False)
        boot_day_idx.sort()

        # --- Path 1: IC from daily_ic_cube ---
        boot_ics = cache.daily_ic_cube[boot_day_idx, :, :]  # [D', F_eval, H]
        p1_pass = _ic_bootstrap_pass(
            boot_ics, config.screening.ic_threshold,
            config.classification.ic_ir_threshold,
            config.screening.bh_fdr_level, config.seed,
        )
        p1_counts += p1_pass

        # --- Path 3a: Temporal IC from daily_temporal_cubes ---
        # Check each temporal metric, pass if ANY metric passes
        p3a_pass_any = np.zeros(n_eval, dtype=bool)
        for metric_name, cube in cache.daily_temporal_cubes.items():
            boot_temporal = cube[boot_day_idx, :, :]
            metric_pass = _ic_bootstrap_pass(
                boot_temporal, config.screening.ic_threshold,
                0.0,  # No IC_IR requirement for temporal
                config.screening.bh_fdr_level, config.seed,
            )
            p3a_pass_any |= metric_pass.astype(bool)
        p3a_counts += p3a_pass_any.astype(int)

        # --- Path 2: dCor+MI from pooled data ---
        # Select pooled samples from bootstrapped dates
        boot_pool_mask = np.isin(cache.pooled_date_indices, boot_day_idx)
        boot_pool_features = cache.pooled_features[boot_pool_mask]
        boot_pool_labels = cache.pooled_labels[boot_pool_mask]

        p2_pass = _dcor_bootstrap_pass(
            boot_pool_features, boot_pool_labels,
            cache.evaluable_indices, horizons, config,
        )
        p2_counts += p2_pass

        if config.verbose and (b + 1) % 10 == 0:
            logger.info(
                f"Stability: {b + 1}/{n_bootstraps} iterations complete"
            )

    # Build StabilityDetail per feature
    evaluable_names = [
        feature_names.get(j, f"feature_{j}")
        for j in cache.evaluable_indices
    ]

    result: dict[str, StabilityDetail] = {}
    for pos, name in enumerate(evaluable_names):
        p1 = float(p1_counts[pos]) / n_bootstraps
        p2 = float(p2_counts[pos]) / n_bootstraps
        p3a = float(p3a_counts[pos]) / n_bootstraps
        combined = max(p1, p2, p3a)

        result[name] = StabilityDetail(
            path1_stability=p1,
            path2_stability=p2,
            path3a_stability=p3a,
            combined_stability=combined,
            path4_ci_coverage=0.0,  # Set by pipeline from regime results
            path5_jmi_stability=0.0,  # Set by pipeline from JMI results
        )

    return result


# ---------------------------------------------------------------------------
# Internal helpers for stability bootstraps
# ---------------------------------------------------------------------------


def _ic_bootstrap_pass(
    boot_cube: np.ndarray,
    ic_threshold: float,
    ic_ir_threshold: float,
    bh_fdr_level: float,
    seed: int,
) -> np.ndarray:
    """Check which features pass IC screening on a bootstrapped cube.

    Args:
        boot_cube: [D', F_eval, H] daily IC cube subset.
        ic_threshold: Minimum |IC| threshold.
        ic_ir_threshold: Minimum IC_IR threshold (0 to skip).
        bh_fdr_level: BH FDR level.
        seed: Seed for bootstrap CI.

    Returns:
        Boolean array [F_eval] — True if feature passes at any horizon.
    """
    from scipy import stats as scipy_stats
    from hft_metrics.testing import benjamini_hochberg
    from hft_metrics.bootstrap import block_bootstrap_ci

    n_eval = boot_cube.shape[1]
    n_horizons = boot_cube.shape[2]

    # Collect all (feature, horizon) raw p-values
    pair_keys: list[tuple[int, int]] = []
    raw_ps: list[float] = []
    ic_means: list[float] = []
    ic_ir_vals: list[float] = []
    ci_excludes: list[bool] = []

    for pos in range(n_eval):
        for h_idx in range(n_horizons):
            ic_series = boot_cube[:, pos, h_idx]
            ic_array = ic_series[np.isfinite(ic_series)]
            n_days = len(ic_array)

            if n_days < 2:
                pair_keys.append((pos, h_idx))
                raw_ps.append(1.0)
                ic_means.append(0.0)
                ic_ir_vals.append(0.0)
                ci_excludes.append(False)
                continue

            ic_mean = float(np.mean(ic_array))
            std = float(np.std(ic_array, ddof=1))
            ic_ir_val = abs(ic_mean) / std if std > 0 else 0.0

            # Bootstrap CI
            try:
                _, ci_lo, ci_hi = block_bootstrap_ci(
                    statistic_fn=lambda x, _y: float(np.mean(x)),
                    x=ic_array,
                    y=np.zeros_like(ic_array),
                    n_bootstraps=500,
                    ci=0.95,
                    seed=seed,
                )
                ci_excl = (ci_lo > 0) or (ci_hi < 0)
            except (ValueError, np.linalg.LinAlgError):
                ci_excl = False

            # T-test
            if n_days >= 3 and std > 0:
                _, raw_p = scipy_stats.ttest_1samp(ic_array, 0.0)
                raw_p = float(raw_p)
            else:
                raw_p = 1.0

            pair_keys.append((pos, h_idx))
            raw_ps.append(raw_p)
            ic_means.append(ic_mean)
            ic_ir_vals.append(ic_ir_val)
            ci_excludes.append(ci_excl)

    # BH correction
    raw_p_array = np.array(raw_ps)
    bh_mask = benjamini_hochberg(raw_p_array, q=bh_fdr_level)

    # Check pass criteria per (feature, horizon)
    feature_passes = np.zeros(n_eval, dtype=bool)
    for idx, (pos, h_idx) in enumerate(pair_keys):
        passes = (
            abs(ic_means[idx]) > ic_threshold
            and ci_excludes[idx]
            and ic_ir_vals[idx] > ic_ir_threshold
            and bool(bh_mask[idx])
        )
        if passes:
            feature_passes[pos] = True

    return feature_passes.astype(int)


def _dcor_bootstrap_pass(
    features: np.ndarray,
    labels: np.ndarray,
    evaluable_indices: tuple[int, ...],
    horizons: list[int],
    config: EvaluationConfig,
) -> np.ndarray:
    """Check which features pass dCor+MI on bootstrapped pooled data.

    Args:
        features: [N', F_all] feature subset from bootstrapped dates.
        labels: [N', H] label subset.
        evaluable_indices: Feature indices to evaluate.
        horizons: Horizon values.
        config: EvaluationConfig.

    Returns:
        Boolean array [F_eval] — True if feature passes at any horizon.
    """
    from hft_metrics.dcor import dcor_test
    from hft_metrics.mi import ksg_mi_test
    from hft_metrics.testing import benjamini_hochberg
    from hft_evaluator.screening import _test_seed

    n_total = features.shape[0]
    n_eval = len(evaluable_indices)
    n_subsample = config.screening.dcor_subsample

    if n_total == 0:
        return np.zeros(n_eval, dtype=int)

    # Subsample with FIXED seed (isolate date variation from subsample variation)
    sub_rng = np.random.RandomState(config.seed)
    if n_total > n_subsample:
        sub_idx = sub_rng.choice(n_total, size=n_subsample, replace=False)
        sub_features = features[sub_idx]
        sub_labels = labels[sub_idx]
    else:
        sub_features = features
        sub_labels = labels

    # Run dCor + MI permutation tests
    dcor_raw_ps: list[float] = []
    mi_raw_ps: list[float] = []
    pair_keys: list[tuple[int, int]] = []

    for j in evaluable_indices:
        feature_col = sub_features[:, j]
        for h_idx in range(len(horizons)):
            if h_idx >= sub_labels.shape[1]:
                continue
            label_col = sub_labels[:, h_idx]

            pair_seed = _test_seed(config.seed, j, h_idx)
            _, dc_p = dcor_test(
                feature_col, label_col,
                n_permutations=config.screening.dcor_permutations,
                seed=pair_seed,
            )
            _, mi_p = ksg_mi_test(
                feature_col, label_col,
                k=config.screening.mi_k,
                n_permutations=config.screening.mi_permutations,
                seed=pair_seed + 1_000_000,
            )

            pair_keys.append((list(evaluable_indices).index(j), h_idx))
            dcor_raw_ps.append(dc_p)
            mi_raw_ps.append(mi_p)

    if not pair_keys:
        return np.zeros(n_eval, dtype=int)

    # BH correction — separate families
    dcor_bh = benjamini_hochberg(
        np.array(dcor_raw_ps), q=config.screening.bh_fdr_level
    )
    mi_bh = benjamini_hochberg(
        np.array(mi_raw_ps), q=config.screening.bh_fdr_level
    )

    # Feature passes if BOTH dCor AND MI pass at any horizon
    feature_passes = np.zeros(n_eval, dtype=bool)
    for idx, (pos, h_idx) in enumerate(pair_keys):
        if bool(dcor_bh[idx]) and bool(mi_bh[idx]):
            feature_passes[pos] = True

    return feature_passes.astype(int)
