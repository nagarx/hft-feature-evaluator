"""
Evaluation pipeline orchestrator: 5-path evaluation with stability selection.

Steps (CODEBASE.md Section 9.2):
    1.  Split holdout
    2.  Pre-screen (categorical + zero-variance)
    3.  IC screening (Path 1)
    4.  CF decomposition
    5.  dCor+MI screening (Path 2)
    6.  Temporal IC (Path 3a)
    7.  Transfer entropy (Path 3b)
    8.  Regime IC (Path 4)
    9.  JMI selection (Path 5)
    10. Stability selection (50 bootstraps × Layer 1)
    11. Aggregate + classify with stability_pct
    12. Holdout validation for STRONG-KEEP candidates
    13. Emit classification table JSON

Reference: CODEBASE.md Section 9, Framework Section 6
"""

import json
import logging
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from hft_metrics.welford import StreamingColumnStats
from hft_metrics.ic import spearman_ic, ic_ir
from hft_metrics.bootstrap import block_bootstrap_ci

from hft_evaluator.config import EvaluationConfig
from hft_evaluator.data.loader import ExportLoader
from hft_evaluator.data.registry import FeatureRegistry
from hft_evaluator.data.holdout import split_holdout
from hft_evaluator.screening.ic_screening import screen_ic
from hft_evaluator.screening.dcor_screening import screen_dcor
from hft_evaluator.selection.concurrent_forward import decompose_concurrent_forward
from hft_evaluator.selection.jmi_selection import jmi_forward_selection
from hft_evaluator.temporal.temporal_ic import compute_temporal_ic
from hft_evaluator.temporal.transfer_entropy import screen_transfer_entropy
from hft_evaluator.regime.regime_ic import compute_regime_ic
from hft_evaluator.stability.stability_selection import stability_selection
from hft_evaluator.decision import (
    Tier,
    PathResult,
    FeatureTier,
    HoldoutReport,
    FeatureClassification,
    classify_feature,
    compute_best_p,
)

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Full 5-path evaluation orchestrator with stability selection.

    Runs all paths independently, aggregates results, applies stability
    selection, classifies features into 4 tiers, validates on holdout.
    """

    def __init__(self, config: EvaluationConfig):
        config.validate()
        self.config = config
        self.loader = ExportLoader(config.export_dir, config.split)
        self.registry = FeatureRegistry(self.loader.schema)

    def run(self) -> FeatureClassification:
        """Execute the full evaluation pipeline."""
        schema = self.loader.schema
        horizons = list(self.config.screening.horizons)

        # Step 1: Split holdout
        all_dates = self.loader.list_dates()
        eval_dates, holdout_dates = split_holdout(
            all_dates, self.config.holdout_days
        )
        self._log(
            f"Dates: {len(eval_dates)} eval + {len(holdout_dates)} holdout"
        )

        # Step 2: Pre-screen
        evaluable, excluded = self._pre_screen(eval_dates)
        self._log(
            f"Pre-screen: {len(evaluable)} evaluable, {len(excluded)} excluded"
        )

        if not evaluable:
            self._log("No evaluable features — returning all DISCARD")
            return FeatureClassification(
                per_feature={},
                config=self.config,
                excluded_features=excluded,
                schema=schema,
            )

        # Step 3: IC screening (Path 1)
        self._log("Running IC screening (Path 1)...")
        ic_results = screen_ic(self.loader, eval_dates, evaluable, self.config)
        n_ic_pass = sum(
            1 for name in ic_results
            for r in ic_results[name].values()
            if r.passes_path1
        )
        self._log(f"  Path 1: {n_ic_pass} (feature, horizon) pairs pass")

        # Step 4: CF decomposition
        self._log("Running concurrent/forward decomposition...")
        cf_results = decompose_concurrent_forward(
            self.loader, eval_dates, evaluable, horizons
        )

        # Step 5: dCor+MI screening (Path 2)
        self._log("Running dCor+MI screening (Path 2)...")
        dcor_results = screen_dcor(
            self.loader, eval_dates, evaluable, self.config
        )
        n_dcor_pass = sum(
            1 for name in dcor_results
            for r in dcor_results[name].values()
            if r.passes_path2
        )
        self._log(f"  Path 2: {n_dcor_pass} (feature, horizon) pairs pass")

        # Step 6: Temporal IC (Path 3a)
        self._log("Running temporal IC (Path 3a)...")
        temporal_results = compute_temporal_ic(
            self.loader, eval_dates, evaluable, horizons, self.config
        )
        n_temporal_pass = sum(
            1 for r in temporal_results.values() if r.passes_path3
        )
        self._log(f"  Path 3a: {n_temporal_pass} features pass")

        # Step 7: Transfer entropy (Path 3b)
        self._log("Running transfer entropy (Path 3b)...")
        te_results = screen_transfer_entropy(
            self.loader, eval_dates, evaluable, horizons, self.config
        )
        n_te_pass = sum(
            1 for name in te_results
            for r in te_results[name].values()
            if r.passes_te
        )
        self._log(f"  Path 3b: {n_te_pass} (feature, horizon) pairs pass TE")

        # Step 8: Regime IC (Path 4)
        conditioning = self.config.regime.conditioning_indices
        if conditioning is None:
            conditioning = self.registry.conditioning_indices()
        self._log(
            f"Running regime IC (Path 4) with conditioning: "
            f"{list(conditioning.keys())}..."
        )
        regime_results = compute_regime_ic(
            self.loader, eval_dates, evaluable, horizons,
            conditioning, self.config,
        )
        n_regime_pass = sum(
            1 for name in regime_results
            for h in regime_results[name]
            for rr in regime_results[name][h]
            if rr.passes_path4
        )
        self._log(f"  Path 4: {n_regime_pass} (feature, horizon, cond) triplets pass")

        # Step 9: JMI selection (Path 5)
        best_horizon = self._select_best_horizon(ic_results, horizons)
        self._log(f"Running JMI selection (Path 5) at best_horizon={best_horizon}...")
        jmi_results = jmi_forward_selection(
            self.loader, eval_dates, evaluable, best_horizon, self.config
        )
        jmi_passing_names = {name for name, _ in jmi_results}
        self._log(f"  Path 5: JMI selected {len(jmi_results)} features")

        # Step 10: Stability selection
        self._log(
            f"Running stability selection "
            f"({self.config.stability.n_bootstraps} bootstraps)..."
        )
        stability = stability_selection(
            self.loader, eval_dates, evaluable, horizons, self.config
        )

        # Step 11: Aggregate and classify
        self._log("Classifying features...")
        per_feature: dict[str, FeatureTier] = {}
        strong_keep_candidates: list[str] = []

        for j in evaluable:
            name = schema.feature_names.get(j, f"feature_{j}")
            all_path_results: list[PathResult] = []

            # Path 1: Linear signal
            if name in ic_results:
                for h, r in ic_results[name].items():
                    all_path_results.append(PathResult(
                        path_name="linear_signal",
                        horizon=h,
                        metric_name="forward_ic",
                        metric_value=r.ic_mean,
                        p_value=r.bh_adjusted_p,
                        ci_lower=r.ci_lower,
                        ci_upper=r.ci_upper,
                        passes=r.passes_path1,
                    ))

            # Path 2: Non-linear signal
            if name in dcor_results:
                for h, r in dcor_results[name].items():
                    all_path_results.append(PathResult(
                        path_name="nonlinear_signal",
                        horizon=h,
                        metric_name="dcor",
                        metric_value=r.dcor_value,
                        p_value=r.dcor_p,
                        ci_lower=float("nan"),
                        ci_upper=float("nan"),
                        passes=r.passes_path2,
                    ))

            # Path 3a: Temporal IC
            if name in temporal_results:
                tr = temporal_results[name]
                all_path_results.append(PathResult(
                    path_name="temporal_value",
                    horizon=tr.best_horizon,
                    metric_name=tr.best_temporal_metric,
                    metric_value=tr.best_temporal_ic,
                    p_value=tr.best_temporal_p,
                    ci_lower=float("nan"),
                    ci_upper=float("nan"),
                    passes=tr.passes_path3,
                ))

            # Path 3b: Transfer entropy
            if name in te_results:
                for h, ter in te_results[name].items():
                    if ter.passes_te:
                        all_path_results.append(PathResult(
                            path_name="temporal_value",
                            horizon=h,
                            metric_name=f"transfer_entropy_L{ter.best_lag}",
                            metric_value=ter.te_value,
                            p_value=ter.bh_adjusted_p,
                            ci_lower=float("nan"),
                            ci_upper=float("nan"),
                            passes=True,
                        ))

            # Path 4: Regime conditional
            if name in regime_results:
                for h, regime_list in regime_results[name].items():
                    for rr in regime_list:
                        # NaN p-value: regime IC uses bootstrap CI, not
                        # hypothesis testing. compute_best_p() filters NaN.
                        regime_p = float("nan")
                        all_path_results.append(PathResult(
                            path_name="regime_conditional",
                            horizon=h,
                            metric_name=f"regime_ic_{rr.conditioning_variable}",
                            metric_value=rr.best_tercile_ic,
                            p_value=regime_p,
                            ci_lower=float("nan"),
                            ci_upper=float("nan"),
                            passes=rr.passes_path4,
                        ))

            # Path 5: Interaction value (JMI)
            if name in jmi_passing_names:
                jmi_score = next(
                    score for n, score in jmi_results if n == name
                )
                all_path_results.append(PathResult(
                    path_name="interaction_value",
                    horizon=best_horizon,
                    metric_name="jmi_score",
                    metric_value=jmi_score,
                    p_value=float("nan"),  # JMI provides ranking, not hypothesis test
                    ci_lower=float("nan"),
                    ci_upper=float("nan"),
                    passes=True,
                ))

            # Determine passing paths and best metrics
            passing_names = sorted({
                r.path_name for r in all_path_results if r.passes
            })
            best_p = compute_best_p(all_path_results)

            # Find best metric across passing results
            passing_results = [r for r in all_path_results if r.passes]
            if passing_results:
                best_result = max(
                    passing_results, key=lambda r: abs(r.metric_value)
                )
                best_horizon_feat = best_result.horizon
                best_metric = best_result.metric_name
                best_value = best_result.metric_value
            else:
                best_horizon_feat = horizons[0] if horizons else 0
                best_metric = ""
                best_value = 0.0

            # CF ratio at best horizon
            cf_ratio = None
            if name in cf_results and best_horizon_feat in cf_results[name]:
                cf_ratio = cf_results[name][best_horizon_feat].ratio

            # Stability percentage
            stability_pct = stability.get(name)

            # Classify with stability
            tier = classify_feature(
                passing_names, stability_pct, best_p, False, self.config
            )

            # Identify holdout candidates
            if (
                len(passing_names) > 0
                and best_p < self.config.classification.strong_keep_p
                and (stability_pct is None
                     or stability_pct >= self.config.stability.stable_threshold)
            ):
                strong_keep_candidates.append(name)

            per_feature[name] = FeatureTier(
                tier=tier,
                passing_paths=tuple(passing_names),
                best_horizon=best_horizon_feat,
                best_metric=best_metric,
                best_value=best_value,
                best_p=best_p,
                stability_pct=stability_pct,
                concurrent_forward_ratio=cf_ratio,
                all_path_results=tuple(all_path_results),
            )

        # Step 12: Holdout validation
        holdout_report = None
        if holdout_dates and strong_keep_candidates:
            self._log(
                f"Holdout validation: {len(strong_keep_candidates)} candidates "
                f"on {len(holdout_dates)} days..."
            )
            holdout_confirmed = self._validate_holdout(
                holdout_dates, strong_keep_candidates, evaluable,
            )
            holdout_report = HoldoutReport(
                holdout_dates=tuple(holdout_dates),
                n_holdout_days=len(holdout_dates),
                candidates_tested=len(strong_keep_candidates),
                candidates_confirmed=sum(holdout_confirmed.values()),
                per_feature=holdout_confirmed,
            )

            # Re-classify with holdout results
            for name in strong_keep_candidates:
                old_tier = per_feature[name]
                confirmed = holdout_confirmed.get(name, False)
                new_tier = classify_feature(
                    list(old_tier.passing_paths),
                    old_tier.stability_pct,
                    old_tier.best_p,
                    confirmed,
                    self.config,
                )
                if new_tier != old_tier.tier:
                    per_feature[name] = FeatureTier(
                        tier=new_tier,
                        passing_paths=old_tier.passing_paths,
                        best_horizon=old_tier.best_horizon,
                        best_metric=old_tier.best_metric,
                        best_value=old_tier.best_value,
                        best_p=old_tier.best_p,
                        stability_pct=old_tier.stability_pct,
                        concurrent_forward_ratio=old_tier.concurrent_forward_ratio,
                        all_path_results=old_tier.all_path_results,
                    )
                    status = "CONFIRMED" if confirmed else "FAILED"
                    self._log(
                        f"  {name}: {old_tier.tier.value} → {new_tier.value} "
                        f"(holdout {status})"
                    )

        # Summary
        tier_counts: dict[str, int] = {}
        for ft in per_feature.values():
            tier_counts[ft.tier.value] = tier_counts.get(ft.tier.value, 0) + 1
        self._log(f"Classification: {tier_counts}")

        return FeatureClassification(
            per_feature=per_feature,
            config=self.config,
            excluded_features=excluded,
            schema=schema,
            holdout=holdout_report,
        )

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _select_best_horizon(
        self,
        ic_results: dict[str, dict],
        horizons: list[int],
    ) -> int:
        """Pick horizon with most Path 1 passing features.

        Ties broken by maximum mean |IC| across passing features.
        Fallback: first horizon in config.
        """
        horizon_counts: dict[int, int] = defaultdict(int)
        horizon_ic_sum: dict[int, float] = defaultdict(float)

        for name, h_results in ic_results.items():
            for h, r in h_results.items():
                if r.passes_path1:
                    horizon_counts[h] += 1
                    horizon_ic_sum[h] += abs(r.ic_mean)

        if not horizon_counts:
            return horizons[0] if horizons else 1

        return max(
            horizon_counts,
            key=lambda h: (horizon_counts[h], horizon_ic_sum[h]),
        )

    # -------------------------------------------------------------------
    # Pre-screening
    # -------------------------------------------------------------------

    def _pre_screen(
        self, evaluation_dates: list[str],
    ) -> tuple[list[int], dict[str, str]]:
        """Exclude categorical and zero-variance features.

        Uses StreamingColumnStats to compute variance in a single streaming
        pass. A feature is zero-variance if variance < 1e-10 across all days.

        Returns:
            (evaluable_indices, excluded_dict)
        """
        schema = self.loader.schema
        streaming_stats = StreamingColumnStats(n_columns=schema.n_features)

        for bundle in self.loader.iter_days(evaluation_dates):
            features_2d = bundle.sequences[:, -1, :]
            streaming_stats.update_batch(
                np.asarray(features_2d, dtype=np.float64)
            )

        summary = streaming_stats.get_summary()
        excluded: dict[str, str] = {}
        evaluable: list[int] = []

        for j in range(schema.n_features):
            name = schema.feature_names.get(j, f"feature_{j}")
            if j in schema.categorical_indices:
                excluded[name] = "categorical"
            elif summary[j]["n"] == 0:
                excluded[name] = "no_data"
            elif summary[j]["std"] ** 2 < 1e-10:
                excluded[name] = "zero_variance"
            else:
                evaluable.append(j)

        return evaluable, excluded

    # -------------------------------------------------------------------
    # Holdout validation
    # -------------------------------------------------------------------

    def _validate_holdout(
        self,
        holdout_dates: list[str],
        candidates: list[str],
        evaluable_indices: list[int],
    ) -> dict[str, bool]:
        """Re-run Layer 1 screening on holdout data for STRONG-KEEP candidates.

        No BH correction on holdout (confirming, not discovering).
        A candidate is confirmed if at any horizon:
            |IC_mean| > ic_threshold AND bootstrap CI excludes zero.
        """
        schema = self.loader.schema
        name_to_idx = {
            schema.feature_names.get(j, f"feature_{j}"): j
            for j in evaluable_indices
        }

        candidate_indices = [
            name_to_idx[name] for name in candidates
            if name in name_to_idx
        ]

        if not candidate_indices:
            return {name: False for name in candidates}

        # Compute per-day ICs on holdout
        horizons = list(self.config.screening.horizons)
        daily_ics: dict[int, dict[int, list[float]]] = {
            j: {h_idx: [] for h_idx in range(len(horizons))}
            for j in candidate_indices
        }

        for bundle in self.loader.iter_days(holdout_dates):
            if bundle.sequences.shape[0] < 3:
                continue
            features_2d = bundle.sequences[:, -1, :]
            labels_2d = bundle.labels

            for j in candidate_indices:
                feature_col = np.asarray(features_2d[:, j], dtype=np.float64)
                for h_idx in range(len(horizons)):
                    if h_idx >= labels_2d.shape[1]:
                        continue
                    label_col = np.asarray(labels_2d[:, h_idx], dtype=np.float64)
                    rho, p = spearman_ic(feature_col, label_col)
                    if not (rho == 0.0 and p == 1.0):
                        daily_ics[j][h_idx].append(rho)

        # Confirm: |IC_mean| > threshold AND bootstrap CI excludes zero
        confirmed: dict[str, bool] = {}
        ic_threshold = self.config.screening.ic_threshold

        for name in candidates:
            if name not in name_to_idx:
                confirmed[name] = False
                continue
            j = name_to_idx[name]
            feature_confirmed = False

            for h_idx in range(len(horizons)):
                ic_array = np.array(daily_ics[j][h_idx])
                if len(ic_array) < 3:
                    continue
                ic_mean = float(np.mean(ic_array))
                if abs(ic_mean) > ic_threshold:
                    try:
                        _, ci_lo, ci_hi = block_bootstrap_ci(
                            statistic_fn=lambda x, _: float(np.mean(x)),
                            x=ic_array,
                            y=np.zeros_like(ic_array),
                            n_bootstraps=500,
                            ci=0.95,
                            seed=self.config.seed,
                        )
                        if (ci_lo > 0) or (ci_hi < 0):
                            feature_confirmed = True
                            break
                    except (ValueError, np.linalg.LinAlgError) as exc:
                        logger.debug(
                            f"Bootstrap CI failed for {name} h={h_idx}: {exc}"
                        )

            confirmed[name] = feature_confirmed

        return confirmed

    # -------------------------------------------------------------------
    # JSON output
    # -------------------------------------------------------------------

    def to_json(
        self, result: FeatureClassification, output_path: str,
    ) -> None:
        """Write classification_table.json.

        Matches pipeline_contract.toml [evaluation.output.required_fields].
        """
        tier_summary: dict[str, int] = {}
        for ft in result.per_feature.values():
            key = ft.tier.value
            tier_summary[key] = tier_summary.get(key, 0) + 1

        features_dict = {}
        for name, ft in sorted(result.per_feature.items()):
            features_dict[name] = {
                "tier": ft.tier.value,
                "passing_paths": list(ft.passing_paths),
                "best_horizon": ft.best_horizon,
                "best_metric": ft.best_metric,
                "best_value": round(ft.best_value, 6),
                "best_p": round(ft.best_p, 6),
                "stability_pct": (
                    round(ft.stability_pct * 100, 1)
                    if ft.stability_pct is not None
                    else None
                ),
                "concurrent_forward_ratio": (
                    round(ft.concurrent_forward_ratio, 3)
                    if ft.concurrent_forward_ratio is not None
                    else None
                ),
            }

        output = {
            "schema": "feature_evaluation_v1",
            "export_dir": self.config.export_dir,
            "export_schema": (
                result.schema.contract_version or result.schema.schema_version
            ),
            "n_features_evaluated": len(result.per_feature),
            "n_features_excluded": len(result.excluded_features),
            "evaluation_date": date.today().isoformat(),
            "seed": self.config.seed,
            "holdout_days": self.config.holdout_days,
            "n_bootstraps": self.config.stability.n_bootstraps,
            "tier_summary": tier_summary,
            "features": features_dict,
            "excluded_features": result.excluded_features,
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

    # -------------------------------------------------------------------
    # v2 Pipeline: Profile-based evaluation
    # -------------------------------------------------------------------

    def run_v2(self) -> dict[str, "FeatureProfile"]:
        """Profile-based evaluation using unified data cache.

        Produces FeatureProfile per feature (not FeatureTier). Uses
        2-pass data loading (vs 49-109 passes in v1) and per-path
        stability (fixes Bug #1: features only passing Paths 3-5
        were auto-DISCARDed in v1).

        Steps:
            1. Split holdout
            2. build_cache() — 2-pass streaming
            3. Run all _from_cache path functions
            4. compute_stability_from_cache() — per-path breakdown
            5. Redundancy: Spearman corr matrix, clustering, VIF
            6. ACF of daily IC series per feature
            7. Build FeatureProfile per feature
            8. Holdout validation for STRONG-KEEP candidates
            9. Return profiles

        Returns:
            dict[feature_name -> FeatureProfile]
        """
        from dataclasses import replace as dc_replace
        import math
        from scipy.stats import rankdata

        from hft_metrics.correlation import (
            correlation_matrix,
            redundant_pairs,
            cluster_by_correlation,
            vif as compute_vif,
        )
        from hft_metrics.acf import autocorrelation

        from hft_evaluator.data.cache import build_cache
        from hft_evaluator.screening.ic_screening import screen_ic_from_cache
        from hft_evaluator.screening.dcor_screening import screen_dcor_from_cache
        from hft_evaluator.selection.concurrent_forward import compute_cf_from_cache
        from hft_evaluator.selection.jmi_selection import jmi_from_cache
        from hft_evaluator.temporal.temporal_ic import compute_temporal_ic_from_cache
        from hft_evaluator.temporal.transfer_entropy import screen_te_from_cache
        from hft_evaluator.regime.regime_ic import compute_regime_ic_from_cache
        from hft_evaluator.stability.stability_selection import (
            compute_stability_from_cache,
        )
        from hft_evaluator.profile import (
            FeatureProfile, PathEvidence, StabilityDetail, compute_tier,
        )

        schema = self.loader.schema
        horizons = list(self.config.screening.horizons)

        # Step 1: Split holdout
        all_dates = self.loader.list_dates()
        eval_dates, holdout_dates = split_holdout(
            all_dates, self.config.holdout_days
        )
        self._log(
            f"v2: {len(eval_dates)} eval + {len(holdout_dates)} holdout days"
        )

        # Step 2: Build cache (2-pass streaming)
        self._log("v2: Building data cache...")
        cache = build_cache(self.loader, eval_dates, self.config)
        self._log(
            f"v2: Cache built: {len(cache.evaluable_indices)} evaluable, "
            f"{cache.n_total_samples} pooled samples"
        )

        # Store excluded for to_json_v2
        self._v2_excluded = cache.excluded_features

        if not cache.evaluable_indices:
            self._log("v2: No evaluable features — returning empty")
            return {}

        # Step 3: Run all paths from cache
        self._log("v2: Running IC screening (Path 1)...")
        ic_results = screen_ic_from_cache(cache, self.config)

        self._log("v2: Running dCor+MI screening (Path 2)...")
        dcor_results = screen_dcor_from_cache(cache, self.config)

        self._log("v2: Running temporal IC (Path 3a)...")
        temporal_results = compute_temporal_ic_from_cache(cache, self.config)

        self._log("v2: Running TE (Path 3b, informational)...")
        te_results = screen_te_from_cache(cache, horizons, self.config)

        conditioning = self.config.regime.conditioning_indices
        if conditioning is None:
            conditioning = self.registry.conditioning_indices()
        self._log("v2: Running regime IC (Path 4)...")
        regime_results = compute_regime_ic_from_cache(
            cache, horizons, conditioning, self.config
        )

        best_horizon = self._select_best_horizon(ic_results, horizons)
        self._log(f"v2: Running JMI (Path 5) at horizon={best_horizon}...")
        jmi_results = jmi_from_cache(cache, best_horizon, self.config)
        jmi_names = {name for name, _ in jmi_results}

        self._log("v2: Running CF decomposition...")
        cf_results = compute_cf_from_cache(cache, horizons)

        # Step 4: Per-path stability
        self._log(
            f"v2: Stability ({self.config.stability.n_bootstraps} bootstraps)..."
        )
        stability_map = compute_stability_from_cache(cache, self.config)

        # Update Path 5 JMI stability in StabilityDetail
        for name, sd in stability_map.items():
            jmi_stab = 1.0 if name in jmi_names else 0.0
            stability_map[name] = StabilityDetail(
                path1_stability=sd.path1_stability,
                path2_stability=sd.path2_stability,
                path3a_stability=sd.path3a_stability,
                combined_stability=sd.combined_stability,
                path4_ci_coverage=sd.path4_ci_coverage,
                path5_jmi_stability=jmi_stab,
            )

        # Step 5: Redundancy (Spearman correlation matrix)
        self._log("v2: Computing redundancy...")
        eval_list = list(cache.evaluable_indices)
        feature_names_list = [
            schema.feature_names.get(j, f"feature_{j}")
            for j in eval_list
        ]

        # Rank-transform for Spearman consistency
        ranked_features = np.apply_along_axis(
            rankdata, 0, cache.pooled_features
        )
        corr_matrix, valid_corr_idx = correlation_matrix(
            ranked_features, eval_list
        )

        clusters = cluster_by_correlation(
            corr_matrix, feature_names_list, valid_corr_idx, threshold=0.7
        )
        # Build name → cluster_id lookup
        name_to_cluster: dict[str, int] = {}
        for cl in clusters:
            for sig_name in cl.signals:
                name_to_cluster[sig_name] = cl.cluster_id

        # Max pairwise correlation per feature
        n_valid = len(valid_corr_idx)
        max_corr_map: dict[int, float] = {}
        for ci, feat_pos in enumerate(valid_corr_idx):
            row = np.abs(corr_matrix[ci, :])
            row[ci] = 0.0  # Exclude self
            max_corr_map[feat_pos] = float(np.max(row)) if n_valid > 1 else 0.0

        # VIF
        try:
            vif_results = compute_vif(
                ranked_features, eval_list, feature_names_list
            )
            name_to_vif = {v.signal_name: v.vif for v in vif_results}
        except (ValueError, np.linalg.LinAlgError):
            name_to_vif = {}

        # Step 6: ACF of daily IC series at each feature's best horizon
        name_to_acf_hl: dict[str, int | None] = {}
        for pos, j in enumerate(cache.evaluable_indices):
            name = schema.feature_names.get(j, f"feature_{j}")
            # Find horizon with highest mean |IC| for this feature
            best_h_idx = 0
            best_abs_ic = 0.0
            for h_idx in range(len(horizons)):
                series = cache.daily_ic_cube[:, pos, h_idx]
                valid = series[np.isfinite(series)]
                if len(valid) > 0:
                    abs_mean = abs(float(np.mean(valid)))
                    if abs_mean > best_abs_ic:
                        best_abs_ic = abs_mean
                        best_h_idx = h_idx

            ic_series = cache.daily_ic_cube[:, pos, best_h_idx]
            ic_valid = ic_series[np.isfinite(ic_series)]
            if len(ic_valid) >= 10:
                _, half_life, _ = autocorrelation(ic_valid, max_lag=50)
                name_to_acf_hl[name] = half_life
            else:
                name_to_acf_hl[name] = None

        # Step 7: Build FeatureProfile per feature
        self._log("v2: Building profiles...")
        profiles: dict[str, FeatureProfile] = {}
        strong_keep_candidates: list[str] = []

        for pos, j in enumerate(cache.evaluable_indices):
            name = schema.feature_names.get(j, f"feature_{j}")
            all_evidence: list[PathEvidence] = []

            # Path 1
            if name in ic_results:
                for h, r in ic_results[name].items():
                    all_evidence.append(PathEvidence(
                        path_name="linear_signal", horizon=h,
                        metric_name="forward_ic",
                        metric_value=r.ic_mean,
                        p_value=r.bh_adjusted_p,
                        ci_lower=r.ci_lower, ci_upper=r.ci_upper,
                        passes=r.passes_path1, is_informational=False,
                    ))

            # Path 2
            if name in dcor_results:
                for h, r in dcor_results[name].items():
                    all_evidence.append(PathEvidence(
                        path_name="nonlinear_signal", horizon=h,
                        metric_name="dcor",
                        metric_value=r.dcor_value,
                        p_value=r.dcor_p,
                        ci_lower=float("nan"), ci_upper=float("nan"),
                        passes=r.passes_path2, is_informational=False,
                    ))

            # Path 3a
            if name in temporal_results:
                tr = temporal_results[name]
                all_evidence.append(PathEvidence(
                    path_name="temporal_value", horizon=tr.best_horizon,
                    metric_name=tr.best_temporal_metric,
                    metric_value=tr.best_temporal_ic,
                    p_value=tr.best_temporal_p,
                    ci_lower=float("nan"), ci_upper=float("nan"),
                    passes=tr.passes_path3, is_informational=False,
                ))

            # Path 3b: TE — INFORMATIONAL only
            if name in te_results:
                for h, ter in te_results[name].items():
                    all_evidence.append(PathEvidence(
                        path_name="transfer_entropy", horizon=h,
                        metric_name=f"te_L{ter.best_lag}",
                        metric_value=ter.te_value,
                        p_value=ter.bh_adjusted_p,
                        ci_lower=float("nan"), ci_upper=float("nan"),
                        passes=ter.passes_te,
                        is_informational=True,  # Stored but NOT gated
                    ))

            # Path 4: Regime
            if name in regime_results:
                for h, regime_list in regime_results[name].items():
                    for rr in regime_list:
                        all_evidence.append(PathEvidence(
                            path_name="regime_conditional", horizon=h,
                            metric_name=f"regime_ic_{rr.conditioning_variable}",
                            metric_value=rr.best_tercile_ic,
                            p_value=float("nan"),
                            ci_lower=float("nan"), ci_upper=float("nan"),
                            passes=rr.passes_path4,
                            is_informational=False,
                        ))

            # Path 5: JMI
            if name in jmi_names:
                jmi_score = next(s for n, s in jmi_results if n == name)
                all_evidence.append(PathEvidence(
                    path_name="interaction_value", horizon=best_horizon,
                    metric_name="jmi_score",
                    metric_value=jmi_score,
                    p_value=float("nan"),
                    ci_lower=float("nan"), ci_upper=float("nan"),
                    passes=True, is_informational=False,
                ))

            # Passing paths: non-informational only
            passing = sorted({
                e.path_name for e in all_evidence
                if e.passes and not e.is_informational
            })

            # Best metric across passing non-informational evidence
            passing_evidence = [
                e for e in all_evidence
                if e.passes and not e.is_informational
            ]
            if passing_evidence:
                best_ev = max(passing_evidence,
                              key=lambda e: abs(e.metric_value))
                best_horizon_feat = best_ev.horizon
                best_metric = best_ev.metric_name
                best_value = best_ev.metric_value
                # Best p: minimum finite p across passing evidence
                finite_ps = [
                    e.p_value for e in passing_evidence
                    if math.isfinite(e.p_value)
                ]
                best_p = min(finite_ps) if finite_ps else float("nan")
            else:
                best_horizon_feat = horizons[0] if horizons else 0
                best_metric = ""
                best_value = 0.0
                best_p = float("nan")

            # CF decomposition at best horizon
            cf_ratio = None
            cf_class = None
            if name in cf_results and best_horizon_feat in cf_results[name]:
                cf = cf_results[name][best_horizon_feat]
                cf_ratio = cf.ratio
                cf_class = cf.classification

            # Stability
            stability = stability_map.get(name, StabilityDetail(
                path1_stability=0.0, path2_stability=0.0,
                path3a_stability=0.0, combined_stability=0.0,
                path4_ci_coverage=0.0, path5_jmi_stability=0.0,
            ))

            # Update Path 4 CI coverage from regime results
            if name in regime_results:
                n_pass = sum(
                    1 for h_list in regime_results[name].values()
                    for rr in h_list if rr.passes_path4
                )
                n_total_regime = sum(
                    len(h_list) for h_list in regime_results[name].values()
                )
                p4_coverage = n_pass / max(n_total_regime, 1)
                stability = StabilityDetail(
                    path1_stability=stability.path1_stability,
                    path2_stability=stability.path2_stability,
                    path3a_stability=stability.path3a_stability,
                    combined_stability=stability.combined_stability,
                    path4_ci_coverage=p4_coverage,
                    path5_jmi_stability=stability.path5_jmi_stability,
                )

            # Redundancy — max_corr_map is keyed by feature index (from valid_corr_idx)
            max_corr = max_corr_map.get(j)

            profile = FeatureProfile(
                feature_name=name,
                feature_index=j,
                best_horizon=best_horizon_feat,
                best_metric=best_metric,
                best_value=best_value,
                best_p=best_p,
                passing_paths=tuple(passing),
                stability=stability,
                concurrent_forward_ratio=cf_ratio,
                cf_classification=cf_class,
                redundancy_cluster_id=name_to_cluster.get(name),
                max_pairwise_correlation=max_corr,
                vif=name_to_vif.get(name),
                ic_acf_half_life=name_to_acf_hl.get(name),
                all_evidence=tuple(all_evidence),
            )

            profiles[name] = profile

            # Identify holdout candidates
            if (
                len(passing) > 0
                and math.isfinite(best_p)
                and best_p < self.config.classification.strong_keep_p
                and stability.combined_stability
                >= self.config.stability.stable_threshold
            ):
                strong_keep_candidates.append(name)

        # Step 8: Holdout validation
        if holdout_dates and strong_keep_candidates:
            self._log(
                f"v2: Holdout validation: {len(strong_keep_candidates)} "
                f"candidates on {len(holdout_dates)} days"
            )
            holdout_confirmed = self._validate_holdout(
                holdout_dates, strong_keep_candidates,
                list(cache.evaluable_indices),
            )
            for name in strong_keep_candidates:
                if holdout_confirmed.get(name, False):
                    profiles[name] = dc_replace(
                        profiles[name], holdout_confirmed=True
                    )

        # Summary
        tier_counts: dict[str, int] = {}
        for p in profiles.values():
            tier = compute_tier(
                p,
                self.config.stability.stable_threshold,
                self.config.stability.investigate_threshold,
                self.config.classification.strong_keep_p,
            )
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        self._log(f"v2: Classification: {tier_counts}")

        return profiles

    def to_json_v2(
        self,
        profiles: dict[str, "FeatureProfile"],
        output_path: str,
        excluded_features: dict[str, str] | None = None,
    ) -> None:
        """Write feature_profiles.json with v2 schema.

        Backward compatible: includes all v1 fields plus new profile fields.
        """
        import math
        from hft_evaluator.profile import compute_tier

        tier_summary: dict[str, int] = {}
        features_dict = {}

        for name, p in sorted(profiles.items()):
            tier = compute_tier(
                p,
                self.config.stability.stable_threshold,
                self.config.stability.investigate_threshold,
                self.config.classification.strong_keep_p,
            )
            tier_summary[tier] = tier_summary.get(tier, 0) + 1

            features_dict[name] = {
                # v1 fields (backward compat)
                "tier": tier,
                "passing_paths": list(p.passing_paths),
                "best_horizon": p.best_horizon,
                "best_metric": p.best_metric,
                "best_value": round(p.best_value, 6),
                "best_p": (
                    round(p.best_p, 6)
                    if math.isfinite(p.best_p) else None
                ),
                "stability_pct": round(
                    p.stability.combined_stability * 100, 1
                ),
                "concurrent_forward_ratio": (
                    round(p.concurrent_forward_ratio, 3)
                    if p.concurrent_forward_ratio is not None else None
                ),
                # v2 fields
                "stability_detail": {
                    "path1": round(p.stability.path1_stability, 3),
                    "path2": round(p.stability.path2_stability, 3),
                    "path3a": round(p.stability.path3a_stability, 3),
                    "combined": round(p.stability.combined_stability, 3),
                    "path4_ci_coverage": round(
                        p.stability.path4_ci_coverage, 3
                    ),
                    "path5_jmi": round(p.stability.path5_jmi_stability, 3),
                },
                "cf_classification": p.cf_classification,
                "redundancy_cluster_id": p.redundancy_cluster_id,
                "max_pairwise_correlation": (
                    round(p.max_pairwise_correlation, 3)
                    if p.max_pairwise_correlation is not None else None
                ),
                "vif": (
                    round(p.vif, 2) if p.vif is not None else None
                ),
                "ic_acf_half_life": p.ic_acf_half_life,
                "holdout_confirmed": p.holdout_confirmed,
            }

        output = {
            "schema": "feature_evaluation_v2",
            "export_dir": self.config.export_dir,
            "export_schema": (
                self.loader.schema.contract_version
                or self.loader.schema.schema_version
            ),
            "n_features_evaluated": len(profiles),
            "n_features_excluded": len(
                excluded_features
                or getattr(self, "_v2_excluded", {})
            ),
            "evaluation_date": date.today().isoformat(),
            "seed": self.config.seed,
            "holdout_days": self.config.holdout_days,
            "n_bootstraps": self.config.stability.n_bootstraps,
            "tier_summary": tier_summary,
            "features": features_dict,
            "excluded_features": (
                excluded_features
                or getattr(self, "_v2_excluded", {})
            ),
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

    # -------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(f"[evaluate] {msg}", flush=True)
        logger.info(msg)
