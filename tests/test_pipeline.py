"""Tests for EvaluationPipeline: pre-screen, end-to-end, holdout, JSON output."""

import json

import pytest

from hft_evaluator.config import EvaluationConfig
from hft_evaluator.pipeline import EvaluationPipeline
from hft_evaluator.decision import Tier
from hft_evaluator.profile import FeatureProfile, StabilityDetail


class TestPreScreen:
    def test_excludes_zero_variance(self, synthetic_config_dict):
        """Feature 1 (mroib) is constant 1.0 → excluded as zero_variance."""
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        pipeline = EvaluationPipeline(config)
        dates = pipeline.loader.list_dates()[:4]
        evaluable, excluded = pipeline._pre_screen(dates)
        assert "mroib" in excluded
        assert excluded["mroib"] == "zero_variance"

    def test_excludes_categoricals(self, synthetic_config_dict):
        """Features at indices {29, 30, 32, 33} excluded as categorical."""
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        pipeline = EvaluationPipeline(config)
        dates = pipeline.loader.list_dates()[:4]
        evaluable, excluded = pipeline._pre_screen(dates)
        assert "bin_valid" in excluded
        assert excluded["bin_valid"] == "categorical"
        assert "schema_version" in excluded
        assert excluded["schema_version"] == "categorical"

    def test_evaluable_count(self, synthetic_config_dict):
        """34 total - 4 categorical - 1 zero-variance = 29 evaluable."""
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        pipeline = EvaluationPipeline(config)
        dates = pipeline.loader.list_dates()[:4]
        evaluable, excluded = pipeline._pre_screen(dates)
        # Feature 1 is zero-variance, indices 29,30,32,33 are categorical
        assert 1 not in evaluable  # zero-variance
        assert 29 not in evaluable  # categorical
        assert 0 in evaluable  # signal feature


class TestEndToEnd:
    def test_runs_without_error(self, synthetic_config_dict):
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        pipeline = EvaluationPipeline(config)
        result = pipeline.run()
        assert result is not None
        assert len(result.per_feature) > 0

    def test_feature_0_evaluated(self, synthetic_config_dict):
        """Feature 0 (trf_signed_imbalance) should be in results."""
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        pipeline = EvaluationPipeline(config)
        result = pipeline.run()
        assert "trf_signed_imbalance" in result.per_feature

    def test_excluded_features_populated(self, synthetic_config_dict):
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        pipeline = EvaluationPipeline(config)
        result = pipeline.run()
        assert len(result.excluded_features) >= 4  # At least 4 categoricals

    def test_all_tiers_valid(self, synthetic_config_dict):
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        pipeline = EvaluationPipeline(config)
        result = pipeline.run()
        for name, ft in result.per_feature.items():
            assert ft.tier in Tier

    def test_passing_paths_are_strings(self, synthetic_config_dict):
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        pipeline = EvaluationPipeline(config)
        result = pipeline.run()
        for name, ft in result.per_feature.items():
            assert isinstance(ft.passing_paths, tuple)
            for p in ft.passing_paths:
                assert isinstance(p, str)


class TestHoldout:
    def test_holdout_zero_no_validation(self, synthetic_config_dict):
        """holdout_days=0 → no holdout validation."""
        synthetic_config_dict["holdout_days"] = 0
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        pipeline = EvaluationPipeline(config)
        result = pipeline.run()
        assert result.holdout is None


class TestJsonOutput:
    def test_json_schema(self, synthetic_config_dict, tmp_path):
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        pipeline = EvaluationPipeline(config)
        result = pipeline.run()

        output_path = str(tmp_path / "classification_table.json")
        pipeline.to_json(result, output_path)

        with open(output_path) as f:
            data = json.load(f)

        # Check required fields from pipeline_contract.toml [evaluation.output]
        required_output = [
            "schema", "export_dir", "export_schema",
            "n_features_evaluated", "n_features_excluded",
            "evaluation_date", "seed", "holdout_days", "n_bootstraps",
            "tier_summary", "features",
        ]
        for field in required_output:
            assert field in data, f"Missing required field: {field}"

        assert data["schema"] == "feature_evaluation_v1"
        assert data["seed"] == 42
        assert isinstance(data["features"], dict)

        # Check per-feature required fields
        required_per_feature = [
            "tier", "passing_paths", "best_horizon",
            "best_metric", "best_value", "best_p",
            "stability_pct", "concurrent_forward_ratio",
        ]
        for name, feat in data["features"].items():
            for field in required_per_feature:
                assert field in feat, (
                    f"Feature {name} missing field: {field}"
                )

    def test_deterministic(self, synthetic_config_dict, tmp_path):
        """Two runs produce identical JSON."""
        config = EvaluationConfig.from_dict(synthetic_config_dict)

        p1 = EvaluationPipeline(config)
        r1 = p1.run()
        path1 = str(tmp_path / "r1.json")
        p1.to_json(r1, path1)

        p2 = EvaluationPipeline(config)
        r2 = p2.run()
        path2 = str(tmp_path / "r2.json")
        p2.to_json(r2, path2)

        with open(path1) as f:
            d1 = json.load(f)
        with open(path2) as f:
            d2 = json.load(f)

        # Compare features (exclude evaluation_date which varies)
        assert d1["features"] == d2["features"]
        assert d1["tier_summary"] == d2["tier_summary"]


# ---------------------------------------------------------------------------
# v2 Pipeline: Profile-based evaluation
# ---------------------------------------------------------------------------


class TestV2Pipeline:
    """Tests for run_v2() — profile-based evaluation.

    Single comprehensive test to avoid repeating the expensive run_v2()
    call (which includes stability bootstraps with permutation tests).
    """

    def test_v2_end_to_end(self, synthetic_config_dict):
        """Comprehensive v2 pipeline test — runs once, checks everything."""
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        pipeline = EvaluationPipeline(config)

        # Phase 4: last_profile_hash must be None before first run_v2 call
        assert pipeline.last_profile_hash is None

        profiles = pipeline.run_v2()

        # Phase 4: after run_v2, last_profile_hash is a 64-char hex digest
        # and matches an independent compute_profile_hash call on the
        # returned profiles (lock the integration point).
        from hft_evaluator.pipeline import compute_profile_hash
        assert pipeline.last_profile_hash is not None
        assert len(pipeline.last_profile_hash) == 64
        assert all(c in "0123456789abcdef" for c in pipeline.last_profile_hash)
        assert pipeline.last_profile_hash == compute_profile_hash(profiles)

        # Basic structure
        assert isinstance(profiles, dict)
        assert len(profiles) > 0

        # All entries are FeatureProfile
        for name, p in profiles.items():
            assert isinstance(p, FeatureProfile), f"{name} is not FeatureProfile"
            assert p.feature_name == name

        # Signal feature present, excluded features absent
        assert "trf_signed_imbalance" in profiles
        assert "mroib" not in profiles      # zero-variance
        assert "bin_valid" not in profiles   # categorical

        # Stability detail populated with correct invariant
        for name, p in profiles.items():
            sd = p.stability
            assert isinstance(sd, StabilityDetail)
            assert 0.0 <= sd.path1_stability <= 1.0
            assert 0.0 <= sd.path2_stability <= 1.0
            assert 0.0 <= sd.path3a_stability <= 1.0
            assert 0.0 <= sd.combined_stability <= 1.0
            assert sd.combined_stability == max(
                sd.path1_stability, sd.path2_stability, sd.path3a_stability
            ), (
                f"{name}: combined={sd.combined_stability} != "
                f"max({sd.path1_stability}, {sd.path2_stability}, "
                f"{sd.path3a_stability})"
            )

        # At least some features have evidence
        has_evidence = sum(
            1 for p in profiles.values() if len(p.all_evidence) > 0
        )
        assert has_evidence > 0

        # TE is informational only — never in passing_paths
        for p in profiles.values():
            assert "transfer_entropy" not in p.passing_paths
            for ev in p.all_evidence:
                if ev.path_name == "transfer_entropy":
                    assert ev.is_informational is True


class TestV2JsonOutput:
    def test_v2_json_schema(self, synthetic_config_dict, tmp_path):
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        pipeline = EvaluationPipeline(config)
        profiles = pipeline.run_v2()

        output_path = str(tmp_path / "profiles.json")
        pipeline.to_json_v2(profiles, output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert data["schema"] == "feature_evaluation_v2"
        assert data["seed"] == 42
        assert isinstance(data["features"], dict)

        # v1 backward-compat fields
        v1_fields = [
            "tier", "passing_paths", "best_horizon",
            "best_metric", "best_value", "best_p",
            "stability_pct", "concurrent_forward_ratio",
        ]
        for name, feat in data["features"].items():
            for field in v1_fields:
                assert field in feat, f"Feature {name} missing v1 field: {field}"

        # v2 fields
        v2_fields = [
            "stability_detail", "cf_classification",
            "redundancy_cluster_id", "vif", "ic_acf_half_life",
            "holdout_confirmed",
        ]
        for name, feat in data["features"].items():
            for field in v2_fields:
                assert field in feat, f"Feature {name} missing v2 field: {field}"

    def test_v2_stability_detail_structure(self, synthetic_config_dict, tmp_path):
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        pipeline = EvaluationPipeline(config)
        profiles = pipeline.run_v2()

        output_path = str(tmp_path / "profiles.json")
        pipeline.to_json_v2(profiles, output_path)

        with open(output_path) as f:
            data = json.load(f)

        for name, feat in data["features"].items():
            sd = feat["stability_detail"]
            assert "path1" in sd
            assert "path2" in sd
            assert "path3a" in sd
            assert "combined" in sd
            assert "path4_ci_coverage" in sd
            assert "path5_jmi" in sd
