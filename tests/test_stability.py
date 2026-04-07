"""Tests for stability selection: bootstrap subsampling of Layer 1."""

import pytest
from hft_evaluator.data.loader import ExportLoader
from hft_evaluator.data.holdout import split_holdout
from hft_evaluator.data.cache import build_cache
from hft_evaluator.config import EvaluationConfig
from hft_evaluator.profile import StabilityDetail
from hft_evaluator.stability.stability_selection import (
    stability_selection,
    compute_stability_from_cache,
)


class TestStabilitySelection:
    def test_returns_dict(self, synthetic_export, synthetic_config_dict):
        """Uses minimal config: 10 bootstraps, small subsample for speed."""
        synthetic_config_dict["screening"]["dcor_permutations"] = 50
        synthetic_config_dict["screening"]["mi_permutations"] = 50
        synthetic_config_dict["screening"]["dcor_subsample"] = 500
        synthetic_config_dict["stability"]["n_bootstraps"] = 10
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0, 2]

        results = stability_selection(loader, dates, evaluable, [1, 5, 10], config)
        assert isinstance(results, dict)
        assert len(results) == 2

    def test_stability_bounded(self, synthetic_export, synthetic_config_dict):
        synthetic_config_dict["screening"]["dcor_permutations"] = 50
        synthetic_config_dict["screening"]["mi_permutations"] = 50
        synthetic_config_dict["screening"]["dcor_subsample"] = 500
        synthetic_config_dict["stability"]["n_bootstraps"] = 10
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0, 2]

        results = stability_selection(loader, dates, evaluable, [1], config)
        for name, pct in results.items():
            assert 0.0 <= pct <= 1.0

    def test_deterministic(self, synthetic_export, synthetic_config_dict):
        synthetic_config_dict["screening"]["dcor_permutations"] = 50
        synthetic_config_dict["screening"]["mi_permutations"] = 50
        synthetic_config_dict["screening"]["dcor_subsample"] = 500
        synthetic_config_dict["stability"]["n_bootstraps"] = 10
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0]

        r1 = stability_selection(loader, dates, evaluable, [1], config)
        r2 = stability_selection(loader, dates, evaluable, [1], config)
        assert r1 == r2

    def test_all_features_present(self, synthetic_export, synthetic_config_dict):
        synthetic_config_dict["screening"]["dcor_permutations"] = 50
        synthetic_config_dict["screening"]["mi_permutations"] = 50
        synthetic_config_dict["screening"]["dcor_subsample"] = 500
        synthetic_config_dict["stability"]["n_bootstraps"] = 10
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0, 2, 3]

        results = stability_selection(loader, dates, evaluable, [1], config)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Cache-based per-path stability tests
# ---------------------------------------------------------------------------


class TestStabilityFromCache:
    """Tests for compute_stability_from_cache (Phase 3)."""

    @pytest.fixture
    def cache_and_config(self, synthetic_export, synthetic_config_dict):
        """Build cache with lean stability settings for testing."""
        synthetic_config_dict["screening"]["dcor_permutations"] = 50
        synthetic_config_dict["screening"]["mi_permutations"] = 50
        synthetic_config_dict["screening"]["dcor_subsample"] = 500
        synthetic_config_dict["stability"]["n_bootstraps"] = 10
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        loader = ExportLoader(config.export_dir, config.split)
        all_dates = loader.list_dates()
        eval_dates, _ = split_holdout(all_dates, config.holdout_days)
        cache = build_cache(loader, eval_dates, config)
        return cache, config

    def test_returns_stability_detail(self, cache_and_config):
        cache, config = cache_and_config
        result = compute_stability_from_cache(cache, config)
        assert isinstance(result, dict)
        for name, sd in result.items():
            assert isinstance(sd, StabilityDetail)

    def test_all_evaluable_features_present(self, cache_and_config):
        cache, config = cache_and_config
        result = compute_stability_from_cache(cache, config)
        feature_names = cache.schema.feature_names
        expected_names = {
            feature_names.get(j, f"feature_{j}")
            for j in cache.evaluable_indices
        }
        assert set(result.keys()) == expected_names

    def test_stability_values_bounded(self, cache_and_config):
        cache, config = cache_and_config
        result = compute_stability_from_cache(cache, config)
        for sd in result.values():
            assert 0.0 <= sd.path1_stability <= 1.0
            assert 0.0 <= sd.path2_stability <= 1.0
            assert 0.0 <= sd.path3a_stability <= 1.0
            assert 0.0 <= sd.combined_stability <= 1.0

    def test_combined_is_max_of_paths(self, cache_and_config):
        cache, config = cache_and_config
        result = compute_stability_from_cache(cache, config)
        for sd in result.values():
            expected = max(sd.path1_stability, sd.path2_stability,
                          sd.path3a_stability)
            assert sd.combined_stability == expected, (
                f"combined={sd.combined_stability} != max({sd.path1_stability}, "
                f"{sd.path2_stability}, {sd.path3a_stability})={expected}"
            )

    def test_deterministic(self, cache_and_config):
        cache, config = cache_and_config
        r1 = compute_stability_from_cache(cache, config)
        r2 = compute_stability_from_cache(cache, config)
        for name in r1:
            assert r1[name].path1_stability == r2[name].path1_stability
            assert r1[name].path2_stability == r2[name].path2_stability
            assert r1[name].path3a_stability == r2[name].path3a_stability
