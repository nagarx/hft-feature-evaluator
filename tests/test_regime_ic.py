"""Tests for regime-conditional IC (Path 4)."""

import numpy as np
import pytest

from hft_evaluator.data.loader import ExportLoader
from hft_evaluator.data.registry import FeatureRegistry
from hft_evaluator.config import EvaluationConfig
from hft_evaluator.regime.regime_ic import compute_regime_ic


class TestRegimeIC:
    def test_returns_results(self, synthetic_export, synthetic_config_dict):
        loader = ExportLoader(synthetic_export, "train")
        registry = FeatureRegistry(loader.schema)
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0, 2, 3]
        conditioning = {"spread_bps": 12}

        results = compute_regime_ic(
            loader, dates, evaluable, [1, 5, 10], conditioning, config
        )
        assert len(results) > 0

    def test_per_conditioning_variable(self, synthetic_export, synthetic_config_dict):
        """Each feature+horizon should have one RegimeICResult per conditioning var."""
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0]
        conditioning = {"spread_bps": 12, "session_progress": 31}

        results = compute_regime_ic(
            loader, dates, evaluable, [1], conditioning, config
        )
        name_0 = loader.schema.feature_names[0]
        # 2 conditioning vars → 2 RegimeICResult per (feature, horizon)
        assert len(results[name_0][1]) == 2

    def test_tercile_names(self, synthetic_export, synthetic_config_dict):
        # Use low min_samples to allow terciles with ~67 samples each
        synthetic_config_dict["regime"] = {"min_samples_per_bin": 10}
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0]
        conditioning = {"spread_bps": 12}

        results = compute_regime_ic(
            loader, dates, evaluable, [1], conditioning, config
        )
        name_0 = loader.schema.feature_names[0]
        regime_result = results[name_0][1][0]
        tercile_names = {t.tercile_name for t in regime_result.per_tercile}
        assert tercile_names == {"LOW", "MEDIUM", "HIGH"}

    def test_sample_counts(self, synthetic_export, synthetic_config_dict):
        """~200 pooled samples / 3 terciles = ~66-67 per cell."""
        # Use low min_samples to allow terciles with ~67 samples each
        synthetic_config_dict["regime"] = {"min_samples_per_bin": 10}
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0]
        conditioning = {"spread_bps": 12}

        results = compute_regime_ic(
            loader, dates, evaluable, [1], conditioning, config
        )
        name_0 = loader.schema.feature_names[0]
        regime_result = results[name_0][1][0]
        total_samples = sum(t.n_samples for t in regime_result.per_tercile)
        # 4 days * 50 seqs = 200 total
        assert total_samples > 100

    def test_ci_bounds_valid(self, synthetic_export, synthetic_config_dict):
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0]
        conditioning = {"spread_bps": 12}

        results = compute_regime_ic(
            loader, dates, evaluable, [1], conditioning, config
        )
        name_0 = loader.schema.feature_names[0]
        for regime_result in results[name_0][1]:
            for tr in regime_result.per_tercile:
                assert tr.ci_lower <= tr.ic <= tr.ci_upper

    def test_passes_bool(self, synthetic_export, synthetic_config_dict):
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0, 2]
        conditioning = {"spread_bps": 12}

        results = compute_regime_ic(
            loader, dates, evaluable, [1], conditioning, config
        )
        for name, horizons in results.items():
            for h, regime_list in horizons.items():
                for rr in regime_list:
                    assert isinstance(rr.passes_path4, bool)

    def test_deterministic(self, synthetic_export, synthetic_config_dict):
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0]
        conditioning = {"spread_bps": 12}

        r1 = compute_regime_ic(loader, dates, evaluable, [1], conditioning, config)
        r2 = compute_regime_ic(loader, dates, evaluable, [1], conditioning, config)

        name_0 = loader.schema.feature_names[0]
        for i, (t1, t2) in enumerate(
            zip(r1[name_0][1][0].per_tercile, r2[name_0][1][0].per_tercile)
        ):
            assert t1.ic == t2.ic

    def test_all_horizons_covered(self, synthetic_export, synthetic_config_dict):
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0]
        conditioning = {"spread_bps": 12}

        results = compute_regime_ic(
            loader, dates, evaluable, [1, 5, 10], conditioning, config
        )
        name_0 = loader.schema.feature_names[0]
        assert set(results[name_0].keys()) == {1, 5, 10}
