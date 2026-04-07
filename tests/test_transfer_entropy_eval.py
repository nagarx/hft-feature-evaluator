"""Tests for transfer entropy screening."""

import pytest
from hft_evaluator.data.loader import ExportLoader
from hft_evaluator.config import EvaluationConfig
from hft_evaluator.temporal.transfer_entropy import screen_transfer_entropy


class TestScreenTE:
    def test_returns_dict(self, synthetic_export, synthetic_config_dict):
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        results = screen_transfer_entropy(loader, dates, [0, 2], [1, 5], config)
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_all_horizons_present(self, synthetic_export, synthetic_config_dict):
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        results = screen_transfer_entropy(loader, dates, [0], [1, 5, 10], config)
        name_0 = loader.schema.feature_names[0]
        assert set(results[name_0].keys()) == {1, 5, 10}

    def test_best_lag_in_range(self, synthetic_export, synthetic_config_dict):
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        results = screen_transfer_entropy(loader, dates, [0], [1], config)
        name_0 = loader.schema.feature_names[0]
        assert results[name_0][1].best_lag in (1, 2, 3)

    def test_p_values_bounded(self, synthetic_export, synthetic_config_dict):
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        results = screen_transfer_entropy(loader, dates, [0, 2], [1], config)
        for name, horizons in results.items():
            for h, r in horizons.items():
                assert 0.0 < r.bh_adjusted_p <= 1.0

    def test_deterministic(self, synthetic_export, synthetic_config_dict):
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        r1 = screen_transfer_entropy(loader, dates, [0], [1], config)
        r2 = screen_transfer_entropy(loader, dates, [0], [1], config)
        name_0 = loader.schema.feature_names[0]
        assert r1[name_0][1].te_value == r2[name_0][1].te_value
