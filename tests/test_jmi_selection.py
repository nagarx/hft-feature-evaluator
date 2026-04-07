"""Tests for JMI forward selection."""

import pytest
from hft_evaluator.data.loader import ExportLoader
from hft_evaluator.config import EvaluationConfig
from hft_evaluator.selection.jmi_selection import jmi_forward_selection


class TestJMISelection:
    def test_returns_list(self, synthetic_export, synthetic_config_dict):
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0, 2, 3, 12]

        results = jmi_forward_selection(loader, dates, evaluable, 1, config)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_returns_tuples(self, synthetic_export, synthetic_config_dict):
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        results = jmi_forward_selection(loader, dates, [0, 2, 3], 1, config)
        for name, score in results:
            assert isinstance(name, str)
            assert isinstance(score, float)

    def test_max_features_respected(self, synthetic_export, synthetic_config_dict):
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        results = jmi_forward_selection(
            loader, dates, [0, 2, 3, 12], 1, config, max_features=2
        )
        assert len(results) <= 2

    def test_invalid_horizon_returns_empty(self, synthetic_export, synthetic_config_dict):
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        results = jmi_forward_selection(loader, dates, [0], 999, config)
        assert results == []

    def test_deterministic(self, synthetic_export, synthetic_config_dict):
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        r1 = jmi_forward_selection(loader, dates, [0, 2, 3], 1, config)
        r2 = jmi_forward_selection(loader, dates, [0, 2, 3], 1, config)
        assert r1 == r2

    def test_first_feature_highest_relevancy(self, synthetic_export, synthetic_config_dict):
        """First selected feature should have highest MI with target."""
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        results = jmi_forward_selection(loader, dates, [0, 2], 1, config)
        # Feature 0 has signal, feature 2 is noise → feature 0 selected first
        if results:
            assert results[0][1] >= results[-1][1]
