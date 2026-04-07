"""Tests for temporal IC: rolling feature IC, per-day streaming, BH correction."""

import numpy as np
import pytest

from hft_evaluator.data.loader import ExportLoader
from hft_evaluator.config import EvaluationConfig
from hft_evaluator.temporal.temporal_ic import compute_temporal_ic


class TestTemporalIC:
    def test_returns_dict(self, synthetic_export, synthetic_config_dict):
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0, 2, 3]

        results = compute_temporal_ic(loader, dates, evaluable, [1, 5, 10], config)
        assert isinstance(results, dict)
        assert len(results) == 3

    def test_all_features_present(self, synthetic_export, synthetic_config_dict):
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0, 2]

        results = compute_temporal_ic(loader, dates, evaluable, [1, 5], config)
        name_0 = loader.schema.feature_names[0]
        name_2 = loader.schema.feature_names[2]
        assert name_0 in results
        assert name_2 in results

    def test_rolling_ics_populated(self, synthetic_export, synthetic_config_dict):
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0]

        results = compute_temporal_ic(loader, dates, evaluable, [1], config)
        name_0 = loader.schema.feature_names[0]
        r = results[name_0]
        # All three rolling ICs should have some value (could be 0 for noise)
        assert isinstance(r.rolling_mean_ic, float)
        assert isinstance(r.rolling_slope_ic, float)
        assert isinstance(r.rate_of_change_ic, float)

    def test_best_metric_selected(self, synthetic_export, synthetic_config_dict):
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0]

        results = compute_temporal_ic(loader, dates, evaluable, [1, 5, 10], config)
        name_0 = loader.schema.feature_names[0]
        r = results[name_0]
        assert r.best_temporal_metric in ("rolling_mean", "rolling_slope", "rate_of_change")
        assert abs(r.best_temporal_ic) >= 0

    def test_passes_bool(self, synthetic_export, synthetic_config_dict):
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0, 2]

        results = compute_temporal_ic(loader, dates, evaluable, [1], config)
        for name, r in results.items():
            assert isinstance(r.passes_path3, bool)

    def test_n_days_correct(self, synthetic_export, synthetic_config_dict):
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0]

        results = compute_temporal_ic(loader, dates, evaluable, [1], config)
        name_0 = loader.schema.feature_names[0]
        assert results[name_0].n_days == 4

    def test_deterministic(self, synthetic_export, synthetic_config_dict):
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0]

        r1 = compute_temporal_ic(loader, dates, evaluable, [1], config)
        r2 = compute_temporal_ic(loader, dates, evaluable, [1], config)
        name_0 = loader.schema.feature_names[0]
        assert r1[name_0].best_temporal_ic == r2[name_0].best_temporal_ic
