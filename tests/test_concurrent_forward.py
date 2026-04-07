"""Tests for concurrent/forward IC decomposition."""

import numpy as np
import pytest

from hft_evaluator.data.loader import ExportLoader
from hft_evaluator.selection.concurrent_forward import decompose_concurrent_forward


class TestConcurrentForward:
    def test_returns_all_features(self, synthetic_export):
        loader = ExportLoader(synthetic_export, "train")
        dates = loader.list_dates()
        evaluable = [0, 2, 3]
        horizons = [1, 5, 10]

        results = decompose_concurrent_forward(loader, dates, evaluable, horizons)
        assert len(results) == 3

    def test_returns_all_horizons(self, synthetic_export):
        loader = ExportLoader(synthetic_export, "train")
        dates = loader.list_dates()
        evaluable = [0]
        horizons = [1, 5, 10]

        results = decompose_concurrent_forward(loader, dates, evaluable, horizons)
        name_0 = loader.schema.feature_names[0]
        assert set(results[name_0].keys()) == {1, 5, 10}

    def test_n_days_correct(self, synthetic_export):
        loader = ExportLoader(synthetic_export, "train")
        dates = loader.list_dates()
        evaluable = [0]
        horizons = [1]

        results = decompose_concurrent_forward(loader, dates, evaluable, horizons)
        name_0 = loader.schema.feature_names[0]
        assert results[name_0][1].n_days == 5

    def test_classification_values(self, synthetic_export):
        """All classifications should be valid strings."""
        loader = ExportLoader(synthetic_export, "train")
        dates = loader.list_dates()
        evaluable = [0, 2, 3, 12]
        horizons = [1]

        results = decompose_concurrent_forward(loader, dates, evaluable, horizons)
        valid = {"contemporaneous", "partially_forward", "forward", "state_variable"}
        for name, horizon_results in results.items():
            for h, r in horizon_results.items():
                assert r.classification in valid, (
                    f"{name} h={h}: {r.classification} not in {valid}"
                )

    def test_ratio_non_negative(self, synthetic_export):
        """Ratio uses abs() for both ICs → always non-negative."""
        loader = ExportLoader(synthetic_export, "train")
        dates = loader.list_dates()
        evaluable = [0, 2]
        horizons = [1, 5]

        results = decompose_concurrent_forward(loader, dates, evaluable, horizons)
        for name, horizon_results in results.items():
            for h, r in horizon_results.items():
                assert r.ratio >= 0

    def test_label_shift_direction(self, synthetic_export):
        """Forward IC uses label[t], concurrent uses label[t-1].
        For our signal feature (correlated with label), forward should be nonzero."""
        loader = ExportLoader(synthetic_export, "train")
        dates = loader.list_dates()
        evaluable = [0]  # Signal feature
        horizons = [1]

        results = decompose_concurrent_forward(loader, dates, evaluable, horizons)
        name_0 = loader.schema.feature_names[0]
        r = results[name_0][1]
        # Feature 0 is correlated with label[:,0] → forward IC should be nonzero
        assert abs(r.forward_ic) > 0.01

    def test_noise_feature_near_zero(self, synthetic_export):
        """Feature 2 is pure noise → both ICs should be near zero."""
        loader = ExportLoader(synthetic_export, "train")
        dates = loader.list_dates()
        evaluable = [2]
        horizons = [1]

        results = decompose_concurrent_forward(loader, dates, evaluable, horizons)
        name_2 = loader.schema.feature_names[2]
        r = results[name_2][1]
        assert abs(r.forward_ic) < 0.3
        assert abs(r.concurrent_ic) < 0.3

    def test_deterministic(self, synthetic_export):
        """Two runs produce identical results."""
        loader = ExportLoader(synthetic_export, "train")
        dates = loader.list_dates()
        evaluable = [0, 2]
        horizons = [1]

        r1 = decompose_concurrent_forward(loader, dates, evaluable, horizons)
        r2 = decompose_concurrent_forward(loader, dates, evaluable, horizons)
        for name in r1:
            for h in r1[name]:
                assert r1[name][h].forward_ic == r2[name][h].forward_ic
                assert r1[name][h].concurrent_ic == r2[name][h].concurrent_ic

    def test_eps_guard(self, synthetic_export):
        """Feature with IC=0 should not cause division error."""
        loader = ExportLoader(synthetic_export, "train")
        dates = loader.list_dates()
        evaluable = [1]  # Zero variance → IC will be 0
        horizons = [1]

        # Feature 1 is constant → spearman_ic returns 0.0
        results = decompose_concurrent_forward(loader, dates, evaluable, horizons)
        name_1 = loader.schema.feature_names[1]
        r = results[name_1][1]
        assert np.isfinite(r.ratio)
