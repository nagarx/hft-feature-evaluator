"""Tests for IC screening: per-day IC, bootstrap CI, BH correction, pass/fail."""

import numpy as np
import pytest

from hft_evaluator.data.loader import ExportLoader
from hft_evaluator.config import EvaluationConfig
from hft_evaluator.screening import bh_adjusted_pvalues
from hft_evaluator.screening.ic_screening import screen_ic


class TestBHAdjustedPValues:
    """Golden tests for BH-adjusted p-value computation."""

    def test_simple_case(self):
        """3 p-values: BH adjusts by m/rank."""
        p = np.array([0.01, 0.03, 0.05])
        adj = bh_adjusted_pvalues(p)
        # p_adj = [0.01*3/1, 0.03*3/2, 0.05*3/3]
        #       = [0.03, 0.045, 0.05]
        # Monotonicity: [0.03, 0.045, 0.05] — already monotone
        np.testing.assert_allclose(adj, [0.03, 0.045, 0.05])

    def test_monotonicity_enforcement(self):
        """BH step-down ensures adjusted[i] <= adjusted[i+1]."""
        p = np.array([0.01, 0.80, 0.05])
        adj = bh_adjusted_pvalues(p)
        # Sorted: [0.01, 0.05, 0.80] at ranks 1, 2, 3
        # Raw adj: [0.01*3/1, 0.05*3/2, 0.80*3/3] = [0.03, 0.075, 0.80]
        # Step-down: [min(0.03, 0.075), min(0.075, 0.80), 0.80] = [0.03, 0.075, 0.80]
        # Unsort to original order: [0.03, 0.80, 0.075]
        np.testing.assert_allclose(adj, [0.03, 0.80, 0.075])

    def test_capped_at_one(self):
        p = np.array([0.5, 0.9])
        adj = bh_adjusted_pvalues(p)
        assert all(a <= 1.0 for a in adj)

    def test_empty_array(self):
        adj = bh_adjusted_pvalues(np.array([]))
        assert len(adj) == 0

    def test_single_pvalue(self):
        adj = bh_adjusted_pvalues(np.array([0.03]))
        np.testing.assert_allclose(adj, [0.03])

    def test_all_significant(self):
        """Very small p-values should all remain significant."""
        p = np.array([0.001, 0.002, 0.003])
        adj = bh_adjusted_pvalues(p)
        assert all(a < 0.05 for a in adj)


class TestScreenIC:
    """Integration tests using the synthetic export fixture."""

    def test_returns_dict(self, synthetic_export, synthetic_config_dict):
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]  # Use 4 days for eval (1 holdout)
        evaluable = [0, 2, 3, 12]  # Signal, noise, regime, conditioning

        results = screen_ic(loader, dates, evaluable, config)
        assert isinstance(results, dict)
        # Should have entries for each feature name
        assert len(results) > 0

    def test_feature_0_has_signal(self, synthetic_export, synthetic_config_dict):
        """Feature 0 is designed with IC ~ 0.3 → should show nonzero IC."""
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0, 2]

        results = screen_ic(loader, dates, evaluable, config)
        name_0 = loader.schema.feature_names[0]  # trf_signed_imbalance
        assert name_0 in results
        # At horizon 0 (index 0, horizon=1): IC should be positive
        h1_result = results[name_0].get(1)  # horizon 1
        if h1_result is not None:
            assert abs(h1_result.ic_mean) > 0.05, (
                f"Feature 0 should have |IC| > 0.05, got {h1_result.ic_mean:.4f}"
            )

    def test_noise_feature_low_ic(self, synthetic_export, synthetic_config_dict):
        """Feature 2 is pure noise → IC should be near zero."""
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [2]

        results = screen_ic(loader, dates, evaluable, config)
        name_2 = loader.schema.feature_names[2]
        assert name_2 in results
        for h, r in results[name_2].items():
            assert abs(r.ic_mean) < 0.3, (
                f"Noise feature IC should be near 0, got {r.ic_mean:.4f} at h={h}"
            )

    def test_ic_ir_computation(self):
        """Golden test: known daily ICs → known IC_IR.
        ic_ir uses sample std (ddof=1): mean/std = 0.1/0.015811 = 6.3246."""
        from hft_metrics.ic import ic_ir
        daily = np.array([0.1, 0.12, 0.08, 0.11, 0.09])
        # ic_ir uses np.std with ddof=1 (sample std)
        expected_ir = np.mean(daily) / np.std(daily, ddof=1)
        actual = ic_ir(daily)
        np.testing.assert_allclose(actual, expected_ir, rtol=1e-6)

    def test_n_days_correct(self, synthetic_export, synthetic_config_dict):
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0]

        results = screen_ic(loader, dates, evaluable, config)
        name_0 = loader.schema.feature_names[0]
        for h, r in results[name_0].items():
            assert r.n_days == 4

    def test_bh_adjusted_p_present(self, synthetic_export, synthetic_config_dict):
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0, 2]

        results = screen_ic(loader, dates, evaluable, config)
        for name, horizons in results.items():
            for h, r in horizons.items():
                assert 0.0 <= r.bh_adjusted_p <= 1.0
                assert 0.0 <= r.raw_p <= 1.0

    def test_deterministic(self, synthetic_export, synthetic_config_dict):
        """Two runs produce identical results."""
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0, 2]

        r1 = screen_ic(loader, dates, evaluable, config)
        r2 = screen_ic(loader, dates, evaluable, config)

        for name in r1:
            for h in r1[name]:
                assert r1[name][h].ic_mean == r2[name][h].ic_mean
                assert r1[name][h].ci_lower == r2[name][h].ci_lower

    def test_all_horizons_covered(self, synthetic_export, synthetic_config_dict):
        """Each feature should have results for all horizons in the config."""
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0]

        results = screen_ic(loader, dates, evaluable, config)
        name_0 = loader.schema.feature_names[0]
        assert set(results[name_0].keys()) == {1, 5, 10}
