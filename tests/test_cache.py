"""Tests for DataCache and build_cache().

Validates:
    - Cache shapes and dtypes
    - Pre-screen correctness (evaluable indices)
    - Pooled data integrity
    - Determinism
"""

import numpy as np
import pytest

from hft_evaluator.config import EvaluationConfig
from hft_evaluator.data.cache import DataCache, build_cache, TEMPORAL_METRICS
from hft_evaluator.data.loader import ExportLoader
from hft_evaluator.data.holdout import split_holdout


@pytest.fixture
def cache(synthetic_export, synthetic_config_dict):
    """Build cache from synthetic fixture."""
    config = EvaluationConfig.from_dict(synthetic_config_dict)
    loader = ExportLoader(config.export_dir, config.split)
    all_dates = loader.list_dates()
    eval_dates, _ = split_holdout(all_dates, config.holdout_days)
    return build_cache(loader, eval_dates, config), config, eval_dates


class TestBuildCache:
    """Test cache construction from synthetic fixture."""

    def test_returns_data_cache(self, cache):
        dc, _, _ = cache
        assert isinstance(dc, DataCache)

    def test_evaluation_dates(self, cache):
        dc, _, eval_dates = cache
        assert dc.evaluation_dates == tuple(eval_dates)

    def test_evaluable_excludes_categorical(self, cache):
        dc, _, _ = cache
        # Synthetic fixture: categoricals at {29, 30, 32, 33}
        for cat_idx in [29, 30, 32, 33]:
            assert cat_idx not in dc.evaluable_indices

    def test_evaluable_excludes_zero_variance(self, cache):
        dc, _, _ = cache
        # Synthetic fixture: feature 1 is constant 1.0
        assert 1 not in dc.evaluable_indices

    def test_evaluable_includes_signal(self, cache):
        dc, _, _ = cache
        # Synthetic fixture: feature 0 has injected signal
        assert 0 in dc.evaluable_indices

    def test_daily_ic_cube_shape(self, cache):
        dc, config, eval_dates = cache
        n_days = len(eval_dates)
        n_eval = len(dc.evaluable_indices)
        n_horizons = len(config.screening.horizons)
        assert dc.daily_ic_cube.shape == (n_days, n_eval, n_horizons)

    def test_daily_ic_cube_dtype(self, cache):
        dc, _, _ = cache
        assert dc.daily_ic_cube.dtype == np.float64

    def test_temporal_cubes_present(self, cache):
        dc, _, _ = cache
        for metric in TEMPORAL_METRICS:
            assert metric in dc.daily_temporal_cubes
            assert dc.daily_temporal_cubes[metric].shape == dc.daily_ic_cube.shape

    def test_forward_concurrent_cubes_shape(self, cache):
        dc, _, _ = cache
        assert dc.daily_forward_ic_cube.shape == dc.daily_ic_cube.shape
        assert dc.daily_concurrent_ic_cube.shape == dc.daily_ic_cube.shape

    def test_pooled_features_shape(self, cache):
        dc, _, _ = cache
        assert dc.pooled_features.ndim == 2
        assert dc.pooled_features.shape[1] == dc.schema.n_features

    def test_pooled_labels_shape(self, cache):
        dc, _, _ = cache
        assert dc.pooled_labels.ndim == 2
        assert dc.pooled_labels.shape[0] == dc.pooled_features.shape[0]

    def test_n_total_samples(self, cache):
        dc, _, _ = cache
        assert dc.n_total_samples == dc.pooled_features.shape[0]
        assert dc.n_total_samples > 0

    def test_signal_feature_has_finite_ics(self, cache):
        dc, _, _ = cache
        # Feature 0 (trf_signed_imbalance) should have some finite ICs
        pos = list(dc.evaluable_indices).index(0)
        finite_count = np.sum(np.isfinite(dc.daily_ic_cube[:, pos, :]))
        assert finite_count > 0, "Signal feature should have finite ICs"

    def test_deterministic(self, synthetic_export, synthetic_config_dict):
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        loader = ExportLoader(config.export_dir, config.split)
        all_dates = loader.list_dates()
        eval_dates, _ = split_holdout(all_dates, config.holdout_days)

        cache1 = build_cache(loader, eval_dates, config)
        cache2 = build_cache(loader, eval_dates, config)

        np.testing.assert_array_equal(
            cache1.daily_ic_cube, cache2.daily_ic_cube
        )
        np.testing.assert_array_equal(
            cache1.pooled_features, cache2.pooled_features
        )

    def test_excluded_features_populated(self, cache):
        dc, _, _ = cache
        # At least categorical features should be excluded
        assert len(dc.excluded_features) >= 4
