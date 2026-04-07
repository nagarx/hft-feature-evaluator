"""Tests for FeatureRegistry: group mapping, evaluation properties."""

import pytest
from hft_evaluator.data.loader import ExportLoader
from hft_evaluator.data.registry import FeatureRegistry


@pytest.fixture
def registry(synthetic_export):
    loader = ExportLoader(synthetic_export, "train")
    return FeatureRegistry(loader.schema)


class TestOffExchangeRegistry:
    def test_total_features(self, registry):
        assert len([registry.get(i) for i in range(34)]) == 34

    def test_evaluable_count(self, registry):
        """34 total - 4 categoricals = 30 evaluable."""
        evaluable = registry.evaluable_indices()
        assert len(evaluable) == 30

    def test_categoricals_not_evaluable(self, registry):
        for idx in [29, 30, 32, 33]:
            assert registry.get(idx).evaluable is False

    def test_non_categoricals_evaluable(self, registry):
        for idx in [0, 1, 2, 3, 12, 27, 31]:
            assert registry.get(idx).evaluable is True

    def test_group_signed_flow(self, registry):
        assert registry.group_indices("signed_flow") == [0, 1, 2, 3]

    def test_group_bbo_dynamics(self, registry):
        assert registry.group_indices("bbo_dynamics") == [12, 13, 14, 15, 16, 17]

    def test_feature_info_trf_signed(self, registry):
        info = registry.get(0)
        assert info.name == "trf_signed_imbalance"
        assert info.group == "signed_flow"
        assert info.evaluable is True
        assert info.signed is True

    def test_feature_info_unsigned(self, registry):
        # dark_share (index 4) is unsigned
        info = registry.get(4)
        assert info.name == "dark_share"
        assert info.signed is False

    def test_conditioning_indices(self, registry):
        cond = registry.conditioning_indices()
        assert cond == {
            "spread_bps": 12,
            "session_progress": 31,
            "bin_trade_count": 27,
        }

    def test_group_names_ordered(self, registry):
        names = registry.group_names()
        assert names[0] == "signed_flow"
        assert names[1] == "venue_metrics"
        assert len(names) == 10
