"""Tests for model feedback protocol (Phase 5 stub)."""

import pytest
from dataclasses import FrozenInstanceError

from hft_evaluator.feedback import (
    FeatureImportance,
    ModelFeedbackProvider,
    merge_feedback_into_profiles,
)


class TestFeatureImportance:
    def test_frozen(self):
        fi = FeatureImportance(
            feature_name="spread_bps",
            importance_score=0.85,
            importance_method="stg_gate",
            model_id="e17_tlob",
            horizon=10,
        )
        with pytest.raises(FrozenInstanceError):
            fi.importance_score = 0.0

    def test_fields(self):
        fi = FeatureImportance(
            feature_name="depth_norm_ofi",
            importance_score=0.42,
            importance_method="loco_group",
            model_id="e17_ridge",
            horizon=60,
        )
        assert fi.feature_name == "depth_norm_ofi"
        assert fi.importance_method == "loco_group"
        assert fi.horizon == 60


class TestModelFeedbackProvider:
    def test_runtime_checkable(self):
        """Protocol should be runtime-checkable."""
        assert hasattr(ModelFeedbackProvider, "__protocol_attrs__") or True
        # The Protocol is runtime_checkable — test that isinstance works
        # on a conforming class
        class MockProvider:
            def get_feature_importances(self, model_id):
                return []
            def get_training_metrics(self, model_id):
                return {}

        assert isinstance(MockProvider(), ModelFeedbackProvider)

    def test_non_conforming_fails(self):
        class NotAProvider:
            pass

        assert not isinstance(NotAProvider(), ModelFeedbackProvider)


class TestMergeFeedback:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            merge_feedback_into_profiles({}, [])
