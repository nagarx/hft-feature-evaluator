"""
Model feedback protocol: structured interface for post-training feature importance.

Defines the artifact protocol that lob-model-trainer will implement to feed
importance scores back into the evaluator for iterative refinement.

NOT YET IMPLEMENTED. This module defines:
    - FeatureImportance: per-feature importance from a trained model
    - ModelFeedbackProvider: Protocol for models to export importance
    - merge_feedback_into_profiles(): stub for enriching profiles

Reference: Plan Phase 5 — Model Feedback Protocol
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class FeatureImportance:
    """Post-training importance score for one feature.

    Produced by lob-model-trainer after a training run.
    Consumed by the evaluator to enrich FeatureProfiles.

    Attrs:
        feature_name: Feature name matching FeatureProfile.feature_name.
        importance_score: Importance metric (interpretation depends on method).
        importance_method: One of "stg_gate", "loco_group", "ig_tsr", "permutation".
        model_id: Identifier for the trained model (e.g., experiment name).
        horizon: Prediction horizon the model was trained for.
    """

    feature_name: str
    importance_score: float
    importance_method: str
    model_id: str
    horizon: int


@runtime_checkable
class ModelFeedbackProvider(Protocol):
    """Protocol for models to report feature importance back to evaluator.

    Implementors: lob-model-trainer training runs.
    The evaluator calls these methods to retrieve post-training importance.
    """

    def get_feature_importances(
        self, model_id: str,
    ) -> list[FeatureImportance]:
        """Return importance scores for all features used by this model."""
        ...

    def get_training_metrics(
        self, model_id: str,
    ) -> dict[str, float]:
        """Return training/validation metrics (loss, IC, R2, DA, etc.)."""
        ...


def merge_feedback_into_profiles(
    profiles: dict,
    importances: list[FeatureImportance],
) -> dict:
    """Merge model feedback into feature profiles.

    Not yet implemented. When implemented, this will:
    1. Match importances to profiles by feature_name
    2. Return new profiles with model_importance fields populated
    3. Enable criteria like min_model_importance in SelectionCriteria

    Raises:
        NotImplementedError: Always (stub).
    """
    raise NotImplementedError(
        "Model feedback integration not yet implemented. "
        "See feedback.py protocol for interface definition. "
        "Implementation deferred to when lob-model-trainer "
        "produces STG/LOCO/IG artifacts."
    )
