"""Tests for EvaluationConfig: YAML parsing, defaults, validation."""

import pytest
import yaml
from pathlib import Path

from hft_evaluator.config import (
    EvaluationConfig,
    ScreeningConfig,
    StabilityConfig,
    ClassificationConfig,
    RegimeConfig,
    TemporalConfig,
    SelectionConfig,
)


# ---------------------------------------------------------------------------
# Valid configs
# ---------------------------------------------------------------------------


class TestFromDict:
    def test_full_config(self, synthetic_config_dict):
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        config.validate()
        assert config.holdout_days == 1
        assert config.seed == 42
        assert config.screening.horizons == (1, 5, 10)
        assert config.screening.bh_fdr_level == 0.05
        assert config.stability.n_bootstraps == 10
        assert config.classification.strong_keep_p == 0.01

    def test_minimal_config(self):
        """Only export_dir required; everything else has defaults."""
        config = EvaluationConfig.from_dict({"export_dir": "/tmp/test"})
        config.validate()
        assert config.export_dir == "/tmp/test"
        assert config.split == "train"
        assert config.holdout_days == 20
        assert config.screening.horizons == (1, 2, 3, 5, 10, 20, 30, 60)

    def test_horizons_as_tuple(self, synthetic_config_dict):
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        assert isinstance(config.screening.horizons, tuple)

    def test_regime_defaults(self):
        config = EvaluationConfig.from_dict({"export_dir": "/tmp"})
        assert config.regime.n_bins == 3
        assert config.regime.min_samples_per_bin == 30
        assert config.regime.conditioning_indices is None


class TestFromYaml:
    def test_offexchange_config(self, tmp_path):
        config_path = tmp_path / "test.yaml"
        config_path.write_text(yaml.dump({
            "export_dir": "/tmp/test",
            "split": "train",
            "holdout_days": 10,
            "seed": 42,
            "screening": {"horizons": [1, 5, 10]},
        }))
        config = EvaluationConfig.from_yaml(str(config_path))
        assert config.holdout_days == 10
        assert config.screening.horizons == (1, 5, 10)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            EvaluationConfig.from_yaml("/nonexistent/path.yaml")


# ---------------------------------------------------------------------------
# Frozen immutability
# ---------------------------------------------------------------------------


class TestFrozen:
    def test_top_level_frozen(self, synthetic_config_dict):
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        with pytest.raises(AttributeError):
            config.holdout_days = 99

    def test_screening_frozen(self, synthetic_config_dict):
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        with pytest.raises(AttributeError):
            config.screening.bh_fdr_level = 0.99


# ---------------------------------------------------------------------------
# Unknown keys
# ---------------------------------------------------------------------------


class TestUnknownKeys:
    def test_unknown_top_level(self):
        with pytest.raises(ValueError, match="Unknown config keys"):
            EvaluationConfig.from_dict({
                "export_dir": "/tmp", "unknown_key": True
            })

    def test_unknown_screening_key(self):
        with pytest.raises(ValueError, match="Unknown keys in screening"):
            EvaluationConfig.from_dict({
                "export_dir": "/tmp",
                "screening": {"horizons": [1], "bad_key": 5},
            })

    def test_unknown_stability_key(self):
        with pytest.raises(ValueError, match="Unknown keys in stability"):
            EvaluationConfig.from_dict({
                "export_dir": "/tmp",
                "stability": {"extra": True},
            })


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def _make(self, **overrides):
        d = {"export_dir": "/tmp/test"}
        d.update(overrides)
        return EvaluationConfig.from_dict(d)

    def test_empty_export_dir(self):
        config = self._make(export_dir="")
        with pytest.raises(ValueError, match="export_dir"):
            config.validate()

    def test_invalid_split(self):
        config = self._make(split="invalid")
        with pytest.raises(ValueError, match="split"):
            config.validate()

    def test_holdout_negative(self):
        config = self._make(holdout_days=-1)
        with pytest.raises(ValueError, match="holdout_days"):
            config.validate()

    def test_holdout_too_large(self):
        config = self._make(holdout_days=101)
        with pytest.raises(ValueError, match="holdout_days"):
            config.validate()

    def test_horizons_empty(self):
        config = self._make(screening={"horizons": []})
        with pytest.raises(ValueError, match="horizons.*non-empty"):
            config.validate()

    def test_horizons_unsorted(self):
        config = self._make(screening={"horizons": [10, 5, 1]})
        with pytest.raises(ValueError, match="sorted ascending"):
            config.validate()

    def test_horizons_negative(self):
        config = self._make(screening={"horizons": [-1, 5]})
        with pytest.raises(ValueError, match="positive"):
            config.validate()

    def test_fdr_zero(self):
        config = self._make(screening={"horizons": [1], "bh_fdr_level": 0.0})
        with pytest.raises(ValueError, match="bh_fdr_level"):
            config.validate()

    def test_thresholds_inverted(self):
        config = self._make(stability={
            "stable_threshold": 0.3,
            "investigate_threshold": 0.5,
        })
        with pytest.raises(ValueError, match="stable_threshold.*must be >"):
            config.validate()

    def test_strong_keep_above_fdr(self):
        config = self._make(
            screening={"horizons": [1], "bh_fdr_level": 0.05},
            classification={"strong_keep_p": 0.1},
        )
        with pytest.raises(ValueError, match="strong_keep_p.*must be <"):
            config.validate()

    def test_regime_bins_too_small(self):
        config = self._make(regime={"n_bins": 1})
        with pytest.raises(ValueError, match="n_bins"):
            config.validate()

    def test_regime_min_samples_too_small(self):
        config = self._make(regime={"min_samples_per_bin": 5})
        with pytest.raises(ValueError, match="min_samples_per_bin"):
            config.validate()

    def test_dcor_permutations_too_small(self):
        config = self._make(screening={"horizons": [1], "dcor_permutations": 50})
        with pytest.raises(ValueError, match="dcor_permutations"):
            config.validate()

    def test_valid_config_passes(self, synthetic_config_dict):
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        config.validate()  # Should not raise

    # --- Temporal config validation ---

    def test_temporal_rolling_window_too_small(self):
        config = self._make(temporal={"rolling_window": 1})
        with pytest.raises(ValueError, match="rolling_window"):
            config.validate()

    def test_temporal_te_lags_empty(self):
        config = self._make(temporal={"te_lags": []})
        with pytest.raises(ValueError, match="te_lags.*non-empty"):
            config.validate()

    def test_temporal_te_lags_unsorted(self):
        config = self._make(temporal={"te_lags": [3, 1, 2]})
        with pytest.raises(ValueError, match="te_lags.*sorted"):
            config.validate()

    def test_temporal_te_lags_negative(self):
        config = self._make(temporal={"te_lags": [0, 1]})
        with pytest.raises(ValueError, match="te_lags.*>= 1"):
            config.validate()

    # --- Selection config validation ---

    def test_selection_elbow_threshold_zero(self):
        config = self._make(selection={"jmi_elbow_threshold": 0.0})
        with pytest.raises(ValueError, match="jmi_elbow_threshold"):
            config.validate()

    def test_selection_max_features_zero(self):
        config = self._make(selection={"jmi_max_features": 0})
        with pytest.raises(ValueError, match="jmi_max_features"):
            config.validate()

    def test_mi_permutations_too_small(self):
        config = self._make(screening={"horizons": [1], "mi_permutations": 10})
        with pytest.raises(ValueError, match="mi_permutations"):
            config.validate()


class TestPhase3Defaults:
    """Test that Phase 3 config sections have correct defaults."""

    def test_temporal_defaults(self):
        config = EvaluationConfig.from_dict({"export_dir": "/tmp"})
        assert config.temporal.rolling_window == 5
        assert config.temporal.te_lags == (1, 2, 3)

    def test_selection_defaults(self):
        config = EvaluationConfig.from_dict({"export_dir": "/tmp"})
        assert config.selection.jmi_max_features is None
        assert config.selection.jmi_elbow_threshold == 0.05

    def test_mi_permutations_default(self):
        config = EvaluationConfig.from_dict({"export_dir": "/tmp"})
        assert config.screening.mi_permutations == 200

    def test_temporal_from_yaml(self, tmp_path):
        config_path = tmp_path / "test.yaml"
        config_path.write_text("export_dir: /tmp\ntemporal:\n  rolling_window: 7\n  te_lags: [1, 2]\n")
        config = EvaluationConfig.from_yaml(str(config_path))
        assert config.temporal.rolling_window == 7
        assert config.temporal.te_lags == (1, 2)
