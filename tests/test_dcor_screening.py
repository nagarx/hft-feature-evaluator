"""Tests for dCor+MI screening: subsampled pooled permutation tests, BH correction."""

import numpy as np
import pytest

from hft_evaluator.data.loader import ExportLoader
from hft_evaluator.config import EvaluationConfig
from hft_evaluator.screening.dcor_screening import screen_dcor


class TestScreenDcor:
    """Integration tests using synthetic export fixture."""

    def test_returns_dict(self, synthetic_export, synthetic_config_dict):
        synthetic_config_dict["screening"]["dcor_permutations"] = 100
        synthetic_config_dict["screening"]["mi_permutations"] = 50
        synthetic_config_dict["screening"]["dcor_subsample"] = 500
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0, 2, 3, 12]

        results = screen_dcor(loader, dates, evaluable, config)
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_all_horizons_present(self, synthetic_export, synthetic_config_dict):
        synthetic_config_dict["screening"]["dcor_permutations"] = 100
        synthetic_config_dict["screening"]["mi_permutations"] = 50
        synthetic_config_dict["screening"]["dcor_subsample"] = 500
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0]

        results = screen_dcor(loader, dates, evaluable, config)
        name_0 = loader.schema.feature_names[0]
        assert set(results[name_0].keys()) == {1, 5, 10}

    def test_p_values_bounded(self, synthetic_export, synthetic_config_dict):
        synthetic_config_dict["screening"]["dcor_permutations"] = 100
        synthetic_config_dict["screening"]["mi_permutations"] = 50
        synthetic_config_dict["screening"]["dcor_subsample"] = 500
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0, 2]

        results = screen_dcor(loader, dates, evaluable, config)
        for name, horizons in results.items():
            for h, r in horizons.items():
                assert 0.0 < r.dcor_p <= 1.0
                assert 0.0 < r.mi_p <= 1.0

    def test_passes_requires_both(self, synthetic_export, synthetic_config_dict):
        """passes_path2 requires BOTH dcor AND mi BH-rejected."""
        synthetic_config_dict["screening"]["dcor_permutations"] = 100
        synthetic_config_dict["screening"]["mi_permutations"] = 50
        synthetic_config_dict["screening"]["dcor_subsample"] = 500
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0, 2]

        results = screen_dcor(loader, dates, evaluable, config)
        for name, horizons in results.items():
            for h, r in horizons.items():
                if r.passes_path2:
                    assert r.dcor_bh_rejected and r.mi_bh_rejected

    def test_dcor_value_non_negative(self, synthetic_export, synthetic_config_dict):
        synthetic_config_dict["screening"]["dcor_permutations"] = 100
        synthetic_config_dict["screening"]["mi_permutations"] = 50
        synthetic_config_dict["screening"]["dcor_subsample"] = 500
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0]

        results = screen_dcor(loader, dates, evaluable, config)
        for name, horizons in results.items():
            for h, r in horizons.items():
                assert r.dcor_value >= 0.0

    def test_n_subsample_correct(self, synthetic_export, synthetic_config_dict):
        synthetic_config_dict["screening"]["dcor_permutations"] = 100
        synthetic_config_dict["screening"]["mi_permutations"] = 50
        synthetic_config_dict["screening"]["dcor_subsample"] = 500
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0]

        results = screen_dcor(loader, dates, evaluable, config)
        name_0 = loader.schema.feature_names[0]
        # 4 days × 50 seqs = 200 total < 500 subsample → uses all
        assert results[name_0][1].n_subsample == 200

    def test_deterministic(self, synthetic_export, synthetic_config_dict):
        synthetic_config_dict["screening"]["dcor_permutations"] = 100
        synthetic_config_dict["screening"]["mi_permutations"] = 50
        synthetic_config_dict["screening"]["dcor_subsample"] = 500
        loader = ExportLoader(synthetic_export, "train")
        config = EvaluationConfig.from_dict(synthetic_config_dict)
        dates = loader.list_dates()[:4]
        evaluable = [0, 2]

        r1 = screen_dcor(loader, dates, evaluable, config)
        r2 = screen_dcor(loader, dates, evaluable, config)

        for name in r1:
            for h in r1[name]:
                assert r1[name][h].dcor_value == r2[name][h].dcor_value
                assert r1[name][h].dcor_p == r2[name][h].dcor_p
