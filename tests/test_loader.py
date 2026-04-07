"""Tests for ExportLoader: schema detection, shape validation, streaming."""

from pathlib import Path

import numpy as np
import pytest

from hft_evaluator.data.loader import ExportLoader, ExportSchema, DayBundle


class TestAutoDetect:
    def test_offexchange_detected(self, synthetic_export):
        loader = ExportLoader(synthetic_export, "train")
        assert loader.schema.contract_version == "off_exchange_1.0"
        assert loader.schema.schema_version == "1.0"

    def test_schema_properties(self, synthetic_export):
        loader = ExportLoader(synthetic_export, "train")
        schema = loader.schema
        assert schema.n_features == 34
        assert schema.window_size == 5
        assert schema.horizons == (1, 5, 10)
        assert schema.bin_size_seconds == 60

    def test_feature_names_dict(self, synthetic_export):
        loader = ExportLoader(synthetic_export, "train")
        # Off-exchange feature names come from hft-contracts
        # Synthetic uses 8 features but contract defines 34
        # The schema still provides the full contract names
        assert 0 in loader.schema.feature_names
        assert loader.schema.feature_names[0] == "trf_signed_imbalance"

    def test_categorical_indices(self, synthetic_export):
        loader = ExportLoader(synthetic_export, "train")
        # Off-exchange categoricals: {29, 30, 32, 33}
        assert 29 in loader.schema.categorical_indices
        assert 33 in loader.schema.categorical_indices


class TestListDates:
    def test_sorted_ascending(self, synthetic_export):
        loader = ExportLoader(synthetic_export, "train")
        dates = loader.list_dates()
        assert dates == sorted(dates)

    def test_count(self, synthetic_export):
        loader = ExportLoader(synthetic_export, "train")
        assert len(loader.list_dates()) == 5


class TestLoadDay:
    def test_shape(self, synthetic_export):
        loader = ExportLoader(synthetic_export, "train")
        bundle = loader.load_day(loader.list_dates()[0])
        assert bundle.sequences.shape == (50, 5, 34)
        assert bundle.labels.shape == (50, 3)

    def test_dtype(self, synthetic_export):
        loader = ExportLoader(synthetic_export, "train")
        bundle = loader.load_day(loader.list_dates()[0])
        assert bundle.sequences.dtype == np.float32
        assert bundle.labels.dtype == np.float64

    def test_date_stored(self, synthetic_export):
        loader = ExportLoader(synthetic_export, "train")
        dates = loader.list_dates()
        bundle = loader.load_day(dates[0])
        assert bundle.date == dates[0]

    def test_metadata_present(self, synthetic_export):
        loader = ExportLoader(synthetic_export, "train")
        bundle = loader.load_day(loader.list_dates()[0])
        assert "contract_version" in bundle.metadata
        assert bundle.metadata["contract_version"] == "off_exchange_1.0"

    def test_nonexistent_date(self, synthetic_export):
        loader = ExportLoader(synthetic_export, "train")
        with pytest.raises(FileNotFoundError):
            loader.load_day("2099-01-01")


class TestIterDays:
    def test_count(self, synthetic_export):
        loader = ExportLoader(synthetic_export, "train")
        bundles = list(loader.iter_days())
        assert len(bundles) == 5

    def test_subset(self, synthetic_export):
        loader = ExportLoader(synthetic_export, "train")
        dates = loader.list_dates()[:2]
        bundles = list(loader.iter_days(dates))
        assert len(bundles) == 2
        assert bundles[0].date == dates[0]
        assert bundles[1].date == dates[1]


class TestErrors:
    def test_invalid_split(self, synthetic_export):
        with pytest.raises(FileNotFoundError, match="Split directory"):
            ExportLoader(synthetic_export, "nonexistent_split")

    def test_invalid_export_dir(self):
        with pytest.raises(FileNotFoundError):
            ExportLoader("/nonexistent/path", "train")
