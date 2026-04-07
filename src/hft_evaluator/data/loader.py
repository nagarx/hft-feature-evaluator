"""
Schema-aware NPY loader with auto-detection of MBO vs off-exchange exports.

Loads exported NPY sequences, labels, and metadata one day at a time.
Validates against hft-contracts at boundary (load time), trusts internal data.

Reference: CODEBASE.md Section 3.1
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

from hft_contracts import (
    # Off-exchange
    OFF_EXCHANGE_FEATURE_NAMES,
    OFF_EXCHANGE_CATEGORICAL_INDICES,
    OFF_EXCHANGE_FEATURE_COUNT,
    OFF_EXCHANGE_SCHEMA_VERSION,
    validate_off_exchange_export_contract,
    # MBO
    FEATURE_COUNT,
    FULL_FEATURE_COUNT,
    CATEGORICAL_INDICES,
    FeatureIndex,
    validate_export_contract,
)
from hft_contracts.validation import ContractError


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportSchema:
    """Detected schema properties for an export directory."""

    schema_version: str
    contract_version: str         # "off_exchange_1.0" or "" for MBO
    n_features: int
    window_size: int
    horizons: tuple[int, ...]
    bin_size_seconds: int | None  # 60 for time-based, None for event-based
    feature_names: dict[int, str]
    categorical_indices: frozenset[int]


@dataclass
class DayBundle:
    """Single day's loaded data."""

    date: str
    sequences: np.ndarray   # [N, T, F] float32
    labels: np.ndarray      # [N, H] float64
    metadata: dict


# ---------------------------------------------------------------------------
# ExportLoader
# ---------------------------------------------------------------------------

# Pattern to extract date from filenames like "2025-02-03_metadata.json"
# or "20250203_metadata.json"
_DATE_PATTERN = re.compile(r"^(\d{4}-?\d{2}-?\d{2})_metadata\.json$")


class ExportLoader:
    """Schema-aware loader for ANY pipeline export (MBO or off-exchange).

    Auto-detects schema from the first metadata file in the split directory.
    Validates contract at construction time. Streams day-by-day.

    Args:
        export_dir: Path to export root (e.g., "data/exports/basic_nvda_60s").
        split: "train", "val", or "test".

    Raises:
        FileNotFoundError: If split directory or metadata files not found.
        ContractError: If metadata fails contract validation.
    """

    def __init__(self, export_dir: str, split: str = "train"):
        self._split_dir = Path(export_dir) / split
        if not self._split_dir.is_dir():
            raise FileNotFoundError(
                f"Split directory not found: {self._split_dir}"
            )

        # Discover dates from metadata files
        meta_files = sorted(self._split_dir.glob("*_metadata.json"))
        if not meta_files:
            raise FileNotFoundError(
                f"No metadata files found in {self._split_dir}"
            )

        self._dates: list[str] = []
        for mf in meta_files:
            m = _DATE_PATTERN.match(mf.name)
            if m:
                self._dates.append(m.group(1))

        if not self._dates:
            raise FileNotFoundError(
                f"No valid date metadata files in {self._split_dir}"
            )

        # Read first metadata to detect schema
        first_meta = self._load_metadata(self._dates[0])
        self._schema = self._detect_schema(first_meta)

    @property
    def schema(self) -> ExportSchema:
        """Detected export schema."""
        return self._schema

    def list_dates(self) -> list[str]:
        """All dates in the split, sorted ascending."""
        return list(self._dates)

    def load_day(self, date: str) -> DayBundle:
        """Load one day's data.

        Args:
            date: Date string matching the file naming convention.

        Returns:
            DayBundle with sequences, labels, metadata.

        Raises:
            FileNotFoundError: If day files are missing.
            ValueError: If shapes don't match schema.
        """
        seq_path = self._split_dir / f"{date}_sequences.npy"
        if not seq_path.exists():
            raise FileNotFoundError(f"Sequences not found: {seq_path}")

        # Load sequences (mmap for memory efficiency on large files)
        sequences = np.load(str(seq_path), mmap_mode="r")

        # Load labels — try standard name, then regression name for MBO
        label_path = self._split_dir / f"{date}_labels.npy"
        if not label_path.exists():
            label_path = self._split_dir / f"{date}_regression_labels.npy"
            if label_path.exists():
                logger.warning(
                    f"Using regression_labels.npy for {date} "
                    f"(labels.npy not found)"
                )
        if not label_path.exists():
            raise FileNotFoundError(
                f"Labels not found for {date} in {self._split_dir}"
            )
        labels = np.load(str(label_path))

        metadata = self._load_metadata(date)

        # Shape validation
        if sequences.shape[2] != self._schema.n_features:
            raise ValueError(
                f"Feature count mismatch for {date}: "
                f"sequences has {sequences.shape[2]}, "
                f"schema expects {self._schema.n_features}"
            )
        if sequences.shape[1] != self._schema.window_size:
            raise ValueError(
                f"Window size mismatch for {date}: "
                f"sequences has {sequences.shape[1]}, "
                f"schema expects {self._schema.window_size}"
            )
        if labels.shape[0] != sequences.shape[0]:
            raise ValueError(
                f"Sample count mismatch for {date}: "
                f"sequences has {sequences.shape[0]}, "
                f"labels has {labels.shape[0]}"
            )

        # Ensure correct dtypes
        if sequences.dtype != np.float32:
            sequences = sequences.astype(np.float32)
        if labels.dtype != np.float64:
            labels = labels.astype(np.float64)

        return DayBundle(
            date=date,
            sequences=sequences,
            labels=labels,
            metadata=metadata,
        )

    def iter_days(self, dates: list[str] | None = None) -> Iterator[DayBundle]:
        """Stream day-by-day. Previous bundle is GC-eligible after yield.

        Args:
            dates: Optional subset of dates. None → all dates in split.
        """
        for date in (dates or self._dates):
            yield self.load_day(date)

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _load_metadata(self, date: str) -> dict:
        meta_path = self._split_dir / f"{date}_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        with open(meta_path) as f:
            return json.load(f)

    def _detect_schema(self, metadata: dict) -> ExportSchema:
        """Auto-detect schema from first metadata file and validate."""
        contract_version = metadata.get("contract_version", "")

        if str(contract_version).startswith("off_exchange"):
            return self._build_offexchange_schema(metadata)
        else:
            return self._build_mbo_schema(metadata)

    def _build_offexchange_schema(self, metadata: dict) -> ExportSchema:
        """Build schema for off-exchange exports."""
        # Validate against contract
        warnings = validate_off_exchange_export_contract(metadata)
        for w in warnings:
            logger.warning(f"Off-exchange contract warning: {w}")

        horizons = tuple(metadata.get("horizons", []))
        return ExportSchema(
            schema_version=metadata.get("schema_version", "1.0"),
            contract_version=metadata.get("contract_version", "off_exchange_1.0"),
            n_features=metadata.get("n_features", OFF_EXCHANGE_FEATURE_COUNT),
            window_size=metadata.get("window_size", 20),
            horizons=horizons,
            bin_size_seconds=metadata.get("bin_size_seconds"),
            feature_names=dict(OFF_EXCHANGE_FEATURE_NAMES),
            categorical_indices=frozenset(OFF_EXCHANGE_CATEGORICAL_INDICES),
        )

    def _build_mbo_schema(self, metadata: dict) -> ExportSchema:
        """Build schema for MBO exports."""
        # Validate against contract
        warnings = validate_export_contract(metadata)
        for w in warnings:
            logger.warning(f"MBO contract warning: {w}")

        # Horizons: try top-level, then nested in labeling
        horizons_raw = metadata.get("horizons")
        if horizons_raw is None:
            labeling = metadata.get("labeling", {})
            horizons_raw = labeling.get("horizons", [])
        horizons = tuple(horizons_raw)

        # Build feature names from FeatureIndex enum
        feature_names = {int(fi): fi.name.lower() for fi in FeatureIndex}

        return ExportSchema(
            schema_version=metadata.get("schema_version", "2.2"),
            contract_version=metadata.get("contract_version", ""),
            n_features=metadata.get("n_features", FEATURE_COUNT),
            window_size=metadata.get("window_size", 100),
            horizons=horizons,
            bin_size_seconds=metadata.get("bin_size_seconds"),
            feature_names=feature_names,
            categorical_indices=frozenset(CATEGORICAL_INDICES),
        )
