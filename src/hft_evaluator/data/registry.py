"""
Feature group registry: maps feature indices to groups, evaluation properties.

Resolves all information from hft-contracts. No hardcoded indices.

Reference: CODEBASE.md Section 3.2
"""

from dataclasses import dataclass

from hft_contracts import (
    # Off-exchange
    OffExchangeFeatureIndex,
    OFF_EXCHANGE_FEATURE_NAMES,
    OFF_EXCHANGE_CATEGORICAL_INDICES,
    OFF_EXCHANGE_UNSIGNED_FEATURES,
    OFF_EXCHANGE_SAFETY_GATES,
    OFF_EXCHANGE_SIGNED_FLOW,
    OFF_EXCHANGE_VENUE_METRICS,
    OFF_EXCHANGE_RETAIL_METRICS,
    OFF_EXCHANGE_BBO_DYNAMICS,
    OFF_EXCHANGE_VPIN,
    OFF_EXCHANGE_TRADE_SIZE,
    OFF_EXCHANGE_CROSS_VENUE,
    OFF_EXCHANGE_ACTIVITY,
    OFF_EXCHANGE_SAFETY_GATES_SLICE,
    OFF_EXCHANGE_CONTEXT,
    # MBO
    FeatureIndex,
    CATEGORICAL_INDICES,
    UNSIGNED_FEATURES,
)

from hft_evaluator.data.loader import ExportSchema


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureInfo:
    """Properties of a single feature for evaluation."""

    index: int
    name: str
    group: str
    evaluable: bool   # False if categorical or safety gate
    signed: bool      # True if directional (+ = bullish)


# ---------------------------------------------------------------------------
# Group mappings from hft-contracts slices
# ---------------------------------------------------------------------------

_OFF_EXCHANGE_GROUPS: dict[str, slice] = {
    "signed_flow": OFF_EXCHANGE_SIGNED_FLOW,
    "venue_metrics": OFF_EXCHANGE_VENUE_METRICS,
    "retail_metrics": OFF_EXCHANGE_RETAIL_METRICS,
    "bbo_dynamics": OFF_EXCHANGE_BBO_DYNAMICS,
    "vpin": OFF_EXCHANGE_VPIN,
    "trade_size": OFF_EXCHANGE_TRADE_SIZE,
    "cross_venue": OFF_EXCHANGE_CROSS_VENUE,
    "activity": OFF_EXCHANGE_ACTIVITY,
    "safety_gates": OFF_EXCHANGE_SAFETY_GATES_SLICE,
    "context": OFF_EXCHANGE_CONTEXT,
}

# Conditioning variables for regime-conditional IC
_OFF_EXCHANGE_CONDITIONING = {
    "spread_bps": int(OffExchangeFeatureIndex.SPREAD_BPS),
    "session_progress": int(OffExchangeFeatureIndex.SESSION_PROGRESS),
    "bin_trade_count": int(OffExchangeFeatureIndex.BIN_TRADE_COUNT),
}

_MBO_CONDITIONING = {
    "spread_bps": int(FeatureIndex.SPREAD_BPS),
    "time_regime": int(FeatureIndex.TIME_REGIME),
    "active_order_count": int(FeatureIndex.ACTIVE_ORDER_COUNT),
}


# ---------------------------------------------------------------------------
# FeatureRegistry
# ---------------------------------------------------------------------------


class FeatureRegistry:
    """Feature group registry. Constructed from ExportSchema.

    Resolves all feature metadata from hft-contracts.
    No hardcoded indices — everything from the contract plane.
    """

    def __init__(self, schema: ExportSchema):
        self._schema = schema
        self._features: dict[int, FeatureInfo] = {}

        is_offex = schema.contract_version.startswith("off_exchange")

        if is_offex:
            self._build_offexchange(schema)
        else:
            self._build_mbo(schema)

    def get(self, index: int) -> FeatureInfo:
        """Get FeatureInfo for a given index.

        Raises:
            KeyError: If index is out of range for this schema.
        """
        return self._features[index]

    def evaluable_indices(self) -> list[int]:
        """Indices eligible for evaluation (not categorical/safety)."""
        return [i for i, f in sorted(self._features.items()) if f.evaluable]

    def group_indices(self, group_name: str) -> list[int]:
        """Indices belonging to a named group."""
        return [
            i for i, f in sorted(self._features.items())
            if f.group == group_name
        ]

    def group_names(self) -> list[str]:
        """All group names in index order (deduped)."""
        seen = set()
        result = []
        for _, f in sorted(self._features.items()):
            if f.group not in seen:
                seen.add(f.group)
                result.append(f.group)
        return result

    def conditioning_indices(self) -> dict[str, int]:
        """Default conditioning variable indices for regime IC.

        Off-exchange: {spread_bps: 12, session_progress: 31, bin_trade_count: 27}
        MBO: {spread_bps: 42, time_regime: 93, active_order_count: 83}
        """
        if self._schema.contract_version.startswith("off_exchange"):
            return dict(_OFF_EXCHANGE_CONDITIONING)
        else:
            return dict(_MBO_CONDITIONING)

    # -------------------------------------------------------------------
    # Internal builders
    # -------------------------------------------------------------------

    def _build_offexchange(self, schema: ExportSchema) -> None:
        """Build registry for off-exchange features."""
        unsigned = OFF_EXCHANGE_UNSIGNED_FEATURES
        categorical = schema.categorical_indices

        # Build index → group mapping from slices
        index_to_group: dict[int, str] = {}
        for group_name, slc in _OFF_EXCHANGE_GROUPS.items():
            for i in range(slc.start, slc.stop):
                index_to_group[i] = group_name

        for i in range(schema.n_features):
            name = schema.feature_names.get(i, f"feature_{i}")
            group = index_to_group.get(i, "unknown")
            self._features[i] = FeatureInfo(
                index=i,
                name=name,
                group=group,
                evaluable=(i not in categorical),
                signed=(i not in unsigned),
            )

    def _build_mbo(self, schema: ExportSchema) -> None:
        """Build registry for MBO features."""
        unsigned = UNSIGNED_FEATURES
        categorical = schema.categorical_indices

        # MBO group mapping derived from FeatureIndex enum boundaries.
        # Ranges are [start, stop) matching the documented layout in CLAUDE.md.
        _MBO_GROUP_RANGES: dict[str, range] = {
            "ask_prices": range(
                int(FeatureIndex.ASK_PRICE_L0),
                int(FeatureIndex.ASK_PRICE_L9) + 1,
            ),
            "ask_sizes": range(
                int(FeatureIndex.ASK_SIZE_L0),
                int(FeatureIndex.ASK_SIZE_L9) + 1,
            ),
            "bid_prices": range(
                int(FeatureIndex.BID_PRICE_L0),
                int(FeatureIndex.BID_PRICE_L9) + 1,
            ),
            "bid_sizes": range(
                int(FeatureIndex.BID_SIZE_L0),
                int(FeatureIndex.BID_SIZE_L9) + 1,
            ),
            "derived": range(
                int(FeatureIndex.MID_PRICE),
                int(FeatureIndex.PRICE_IMPACT) + 1,
            ),
            "mbo_order_flow": range(
                int(FeatureIndex.ADD_RATE_BID),
                int(FeatureIndex.FLOW_REGIME_INDICATOR) + 1,
            ),
            "mbo_size_dist": range(
                int(FeatureIndex.SIZE_P25),
                int(FeatureIndex.SIZE_CONCENTRATION) + 1,
            ),
            "mbo_queue_depth": range(
                int(FeatureIndex.AVG_QUEUE_POSITION),
                int(FeatureIndex.DEPTH_TICKS_ASK) + 1,
            ),
            "mbo_institutional": range(
                int(FeatureIndex.LARGE_ORDER_FREQUENCY),
                int(FeatureIndex.ICEBERG_PROXY) + 1,
            ),
            "mbo_core": range(
                int(FeatureIndex.AVG_ORDER_AGE),
                int(FeatureIndex.ACTIVE_ORDER_COUNT) + 1,
            ),
            "trading_signals": range(
                int(FeatureIndex.TRUE_OFI),
                int(FeatureIndex.DEPTH_ASYMMETRY) + 1,
            ),
            "control_metadata": range(
                int(FeatureIndex.BOOK_VALID),
                int(FeatureIndex.SCHEMA_VERSION) + 1,
            ),
        }

        # Build index → group lookup
        index_to_group: dict[int, str] = {}
        for group_name, idx_range in _MBO_GROUP_RANGES.items():
            for i in idx_range:
                index_to_group[i] = group_name

        for i in range(schema.n_features):
            name = schema.feature_names.get(i, f"feature_{i}")
            group = index_to_group.get(i, "experimental")

            self._features[i] = FeatureInfo(
                index=i,
                name=name,
                group=group,
                evaluable=(i not in categorical),
                signed=(i not in unsigned),
            )
