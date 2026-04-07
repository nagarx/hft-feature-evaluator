"""Screening layer: IC and dCor+MI independence screening with BH correction."""

import numpy as np


def _test_seed(base_seed: int, feature_idx: int, horizon_idx: int,
               lag_idx: int = 0) -> int:
    """Deterministic per-test seed for permutation tests.

    Order-independent: changing iteration order of features/horizons
    does not change the seed for any (feature, horizon, lag) triple.
    Does not consume RNG state (no RandomState object needed).

    Args:
        base_seed: Global seed from config (e.g. 42).
        feature_idx: Raw feature index (e.g. 0-97 for MBO).
        horizon_idx: Horizon position in the horizons list (0-based).
        lag_idx: Lag index for TE (0 for non-TE tests).

    Returns:
        Unique deterministic seed in [0, 2^31 - 2].
    """
    return (base_seed + feature_idx * 10007
            + horizon_idx * 31 + lag_idx * 7) % (2**31 - 1)


def bh_adjusted_pvalues(p_values: np.ndarray) -> np.ndarray:
    """Compute Benjamini-Hochberg adjusted p-values.

    Standard step-up procedure (Benjamini & Hochberg 1995, JRSS-B 57(1)):
        1. Sort p-values ascending: p_(1) <= ... <= p_(m)
        2. Compute raw adjusted: p_adj_(i) = p_(i) * m / i
        3. Enforce monotonicity via cumulative min from right to left
        4. Cap at 1.0

    Args:
        p_values: Raw p-values (1D array).

    Returns:
        BH-adjusted p-values in the same order as input. Capped at 1.0.
    """
    m = len(p_values)
    if m == 0:
        return np.array([], dtype=np.float64)

    order = np.argsort(p_values)
    sorted_p = p_values[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)

    adjusted = sorted_p * m / ranks

    # Step-down monotonicity: adjusted[i] <= adjusted[i+1]
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    adjusted = np.minimum(adjusted, 1.0)

    # Unsort to original order
    result = np.empty(m, dtype=np.float64)
    result[order] = adjusted
    return result
