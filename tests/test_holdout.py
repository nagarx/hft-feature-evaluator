"""Tests for split_holdout: boundary conditions, ordering, errors."""

import pytest
from hft_evaluator.data.holdout import split_holdout


class TestSplitHoldout:
    def test_basic_split(self):
        dates = [f"2025-01-{d:02d}" for d in range(1, 11)]
        eval_d, hold_d = split_holdout(dates, 3)
        assert len(eval_d) == 7
        assert len(hold_d) == 3

    def test_holdout_zero(self):
        dates = ["2025-01-01", "2025-01-02", "2025-01-03"]
        eval_d, hold_d = split_holdout(dates, 0)
        assert eval_d == dates
        assert hold_d == []

    def test_holdout_dates_are_last(self):
        dates = [f"2025-01-{d:02d}" for d in range(1, 11)]
        _, hold_d = split_holdout(dates, 3)
        assert hold_d == ["2025-01-08", "2025-01-09", "2025-01-10"]

    def test_eval_dates_are_first(self):
        dates = [f"2025-01-{d:02d}" for d in range(1, 6)]
        eval_d, _ = split_holdout(dates, 2)
        assert eval_d == ["2025-01-01", "2025-01-02", "2025-01-03"]

    def test_no_overlap(self):
        dates = [f"2025-01-{d:02d}" for d in range(1, 11)]
        eval_d, hold_d = split_holdout(dates, 3)
        assert set(eval_d) & set(hold_d) == set()

    def test_preserves_order(self):
        dates = [f"2025-01-{d:02d}" for d in range(1, 11)]
        eval_d, hold_d = split_holdout(dates, 3)
        assert eval_d == sorted(eval_d)
        assert hold_d == sorted(hold_d)

    def test_union_equals_input(self):
        dates = [f"2025-01-{d:02d}" for d in range(1, 11)]
        eval_d, hold_d = split_holdout(dates, 3)
        assert eval_d + hold_d == dates

    def test_empty_dates_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            split_holdout([], 0)

    def test_holdout_exceeds_length(self):
        dates = ["2025-01-01", "2025-01-02"]
        with pytest.raises(ValueError, match="must be <"):
            split_holdout(dates, 2)

    def test_holdout_equals_length(self):
        dates = ["2025-01-01", "2025-01-02", "2025-01-03"]
        with pytest.raises(ValueError, match="must be <"):
            split_holdout(dates, 3)

    def test_holdout_negative(self):
        with pytest.raises(ValueError, match=">= 0"):
            split_holdout(["2025-01-01"], -1)

    def test_single_date_no_holdout(self):
        eval_d, hold_d = split_holdout(["2025-01-01"], 0)
        assert eval_d == ["2025-01-01"]
        assert hold_d == []
