"""
Holdout reservation: split evaluation dates into main + holdout subsets.

The holdout is the chronologically LAST holdout_days dates. This prevents
lookahead: evaluation uses earlier dates, holdout validates on later dates.

Reference: CODEBASE.md Section 3.3, Framework Section 6.5
"""


def split_holdout(
    dates: list[str],
    holdout_days: int,
) -> tuple[list[str], list[str]]:
    """Reserve the LAST holdout_days dates as meta-holdout.

    Args:
        dates: Sorted date strings (YYYY-MM-DD or YYYYMMDD).
        holdout_days: Number of days to hold out. 0 = no holdout.

    Returns:
        (evaluation_dates, holdout_dates)

    Raises:
        ValueError: If holdout_days >= len(dates), dates is empty, or
                    holdout_days < 0.
    """
    if not dates:
        raise ValueError("dates must be non-empty")
    if holdout_days < 0:
        raise ValueError(f"holdout_days must be >= 0, got {holdout_days}")
    if holdout_days >= len(dates):
        raise ValueError(
            f"holdout_days ({holdout_days}) must be < number of dates "
            f"({len(dates)}). At least 1 day must remain for evaluation."
        )
    if holdout_days == 0:
        return list(dates), []
    return list(dates[:-holdout_days]), list(dates[-holdout_days:])
