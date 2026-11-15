"""Frozen chronological split definitions."""


def build_split_config(n_hours: int, continuity_verified: bool = False) -> dict:
    """Return inclusive split bounds; reject lengths inconsistent with selected protocol."""
    expected = 1008 if continuity_verified else 840
    if n_hours != expected:
        raise ValueError(f"Expected {expected} hourly rows; received {n_hours}")
    fit_end = 671 if continuity_verified else 503
    split = {"mode": "primary" if continuity_verified else "fallback",
            "bounds": "inclusive", "lookback": 168, "horizons": [1, 6, 24],
            "fit": [0, fit_end], "validation": [fit_end + 1, fit_end + 168],
            "holdout": [fit_end + 169, fit_end + 336],
            "final_refit": [0, fit_end + 168], "week11": "excluded"}
    from src.evaluation.origins import split_origins

    split["origins"] = {segment: {str(h): origins.tolist() for h, origins in horizons.items()}
                        for segment, horizons in split_origins(split).items()}
    return split
