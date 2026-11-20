"""Validation-only family selection with paired-CI preference for cheaper models."""

import numpy as np

from src.evaluation.bootstrap import paired_multihorizon_ci


def choose_candidate(
    families: dict,
    clusters: np.ndarray,
    order: list[str],
    repetitions: int = 1000,
    seed: int = 17,
    confidence: float = 0.95,
) -> tuple[str, list[dict]]:
    """Rank complete three-horizon families; fall back to cheaper if CI cannot distinguish."""
    horizons = (1, 6, 24)
    if not families or any(set(records) != set(horizons) for records in families.values()):
        raise ValueError("Every candidate family must provide exactly horizons 1,6,24")
    reference = next(iter(families.values()))
    for records in families.values():
        for h in horizons:
            if not np.array_equal(
                records[h][0], reference[h][0], equal_nan=True
            ) or not np.array_equal(records[h][2], reference[h][2]):
                raise ValueError("Candidate families must use identical actuals and common masks")
    baseline_names = [
        n for n in families if n in ("persistence", "seasonal_naive_24", "seasonal_naive_168")
    ]
    if not baseline_names:
        raise ValueError("Baseline fallback is mandatory")
    scales = [
        min(
            float(np.abs(families[n][h][1] - families[n][h][0])[families[n][h][2]].mean())
            for n in baseline_names
        )
        for h in horizons
    ]
    if any(s <= 0 or not np.isfinite(s) for s in scales):
        raise ValueError("Selection requires positive comparable baseline MAE")
    scores = {
        n: float(
            np.mean(
                [
                    np.abs(record[1] - record[0])[record[2]].mean() / scales[i]
                    for i, (_, record) in enumerate(sorted(records.items()))
                ]
            )
        )
        for n, records in families.items()
    }
    if any(n not in order for n in families):
        raise ValueError("All families require a predeclared complexity order")
    best = min(scores, key=lambda n: (scores[n], order.index(n)))
    rows = []
    for name in sorted(families, key=order.index):
        pairs = [
            (families[name][h][0], families[name][h][1], families[best][h][1], families[name][h][2])
            for h in horizons
        ]
        ci = paired_multihorizon_ci(pairs, clusters, scales, repetitions, seed, confidence)
        eligible = name == best or (
            order.index(name) < order.index(best) and ci["status"] == "ok" and ci["lower"] <= 0
        )
        rows.append(
            {
                "family": name,
                "normalized_mae": scores[name],
                "nominal_best": best,
                "ci_lower_vs_best": ci["lower"],
                "ci_upper_vs_best": ci["upper"],
                "ci_status": ci["status"],
                "eligible_cheaper_fallback": bool(eligible),
                "complexity_rank": order.index(name),
            }
        )
    selected = next(row["family"] for row in rows if row["eligible_cheaper_fallback"])
    for row in rows:
        row["recommended_candidate"] = row["family"] == selected
    return selected, rows
