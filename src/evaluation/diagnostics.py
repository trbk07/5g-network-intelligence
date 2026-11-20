"""Deterministic validation error atlas; selection rules are fixed, not handpicked."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def error_atlas(
    panel,
    records: dict,
    masks: dict,
    tiers: np.ndarray,
    reference: str,
    comparison: list[str],
    reports: Path,
    fit_end: int,
) -> None:
    """Write top-5 errors, five representatives per tier, and two peaks plus one transition."""
    figures = reports / "figures"
    figures.mkdir(exist_ok=True)
    per_beam = []
    for h in (1, 6, 24):
        record = records[reference][h]
        mask = masks[h]
        count = mask.sum(0)
        per_beam.append(
            np.divide(
                np.where(mask, np.abs(record.predicted - record.actual), 0).sum(0),
                count,
                out=np.full(len(count), np.nan),
                where=count > 0,
            )
        )
    errors = np.nanmean(np.stack(per_beam), axis=0)
    valid = np.flatnonzero(np.isfinite(errors))
    top = valid[np.argsort(-errors[valid], kind="stable")[:5]]
    rows = [
        {
            "selection": "top5_reference_mean_horizon_mae",
            "beam_id": panel.beam_ids[i],
            "load_tier": tiers[i],
            "mae": errors[i],
        }
        for i in top
    ]
    # Stable, equally spaced ranks inside each fit-only load tier; no manual examples.
    fit = panel.values[: fit_end + 1, :, 0]
    count = np.isfinite(fit).sum(0)
    means = np.divide(
        np.nansum(fit, axis=0), count, out=np.full(len(count), np.nan), where=count > 0
    )
    for tier in ("Low", "Mid", "High"):
        indices = np.flatnonzero((tiers == tier) & np.isfinite(means))
        indices = indices[np.argsort(means[indices], kind="stable")]
        if len(indices):
            chosen = indices[
                np.unique(np.linspace(0, len(indices) - 1, min(5, len(indices))).astype(int))
            ]
            rows.extend(
                {
                    "selection": "five_evenly_spaced_fit_load_ranks",
                    "beam_id": panel.beam_ids[i],
                    "load_tier": tier,
                    "mae": errors[i],
                }
                for i in chosen
            )
    selections = pd.DataFrame(rows)
    selections.to_csv(reports / "error_atlas_beams.csv", index=False)
    h = 24
    record = records[reference][h]
    actual, prediction = record.actual, record.predicted
    under = actual - prediction
    finite = masks[h]
    cutoff = np.quantile(actual[finite], 0.9)
    peak = finite & (actual >= cutoff) & (under > 0)
    previous = panel.values[record.origins + h - 1, :, 0]
    transition = finite & (previous == 0) & (actual > 0) & (under > 0)
    cases, used = [], set()
    for kind, eligible in (
        ("miss_peak", peak),
        ("miss_peak", peak),
        ("zero_to_active", transition),
    ):
        positions = np.flatnonzero(eligible)
        positions = positions[np.argsort(-under.reshape(-1)[positions], kind="stable")]
        selected = next((int(p) for p in positions if int(p) not in used), None)
        if selected is None:
            continue
        used.add(selected)
        oi, bi = np.unravel_index(selected, actual.shape)
        cases.append(
            {
                "case": kind,
                "beam_id": panel.beam_ids[bi],
                "beam_index": int(bi),
                "origin_row": int(oi),
                "origin": int(record.origins[oi]),
                "target": int(record.origins[oi] + h),
                "actual": actual[oi, bi],
                "prediction": prediction[oi, bi],
                "underprediction": under[oi, bi],
            }
        )
    case_frame = pd.DataFrame(cases)
    case_frame.to_csv(reports / "error_atlas_cases.csv", index=False)
    if cases:
        fig, axes = plt.subplots(len(cases), 1, figsize=(12, 3.4 * len(cases)), squeeze=False)
        for ax, case in zip(axes[:, 0], cases):
            bi, oi = case["beam_index"], case["origin_row"]
            window = slice(max(0, oi - 24), min(len(record.origins), oi + 25))
            x = record.origins[window] + h
            ax.plot(
                x,
                np.where(finite[window, bi], actual[window, bi], np.nan),
                label="Actual",
                color="black",
                linewidth=2,
            )
            for name in comparison:
                ax.plot(
                    x,
                    np.where(finite[window, bi], records[name][h].predicted[window, bi], np.nan),
                    label=name,
                )
            ax.axvline(case["target"], color="gray", linestyle="--")
            ax.set_title(f"{case['case']} | beam {case['beam_id']} | +24h")
            ax.set_xlabel("Relative target hour")
            ax.set_ylabel("DLPRB usage")
            ax.legend()
        fig.tight_layout()
        fig.savefig(figures / "actual_vs_forecast.png", dpi=140)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    for index, name in enumerate(comparison):
        values = [
            np.abs(records[name][hh].predicted - records[name][hh].actual)[masks[hh]].mean()
            for hh in (1, 6, 24)
        ]
        ax.bar(np.arange(3) + (index - 1) * 0.24, values, width=0.24, label=name)
    ax.set_xticks(np.arange(3), ["+1h", "+6h", "+24h"])
    ax.set_ylabel("Common-mask MAE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "mae_by_horizon.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, hh in zip(axes, (1, 6, 24)):
        for index, name in enumerate(comparison):
            values = []
            for tier in ("Low", "Mid", "High"):
                mask = masks[hh] & (tiers == tier)[None, :]
                values.append(
                    np.abs(records[name][hh].predicted - records[name][hh].actual)[mask].mean()
                    if mask.any()
                    else np.nan
                )
            ax.bar(np.arange(3) + (index - 1) * 0.24, values, width=0.24, label=name)
        ax.set_xticks(np.arange(3), ["Low", "Mid", "High"])
        ax.set_title(f"+{hh}h")
        ax.set_ylabel("Common-mask MAE")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(figures / "mae_by_load_tier.png", dpi=140)
    plt.close(fig)
    lines = [
        "# Validation error atlas",
        "",
        f"Reference family: {reference}.",
        "Selection rules: top 5 beams by equal-weight mean per-horizon reference MAE;",
        "five equally spaced fit-load ranks per Low/Mid/High tier; two largest distinct",
        "+24h underpredictions among actual >= validation P90, plus one largest zero-to-active miss.",
        "Ties use original numeric beam/time order. No manual/cherry-picked cases.",
        "Cases are retrospective error diagnostics, not incident/congestion labels.",
        "",
    ]
    for title, frame in (("Beam selections", selections), ("Peak/transition cases", case_frame)):
        lines += [f"## {title}", "", "```csv", frame.to_csv(index=False).strip(), "```", ""]
    lines += [
        "![Actual and forecasts](figures/actual_vs_forecast.png)",
        "![Horizon MAE](figures/mae_by_horizon.png)",
        "![Load-tier MAE](figures/mae_by_load_tier.png)",
    ]
    (reports / "error_atlas.md").write_text("\n".join(lines), encoding="utf-8")
