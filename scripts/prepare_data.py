"""Run Week 1 ingestion and audit: python -m scripts.prepare_data --step all."""
import argparse
import json
import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.audit import (
    audit_beams, audit_duplicates, audit_missing, audit_range, audit_zeros,
    check_temporal_boundary,
)
from src.data.io import build_manifest, download_dataset
from src.data.panel import Panel, build_panel, load_panel, save_panel
from src.data.schema import KPIS, SEGMENTS, csv_path
from src.data.splits import build_split_config
from src.utils.config import load_config
from src.utils.metadata import collect_run_metadata
from src.utils.seed import set_seed


def write_json(path: Path, value: dict) -> None:
    """Write a UTF-8 JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def plot_audit(panel: Panel, reports: Path) -> None:
    """Save distribution, missing/zero profile, series and phase figures."""
    folder = reports / "figures"
    folder.mkdir(exist_ok=True)
    target = panel.values[:, :, 0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    observed = target[np.isfinite(target)]
    axes[0, 0].hist(observed, bins=60, log=True)
    axes[0, 0].set_title("DLPRB usage distribution (log count)")
    axes[0, 1].hist((target == 0).sum(0) / np.maximum(np.isfinite(target).sum(0), 1), bins=40)
    axes[0, 1].set_title("Per-beam zero ratio (observed denominator)")
    axes[1, 0].plot(panel.time, target[:, :3])
    axes[1, 0].set_title("Sample beams: DLPRB usage")
    for period in (24, 168):
        profile = [float(np.nanmean(target[panel.time % period == phase]))
                   if np.isfinite(target[panel.time % period == phase]).any() else np.nan
                   for phase in range(period)]
        axes[1, 1].plot(range(period), profile, label=f"{period}h phase")
    axes[1, 1].legend()
    axes[1, 1].set_title("Phase index, not calendar time")
    fig.tight_layout()
    fig.savefig(folder / "data_audit.png", dpi=140)
    plt.close(fig)


def main() -> None:
    """Execute download, panel construction, audit, or all stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--step", choices=["download", "panel", "audit", "all"], default="all")
    args = parser.parse_args()
    config = load_config(args.config)
    raw = Path(config["raw_dir"])
    processed = Path(config["processed_dir"])
    reports = Path(config["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
                        logging.FileHandler(reports / "prepare_data.log", encoding="utf-8")])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    set_seed(config["seed"])
    logging.info("Starting step=%s revision=%s", args.step, config["revision"])
    if args.step in ("download", "all"):
        download_dataset(config["repo_id"], config["revision"], raw)
        write_json(reports / "data_manifest.json", build_manifest(raw))
    if args.step in ("panel", "all"):
        manifest = build_manifest(raw)
        write_json(reports / "data_manifest.json", manifest)
        metadata = {"dataset_revision": manifest["revision"], "license": manifest["license"]}
        panels = {segment: build_panel({k: raw / csv_path(k, segment) for k in KPIS},
                                       {**metadata, "segment": segment}) for segment in SEGMENTS}
        boundary = check_temporal_boundary(panels["train"], panels["week6"],
                    verified=config["continuity_verified"], evidence=config["continuity_evidence"])
        panel = panels["train"]
        if boundary["continuity_verified"]:
            panel = Panel(np.concatenate([panel.values, panels["week6"].values]),
                          np.arange(1008), panel.beam_ids,
                          {**metadata, "segment": "train+week6", "boundary": boundary})
        if not np.array_equal(panel.time, np.arange(len(panel.time))):
            raise ValueError("Main panel requires contiguous hourly indices starting at zero")
        split = build_split_config(len(panel.time), boundary["continuity_verified"])
        save_panel(panel, processed / "panel.npz")
        save_panel(panels["week6"], processed / "week6.npz")
        save_panel(panels["week11"], processed / "week11.npz")
        write_json(processed / "split.json", split)
        write_json(reports / "temporal_boundary.json", boundary)
        (reports / "split_decision.md").write_text(
            f"# Temporal split decision\n\nMode: **{split['mode']}**.\n\n"
            f"{boundary['reason']}.\n\nEvidence: {boundary['evidence'] or 'Not approved'}.\n\n"
            "Source: dataset_card.md at pinned revision. Source indices reset for week 6; "
            "no calendar timestamps are supplied. No profile correlation is used as evidence.\n\n"
            f"Inclusive bounds:\n```json\n{json.dumps(split, indent=2)}\n```\n"
            "Week 11 is separate. Model selection uses validation only.\n", encoding="utf-8")
    if args.step in ("audit", "all"):
        panel = load_panel(processed / "panel.npz")
        audit = {"shape": list(panel.values.shape), "missing": audit_missing(panel),
                 "zeros": audit_zeros(panel), "range": audit_range(panel),
                 "duplicates": audit_duplicates(panel), "beams": audit_beams(panel)}
        write_json(reports / "data_audit.json", audit)
        (reports / "data_audit.md").write_text(
            "# Data audit\n\nDescriptive audit of the main panel; no imputation or filtering.\n\n"
            "NaN is preserved; zero is a valid observation. DLPRB unit remains unverified; "
            "use **DLPRB usage**. Active means value > 0 among observed values.\n\n"
            f"```json\n{json.dumps({k: v for k, v in audit.items() if k != 'missing'}, indent=2)}\n```\n"
            f"\nMissing ratios per KPI: {audit['missing']['per_kpi']}\n\n"
            "Full per-beam missing ratios: data_audit.json.\n\n"
            "![Audit figures](figures/data_audit.png)\n", encoding="utf-8")
        plot_audit(panel, reports)
    write_json(reports / "run_metadata.json", collect_run_metadata(config))
    logging.info("Completed step=%s", args.step)


if __name__ == "__main__":
    main()
