"""Finalize validation-only champion config; Git tagging is a separate explicit step."""

import json
from pathlib import Path

import pandas as pd
import yaml

from src.data.io import sha256
from src.inference.freeze import config_hash, source_hash
from src.utils.config import load_config


def main() -> None:
    """Freeze the Week 5 family and validation iteration/epoch choices without loading holdout."""
    candidate = load_config("configs/champion_candidate.yaml")
    gate = json.loads(Path("reports/model_quality_gate.json").read_text())
    if (
        gate["status"] != "PASS"
        or sha256(Path("reports/model_quality_gate.json")) != candidate["quality_gate_sha256"]
    ):
        raise RuntimeError("Current quality gate does not match approved candidate")
    if not Path("reports/risk_validation.csv").exists():
        raise RuntimeError("Validation risk report required before freeze")
    data = load_config("configs/data.yaml")
    split = json.loads(Path(data["processed_dir"], "split.json").read_text())
    lgb = load_config("configs/lightgbm.yaml")
    gru = load_config("configs/gru.yaml")
    risk = load_config("configs/risk.yaml")
    family = candidate["family"]
    baseline = pd.read_csv("reports/baseline_benchmark_val.csv")
    baseline = baseline[baseline.load_tier == "All"]
    order = ["persistence", "seasonal_naive_24", "seasonal_naive_168"]
    comparators = {
        h: min(
            order,
            key=lambda n: float(
                baseline[(baseline.horizon == h) & (baseline.model == n)].mae.iloc[0]
            ),
        )
        for h in (1, 6, 24)
    }
    routes = {}
    if family.startswith("lightgbm_"):
        name = family.removeprefix("lightgbm_")
        for h in (1, 6, 24):
            meta = json.loads(Path(lgb["artifact_dir"], name, f"h{h}", "meta.json").read_text())
            routes[h] = {
                "family": "lightgbm",
                "iterations": meta["best_iteration"],
                "parameters": lgb["configs"][name],
                "seed": lgb["seed"],
                "feature_schema": meta["feature_schema"],
            }
    elif family in order:
        routes = {h: {"family": "baseline", "definition": family} for h in (1, 6, 24)}
    elif family.startswith("gru_"):
        variant = family[-1]
        seeds = {
            seed: json.loads(
                Path(gru["artifact_dir"], variant, f"seed{seed}", "meta.json").read_text()
            )["best_epoch"]
            + 1
            for seed in gru["seeds"]
        }
        routes = {
            h: {
                "family": "gru",
                "variant": variant,
                "seeds_epochs": seeds,
                "ensemble": "arithmetic_mean",
                "phase_features": 4,
                "config": gru,
            }
            for h in (1, 6, 24)
        }
    else:
        raise ValueError("Unsupported frozen family")
    config = {
        "frozen": True,
        "holdout_allowed": True,
        "freeze_tag": "pre-holdout",
        "family": family,
        "dataset_revision": data["revision"],
        "panel_sha256": sha256(Path(data["processed_dir"]) / "panel.npz"),
        "split_mode": split["mode"],
        "lookback": 168,
        "horizons": [1, 6, 24],
        "final_fit": split["final_refit"],
        "holdout": split["holdout"],
        "training_stride": lgb["training_stride"],
        "num_threads": lgb["num_threads"],
        "routes": routes,
        "baseline_comparators": comparators,
        "risk_policy": risk,
        "quantile_models": None,
        "bootstrap": {"repetitions": 1000, "seed": 17, "confidence": 0.95},
        "selection_evidence": {
            "candidate_sha256": source_hash(Path("configs/champion_candidate.yaml")),
            "quality_gate_sha256": candidate["quality_gate_sha256"],
            "risk_validation_sha256": source_hash(Path("reports/risk_validation.csv")),
        },
    }
    Path("configs/champion.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n"
    )
    text = [
        "# Frozen champion selection",
        "",
        f"Family: **{family}**. Config hash: `{config_hash(config)}`.",
        "Applied the predeclared Week 5 normalized-MAE/paired-CI rule; baseline fallback was eligible.",
        "Risk policy was fixed from fit-segment units/sampling before validation risk metrics; it did not change the point champion.",
        f"Final fit bounds: {split['final_refit']}; holdout bounds: {split['holdout']}.",
        "Refit rounds/epochs are copied from validation artifacts. No holdout early stopping or tuning.",
        "Create the annotated pre-holdout tag before any final evaluation, then generate freeze_manifest.json.",
        "",
        "```yaml",
        yaml.safe_dump(routes, sort_keys=False).strip(),
        "```",
    ]
    Path("reports/champion_selection.md").write_text("\n".join(text), encoding="utf-8")
    print("Frozen config prepared:", family, "; config SHA:", config_hash(config))


if __name__ == "__main__":
    main()
