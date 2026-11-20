"""Week 5 common-mask comparison, paired CIs, ablation and champion-ready candidate."""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.quality_audit import audit
from src.data.audit import check_mapping
from src.data.io import sha256
from src.data.schema import parse_beam_id
from src.evaluation.bootstrap import paired_cluster_ci
from src.evaluation.common_mask import comparable_mask, coverage_report
from src.evaluation.diagnostics import error_atlas
from src.evaluation.evaluator import ForecastRecord, to_benchmark_rows
from src.evaluation.metrics import assign_load_tiers
from src.evaluation.validation import validation_inputs
from src.inference.registry import choose_candidate
from src.utils.config import load_config


def compare(diagnostics: bool = True) -> None:
    """Require current quality PASS; compare validation arrays only and never freeze a champion."""
    gate = audit()
    if gate["status"] != "PASS":
        raise RuntimeError("Quality gate FAIL: champion selection and comparison blocked")
    config = load_config("configs/data.yaml")
    rules = load_config("configs/quality.yaml")
    panel, split = validation_inputs(config)
    reports = Path(config["reports_dir"])
    records = {}
    canonical_actuals = {}
    source_hashes = {}
    for filename in ("baselines_val.npz", "lightgbm_val.npz", "gru_val.npz"):
        path = reports / "forecasts" / filename
        source_hashes[filename] = sha256(path)
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
            if meta["dataset_revision"] != config["revision"]:
                raise ValueError("Forecast archive dataset revision differs")
            if (
                filename == "lightgbm_val.npz"
                and meta["config"]["model_config"]["artifact_dir"]
                != load_config("configs/lightgbm.yaml")["artifact_dir"]
            ):
                raise ValueError("Legacy LightGBM forecast archive rejected")
            if (
                filename == "gru_val.npz"
                and meta["config"]["gru"]["artifact_dir"]
                != load_config("configs/gru.yaml")["artifact_dir"]
            ):
                raise ValueError("Legacy GRU forecast archive rejected")
            for entry in meta["records"]:
                p = entry["prefix"]
                h = entry["horizon"]
                name = entry["model"]
                record = ForecastRecord(
                    name,
                    h,
                    archive[p + "_origins"],
                    tuple(archive[p + "_beam_ids"].tolist()),
                    archive[p + "_actual"],
                    archive[p + "_predicted"],
                )
                expected = np.arange(split["fit"][1], split["validation"][1] - h + 1)
                if record.beam_ids != panel.beam_ids or not np.array_equal(
                    record.origins, expected
                ):
                    raise ValueError("Origins or beam axis mismatch across candidates")
                # CSV parser versions can differ by a few float64 ulps across Windows/Linux.
                # Keep cloud actuals bit-identical across candidates; only the local diagnostic
                # panel comparison permits 1e-12 round-off, never different rows or beam order.
                np.testing.assert_allclose(
                    record.actual,
                    panel.values[record.origins + h, :, 0],
                    rtol=1e-12,
                    atol=1e-12,
                    equal_nan=True,
                )
                if h in canonical_actuals:
                    np.testing.assert_array_equal(record.actual, canonical_actuals[h])
                else:
                    canonical_actuals[h] = record.actual
                records.setdefault(name, {})[h] = record
    for variant in ("A", "B"):
        names = [f"gru_{variant}_seed{seed}" for seed in (17, 42)]
        for h in (1, 6, 24):
            a, b = [records[n][h] for n in names]
            records.setdefault(f"gru_{variant}", {})[h] = ForecastRecord(
                f"gru_{variant}",
                h,
                a.origins,
                a.beam_ids,
                a.actual,
                (a.predicted + b.predicted) / 2,
            )
    masks = {
        h: comparable_mask(
            next(iter(records.values()))[h].actual,
            {n: rs[h].predicted for n, rs in records.items()},
        )
        for h in (1, 6, 24)
    }
    mapping = check_mapping(panel)["complete_expected_mapping"]
    clusters = (
        np.array([parse_beam_id(b)[0] for b in panel.beam_ids])
        if mapping
        else np.arange(len(panel.beam_ids))
    )
    unit = "verified_BS" if mapping else "beam_fallback_dependency_limitation"
    rows = []
    tiers = assign_load_tiers(panel.values[: split["fit"][1] + 1, :, 0])
    for name, horizons in records.items():
        for h, record in horizons.items():
            mask = masks[h]
            masked = ForecastRecord(
                name,
                h,
                record.origins,
                record.beam_ids,
                np.where(mask, record.actual, np.nan),
                np.where(mask, record.predicted, np.nan),
            )
            for row in to_benchmark_rows(masked, tiers):
                beams = (
                    np.ones(len(tiers), dtype=bool)
                    if row["load_tier"] == "All"
                    else tiers == row["load_tier"]
                )
                row.update(
                    coverage_report(
                        record.actual[:, beams], record.predicted[:, beams], mask[:, beams]
                    )
                )
                rows.append(row)
    pd.DataFrame(rows).to_csv(reports / "model_comparison_val.csv", index=False)
    cis = []
    for h in (1, 6, 24):
        for a, b in itertools.combinations(records, 2):
            ci = paired_cluster_ci(
                records[a][h].actual,
                records[a][h].predicted,
                records[b][h].predicted,
                clusters,
                masks[h],
                rules["bootstrap_repetitions"],
                rules["bootstrap_seed"],
                rules["confidence"],
            )
            cis.append({"horizon": h, "model_a": a, "model_b": b, "cluster_unit": unit, **ci})
    pd.DataFrame(cis).to_csv(reports / "paired_model_comparison_val.csv", index=False)
    family_names = rules["family_order"]
    families = {
        n: {h: (records[n][h].actual, records[n][h].predicted, masks[h]) for h in (1, 6, 24)}
        for n in family_names
    }
    selected, table = choose_candidate(
        families,
        clusters,
        family_names,
        rules["bootstrap_repetitions"],
        rules["bootstrap_seed"],
        rules["confidence"],
    )
    pd.DataFrame(table).to_csv(reports / "champion_candidate_table.csv", index=False)
    ablation = []
    for variant in ("A", "B"):
        for h in (1, 6, 24):
            values = [
                np.abs(
                    records[f"gru_{variant}_seed{s}"][h].predicted
                    - records[f"gru_{variant}_seed{s}"][h].actual
                )[masks[h]].mean()
                for s in (17, 42)
            ]
            ablation.append(
                {
                    "variant": variant,
                    "horizon": h,
                    "seeds": 2,
                    "mae_mean": np.mean(values),
                    "mae_sd": np.std(values, ddof=1),
                    "n_pairs": int(masks[h].sum()),
                    "comparison": "same origins, beam sample, seeds and common finite target mask",
                }
            )
    pd.DataFrame(ablation).to_csv(reports / "ablation_study.csv", index=False)
    references = {}
    lgb_config = load_config("configs/lightgbm.yaml")
    gru_config = load_config("configs/gru.yaml")
    for h in (1, 6, 24):
        if selected.startswith("lightgbm_"):
            references[h] = [
                str(Path(lgb_config["artifact_dir"]) / selected.removeprefix("lightgbm_") / f"h{h}")
            ]
        elif selected.startswith("gru_"):
            references[h] = [
                str(Path(gru_config["artifact_dir"]) / selected[-1] / f"seed{s}") for s in (17, 42)
            ]
        else:
            references[h] = [selected]
    candidate = {
        "family": selected,
        "frozen": False,
        "holdout_allowed": False,
        "horizons": references,
        "selection_rule": rules["selection_rule"],
        "cluster_unit": unit,
        "quality_gate_sha256": sha256(reports / "model_quality_gate.json"),
        "forecast_sha256": source_hashes,
        "cost_order": family_names,
    }
    Path("configs/champion_candidate.yaml").write_text(yaml.safe_dump(candidate, sort_keys=False))
    best_lgb = f"lightgbm_{lgb_config['selected_config']}"
    best_gru = min(
        ("gru_A", "gru_B"),
        key=lambda n: next(row["normalized_mae"] for row in table if row["family"] == n),
    )
    records["best_baseline"] = {}
    for h in (1, 6, 24):
        name = min(
            family_names[:3],
            key=lambda n: np.abs(records[n][h].predicted - records[n][h].actual)[masks[h]].mean(),
        )
        records["best_baseline"][h] = records[name][h]
    if diagnostics:
        error_atlas(
            panel,
            records,
            masks,
            tiers,
            selected,
            [best_lgb, best_gru, "best_baseline"],
            reports,
            split["fit"][1],
        )
    text = [
        "# Week 5 model analysis",
        "",
        f"Champion-ready candidate: **{selected}** (not frozen).",
        "Selection uses the predeclared mean normalized MAE rule and cheaper-family fallback when paired CI cannot distinguish.",
        f"Cluster bootstrap: {unit}, {rules['bootstrap_repetitions']} repetitions; paired identical masks.",
        "Costs are an explicit simplicity order, not measured deployment latency.",
        "Intervals are exploratory validation evidence without multiplicity adjustment, not final-test claims.",
        "GRU families use the arithmetic mean of both seed predictions; seed variation is descriptive only.",
        "Shared GRU beam sample and global DLPRB normalization were audited. A/B masks and origins are identical.",
        "Hypotheses for any neural shortfall include short fit history, sampling and bias/variance; these are not proven causes.",
        "No final refit/freeze, holdout evaluation or optional quantile experiment was performed.",
        "Legacy 44/static6 results are superseded; see model_quality_gate_initial.json.",
        "",
        "```csv",
        pd.DataFrame(table).to_csv(index=False).strip(),
        "```",
        "",
        "Paired per-horizon intervals: paired_model_comparison_val.csv. Seed ablation: ablation_study.csv.",
        "Error cases and selection rules: error_atlas.md.",
    ]
    (reports / "model_analysis.md").write_text("\n".join(text), encoding="utf-8")
    (reports / "week5_comparison_manifest.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "forecast_sha256": source_hashes,
                "candidate": selected,
                "cluster_unit": unit,
                "holdout_evaluated": False,
                "frozen": False,
                "p1": "skipped_due_to_retraining",
            },
            indent=2,
        )
    )
    print("Week 5 comparison complete; candidate:", selected, "; holdout remains closed")


if __name__ == "__main__":
    compare()
