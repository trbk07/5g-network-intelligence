"""Single final replay, guarded by frozen source, artifacts and an atomic attempt marker."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.io import sha256
from src.data.panel import load_panel
from src.data.audit import check_mapping
from src.data.schema import parse_beam_id
from src.evaluation.bootstrap import paired_cluster_ci
from src.evaluation.common_mask import comparable_mask
from src.evaluation.evaluator import ForecastRecord, evaluate, save_forecasts, to_benchmark_rows
from src.evaluation.metrics import assign_load_tiers
from src.evaluation.origins import evaluation_origins
from src.features.lightgbm_features import build_features
from src.inference.freeze import claim_holdout, config_hash, verify_artifacts, verify_frozen
from src.models.baselines import Persistence, SeasonalNaive
from src.models.lgbm import load_lgbm
from src.risk.evaluator import evaluate_exceedance
from src.risk.thresholds import Thresholds
from src.utils.config import load_config


def baseline(name):
    """Resolve the baseline definition fixed from validation."""
    return Persistence() if name == "persistence" else SeasonalNaive(int(name.rsplit("_", 1)[1]))


def main():
    """Never select or update parameters after entering this runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", default="configs/champion.yaml")
    args = parser.parse_args()
    frozen, attestation = verify_frozen(args.frozen)
    artifacts = verify_artifacts()
    digest = config_hash(frozen)
    if (
        artifacts["frozen_config_sha256"] != digest
        or artifacts["freeze_commit"] != attestation["commit"]
    ):
        raise RuntimeError("Artifact provenance differs from freeze")
    reports = Path("reports")
    claim_holdout(reports / "holdout_attempt.json", digest)
    data = load_config("configs/data.yaml")
    if data["revision"] != frozen["dataset_revision"]:
        raise RuntimeError("Dataset revision differs")
    # The first access to final targets happens only after every freeze/attempt guard.
    panel_path = Path(data["processed_dir"]) / "panel.npz"
    if sha256(panel_path) != frozen["panel_sha256"]:
        raise RuntimeError("Canonical cloud panel checksum differs")
    panel = load_panel(panel_path)
    end = frozen["final_fit"][1]
    if frozen["holdout"][0] != end + 1 or len(panel.time) != frozen["holdout"][1] + 1:
        raise RuntimeError("Final evaluation split differs")
    thresholds = Thresholds.load("artifacts/champion/thresholds.npz")
    if thresholds.fit_end != end or thresholds.beam_ids != panel.beam_ids:
        raise RuntimeError("Threshold fit boundary/beam axis mismatch")
    tiers = assign_load_tiers(panel.values[: end + 1, :, 0])
    # Dataset topology is verified by the Week 5 quality gate: 30 BS x 3 cells x 32 beams.
    if not check_mapping(panel)["complete_expected_mapping"]:
        raise RuntimeError("Verified BS topology changed")
    clusters = np.array([parse_beam_id(b)[0] for b in panel.beam_ids])
    records, rows, cis, risks, cases = [], [], [], [], []
    for h in frozen["horizons"]:
        origins = evaluation_origins(end, frozen["holdout"][1], h)
        comparator = evaluate(baseline(frozen["baseline_comparators"][h]), panel, origins, h)
        if frozen["family"].startswith("lightgbm_"):
            model = load_lgbm(Path("artifacts/champion") / f"h{h}")
            if model.static.fit_end != end or model.static.beam_ids != panel.beam_ids:
                raise RuntimeError("Final model boundary/beam axis mismatch")
            x = build_features(panel.values, origins, h, model.static)
            predicted = model.booster.predict(x, num_threads=frozen["num_threads"]).reshape(
                len(origins), -1
            )
            for j in (0, len(origins) // 2, len(origins) - 1):
                t = int(origins[j])
                np.testing.assert_allclose(
                    predicted[j],
                    model.predict_at(panel.values[t - 167 : t + 1], t, h),
                    rtol=1e-9,
                    atol=1e-9,
                )
            champion = ForecastRecord(
                frozen["family"], h, origins, panel.beam_ids, comparator.actual.copy(), predicted
            )
        else:
            champion = evaluate(baseline(frozen["family"]), panel, origins, h)
        mask = comparable_mask(
            champion.actual, {"champion": champion.predicted, "baseline": comparator.predicted}
        )
        cis.append(
            {
                "horizon": h,
                "model_a": champion.model,
                "model_b": comparator.model,
                "cluster_unit": "BS",
                **paired_cluster_ci(
                    champion.actual,
                    champion.predicted,
                    comparator.predicted,
                    clusters,
                    mask,
                    **frozen["bootstrap"],
                ),
            }
        )
        for record in (champion, comparator):
            # Both models use exactly the same valid pairs for every reported metric.
            record.actual = np.where(mask, record.actual, np.nan)
            record.predicted = np.where(mask, record.predicted, np.nan)
            records.append(record)
            rows.extend(to_benchmark_rows(record, tiers))
            groups = {"All": np.ones(len(tiers), dtype=bool)}
            groups.update({tier: tiers == tier for tier in np.unique(tiers)})
            for tier, beams in groups.items():
                group_mask = mask & beams[None, :]
                risks.extend(
                    {
                        "model": record.model,
                        "horizon": h,
                        "load_tier": tier,
                        "n_group_beams": int(beams.sum()),
                        **r,
                    }
                    for r in evaluate_exceedance(
                        record.actual,
                        record.predicted,
                        thresholds,
                        group_mask,
                        frozen["risk_policy"]["min_positive_report"],
                    )
                )
        for label, cutoff in (
            ("warning_p90", thresholds.warning),
            ("critical_p95", thresholds.critical),
        ):
            eligible = mask & thresholds.valid[None, :]
            actual_event = champion.actual >= cutoff
            predicted_event = champion.predicted >= cutoff
            for kind, selection in (
                ("FP", predicted_event & ~actual_event),
                ("FN", actual_event & ~predicted_event),
            ):
                for i, b in np.argwhere(eligible & selection)[:3]:
                    cases.append(
                        {
                            "horizon": h,
                            "scenario": label,
                            "kind": kind,
                            "origin": int(origins[i]),
                            "target_time": int(origins[i] + h),
                            "beam_id": panel.beam_ids[b],
                            "actual": float(champion.actual[i, b]),
                            "predicted": float(champion.predicted[i, b]),
                            "threshold": float(cutoff[b]),
                        }
                    )
        print("Final replay completed:", h, "h", flush=True)
    pd.DataFrame(rows).to_csv(reports / "champion_benchmark_holdout.csv", index=False)
    pd.DataFrame(rows).to_csv(reports / "holdout_benchmark.csv", index=False)
    pd.DataFrame(cis).to_csv(reports / "paired_comparison_holdout.csv", index=False)
    pd.DataFrame(risks).to_csv(reports / "risk_holdout.csv", index=False)
    pd.DataFrame(cases).to_csv(reports / "risk_cases_holdout.csv", index=False)
    pd.DataFrame(
        {
            "beam_id": thresholds.beam_ids,
            "eligible": thresholds.valid,
            "reason": thresholds.reasons,
            "P90": thresholds.warning,
            "P95": thresholds.critical,
            "active_ratio": thresholds.active_ratio,
            "n_observed": thresholds.n_observed,
        }
    ).to_csv(reports / "threshold_eligibility.csv", index=False)
    risk_text = "# Historical risk evaluation\n\nP90/P95 were fitted only on final development observations.\n"
    risk_text += "Events use >= for actual and prediction. Equal P90/P95 has no Warning band.\n"
    risk_text += "No operator congestion labels are available; these are statistical exceedances.\n"
    risk_text += "Counts, exclusions and undefined denominator flags are in risk_holdout.csv; load groups use development-only tiers.\n"
    risk_text += "Examples below are the first three available FP/FN per horizon and scenario, without cherry-picking.\n\n```csv\n"
    risk_text += pd.DataFrame(cases).to_csv(index=False) + "```\n"
    (reports / "risk_evaluation.md").write_text(risk_text, encoding="utf-8")
    forecasts = reports / "forecasts/champion_holdout.npz"
    save_forecasts(
        records, forecasts, {"freeze_commit": attestation["commit"], "frozen_config_sha256": digest}
    )
    outputs = [
        reports / n
        for n in (
            "champion_benchmark_holdout.csv",
            "paired_comparison_holdout.csv",
            "risk_holdout.csv",
            "risk_cases_holdout.csv",
        )
    ] + [forecasts]
    (reports / "holdout_completion.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "freeze_commit": attestation["commit"],
                "frozen_config_sha256": digest,
                "files": {p.as_posix(): sha256(p) for p in outputs},
            },
            indent=2,
        )
    )
    (reports / "week6_results.md").write_text(
        "# Final holdout evaluation\n\nFixed-model rolling replay; no parameters were selected from holdout.\n"
        "See champion_benchmark_holdout.csv and paired_comparison_holdout.csv (negative difference favors champion).\n"
        "Risk P90/P95 are historical statistical scenarios, not congestion ground truth.\n"
        "FP/FN examples are descriptive, not incident explanations. Undefined metrics carry flags.\n"
    )
    (reports / "holdout_analysis.md").write_text(
        (reports / "week6_results.md").read_text()
        + "\nMetrics are final and descriptive; no post-holdout tuning is permitted.\n"
        + "Baseline was selected on validation independently for each horizon. CI resamples 30 verified BS clusters, 1000 repetitions.\n"
        + "Fallback split is internal holdout; no temporal continuity claim is made about external weeks.\n"
    )


if __name__ == "__main__":
    main()
