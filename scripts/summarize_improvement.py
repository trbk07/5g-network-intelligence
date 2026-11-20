"""Verify completed cloud outputs and summarize measured gains; no fitting or selection occurs here."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_risk import risk_records
from src.data.io import sha256
from src.models.research_ensemble import ResearchEnsemble


def main():
    reports = Path("reports")
    directory = Path("artifacts/improved_candidate")
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Research artifact checksum mismatch: {name}")
    selection_path = reports / "improvement_selection.json"
    if sha256(selection_path) != manifest["selection_sha256"]:
        raise ValueError("Inner selection hash differs from final model manifest")
    model = ResearchEnsemble(directory)
    replay = risk_records(reports / "forecasts/improvement_audit.npz")
    from src.data.panel import load_panel

    panel = load_panel(Path("data/processed/panel.npz"))
    parity = []
    for h in (1, 6, 24):
        record = replay[h]["improved_ensemble"]
        for beam in ("0_0_0", "5_1_8", "15_2_0", "29_0_8"):
            b = record["beam_ids"].index(beam)
            source_b = panel.beam_ids.index(beam)
            for origin in (503, 527, 551, 575, 599):
                i = int(np.flatnonzero(record["origins"] == origin)[0])
                history = panel.values[origin - 167 : origin + 1, source_b : source_b + 1, :].copy()
                predicted = float(model.predict_at(history, origin, h)[0])
                saved = float(record["predicted"][i, b])
                error = abs(predicted - saved)
                if error > 1e-8:
                    raise AssertionError(
                        f"Research CPU parity differs: {beam}/{origin}/{h}: {error}"
                    )
                parity.append(
                    {
                        "beam_id": beam,
                        "origin": origin,
                        "horizon": h,
                        "saved": saved,
                        "cpu": predicted,
                        "absolute_error": error,
                        "atol": 1e-8,
                    }
                )
    pd.DataFrame(parity).to_csv(reports / "improvement_cpu_parity.csv", index=False)
    frame = pd.read_csv(reports / "improvement_audit.csv")
    pooled = frame[frame.load_tier == "All"]
    mae = pooled.pivot(index="model", columns="horizon", values="mae")
    paired = pd.read_csv(reports / "improvement_paired.csv")
    gains = {
        name: float((1 - mae.loc["improved_ensemble"] / mae.loc[name]).mean() * 100)
        for name in ("lightgbm_c1", "week7_deeper_l1")
    }
    lines = [
        "# Model improvement results",
        "",
        "All training, target-recipe fitting and ensemble calibration ran on hosted Kaggle. Production champion and previously observed final holdout are unchanged.",
        "",
        "## Reused-validation audit",
        "",
        "| Horizon | Original C1 MAE | Week 7 deeper MAE | New ensemble MAE | Reduction vs C1 | Reduction vs Week 7 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for h in (1, 6, 24):
        original, previous, new = (
            mae.loc[n, h] for n in ("lightgbm_c1", "week7_deeper_l1", "improved_ensemble")
        )
        lines.append(
            f"| {h}h | {original:.6f} | {previous:.6f} | {new:.6f} | {100 * (1 - new / original):.3f}% | {100 * (1 - new / previous):.3f}% |"
        )
    lines += [
        "",
        f"Mean normalized MAE reduction: {gains['lightgbm_c1']:.3f}% against C1; {gains['week7_deeper_l1']:.3f}% against Week 7 deeper.",
        "",
        "## What was tested",
        "",
        "- Nine LightGBM recipes screened on the first inner fold; promoted recipes confirmed on a second chronological fold.",
        "- Two CatBoost configurations on both folds, trained on GPU. All models use the original causal40 numerical inputs.",
        "- Separate training, early-stopping and scoring windows; all search scores use targets through hour 503.",
        "- Convex blends, small bias calibration, and final round selection fitted on cloud only. Weights/recipes were written and hashed before outer audit arrays were constructed.",
        "- Final candidate refit on development targets through 503 with fixed rounds; no stopping on audit 504-671.",
        "- Target normalization/residuals, more frequent training origins, recency weighting and model diversity were tested rather than assuming a larger model would help.",
        "",
        "## Paired uncertainty",
        "",
        "```csv",
        paired.to_csv(index=False).strip(),
        "```",
        "",
        "Intervals describe this reused development sample. They are not corrected for model search/multiple comparisons and are not evidence of unseen-week generalization.",
        "",
        "## Saved model and limitations",
        "",
        "`artifacts/improved_candidate` contains a loadable research ensemble, exact weights/bias, member models and checksum manifest. `scripts.predict_research` performs CPU inference only.",
        "Independent CPU inference matched 20 beam/origin pairs across 3 horizons within 1e-8; see improvement_cpu_parity.csv.",
        "Audit 504-671 is previously seen validation. No new final holdout score or global-optimality claim is made. Fresh unseen chronological data and a separately frozen promotion protocol remain necessary.",
        "Forecast MAE and exceedance alerting are different objectives: inspect improvement_risk.csv for precision/recall/FPR. Thresholds are historical scenarios, not verified congestion or predictive quantiles.",
        "No changes were made to artifacts/champion, configs/champion.yaml or Week 6 holdout reports.",
        "",
        "## Reproduction",
        "",
        "Protocol: configs/model_improvement.yaml; explanation: docs/model_improvement.md.",
        "Cloud workers: improve-lgbm (CPU), improve-catboost (GPU), improve-audit (cloud refit/calibration). Checkpoints persist in private Kaggle Output.",
        "Review notebook displays saved reports and never fits models or reopens holdout.",
        "",
    ]
    (reports / "model_improvement_results.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )
    result = {
        "research_only": True,
        "production_changed": False,
        "holdout_reopened": False,
        "mae_reduction_pct": gains,
        "selection_sha256": manifest["selection_sha256"],
        "cpu_parity_max_error": max(r["absolute_error"] for r in parity),
        "cpu_parity_comparisons": len(parity),
        "manifest": manifest,
        "reports": {p.name: sha256(p) for p in reports.glob("improvement_*.csv")},
    }
    (reports / "model_improvement_evidence.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8", newline="\n"
    )
    print("Verified research model + CPU parity; measured MAE reduction:", gains)


if __name__ == "__main__":
    main()
