"""Assess fixed aggregate comparisons only after cloud integrity verification."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.configure_telecom_transfer import ARMS

KEYS = ["fold", "horizon"]
EXPECTED = {(fold, h) for fold in ("inner1", "inner2") for h in (1, 6, 24)}


def paired(candidate, reference):
    """Require complete paired coverage before computing relative MAE gains."""
    for frame in (candidate, reference):
        if (len(frame) != 6 or frame.duplicated(KEYS).any()
                or set(frame[KEYS].itertuples(index=False, name=None)) != EXPECTED):
            raise ValueError("Complete unique fold/horizon population required")
        if not np.isfinite(frame[["mae", "wape", "n_valid"]].to_numpy()).all():
            raise ValueError("Finite aggregate metrics required")
        if (frame.mae < 0).any() or (frame.wape < 0).any() or (frame.n_valid <= 0).any():
            raise ValueError("Valid error and population metrics required")
    result = candidate.merge(reference[KEYS+["mae", "n_valid"]], on=KEYS,
                             suffixes=("", "_reference"), validate="one_to_one")
    if not (result.n_valid == result.n_valid_reference).all() or (result.mae_reference <= 0).any():
        raise ValueError("Identical observed populations and positive reference MAE required")
    result["relative_mae_gain_pct"] = 100*(1-result.mae/result.mae_reference)
    return result


def gate(comparison):
    """No subset selection: every declared fold and horizon contributes."""
    gains = comparison.relative_mae_gain_pct
    no_regression = bool((gains >= 0).all())
    mean = float(gains.mean())
    return {"no_fold_horizon_regression": no_regression, "mean_relative_mae_gain_pct": mean,
            "initial_research_gate_passed": no_regression and mean >= 2,
            "material_point_gain_gate_passed": no_regression and mean >= 5,
            "all_development_WAPE_below_10pct": bool((comparison.wape < .10).all()),
            "production_promotion_permitted": False,
            "scope": "point estimates only; reused development, robustness and uncertainty still required"}


def main():
    path = Path("reports/telecom_transfer_verified_metrics.json")
    evidence = json.loads(path.read_text())
    for key in ("all_metrics_reconstructed", "cold_initialization_and_encoder_transfer_verified",
                "all_arms_have_bitwise_matched_target_cold_states"):
        if evidence.get(key) is not True:
            raise ValueError("Complete cloud verification evidence required")
    if (evidence["holdout_used"] or evidence["production_promoted"] or evidence["training_performed_by_audit"]
            or not np.isfinite(evidence["maximum_cpu_gpu_error"]) or evidence["maximum_cpu_gpu_error"] > 1e-4):
        raise ValueError("Temporal/parity scope differs")
    expected_budgets = {"telecom_matched": 5120, "existing_extended": 15360, "telecom_extended": 15360}
    budgets = evidence["source_budgets"]
    if len(budgets) != 3 or {x["arm"]: x["verified_optimizer_updates"] for x in budgets} != expected_budgets:
        raise ValueError("Declared compute arms incomplete")
    frame = pd.DataFrame(evidence["metrics"])
    if len(frame) != 18 or set(frame.arm) != set(expected_budgets):
        raise ValueError("All declared arm scores required; never select successful subsets")
    proof = json.loads(Path("reports/cell_multisource_v2_multi_source_pretrain_verification.json").read_text())
    if not proof["all_metrics_reconstructed"] or not proof["controlled_neural_initialization_verified"]:
        raise ValueError("Verified historical short-budget control required")
    historical = pd.read_csv("reports/cell_multisource_v2_multi_source_pretrain_comparison.csv")
    strong = historical[KEYS+["wape", "n_valid", "current_mae"]].rename(columns={"current_mae": "mae"})
    tables, results = [], {}
    for arm, _, _ in ARMS:
        comparison = paired(frame[frame.arm == arm], strong)
        comparison["comparison"] = "strong_comparator"
        tables.append(comparison)
        results[arm] = gate(comparison)
    comparisons = (
        ("data_at_short_budget", frame[frame.arm == "telecom_matched"], historical),
        ("data_at_extended_budget", frame[frame.arm == "telecom_extended"], frame[frame.arm == "existing_extended"]),
        ("compute_on_existing_data", frame[frame.arm == "existing_extended"], historical),
        ("compute_with_telecom", frame[frame.arm == "telecom_extended"], frame[frame.arm == "telecom_matched"]),
    )
    for name, candidate, reference in comparisons:
        comparison = paired(candidate, reference)
        comparison["comparison"] = name
        tables.append(comparison)
    pd.concat(tables, ignore_index=True).to_csv("reports/telecom_transfer_paired_comparisons.csv", index=False)
    output = {"arms": results, "quality_improvement_claim": "Evaluate every arm; no best-arm or best-horizon cherry-picking",
              "production_promoted": False, "fresh_generalization_established": False,
              "historical_control_caveat": "Check source/runtime identities before attributing historical differences purely to data or compute"}
    Path("reports/telecom_transfer_quality_assessment.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
