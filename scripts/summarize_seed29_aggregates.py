"""Compare cloud-verified seed-29 aggregates without downloading checkpoints."""

import json
from pathlib import Path

import pandas as pd


def main():
    evidence = json.loads(Path("reports/cell_attention_seed29_aggregate_evidence.json").read_text())
    proof = evidence["verification"]
    if (not all(proof[k] for k in ("all_metrics_reconstructed", "controlled_neural_initialization_verified",
                                  "stopping_choices_verified")) or proof["holdout_used"]
            or proof["maximum_cpu_gpu_error"] > 1e-4):
        raise ValueError("Cloud verification required")
    frame = pd.DataFrame(evidence["metrics"])
    reference = pd.read_csv("reports/cell_attention_v2_comparison.csv")
    reference = reference[["fold", "mode", "horizon", "mae", "n_valid", "current_mae"]].rename(
        columns={"mae": "seed17_mae", "n_valid": "seed17_n_valid"})
    result = frame.merge(reference, on=["fold", "mode", "horizon"], validate="one_to_one")
    if len(result) != 12 or not (result.n_valid == result.seed17_n_valid).all():
        raise ValueError("Complete seed/runtime comparison required")
    result["relative_mae_gain_vs_current_pct"] = 100*(1-result.mae/result.current_mae)
    result["seed29_vs_seed17_gain_pct"] = 100*(1-result.mae/result.seed17_mae)
    result.to_csv("reports/cell_attention_seed29_comparison.csv", index=False)
    print(result[["fold", "mode", "horizon", "mae", "wape", "relative_mae_gain_vs_current_pct"]].to_string(index=False))
    print("No best-seed selection; seed and Torch/runtime change together.")


if __name__ == "__main__":
    main()
