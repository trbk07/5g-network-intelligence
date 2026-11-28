"""Compare verified transfer studies against bitwise-matched cold baselines."""

import json
from pathlib import Path

import pandas as pd
import torch


def main():
    baseline = pd.read_csv("reports/cell_attention_v2_development.csv")
    baseline = baseline[baseline["mode"] == "cell"]
    rows = []
    for study in ("cell_attention_v2", "cell_multisource_v2_multi_source_pretrain",
                  "cell_multisource_v2_target_self_pretrain"):
        proof = json.loads(Path(f"reports/{study}_verification.json").read_text())
        if not all(proof[k] for k in ("all_metrics_reconstructed", "controlled_neural_initialization_verified",
                                     "stopping_choices_verified")) or proof["holdout_used"]:
            raise ValueError("Verified temporal/seed/metric evidence required")
    for case in ("multi_source_pretrain", "target_self_pretrain"):
        for fold in ("inner1", "inner2"):
            a = torch.load(Path("artifacts/cell_attention_v2")/fold/"cell/cold_initialization.pt",
                           weights_only=True, map_location="cpu")["model"]
            b = torch.load(Path("artifacts/cell_multisource_v2")/case/fold/"cell/cold_initialization.pt",
                           weights_only=True, map_location="cpu")["model"]
            if set(a) != set(b) or any(not torch.equal(v, b[k]) for k, v in a.items()):
                raise ValueError("Cross-job cold states differ")
        frame = pd.read_csv(f"reports/cell_multisource_v2_{case}_development.csv").merge(
            baseline, on=["fold", "mode", "horizon"], suffixes=("", "_cold"), validate="one_to_one")
        if len(frame) != 6 or not (frame["n_valid"] == frame["n_valid_cold"]).all():
            raise ValueError("Paired comparison coverage differs")
        frame["case"] = case
        frame["gain_vs_cold_pct"] = 100*(1-frame.mae/frame.mae_cold)
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    result.to_csv("reports/cell_multisource_v2_matched_cold_comparison.csv", index=False)
    print(result[["case", "fold", "horizon", "gain_vs_cold_pct"]].to_string(index=False))
    print("All cross-job cold states match bitwise. Reused development; no promotion.")


if __name__ == "__main__":
    main()
