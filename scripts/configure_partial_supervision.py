"""Declare matched joint/available-label controls without reading observations."""

from copy import deepcopy
from pathlib import Path

import yaml


def protocols(base):
    result = {}
    for arm in ("joint", "available"):
        cfg = deepcopy(base)
        cfg.update(protocol="partial-target-supervision-"+arm, modes=["cell"],
                   training_objective="physical_DLPRB_batch_horizon_balanced_MAE",
                   training_label_policy=arm, stopping_label_policy="complete_joint_windows_unchanged",
                   source_pretraining="none", historical_control_reused_as_matched_arm=False,
                   comparison_scope="paired_new_control_same_training_loss_reused_development",
                   compute_scope="natural_window_counts_differ_report_actual_optimizer_steps")
        result[arm] = cfg
    return result


def main():
    base = yaml.safe_load(Path("configs/cell_attention_v2.yaml").read_text())
    for arm, cfg in protocols(base).items():
        output = Path(f"configs/partial_supervision_{arm}.yaml")
        if output.exists():
            raise FileExistsError("Preserve declared study configuration")
        output.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
