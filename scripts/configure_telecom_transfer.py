"""Freeze data and compute ablations before observing Telecom transfer scores."""

from copy import deepcopy
from pathlib import Path

import yaml

ARMS = (("telecom_matched", True, 20), ("existing_extended", False, 60), ("telecom_extended", True, 60))


def protocols(base):
    """Separate additional-data effects from a tripled pretraining budget."""
    result = {}
    for arm, telecom, epochs in ARMS:
        cfg = deepcopy(base)
        cfg.update(protocol="telecom-encoder-transfer-"+arm, cases=["multi_source_pretrain"],
                   pretraining_epochs=epochs, reference_protocol="configs/cell_multisource_v2.yaml",
                   study_arm=arm, comparison_scope="reused_development_no_production_promotion")
        if telecom:
            cfg["source_reports"].append("reports/telecom_italia_preparation.json")
            cfg["source_weights"].append(1)
        cfg["data_license_policy"] = "ODbL_database_attribution_and_share_alike_separate_from_MIT_code"
        result[arm] = cfg
    return result


def main():
    base = yaml.safe_load(Path("configs/cell_multisource_v2.yaml").read_text())
    for arm, cfg in protocols(base).items():
        path = Path(f"configs/telecom_transfer_{arm}.yaml")
        if path.exists():
            raise FileExistsError("Preserve declared protocol; do not overwrite")
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
