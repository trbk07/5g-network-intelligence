"""Protocol controls and local-training guard only; never fit a model."""

import pytest

from scripts.configure_telecom_transfer import protocols
from src.utils.config import load_config


def test_data_and_budget_arms_have_matched_target_settings():
    base = load_config("configs/cell_multisource_v2.yaml")
    arms = protocols(base)
    assert len(base["source_reports"]) == 9  # Factory must not mutate its control.
    assert len(arms["telecom_matched"]["source_reports"]) == 10
    assert len(arms["existing_extended"]["source_reports"]) == 9
    assert len(arms["telecom_extended"]["source_reports"]) == 10
    assert [a["pretraining_epochs"] for a in arms.values()] == [20, 60, 60]
    for arm in arms.values():
        for key in ("folds", "seed", "width", "learning_rate", "max_epochs", "early_stopping_patience"):
            assert arm[key] == base[key]
        assert arm["source_weights"][:9] == base["source_weights"]
        assert arm["cases"] == ["multi_source_pretrain"]


def test_controller_refuses_local_training(monkeypatch):
    from scripts import train_telecom_transfer

    def reject():
        raise RuntimeError("local training forbidden")

    monkeypatch.setattr(train_telecom_transfer, "require_cloud_training", reject)
    with pytest.raises(RuntimeError, match="local training forbidden"):
        train_telecom_transfer.main()
