"""Verify hosted multisource checkpoints, initialization, metrics and CPU replay."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.summarize_temporal_linear import summarize
from scripts.train_multisource_neural import source_cases
from src.data.io import sha256
from src.features.pretraining_batches import PretrainingBatches
from src.models.temporal_mlp_forecaster import TemporalMLPForecaster
from src.utils.config import load_config


class MultisourceNeuralForecaster(TemporalMLPForecaster):
    protocol_name = "multisource-univariate-mlp-pretraining-v1"
    architecture_name = "multisource-univariate-mlp-v1"


def verify_external_identity(identity, expected_sources, case, cfg):
    """Reject changed source prefixes or preprocessing, not just changed model tensors."""
    if identity["seed"] != cfg["seed"] or identity["run_identity"]["case"] != case:
        raise ValueError("External seed/case identity differs")
    observed = identity["run_identity"]["sources"]
    if len(observed) != len(expected_sources):
        raise ValueError("External source population differs")
    for actual, expected in zip(observed, expected_sources):
        for key in ("report", "sha256", "entities", "fit_scope"):
            if actual[key] != expected[key]:
                raise ValueError("External prefix provenance differs")
        np.testing.assert_allclose([actual["mean"], actual["scale"]],
                                   [expected["mean"], expected["scale"]], rtol=1e-12, atol=1e-12)
    ignored = {"max_epochs", "variants", "seeds", "artifact_dir", "gpu_budget_seconds"}
    expected_config = {**cfg, "early_stopping_patience": cfg["external_epochs"]+1}
    if identity["config"] != {k: v for k, v in expected_config.items() if k not in ignored}:
        raise ValueError("External optimization protocol differs")


def compare_cases(frames):
    """Require the full declared population and controls before computing gains."""
    required = {"target_only", "target_self_pretrain", "radio", "cesnet", "radio_and_cesnet"}
    if set(frames) != required:
        raise ValueError("All five cases required, including matched self-pretraining control")
    keys = ["fold", "variant", "horizon"]
    indexed = {}
    for case, frame in frames.items():
        if frame.duplicated(keys).any() or len(frame) != 6:
            raise ValueError("Expected two folds and three horizons exactly once per case")
        indexed[case] = frame.set_index(keys).sort_index()
    reference = indexed["target_only"]
    expected = {(fold, "A", h) for fold in ("inner1", "inner2") for h in (1, 6, 24)}
    if set(reference.index) != expected:
        raise ValueError("Unexpected development fold/horizon population")
    rows = []
    for case, frame in indexed.items():
        if not frame.index.equals(reference.index):
            raise ValueError("Case coverage differs")
        np.testing.assert_array_equal(frame.n_valid, reference.n_valid)
        if (not np.isfinite(frame[["mae", "wape"]]).all().all()
                or (frame.mae < 0).any() or (frame.n_valid <= 0).any()):
            raise ValueError("Invalid metrics or empty population")
        for index, row in frame.iterrows():
            out = dict(zip(keys, index)) | {"case": case, **row.to_dict()}
            for control in ("target_only", "target_self_pretrain"):
                baseline = indexed[control].loc[index, "mae"]
                if baseline <= 0:
                    raise ValueError("Positive baseline MAE required for relative gains")
                out[f"relative_mae_gain_vs_{control}_pct"] = 100*(1-row.mae/baseline)
            rows.append(out)
    return pd.DataFrame(rows)


def verify_initialization(root, case, fold, cfg):
    """Prove transferred tensors equal the declared fixed-budget source checkpoint."""
    directory = root/case/fold["name"]/"A"
    initial = torch.load(directory/"initialization.pt", map_location="cpu", weights_only=True)
    if initial["case"] != case or initial["fold"] != fold:
        raise ValueError("Initialization case/fold differs")
    if case == "target_only":
        if initial["source_checkpoint"] is not None:
            raise ValueError("Target-only must not load pretrained weights")
        return
    source = (root/"self_pretraining"/fold["name"]/"last.pt" if case == "target_self_pretrain"
              else root/"external"/case/"last.pt")
    if initial["source_checkpoint_sha256"] != sha256(source):
        raise ValueError("Transferred source checkpoint checksum differs")
    state = torch.load(source, map_location="cpu", weights_only=True)
    if state["epoch"]+1 != cfg["external_epochs"]:
        raise ValueError("Fixed pretraining budget incomplete")
    expected_steps = math.ceil(cfg["external_samples_per_epoch"]/cfg["batch_size"])*cfg["external_epochs"]
    actual_steps = {int(item["step"]) for item in state["optimizer"]["state"].values()}
    if actual_steps != {expected_steps}:
        raise ValueError("Pretraining optimizer steps differ from matched compute budget")
    if initial["model"].keys() != state["model"].keys():
        raise ValueError("Transferred model schema differs")
    for key in state["model"]:
        torch.testing.assert_close(initial["model"][key], state["model"][key], rtol=0, atol=0)
    if case == "target_self_pretrain":
        identity = state["identity"]["run_identity"]
        if identity != {"case": case, "fold": fold, "fit_end": fold["fit_end"]}:
            raise ValueError("Self-pretraining used a different fold")


def main():
    cfg = load_config("configs/multisource_neural_pilot.yaml")
    root = Path("artifacts/multisource_neural")
    manifest = json.loads((root/"manifest.json").read_text())
    if manifest["protocol"] != cfg:
        raise ValueError("Root protocol differs")
    for name, digest in manifest["files"].items():
        if sha256(root/name) != digest:
            raise ValueError("Root checkpoint checksum differs")
    frames = {}
    for case in cfg["cases"]:
        if case in source_cases():
            reports, weights = source_cases()[case]
            expected = PretrainingBatches(reports, weights, samples=1).provenance
            checkpoint = torch.load(root/"external"/case/"last.pt", map_location="cpu", weights_only=True)
            verify_external_identity(checkpoint["identity"], expected, case, cfg)
        for fold in cfg["folds"]:
            verify_initialization(root, case, fold, cfg)
        prefix = f"multisource_neural_{case}"
        summarize("configs/multisource_neural_pilot.yaml", root/case,
                  MultisourceNeuralForecaster, prefix, compare_mlp=True)
        frames[case] = pd.read_csv(f"reports/{prefix}_development.csv")
    comparison = compare_cases(frames)
    comparison.to_csv("reports/multisource_neural_ablation.csv", index=False)
    Path("reports/multisource_neural_verification.json").write_text(json.dumps({
        "all_five_cases_verified": True, "initialization_tensors_verified": True,
        "pretraining_optimizer_step_counts_verified": True,
        "external_prefix_checksums_and_global_preprocessing_verified": True,
        "comparison_rows": len(comparison), "holdout_used": False,
        "production_promoted": False, "claim_scope": "one-seed reused development only"}, indent=2))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
