"""Independent in-place Kaggle checkpoint/metric audit; never trains or exports weights."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.configure_telecom_transfer import ARMS, protocols
from src.data.io import sha256
from src.data.panel import load_panel
from src.evaluation.origins import evaluation_origins
from src.features.cell_inputs import cell_groups
from src.features.gru_inputs import GRUPreprocessor
from src.features.pretraining_shards import iter_sharded_segments
from src.models.cell_attention import CellPatchAttention, configure_attention_math
from src.models.cell_pretraining import ENCODER_PREFIXES, SourcePatchForecaster
from src.models.initialization_replay import verify_cold_state
from src.utils.config import load_config


def verify_tree(root):
    """Require exact regular-file coverage and hashes before loading checkpoints."""
    root = Path(root)
    manifest = json.loads((root/"manifest.json").read_text())
    for name, digest in manifest["files"].items():
        path = root/name
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or sha256(path) != digest:
            raise ValueError("Artifact path/hash differs")
    files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if files != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Artifact manifest coverage differs")


def verify_steps(last, cfg):
    """Check actual optimizer steps, not merely the declared epoch budget."""
    expected = cfg["pretraining_epochs"]*int(np.ceil(cfg["pretraining_samples_per_epoch"]/cfg["pretraining_batch_size"]))
    steps = [int(x["step"]) for x in last["optimizer"]["state"].values() if "step" in x]
    if not steps or set(steps) != {expected} or last["epoch"]+1 != cfg["pretraining_epochs"]:
        raise ValueError("Pretraining update budget incomplete")
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("Verify checkpoints in Kaggle Output/Input; no local checkpoint transfer")
    torch.set_num_threads(2)
    configure_attention_math()
    verify_tree(args.artifacts)
    panel = load_panel("data/processed/panel.npz")
    if panel.values.shape != (840, 2880, 4) or not np.isnan(panel.values[504:]).all():
        raise ValueError("Canonical shape/redacted original future required")
    prepared = json.loads(Path("reports/telecom_italia_preparation.json").read_text())
    # Stream and validate the source manifest once, rather than on each target fold.
    source_hours = sum(len(values) for _, values, _ in iter_sharded_segments(prepared))
    if source_hours != prepared["eligible_hours"]:
        raise ValueError("Audited source population differs")
    expected_cfgs = protocols(load_config("configs/cell_multisource_v2.yaml"))
    all_rows, parity, controls, budgets = [], [], {}, []
    groups = cell_groups(panel.beam_ids)[[0, 45, 89]]
    for arm, _, _ in ARMS:
        cfg = load_config(f"configs/telecom_transfer_{arm}.yaml")
        if cfg != expected_cfgs[arm]:
            raise ValueError("Declared arm protocol differs")
        root = args.artifacts/arm/"multi_source_pretrain"
        source = root/"pretraining/external"
        meta = json.loads((source/"source_meta.json").read_text())
        fixed = {**cfg, "batch_size": cfg["pretraining_batch_size"], "max_epochs": cfg["pretraining_epochs"],
                 "early_stopping_patience": cfg["pretraining_epochs"]+1}
        provenance = meta["identity"]["provenance"]
        if (meta["config"] != fixed or meta["checkpoint"] != "last.pt" or provenance["target_network_used"]
                or [x["report"] for x in provenance["sources"]] != cfg["source_reports"]):
            raise ValueError("Source protocol/membership differs")
        for item in provenance["sources"]:
            report = json.loads(Path(item["report"]).read_text())
            if item["sha256"] != report["processed_sha256"] or sha256(Path(report["processed_path"])) != item["sha256"]:
                raise ValueError("Prepared source hash differs")
        source_cold_path = source/"source_initialization.pt"
        source_cold = torch.load(source_cold_path, weights_only=True, map_location="cpu")
        if (source_cold["seed"] != cfg["seed"] or source_cold["torch_initial_seed"] != cfg["seed"]
                or meta["source_initialization_sha256"] != sha256(source_cold_path)
                or meta["torch_initial_seed"] != cfg["seed"]):
            raise ValueError("Source seed identity differs")
        verify_cold_state(lambda: SourcePatchForecaster(cfg), cfg["seed"], source_cold["model"])
        source_last = torch.load(source/"last.pt", weights_only=True, map_location="cpu")
        budgets.append({"arm": arm, "verified_optimizer_updates": verify_steps(source_last, cfg)})
        rows = []
        for fold in cfg["folds"]:
            directory = root/fold["name"]/"cell"
            meta = json.loads((directory/"meta.json").read_text())
            if (meta["protocol"] != cfg or meta["fold"] != fold or meta["mode"] != "cell"
                    or meta["source_commit"] != args.source_commit or not meta["training_complete"]
                    or meta["holdout_used"] or meta["validation_504_671_used"] or meta["gpu_attention_fastpath"] is not False):
                raise ValueError("Target protocol/provenance differs")
            pre = GRUPreprocessor.load(directory/"preprocessor.npz")
            expected = GRUPreprocessor.fit(panel, fold["fit_end"], "B")
            if pre.fit_end != fold["fit_end"] or pre.beam_ids != panel.beam_ids:
                raise ValueError("Preprocessing cutoff/axes differ")
            for key in ("median", "mean", "scale"):
                np.testing.assert_allclose(getattr(pre, key), getattr(expected, key), atol=1e-12, rtol=1e-12)
            model = CellPatchAttention(cfg, pre.mean[0], pre.scale[0], "cell").eval()
            cold_path = directory/"cold_initialization.pt"
            cold = torch.load(cold_path, weights_only=True, map_location="cpu")
            if (cold["seed"] != cfg["seed"] or cold["torch_initial_seed"] != cfg["seed"]
                    or meta["torch_initial_seed"] != cfg["seed"]
                    or meta["cold_initialization_sha256"] != sha256(cold_path)):
                raise ValueError("Target seed/cold identity differs")
            verify_cold_state(lambda: CellPatchAttention(cfg, pre.mean[0], pre.scale[0], "cell"), cfg["seed"], cold["model"])
            if sum(p.numel() for p in model.parameters()) != meta["parameters"]:
                raise ValueError("Target parameter count differs")
            previous = controls.setdefault(fold["name"], cold["model"])
            if any(not torch.equal(value, previous[key]) for key, value in cold["model"].items()):
                raise ValueError("Arms do not share identical target cold weights")
            initialization = torch.load(directory/"initialization.pt", weights_only=True, map_location="cpu")
            keys = {k for k in model.state_dict() if k.startswith(ENCODER_PREFIXES)}
            if (set(initialization["transferred_encoder_keys"]) != keys
                    or initialization["source_checkpoint_sha256"] != sha256(source/"last.pt")
                    or initialization["cold_checkpoint_sha256"] != sha256(cold_path)
                    or initialization["seed"] != cfg["seed"] or initialization["fold"] != fold
                    or initialization["case"] != "multi_source_pretrain"):
                raise ValueError("Encoder transfer identity differs")
            for key, value in initialization["model"].items():
                reference = source_last["model"]["encoder."+key] if key in keys else cold["model"][key]
                if not torch.equal(value, reference):
                    raise ValueError("Encoder-only transfer isolation failed")
            if not any(not torch.equal(source_last["model"]["encoder."+k], source_cold["model"]["encoder."+k]) for k in keys):
                raise ValueError("Source encoder did not change during pretraining")
            best = torch.load(directory/"best.pt", weights_only=True, map_location="cpu")
            last = torch.load(directory/"last.pt", weights_only=True, map_location="cpu")
            history = [json.loads(line) for line in (directory/"history.jsonl").read_text().splitlines()]
            chosen = min(history, key=lambda x: (x["val_l1_scaled"], x["epoch"]))["epoch"]
            if (best["epoch"] != chosen or last["epoch"]+1 != len(history)
                    or last["epoch"]-chosen < cfg["early_stopping_patience"]
                    or meta["training"]["best_epoch"] != chosen or meta["training"]["epochs_completed"] != len(history)):
                raise ValueError("Stopping/termination evidence incomplete")
            model.load_state_dict(best["model"])
            for j, h in enumerate(cfg["horizons"]):
                with np.load(directory/f"score_h{h}.npz", allow_pickle=False) as cache:
                    origins = evaluation_origins(fold["stop_end"], fold["score_end"], h)
                    actual = panel.values[origins+h, :, 0].reshape(-1)
                    np.testing.assert_array_equal(cache["origins"], origins)
                    np.testing.assert_array_equal(cache["actual"], actual)
                    np.testing.assert_array_equal(cache["beam_ids"], panel.beam_ids)
                    predicted = cache["predicted"]
                    if not np.isfinite(predicted).all() or (predicted < 0).any():
                        raise ValueError("Forecast population/constraint differs")
                    mask = np.isfinite(actual)
                    error = np.abs(actual[mask]-predicted[mask])
                    zero = mask & (actual == 0)
                    rows.append({"fold": fold["name"], "mode": "cell", "horizon": h, "mae": float(error.mean()),
                                 "wape": float(error.sum()/np.abs(actual[mask]).sum()), "n_valid": int(mask.sum()),
                                 "zero_fp_rate": float((predicted[zero] > 0).mean())})
                    for origin in (fold["stop_end"], fold["score_end"]-24):
                        history_values = pre.transform(panel.values[origin-167:origin+1], "B")
                        inputs = torch.as_tensor(history_values[:, groups].transpose(1, 0, 2, 3))
                        phase = torch.as_tensor(pre.static_at(np.full(3, origin), np.arange(3)))
                        with torch.no_grad():
                            cpu = model(inputs, phase).numpy()[:, :, j]
                        gpu = predicted.reshape(len(origins), 2880)[origin-origins[0], groups]
                        np.testing.assert_allclose(cpu, gpu, atol=1e-4, rtol=0)
                        parity.append(float(np.abs(cpu-gpu).max()))
        frame = pd.DataFrame(rows)
        saved = args.artifacts.parent.parent/"reports"/f"telecom_transfer_{arm}_multi_source_pretrain_development.csv"
        columns = ["fold", "mode", "horizon", "mae", "wape", "n_valid", "zero_fp_rate"]
        pd.testing.assert_frame_equal(frame[columns], pd.read_csv(saved)[columns], check_dtype=False, atol=1e-12, rtol=1e-12)
        all_rows.extend({"arm": arm, **row} for row in rows)
    output = {"source_commit": args.source_commit, "training_performed_by_audit": False,
              "checkpoint_storage": "Kaggle Input/Output; no local checkpoint transfer",
              "all_metrics_reconstructed": True, "cold_initialization_and_encoder_transfer_verified": True,
              "all_arms_have_bitwise_matched_target_cold_states": True, "source_budgets": budgets,
              "maximum_cpu_gpu_error": max(parity), "history_only_predictions_checked": len(parity)*groups.size,
              "holdout_used": False, "production_promoted": False, "metrics": all_rows,
              "scope": "reused development; quality gate and generalization require separate assessment"}
    Path("/kaggle/working/telecom_transfer_verified_metrics.json").write_text(json.dumps(output, indent=2))
    print(json.dumps({k: v for k, v in output.items() if k != "metrics"}, indent=2))


if __name__ == "__main__":
    main()
