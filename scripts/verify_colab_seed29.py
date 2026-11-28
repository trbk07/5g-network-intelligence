"""Verify completed Colab artifacts in place; export aggregate evidence only."""

import hashlib
import json
import os
from pathlib import Path


def main():
    """Reconstruct all scores and CPU parity without moving checkpoints off Drive."""
    import sys
    project = Path("/content/cell_seed29")
    os.chdir(project)
    sys.path.insert(0, str(project))
    import numpy as np
    import pandas as pd
    import torch
    from src.data.panel import load_panel
    from src.features.cell_inputs import cell_groups
    from src.features.gru_inputs import GRUPreprocessor
    from src.models.cell_attention import CellPatchAttention, configure_attention_math
    from src.utils.config import load_config

    torch.set_num_threads(2)
    configure_attention_math()
    cfg = load_config("configs/cell_attention_seed29.yaml")
    root = Path("artifacts/cell_attention_seed29")
    manifest = json.loads((root/"manifest.json").read_text())
    actual_files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual_files != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Artifact manifest coverage differs")
    for name, digest in manifest["files"].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError("Artifact checksum differs")
    panel = load_panel("data/processed/panel.npz")
    if not np.isnan(panel.values[504:]).all():
        raise ValueError("Original future exclusions must remain redacted")
    groups = cell_groups(panel.beam_ids)[[0, 44, 89]]
    rows, parity, stopping = [], [], []
    for fold in cfg["folds"]:
        for mode in cfg["modes"]:
            directory = root/fold["name"]/mode
            meta = json.loads((directory/"meta.json").read_text())
            if meta["protocol"] != cfg or meta["holdout_used"] or not meta["training_complete"]:
                raise ValueError("Training protocol/state differs")
            pre = GRUPreprocessor.load(directory/"preprocessor.npz")
            expected = GRUPreprocessor.fit(panel, fold["fit_end"], "B")
            for key in ("median", "mean", "scale"):
                np.testing.assert_allclose(getattr(pre, key), getattr(expected, key), atol=1e-12, rtol=1e-12)
            torch.manual_seed(cfg["seed"])
            model = CellPatchAttention(cfg, pre.mean[0], pre.scale[0], mode).eval()
            cold_path = directory/"cold_initialization.pt"
            cold = torch.load(cold_path, weights_only=True, map_location="cpu")
            if (cold["torch_initial_seed"] != 29 or meta["torch_initial_seed"] != 29
                    or meta["gpu_attention_fastpath"] is not False
                    or meta["cold_initialization_sha256"] != hashlib.sha256(cold_path.read_bytes()).hexdigest()
                    or any(not torch.equal(v, cold["model"][k]) for k, v in model.state_dict().items())):
                raise ValueError("Seeded cold state does not match bitwise")
            best = torch.load(directory/"best.pt", weights_only=True, map_location="cpu")
            last = torch.load(directory/"last.pt", weights_only=True, map_location="cpu")
            history = [json.loads(line) for line in (directory/"history.jsonl").read_text().splitlines()]
            chosen = min(history, key=lambda x: (x["val_l1_scaled"], x["epoch"]))["epoch"]
            if best["epoch"] != chosen or last["epoch"]+1 != len(history):
                raise ValueError("Stopping choice/history differs")
            stale = last["epoch"]-chosen
            if stale < cfg["early_stopping_patience"]:
                raise ValueError("Patience incomplete")
            stopping.append({"fold": fold["name"], "mode": mode, "best_epoch": chosen,
                             "epochs_completed": len(history), "stale_epochs": stale})
            model.load_state_dict(best["model"])
            for h in cfg["horizons"]:
                with np.load(directory/f"score_h{h}.npz", allow_pickle=False) as score:
                    origins = np.arange(fold["stop_end"], fold["score_end"]-h+1)
                    np.testing.assert_array_equal(score["origins"], origins)
                    np.testing.assert_array_equal(score["beam_ids"], panel.beam_ids)
                    actual = panel.values[origins+h, :, 0].reshape(-1)
                    np.testing.assert_array_equal(score["actual"], actual)
                    predicted = score["predicted"]
                    if not np.isfinite(predicted).all():
                        raise ValueError("Forecast coverage differs")
                    mask = np.isfinite(actual)
                    error = np.abs(predicted[mask]-actual[mask])
                    zero = mask & (actual == 0)
                    rows.append({"fold": fold["name"], "mode": mode, "horizon": h,
                                 "mae": float(error.mean()), "wape": float(error.sum()/np.abs(actual[mask]).sum()),
                                 "n_valid": int(mask.sum()), "zero_fp_rate": float((predicted[zero] > 0).mean())})
                    for origin in (fold["stop_end"], fold["score_end"]-24):
                        history_values = pre.transform(panel.values[origin-167:origin+1], "B")
                        inputs = torch.as_tensor(history_values[:, groups].transpose(1, 0, 2, 3))
                        phase = torch.as_tensor(pre.static_at(np.full(3, origin), np.arange(3)))
                        with torch.no_grad():
                            cpu = model(inputs, phase).numpy()[:, :, cfg["horizons"].index(h)]
                        gpu = predicted.reshape(len(origins), len(panel.beam_ids))[origin-origins[0], groups]
                        np.testing.assert_allclose(cpu, gpu, atol=1e-4, rtol=0)
                        parity.append(float(np.abs(cpu-gpu).max()))
    frame = pd.DataFrame(rows)
    columns = ["fold", "mode", "horizon", "mae", "wape", "n_valid", "zero_fp_rate"]
    pd.testing.assert_frame_equal(frame[columns], pd.read_csv("reports/cell_attention_seed29_development.csv")[columns],
                                  check_dtype=False, atol=1e-12, rtol=1e-12)
    proof = {"all_metrics_reconstructed": True, "controlled_neural_initialization_verified": True,
             "stopping_choices_verified": True, "maximum_cpu_gpu_error": max(parity),
             "history_only_beam_predictions_checked": len(parity)*groups.size,
             "checkpoint_storage": "Drive; no checkpoint download", "holdout_used": False,
             "production_promoted": False, "cloud_torch": str(torch.__version__), "termination": stopping,
             "claim_scope": "Reused development, seed and runtime robustness; no best-seed selection"}
    output = Path("/content/seed29_aggregate_evidence.json")
    output.write_text(json.dumps({"verification": proof, "metrics": rows}, indent=2))
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
