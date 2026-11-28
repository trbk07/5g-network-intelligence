"""Hosted inference-only backend diagnosis and strict replay of frozen checkpoints."""

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.train_cell_attention import score
from scripts.train_multisource_neural import validate_output_storage
from src.data.io import sha256
from src.evaluation.validation import validation_inputs
from src.features.cell_inputs import cell_groups
from src.features.gru_inputs import GRUPreprocessor
from src.models.cell_attention import CellPatchAttention
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training


def predict(model, panel, pre, origin, groups, device):
    window = pre.transform(panel.values[origin-167:origin+1], "B")
    sequence = torch.as_tensor(window[:, groups].transpose(1, 0, 2, 3), device=device)
    phase = torch.as_tensor(pre.static_at(np.full(len(groups), origin), np.arange(len(groups))), device=device)
    with torch.no_grad():
        return model(sequence, phase).cpu().numpy()


def main():
    require_cloud_training()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-root", type=Path, required=True)
    parser.add_argument("--case", choices=["target_self_pretrain", "multi_source_pretrain"])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Hosted GPU required for backend comparison")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    cfg = load_config("configs/cell_multisource.yaml" if args.case else "configs/cell_attention.yaml")
    panel, _ = validation_inputs(load_config("configs/data.yaml"))
    panel = replace(panel, values=panel.values[:504], time=panel.time[:504])
    groups = cell_groups(panel.beam_ids)[[0, 45, 89]]
    root = Path("artifacts/cell_multisource_precision")/args.case if args.case else Path("artifacts/cell_attention_precision")
    report_prefix = f"cell_multisource_{args.case}_precision" if args.case else "cell_attention_precision"
    validate_output_storage(root)
    root.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    rows, diagnostics = [], []
    # Diagnostic origins and tolerance are fixed before reading any score aggregates.
    for fold in cfg["folds"]:
        for mode in cfg["modes"]:
            trained = args.trained_root/fold["name"]/mode
            pre = GRUPreprocessor.load(trained/"preprocessor.npz")
            state = torch.load(trained/"best.pt", map_location="cpu", weights_only=True)
            cpu = CellPatchAttention(cfg, pre.mean[0], pre.scale[0], mode).eval()
            gpu = CellPatchAttention(cfg, pre.mean[0], pre.scale[0], mode).to("cuda").eval()
            cpu.load_state_dict(state["model"])
            gpu.load_state_dict(state["model"])
            for origin in (fold["stop_end"], fold["score_end"]-24):
                torch.backends.mha.set_fastpath_enabled(False)
                reference = predict(cpu, panel, pre, origin, groups, "cpu")
                for fast in (True, False):
                    torch.backends.mha.set_fastpath_enabled(fast)
                    predicted = predict(gpu, panel, pre, origin, groups, "cuda")
                    saved_columns = []
                    for h in cfg["horizons"]:
                        with np.load(trained/f"score_h{h}.npz", allow_pickle=False) as saved:
                            index = int(np.flatnonzero(saved["origins"] == origin)[0])
                            saved_columns.append(saved["predicted"].reshape(len(saved["origins"]), len(panel.beam_ids))[index, groups])
                    original = np.stack(saved_columns, axis=-1)
                    diagnostics.append({"fold": fold["name"], "mode": mode, "origin": origin,
                        "gpu_fastpath": fast, "max_error_vs_cpu": float(np.abs(predicted-reference).max()),
                        "max_error_vs_original": float(np.abs(predicted-original).max()),
                        "gpu_predictions": predicted.tolist(), "cpu_predictions": reference.tolist()})
            del cpu, gpu
            torch.cuda.empty_cache()
            print(json.dumps({"fold": fold["name"], "mode": mode,
                "backend_diagnosis_complete": True}), flush=True)
    evidence = {"torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
                "case": args.case,
                "training_performed": False, "checkpoint_selection_changed": False,
                "tolerance": 1e-4, "diagnostics": diagnostics}
    (root/"backend_diagnostic.json").write_text(json.dumps(evidence, indent=2))
    for item in diagnostics:
        if item["gpu_fastpath"] and item["max_error_vs_original"] > 1e-5:
            raise ValueError("Original GPU inference could not be reproduced")
        if not item["gpu_fastpath"] and item["max_error_vs_cpu"] > 1e-4:
            raise ValueError("Strict math CPU/GPU parity still fails; preserve diagnostics")
    # The backend is prescribed for numerical consistency, never selected by MAE/WAPE.
    torch.backends.mha.set_fastpath_enabled(False)
    for fold in cfg["folds"]:
        for mode in cfg["modes"]:
            trained = args.trained_root/fold["name"]/mode
            pre = GRUPreprocessor.load(trained/"preprocessor.npz")
            model = CellPatchAttention(cfg, pre.mean[0], pre.scale[0], mode).to("cuda").eval()
            state = torch.load(trained/"best.pt", map_location="cuda", weights_only=True)
            model.load_state_dict(state["model"])
            directory = root/fold["name"]/mode
            directory.mkdir(parents=True, exist_ok=True)
            (directory/"replay_identity.json").write_text(json.dumps({
                "protocol": cfg, "fold": fold, "mode": mode,
                "original_checkpoint_sha256": sha256(trained/"best.pt"),
                "checkpoint_epoch": state["epoch"], "gpu_attention_fastpath": False,
                "checkpoint_selection_changed": False}, indent=2))
            rows.extend({"fold": fold["name"], "mode": mode, **r} for r in score(model, panel, pre, fold, directory))
            pd.DataFrame(rows).to_csv(f"reports/{report_prefix}_development.csv", index=False)
            del model
            torch.cuda.empty_cache()
    files = {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob("*") if p.is_file()}
    (root/"manifest.json").write_text(json.dumps({"files": files}, indent=2))
    print(json.dumps({"strict_replay_complete": True, "training_performed": False,
                      "maximum_strict_cpu_gpu_error": max(r["max_error_vs_cpu"] for r in diagnostics if not r["gpu_fastpath"])}), flush=True)


if __name__ == "__main__":
    main()
