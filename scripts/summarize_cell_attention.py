"""Independent CPU inference and metric reconstruction of saved cell studies only."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.summarize_chronos2_lora import metrics, verify_tree
from src.data.io import sha256
from src.evaluation.origins import evaluation_origins
from src.evaluation.validation import validation_inputs
from src.features.cell_inputs import cell_groups
from src.features.gru_inputs import GRUPreprocessor
from src.models.cell_attention import CellPatchAttention
from src.models.cell_pretraining import ENCODER_PREFIXES, SourcePatchForecaster
from src.models.forecast_blend import blend_forecasts
from src.models.training_status import stopping_status
from src.utils.config import load_config


def predict_cells(model, panel, preprocessing, origin, groups):
    """Only the exact past window is transformed; future values never enter inference."""
    if origin < 167 or origin >= len(panel.time):
        raise ValueError("Valid causal origin required")
    history = preprocessing.transform(panel.values[origin-167:origin+1], "B")
    history = torch.as_tensor(history[:, groups].transpose(1, 0, 2, 3))
    phase = torch.as_tensor(preprocessing.static_at(np.full(len(groups), origin), np.arange(len(groups))))
    with torch.no_grad():
        return model(history, phase).numpy()


def verify_pretraining(root, directory, model, fold, case, cfg):
    cold_path = directory/"cold_initialization.pt"
    if not cold_path.exists():
        raise ValueError("Historical run lacks seeded cold initialization evidence; controlled transfer unproven")
    source = root/"pretraining"/(fold["name"] if case == "target_self_pretrain" else "external")
    metadata = json.loads((source/"source_meta.json").read_text())
    fixed = {**cfg, "batch_size": cfg["pretraining_batch_size"], "max_epochs": cfg["pretraining_epochs"],
             "early_stopping_patience": cfg["pretraining_epochs"]+1}
    if metadata["config"] != fixed or metadata["identity"]["case"] != case or metadata["checkpoint"] != "last.pt":
        raise ValueError("Pretraining protocol differs")
    source_cold_path = source/"source_initialization.pt"
    if (not source_cold_path.exists() or metadata.get("source_initialization_sha256") != sha256(source_cold_path)
            or metadata.get("torch_initial_seed") != cfg["seed"]):
        raise ValueError("Source cold initialization seed/evidence missing")
    source_cold = torch.load(source_cold_path, map_location="cpu", weights_only=True)
    if source_cold["seed"] != cfg["seed"] or source_cold["torch_initial_seed"] != cfg["seed"]:
        raise ValueError("Source Torch seed differs")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg["seed"])
        expected_source = SourcePatchForecaster(cfg)
    if any(not torch.equal(value, source_cold["model"][key]) for key, value in expected_source.state_dict().items()):
        raise ValueError("Source cold weights do not reproduce declared seed")
    if case == "target_self_pretrain":
        provenance = metadata["identity"]["provenance"]
        if provenance["maximum_label_hour"] != fold["fit_end"] or provenance["original_fit_end"] != fold["fit_end"]:
            raise ValueError("Self-pretraining crosses original fit cutoff")
    else:
        provenance = metadata["identity"]["provenance"]
        if provenance["target_network_used"] or [p["report"] for p in provenance["sources"]] != cfg["source_reports"]:
            raise ValueError("External source membership differs")
        for item in provenance["sources"]:
            report = json.loads(Path(item["report"]).read_text())
            if item["sha256"] != report["processed_sha256"] or sha256(Path(report["processed_path"])) != item["sha256"]:
                raise ValueError("Pretraining source payload differs")
    last_path = source/"last.pt"
    last = torch.load(last_path, map_location="cpu", weights_only=True)
    expected_steps = cfg["pretraining_epochs"]*int(np.ceil(cfg["pretraining_samples_per_epoch"]/cfg["pretraining_batch_size"]))
    steps = [int(value["step"]) for value in last["optimizer"]["state"].values() if "step" in value]
    if not steps or set(steps) != {expected_steps} or last["epoch"]+1 != cfg["pretraining_epochs"]:
        raise ValueError("Fixed pretraining optimizer budget incomplete")
    initialization = torch.load(directory/"initialization.pt", map_location="cpu", weights_only=True)
    if (initialization["source_checkpoint_sha256"] != sha256(last_path)
            or initialization["case"] != case or initialization["fold"] != fold):
        raise ValueError("Transfer initialization provenance differs")
    if initialization.get("cold_checkpoint_sha256") != sha256(cold_path) or initialization.get("seed") != cfg["seed"]:
        raise ValueError("Cold target initialization identity differs")
    expected_keys = {key for key in model.state_dict() if key.startswith(ENCODER_PREFIXES)}
    if set(initialization["transferred_encoder_keys"]) != expected_keys:
        raise ValueError("Source head/scales or incomplete encoder transferred")
    for key in expected_keys:
        if not torch.equal(initialization["model"][key], last["model"]["encoder."+key]):
            raise ValueError("Transferred encoder differs from completed source checkpoint")
    if not any(not torch.equal(last["model"]["encoder."+key], source_cold["model"]["encoder."+key]) for key in expected_keys):
        raise ValueError("Pretraining did not change any transferred encoder tensor")
    cold = torch.load(cold_path, map_location="cpu", weights_only=True)
    if cold["seed"] != cfg["seed"] or cold["torch_initial_seed"] != cfg["seed"]:
        raise ValueError("Torch seed at cold initialization differs")
    for key, value in cold["model"].items():
        if key not in expected_keys and not torch.equal(value, initialization["model"][key]):
            raise ValueError("Unexpected source transfer into target head/spatial/scales")
    return {"fold": fold["name"], "case": case, "optimizer_steps": expected_steps,
            "transferred_tensors": len(expected_keys), "source_checkpoint_sha256": sha256(last_path)}


def summarize(root, cfg, report_prefix, case=None, forecast_root=None):
    verify_tree(root)
    forecast_root = root if forecast_root is None else forecast_root
    if forecast_root != root:
        verify_tree(forecast_root)
        diagnosis = json.loads((forecast_root/"backend_diagnostic.json").read_text())
        if diagnosis["training_performed"] or diagnosis["checkpoint_selection_changed"] or diagnosis["tolerance"] != 1e-4:
            raise ValueError("Inference-only backend protocol differs")
        expected_probes = {(f["name"], m, o, fast) for f in cfg["folds"] for m in cfg["modes"]
                           for o in (f["stop_end"], f["score_end"]-24) for fast in (True, False)}
        observed_probes = {(r["fold"], r["mode"], r["origin"], r["gpu_fastpath"]) for r in diagnosis["diagnostics"]}
        if observed_probes != expected_probes or len(diagnosis["diagnostics"]) != len(expected_probes):
            raise ValueError("Backend diagnostic coverage differs")
        for probe in diagnosis["diagnostics"]:
            field, tolerance = ("max_error_vs_original", 1e-5) if probe["gpu_fastpath"] else ("max_error_vs_cpu", 1e-4)
            if not np.isfinite(probe[field]) or probe[field] > tolerance:
                raise ValueError("Backend diagnostic parity/reproducibility failed")
    panel, _ = validation_inputs(load_config("configs/data.yaml"))
    panel = replace(panel, values=panel.values[:504], time=panel.time[:504])
    groups = cell_groups(panel.beam_ids)
    chosen_groups = groups[np.unique([0, len(groups)//2, len(groups)-1])]
    blend_cfg = load_config("configs/mlp_blend.yaml")
    baseline_roots = {"hurdle": Path("artifacts/hurdle_research"), "mlp": Path("artifacts/temporal_mlp"),
                      "blend": Path("artifacts/mlp_blend")}
    pinned = {"hurdle": blend_cfg["hurdle_manifest_sha256"], "mlp": blend_cfg["mlp_manifest_sha256"],
              "blend": load_config("configs/clipped_blend.yaml")["current_manifest_sha256"]}
    for name, path in baseline_roots.items():
        if sha256(path/"manifest.json") != pinned[name]:
            raise ValueError("Pinned historical control differs")
        verify_tree(path)
    selection = json.loads((baseline_roots["blend"]/"selection.json").read_text())
    if selection["protocol"] != blend_cfg:
        raise ValueError("Current baseline selection differs")
    rows, comparisons, parity, termination, transfers = [], [], [], [], []
    for fold in cfg["folds"]:
        for mode in cfg["modes"]:
            directory = root/fold["name"]/mode
            metadata = json.loads((directory/"meta.json").read_text())
            if (metadata["protocol"] != cfg or metadata["fold"] != fold or metadata["mode"] != mode
                    or not metadata["training_complete"] or metadata["holdout_used"] or metadata["validation_504_671_used"]):
                raise ValueError("Cell study temporal/training metadata differs")
            preprocessing = GRUPreprocessor.load(directory/"preprocessor.npz")
            expected = GRUPreprocessor.fit(panel, fold["fit_end"], "B")
            if preprocessing.fit_end != fold["fit_end"] or preprocessing.beam_ids != panel.beam_ids:
                raise ValueError("Preprocessing time/beam identity differs")
            for name in ("median", "mean", "scale"):
                # NumPy reductions across hosted/local builds may differ by one float64 ULP.
                # Identity/cutoff/shape remain exact; only numeric reduction roundoff is tolerated.
                np.testing.assert_allclose(getattr(preprocessing, name), getattr(expected, name),
                                           atol=1e-12, rtol=1e-12)
            model = CellPatchAttention(cfg, preprocessing.mean[0], preprocessing.scale[0], mode).eval()
            if cfg.get("seed_all_torch_generators"):
                cold_path = directory/"cold_initialization.pt"
                cold = torch.load(cold_path, map_location="cpu", weights_only=True)
                if (cold["seed"] != cfg["seed"] or cold["torch_initial_seed"] != cfg["seed"]
                        or metadata.get("torch_initial_seed") != cfg["seed"]
                        or metadata.get("cold_initialization_sha256") != sha256(cold_path)
                        or metadata.get("gpu_attention_fastpath") is not False):
                    raise ValueError("Seeded strict-backend contract differs")
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(cfg["seed"])
                    expected_cold = CellPatchAttention(cfg, preprocessing.mean[0], preprocessing.scale[0], mode)
                if any(not torch.equal(value, cold["model"][key]) for key, value in expected_cold.state_dict().items()):
                    raise ValueError("Target cold weights do not reproduce declared seed")
            best = torch.load(directory/"best.pt", map_location="cpu", weights_only=True)
            last = torch.load(directory/"last.pt", map_location="cpu", weights_only=True)
            history = [json.loads(line) for line in (directory/"history.jsonl").read_text().splitlines()]
            training = metadata["training"]
            outcome = stopping_status(training, cfg)
            if (len(history) != training["epochs_completed"] or last["epoch"]+1 != len(history)
                    or best["epoch"] != training["best_epoch"] or last["stale"] != outcome["stale_epochs"]
                    or min(history, key=lambda r: (r["val_l1_scaled"], r["epoch"]))["epoch"] != best["epoch"]):
                raise ValueError("Saved stopping choice/termination differs")
            if sum(p.numel() for p in model.parameters()) != metadata["parameters"]:
                raise ValueError("Architecture parameter count differs")
            if case:
                transfers.append(verify_pretraining(root, directory, model, fold, case, cfg))
            model.load_state_dict(best["model"])
            forecast_directory = forecast_root/fold["name"]/mode
            if forecast_root != root:
                replay = json.loads((forecast_directory/"replay_identity.json").read_text())
                if (replay["protocol"] != cfg or replay["fold"] != fold or replay["mode"] != mode
                        or replay["original_checkpoint_sha256"] != sha256(directory/"best.pt")
                        or replay["checkpoint_epoch"] != best["epoch"] or replay["gpu_attention_fastpath"]
                        or replay["checkpoint_selection_changed"]):
                    raise ValueError("Strict replay changed selected weights or protocol")
            termination.append({"fold": fold["name"], "mode": mode, **training, **outcome})
            saved = {}
            for h in cfg["horizons"]:
                origins = evaluation_origins(fold["stop_end"], fold["score_end"], h)
                actual = panel.values[origins+h, :, 0].reshape(-1)
                with np.load(forecast_directory/f"score_h{h}.npz", allow_pickle=False) as data:
                    for key, expected_axis in (("origins", origins), ("beam_ids", panel.beam_ids), ("actual", actual)):
                        np.testing.assert_array_equal(data[key], expected_axis)
                    predicted = data["predicted"].copy()
                row = {"fold": fold["name"], "mode": mode, **metrics(actual, predicted), "horizon": h}
                rows.append(row)
                saved[h] = (origins, predicted.reshape(len(origins), len(panel.beam_ids)))
                choice = selection["choices"][str(h)]
                with np.load(baseline_roots["hurdle"]/fold["name"]/f"h{h}"/"score.npz", allow_pickle=False) as hurdle, np.load(
                        baseline_roots["mlp"]/fold["name"]/choice["variant"]/f"score_h{h}.npz", allow_pickle=False) as mlp:
                    for data in (hurdle, mlp):
                        for key, expected_axis in (("origins", origins), ("beam_ids", panel.beam_ids), ("actual", actual)):
                            np.testing.assert_array_equal(data[key], expected_axis)
                    reference = blend_forecasts(hurdle["hurdle_median"], mlp["predicted"], choice["mlp_weight"])
                    current_mae = metrics(actual, reference)["mae"]
                comparisons.append({**row, "current_mae": current_mae,
                    "relative_mae_gain_vs_current_pct": 100*(1-row["mae"]/current_mae)})
            for origin in (fold["stop_end"], fold["score_end"]-24):
                cpu = predict_cells(model, panel, preprocessing, origin, chosen_groups)
                for j, h in enumerate(cfg["horizons"]):
                    origins, prediction = saved[h]
                    gpu = prediction[np.flatnonzero(origins == origin)[0], chosen_groups]
                    np.testing.assert_allclose(cpu[:, :, j], gpu, atol=1e-4, rtol=0)
                    parity.append({"fold": fold["name"], "mode": mode, "origin": origin, "horizon": h,
                        "beams_checked": int(chosen_groups.size), "maximum_error": float(np.abs(cpu[:, :, j]-gpu).max())})
    frame = pd.DataFrame(rows)
    columns = ["fold", "mode", "horizon", "mae", "wape", "n_valid", "zero_fp_rate"]
    pd.testing.assert_frame_equal(frame[columns], pd.read_csv(f"reports/{report_prefix}_development.csv")[columns],
                                  check_dtype=False, atol=1e-12, rtol=1e-12)
    pd.DataFrame(comparisons).to_csv(f"reports/{report_prefix}_comparison.csv", index=False)
    pd.DataFrame(parity).to_csv(f"reports/{report_prefix}_cpu_parity.csv", index=False)
    pd.DataFrame(termination).to_csv(f"reports/{report_prefix}_termination.csv", index=False)
    evidence = {"all_metrics_reconstructed": True, "fit_only_preprocessing_verified": True,
        "cpu_torch": str(torch.__version__), "preprocessing_reduction_tolerance": 1e-12,
        "forecast_root": forecast_root.as_posix(), "original_checkpoint_selection_changed": False,
        "controlled_neural_initialization_verified": bool(cfg.get("seed_all_torch_generators")),
        "stopping_choices_verified": True, "history_only_beam_predictions_checked": sum(r["beams_checked"] for r in parity),
        "maximum_cpu_gpu_error": max(r["maximum_error"] for r in parity),
        "epoch_capped_runs": sum(r["termination_reason"] == "epoch_cap" for r in termination),
        "pretraining_transfer": transfers, "holdout_used": False, "production_promoted": False,
        "claim_scope": "reused development; fresh generalization unproven"}
    Path(f"reports/{report_prefix}_verification.json").write_text(json.dumps(evidence, indent=2))
    print(pd.DataFrame(comparisons).to_string(index=False))
    print(json.dumps(evidence, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multisource", action="store_true")
    parser.add_argument("--precision", action="store_true", help="Verify separate strict-backend forecasts against original weights")
    parser.add_argument("--seeded", action="store_true", help="Verify seeded strict-backend v2 studies")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    if args.seeded and args.precision:
        raise ValueError("Seeded v2 already uses the strict inference backend")
    if args.multisource:
        if args.precision:
            raise ValueError("Multi-corpus strict replay has not been launched")
        study = "cell_multisource_v2" if args.seeded else "cell_multisource"
        verify_tree(Path("artifacts")/study)
        cfg = load_config(f"configs/{study}.yaml")
        for case in cfg["cases"]:
            summarize(Path("artifacts")/study/case, cfg, f"{study}_{case}", case)
    elif args.precision:
        summarize(Path("artifacts/cell_attention"), load_config("configs/cell_attention.yaml"),
                  "cell_attention_precision", forecast_root=Path("artifacts/cell_attention_precision"))
    else:
        study = "cell_attention_v2" if args.seeded else "cell_attention"
        summarize(Path("artifacts")/study, load_config(f"configs/{study}.yaml"), study)


if __name__ == "__main__":
    main()
