"""Independently reconstruct saved development evidence; never fit or select anew."""

import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from src.data.io import sha256
from src.evaluation.origins import evaluation_origins
from src.evaluation.validation import validation_inputs
from src.features.chronos2_inputs import history_tasks
from src.models.forecast_blend import blend_forecasts
from src.utils.config import load_config


def verify_tree(root):
    manifest = json.loads((root/"manifest.json").read_text())
    for name, digest in manifest["files"].items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or ":" in name or "\\" in name:
            raise ValueError("Unsafe manifest path")
        if not (root/name).resolve().is_relative_to(root.resolve()) or sha256(root/name) != digest:
            raise ValueError("Artifact checksum differs: " + name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Unmanifested artifact files")
    return manifest


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.shape != predicted.shape or not np.isfinite(predicted).all() or (predicted < 0).any():
        raise ValueError("Forecast shape, coverage or nonnegativity differs")
    mask = np.isfinite(actual)
    denominator = float(np.abs(actual[mask]).sum())
    if not mask.any() or denominator <= 0:
        raise ValueError("Positive observed target denominator required")
    error = np.abs(predicted[mask]-actual[mask])
    zero = mask & (actual == 0)
    return {"mae": float(error.mean()), "wape": float(error.sum()/denominator),
            "n_valid": int(mask.sum()), "zero_fp_rate": float((predicted[zero] > 0).mean())}


def cached_forecast(path, values, origin, expected_model):
    with np.load(path, allow_pickle=False) as data:
        identity = json.loads(str(data["identity_json"]))
        predicted = data["predicted"].copy()
    history = history_tasks(values, origin, np.arange(values.shape[1]))
    expected = {"model": expected_model, "origin": origin, "beams": values.shape[1],
                "history_sha256": hashlib.sha256(history.tobytes()).hexdigest(),
                "context_length": 168, "prediction_length": 24, "cross_learning": False}
    if identity != expected or predicted.shape != (values.shape[1], 24):
        raise ValueError("Inference cache model/history identity differs")
    if not np.isfinite(predicted).all() or (predicted < 0).any():
        raise ValueError("Invalid cached forecasts")
    return predicted


def main():
    cfg = load_config("configs/chronos2_lora.yaml")
    panel, _ = validation_inputs(load_config("configs/data.yaml"))
    values = panel.values[:504]
    root, trained = Path("artifacts/chronos2_lora_evaluation"), Path("artifacts/chronos2_lora")
    verify_tree(root)
    verify_tree(trained)
    identity = json.loads((root/"evaluation_identity.json").read_text())
    model_cfg = load_config(cfg["model_config"])
    if (identity["protocol"] != cfg or identity["training_manifest_sha256"] != sha256(trained/"manifest.json")
            or identity["base_sha256"] != model_cfg["weights_sha256"]
            or identity["model_revision"] != model_cfg["revision"]
            or identity["panel_prefix_sha256"] != hashlib.sha256(values.tobytes()).hexdigest()
            or identity["beam_ids_sha256"] != hashlib.sha256(json.dumps(panel.beam_ids).encode()).hexdigest()):
        raise ValueError("Evaluation identity differs from local development prefix")
    controls = {"hurdle": Path("artifacts/hurdle_research"),
                "mlp": Path("artifacts/temporal_mlp"), "blend": Path("artifacts/mlp_blend")}
    blend_cfg = load_config("configs/mlp_blend.yaml")
    pinned = {"hurdle": blend_cfg["hurdle_manifest_sha256"], "mlp": blend_cfg["mlp_manifest_sha256"],
              "blend": load_config("configs/clipped_blend.yaml")["current_manifest_sha256"]}
    for key, path in controls.items():
        if sha256(path/"manifest.json") != pinned[key]:
            raise ValueError("Historical control manifest differs")
        verify_tree(path)
    blend_selection = json.loads((controls["blend"]/"selection.json").read_text())
    if blend_selection["protocol"] != blend_cfg:
        raise ValueError("Stored historical blend protocol differs")
    choices = blend_selection["choices"]
    rows, stopping_rows, comparison, parity = [], [], [], []
    for fold in cfg["folds"]:
        directory = root/fold["name"]
        metadata = json.loads((trained/fold["name"]/"training_protocol.json").read_text())
        if (metadata["protocol"] != cfg or metadata["fold"] != fold
                or metadata["maximum_input_hour"] != fold["fit_end"]
                or metadata["input_shape"] != [len(panel.beam_ids), 4, fold["fit_end"]+1]
                or not metadata["fixed_step_training_complete"] or metadata["holdout_used"]
                or metadata["validation_504_671_used"] or metadata["torch"] != identity["torch"]):
            raise ValueError("Training prefix/completion protocol differs")
        selection = json.loads((directory/"selection.json").read_text())
        if selection["protocol"] != cfg or selection["fold"] != fold or selection["selection_data"] != "stopping_only":
            raise ValueError("Stored stopping selection protocol differs")
        candidates = []
        for step in cfg["selection_checkpoints"]:
            adapter = trained/fold["name"]/"trainer"/f"checkpoint-{step}"
            state = json.loads((adapter/"trainer_state.json").read_text())
            if state["global_step"] != step or state["max_steps"] != cfg["max_steps"]:
                raise ValueError("Training optimizer-step identity differs")
            model_identity = {**identity, "adapter_sha256": sha256(adapter/"adapter_model.safetensors")}
            total, count = 0., 0
            for origin in evaluation_origins(fold["fit_end"], fold["stop_end"], 24):
                origin = int(origin)
                prediction = cached_forecast(directory/f"stopping_step{step}"/f"origin_{origin}.npz",
                                             values, origin, model_identity)
                actual = values[origin+np.array([1, 6, 24]), :, 0].T
                mask = np.isfinite(actual)
                total += float(np.abs(prediction[:, [0, 5, 23]][mask]-actual[mask]).sum())
                count += int(mask.sum())
            candidates.append({"fold": fold["name"], "step": step, "mae": total/count, "n_valid": count})
        pd.testing.assert_frame_equal(pd.DataFrame(candidates), pd.DataFrame(selection["stopping_rows"]),
                                      check_dtype=False, atol=1e-12, rtol=1e-12)
        expected_step = min(candidates, key=lambda r: (r["mae"], r["step"]))["step"]
        if selection["selected_step"] != expected_step:
            raise ValueError("Stored checkpoint does not match stopping-only rule")
        stopping_rows.extend(candidates)
        for stage in ("zero_shot", "selected_lora"):
            output = directory/stage
            adapter = trained/fold["name"]/"trainer"/f"checkpoint-{expected_step}"
            model_identity = {**identity, "adapter_sha256": sha256(adapter/"adapter_model.safetensors") if stage == "selected_lora" else None}
            for h in cfg["horizons"]:
                origins = evaluation_origins(fold["stop_end"], fold["score_end"], h)
                actual = values[origins+h, :, 0].reshape(-1)
                with np.load(output/f"score_h{h}.npz", allow_pickle=False) as data:
                    np.testing.assert_array_equal(data["origins"], origins)
                    np.testing.assert_array_equal(data["beam_ids"], panel.beam_ids)
                    np.testing.assert_array_equal(data["actual"], actual)
                    predicted = data["predicted"].copy()
                cached = np.concatenate([cached_forecast(output/"cache"/f"origin_{o}.npz", values,
                                         int(o), model_identity)[:, h-1] for o in origins])
                np.testing.assert_array_equal(predicted, cached)
                row = {"fold": fold["name"], "stage": stage, "horizon": h, **metrics(actual, predicted)}
                rows.append(row)
                choice = choices[str(h)]
                with np.load(controls["hurdle"]/fold["name"]/f"h{h}"/"score.npz", allow_pickle=False) as hurdle, np.load(
                        controls["mlp"]/fold["name"]/choice["variant"]/f"score_h{h}.npz", allow_pickle=False) as mlp:
                    for control in (hurdle, mlp):
                        for key, expected in (("actual", actual), ("origins", origins), ("beam_ids", panel.beam_ids)):
                            np.testing.assert_array_equal(control[key], expected)
                    hurdle_mae = metrics(actual, hurdle["hurdle_median"])["mae"]
                    # Replay the existing frozen choice; no local search or fitting.
                    reference = blend_forecasts(hurdle["hurdle_median"], mlp["predicted"], choice["mlp_weight"])
                    current_mae = metrics(actual, reference)["mae"]
                    if fold["name"] == "inner2":
                        with np.load(controls["blend"]/f"confirmation_h{h}.npz", allow_pickle=False) as fixed:
                            np.testing.assert_array_equal(fixed["predicted"], reference)
                comparison.append({**row, "hurdle_mae": hurdle_mae, "current_mae": current_mae,
                                   "relative_mae_gain_vs_current_pct": 100*(1-row["mae"]/current_mae)})
            report = json.loads((output/"cpu_parity.json").read_text())
            if report["tolerance"] != 1e-4 or len(report["rows"]) != 2:
                raise ValueError("CPU parity contract differs")
            if [r["origin"] for r in report["rows"]] != [fold["stop_end"], fold["score_end"]-24]:
                raise ValueError("CPU parity origins differ")
            for item in report["rows"]:
                expected_beams = [panel.beam_ids[b] for b in (0, len(panel.beam_ids)//2, len(panel.beam_ids)-1)]
                if item["beam_ids"] != expected_beams:
                    raise ValueError("CPU parity beam coverage differs")
                beams = [panel.beam_ids.index(b) for b in item["beam_ids"]]
                saved = cached_forecast(output/"cache"/f"origin_{item['origin']}.npz", values,
                                        item["origin"], model_identity)[beams]
                error = float(np.abs(np.asarray(item["cpu_predictions"])-saved).max())
                if error > 1e-4 or not np.isclose(error, item["maximum_error"], atol=1e-12, rtol=0):
                    raise ValueError("CPU/GPU parity failed")
                parity.append(error)
    for records, name in ((rows, "development"), (stopping_rows, "stopping")):
        pd.testing.assert_frame_equal(pd.DataFrame(records), pd.read_csv(f"reports/chronos2_lora_{name}.csv"),
                                      check_dtype=False, atol=1e-12, rtol=1e-12)
    frame = pd.DataFrame(comparison)
    frame.to_csv("reports/chronos2_lora_control_comparison.csv", index=False)
    gains = frame.loc[frame.stage == "selected_lora", "relative_mae_gain_vs_current_pct"]
    evidence = {"all_metrics_reconstructed": True, "stopping_choices_verified": True,
                "all_frozen_axes_and_cache_histories_verified": True,
                "maximum_cpu_gpu_error": max(parity), "parity_origins_checked": len(parity),
                "passes_initial_research_gate": bool((gains >= 0).all() and gains.mean() >= 2),
                "holdout_used": False, "validation_504_671_used": False, "production_promoted": False,
                "claim_scope": "one-seed reused development; independent generalization unproven"}
    Path("reports/chronos2_lora_verification.json").write_text(json.dumps(evidence, indent=2))
    print(frame.to_string(index=False))
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
