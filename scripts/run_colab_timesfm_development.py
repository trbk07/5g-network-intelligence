"""Hosted, resumable TimesFM median screen; no training or future covariates."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import urllib.request


def main():
    """Keep pretrained weights and causal origin caches on Drive."""
    if os.name == "nt" or "COLAB_RELEASE_TAG" not in os.environ:
        raise RuntimeError("Hosted Colab only")
    project = Path("/content/cell_seed29")
    os.chdir(project)
    sys.path.insert(0, str(project))
    import numpy as np
    import yaml
    cfg = yaml.safe_load(Path("/content/timesfm_development.yaml").read_text())
    source_commit = os.environ["TIMESFM_SOURCE_COMMIT"]
    if len(source_commit) != 40:
        raise ValueError("Exact source commit required")
    drive = Path("/content/drive/MyDrive")
    if not drive.is_dir():
        raise RuntimeError("Mounted Drive required before acquiring weights")
    root = drive/"5g-network-intelligence/checkpoints/timesfm25_development_v1"
    root.mkdir(parents=True, exist_ok=True)
    identity = root/"protocol.json"
    if identity.exists() and json.loads(identity.read_text()) != cfg:
        raise ValueError("Existing protocol differs; never overwrite a different study")
    identity.write_text(json.dumps(cfg, indent=2))
    wheel = root/"timesfm-2.0.2-py3-none-any.whl"
    if not wheel.exists():
        metadata = json.load(urllib.request.urlopen("https://pypi.org/pypi/timesfm/2.0.2/json", timeout=60))
        item = next(x for x in metadata["urls"] if x["filename"] == wheel.name)
        if item["digests"]["sha256"] != cfg["wheel_sha256"]:
            raise ValueError("Publisher wheel identity differs")
        wheel.write_bytes(urllib.request.urlopen(item["url"], timeout=120).read())
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != cfg["wheel_sha256"]:
        raise ValueError("Wheel checksum mismatch")
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", str(wheel)], check=True)
    import torch
    import timesfm
    from huggingface_hub import snapshot_download
    from src.data.io import sha256
    from src.data.panel import load_panel
    if not torch.cuda.is_available():
        raise RuntimeError("Hosted GPU required")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    panel = load_panel("data/processed/panel.npz")
    if not np.isnan(panel.values[504:]).all():
        raise ValueError("Original future observations must remain redacted")
    public = json.loads(Path("public_sources.json").read_text())
    if hashlib.sha256(panel.values[:504].tobytes()).hexdigest() != public["prefix_values_sha256"]:
        raise ValueError("Canonical original prefix differs")
    finished = threading.Event()

    def heartbeat():
        while not finished.wait(20):
            print("[TimesFM] Hosted inference active; weights and origin caches on Drive.", flush=True)

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        model_path = snapshot_download(cfg["model_id"], revision=cfg["revision"], token=False,
            cache_dir=str(root/"hf_cache"), allow_patterns=["config.json", "model.safetensors", "README.md"])
        if sha256(Path(model_path)/"model.safetensors") != cfg["weights_sha256"]:
            raise ValueError("Pinned pretrained checkpoint checksum differs")
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_path, torch_compile=False, local_files_only=True)
        model.compile(timesfm.ForecastConfig(
            max_context=cfg["max_context"], max_horizon=cfg["max_horizon"],
            per_core_batch_size=cfg["batch_size"], normalize_inputs=cfg["normalize_inputs"],
            force_flip_invariance=cfg["force_flip_invariance"], infer_is_positive=cfg["infer_is_positive"],
            use_continuous_quantile_head=cfg["use_continuous_quantile_head"],
            fix_quantile_crossing=cfg["fix_quantile_crossing"]))
        print(json.dumps({"status": "model_loaded", "torch": str(torch.__version__),
                          "revision": cfg["revision"], "weights_sha256": cfg["weights_sha256"]}), flush=True)
        (root/"runtime.json").write_text(json.dumps({"source_commit": source_commit,
            "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
            "training_performed": False, "weights_and_caches_storage": "Drive"}, indent=2))
        summary = []
        for fold in cfg["folds"]:
            for origin in range(fold["stop_end"], fold["score_end"]):
                output = root/fold["name"]/f"origin_{origin}.npz"
                output.parent.mkdir(exist_ok=True)
                if output.exists():
                    with np.load(output, allow_pickle=False) as previous:
                        if int(previous["origin"]) != origin or previous["predicted"].shape != (2880, 3):
                            raise ValueError("Existing origin cache differs")
                    continue
                history = panel.values[origin-167:origin+1, :, 0].T
                if not np.isfinite(history).all():
                    raise ValueError("Missing history needs a separately declared causal policy")
                # Passing only this origin's observed values prevents future leakage.
                with torch.inference_mode():
                    point, quantiles = model.forecast(horizon=24, inputs=[row.copy() for row in history])
                np.testing.assert_array_equal(point, quantiles[:, :, 5])
                predicted = point[:, np.array(cfg["horizons"])-1]
                if predicted.shape != (2880, 3) or not np.isfinite(predicted).all() or (predicted < 0).any():
                    raise ValueError("Invalid full-population forecasts")
                partial = output.with_suffix(".part")
                with partial.open("wb") as handle:
                    np.savez_compressed(handle, origin=origin, predicted=predicted, beam_ids=panel.beam_ids)
                partial.replace(output)
                print(json.dumps({"fold": fold["name"], "origin_completed": origin}), flush=True)
            for j, h in enumerate(cfg["horizons"]):
                actual, predicted = [], []
                for origin in range(fold["stop_end"], fold["score_end"]-h+1):
                    with np.load(root/fold["name"]/f"origin_{origin}.npz", allow_pickle=False) as cache:
                        np.testing.assert_array_equal(cache["beam_ids"], panel.beam_ids)
                        predicted.append(cache["predicted"][:, j])
                    actual.append(panel.values[origin+h, :, 0])
                actual, predicted = np.concatenate(actual), np.concatenate(predicted)
                mask = np.isfinite(actual)
                errors = np.abs(predicted[mask]-actual[mask])
                summary.append({"fold": fold["name"], "horizon": h, "mae": float(errors.mean()),
                                "wape": float(errors.sum()/np.abs(actual[mask]).sum()), "n_valid": int(mask.sum())})
            (root/"aggregate_metrics.json").write_text(json.dumps({"protocol": cfg, "metrics": summary,
                "training_performed": False, "holdout_used": False, "pretraining_overlap_certified_absent": False}, indent=2))
        print(json.dumps({"status": "complete", "metrics": summary}), flush=True)
    finally:
        finished.set()


if __name__ == "__main__":
    main()
