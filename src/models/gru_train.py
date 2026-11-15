"""Masked L1 training with resumable optimizer/RNG state and per-epoch checkpoints."""
import json
import os
import time
from pathlib import Path

import torch

from src.features.gru_inputs import GRUBatches
from src.models.gru import GRUNet


def masked_l1(predicted: torch.Tensor, actual: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Return mean absolute error among finite targets and number of valid targets."""
    mask = torch.isfinite(actual)
    count = int(mask.sum())
    if not count:
        return predicted.sum() * 0, 0
    return torch.abs(predicted[mask] - actual[mask]).mean(), count


def _save(state: dict, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def train_gru(model: GRUNet, train: GRUBatches, validation: GRUBatches, config: dict,
               directory: str | Path, seed: int, resume: bool = False,
               allow_cpu_test: bool = False, run_identity: dict | None = None,
               seconds_limit: float | None = None) -> dict:
    """Train with AdamW, clipping, early stopping; reject full training on CPU."""
    if train.device == "cpu" and not allow_cpu_test:
        raise ValueError("Full GRU training requires GPU; CPU is allowed only for small tests")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.to(train.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    start, best, best_epoch, stale = 0, float("inf"), -1, 0
    ignored = {"max_epochs", "variants", "seeds", "artifact_dir", "gpu_budget_seconds"}
    identity = {"seed": seed, "config": {k: v for k, v in config.items() if k not in ignored},
                "run_identity": run_identity}
    if resume:
        state = torch.load(directory / "last.pt", map_location=train.device, weights_only=True)
        state["identity"]["config"] = {k: v for k, v in state["identity"]["config"].items()
                                           if k not in ignored}
        if state["identity"] != identity:
            raise ValueError("Resume identity differs from checkpoint")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"].cpu())
        if train.device != "cpu":
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
        start, best, best_epoch, stale = state["epoch"]+1, state["best"], state["best_epoch"], state["stale"]
    completed = start
    run_started = time.perf_counter()
    budget_exhausted = False
    for epoch in range(start, config["max_epochs"]):
        if stale >= config["early_stopping_patience"]:
            break
        began = time.perf_counter()
        model.train()
        total, count = 0., 0
        for sequence, static, target in train.batches(config["batch_size"], seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            loss, n = masked_l1(model(sequence, static), target)
            if not n:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            total += float(loss.detach()) * n
            count += n
        model.eval()
        val_total, val_count = 0., 0
        with torch.no_grad():
            for sequence, static, target in validation.batches(config["batch_size"]):
                loss, n = masked_l1(model(sequence, static), target)
                val_total += float(loss) * n
                val_count += n
        if not count or not val_count:
            raise ValueError("No observed training/validation targets")
        score = val_total / val_count
        improved = score < best
        best, best_epoch, stale = (score, epoch, 0) if improved else (best, best_epoch, stale + 1)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                 "best": best, "best_epoch": best_epoch, "stale": stale, "identity": identity,
                 "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()
                 if train.device != "cpu" else []}
        _save(state, directory / "last.pt")
        if improved:
            _save(state, directory / "best.pt")
        row = {"epoch": epoch, "train_l1_scaled": total/count, "val_l1_scaled": score,
               "seconds": time.perf_counter()-began, "best_epoch": best_epoch,
               "train_valid_targets": count, "validation_valid_targets": val_count,
               "peak_vram_bytes": torch.cuda.max_memory_allocated() if train.device != "cpu" else 0}
        with (directory / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row)+"\n")
        print(json.dumps(row), flush=True)
        completed = epoch + 1
        if seconds_limit is not None and time.perf_counter() - run_started >= seconds_limit:
            budget_exhausted = True
            break
    return {"best_epoch": best_epoch, "best_val_l1_scaled": best,
            "epochs_completed": completed, "budget_exhausted": budget_exhausted}
