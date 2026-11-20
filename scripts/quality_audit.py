"""Audit current artifacts against Week 5 causal/preprocessing contracts without fitting."""

import json
from pathlib import Path

import numpy as np
import torch

from src.evaluation.origins import evaluation_origins, train_origins
from src.evaluation.validation import validation_inputs
from src.features.lightgbm_features import FEATURE_NAMES
from src.features.gru_inputs import GRUPreprocessor, SCHEMA
from src.models.lgbm import load_lgbm
from src.data.io import sha256
from src.utils.config import load_config


def audit(data_path: str = "configs/data.yaml", write: bool = True) -> dict:
    """Require all rerun artifacts, exact schema, provenance, origin counts and preprocessing parity."""
    data, lgb_config, gru_config = (
        load_config(p) for p in (data_path, "configs/lightgbm.yaml", "configs/gru.yaml")
    )
    panel, split = validation_inputs(data)
    checks, digests, preprocessors, samples = [], {}, [], []

    def check(label: str, condition: bool, detail: str = "") -> None:
        checks.append(
            {
                "check": label.replace("\\", "/"),
                "status": "PASS" if condition else "FAIL",
                "detail": detail,
            }
        )

    check(
        "feature_schema",
        len(FEATURE_NAMES) == 40 and not any("beam_fit" in n for n in FEATURE_NAMES),
    )
    for config_name in lgb_config["configs"]:
        for h in split["horizons"]:
            directory = Path(lgb_config["artifact_dir"]) / config_name / f"h{h}"
            try:
                model = load_lgbm(directory)
                meta = model.metadata
                check(
                    str(directory),
                    meta["dataset_revision"] == data["revision"]
                    and meta["config"] == lgb_config["configs"][config_name]
                    and meta["seed"] == lgb_config["seed"]
                    and bool(meta["git_commit"])
                    and meta["fit_end"] == split["fit"][1]
                    and meta["feature_names"] == FEATURE_NAMES,
                )
                for name, origins, count_key, missing_key in (
                    (
                        "train",
                        train_origins(split["fit"][1], h, stride=lgb_config["training_stride"]),
                        "train_rows",
                        "train_missing_targets",
                    ),
                    (
                        "validation",
                        evaluation_origins(split["fit"][1], split["validation"][1], h),
                        "validation_rows",
                        "validation_missing_targets",
                    ),
                ):
                    expected = np.isfinite(panel.values[origins + h, :, 0]).sum()
                    check(
                        f"{directory}/{name}_samples",
                        meta[count_key] == int(expected)
                        and meta[missing_key] == len(origins) * len(panel.beam_ids) - int(expected),
                    )
                for filename in ("model.txt", "meta.json", "context.npz"):
                    digests[(directory / filename).as_posix()] = sha256(directory / filename)
            except (OSError, ValueError, KeyError) as exc:
                check(str(directory), False, str(exc))
    for variant in gru_config["variants"]:
        for seed in gru_config["seeds"]:
            directory = Path(gru_config["artifact_dir"]) / variant / f"seed{seed}"
            try:
                meta = json.loads((directory / "meta.json").read_text())
                prep = GRUPreprocessor.load(directory / "preprocessor.npz")
                state = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
                k = 1 if variant == "A" else 4
                check(
                    str(directory),
                    meta.get("preprocessing_schema") == SCHEMA
                    and meta["dataset_revision"] == data["revision"]
                    and bool(meta["git_commit"])
                    and meta["seed"] == seed
                    and state["identity"]["seed"] == seed
                    and state["identity"]["run_identity"]["variant"] == variant
                    and prep.variant == variant
                    and prep.fit_end == split["fit"][1]
                    and meta["phase_features"] == 4
                    and state["model"]["head.0.weight"].shape[1]
                    == meta["architecture"]["hidden_size"] + 4
                    and state["model"]["gru.weight_ih_l0"].shape[1] == 2 * k
                    and all(getattr(prep, p).shape == (k,) for p in ("median", "mean", "scale")),
                )
                identity = state["identity"]["run_identity"]
                samples.append(identity["beam_sample"])
                preprocessors.append((prep.median[0], prep.mean[0], prep.scale[0]))
                from src.evaluation.origins import train_origins_joint

                expected_train = len(
                    train_origins_joint(split["fit"][1], stride=gru_config["training_stride"])
                ) * len(samples[-1])
                expected_val = len(
                    evaluation_origins(split["fit"][1], split["validation"][1], 24)
                ) * len(panel.beam_ids)
                check(
                    f"{directory}/samples",
                    meta["train_samples"] == expected_train
                    and meta["validation_joint_samples"] == expected_val,
                )
                for filename in (
                    "best.pt",
                    "last.pt",
                    "preprocessor.npz",
                    "meta.json",
                    "history.jsonl",
                ):
                    digests[(directory / filename).as_posix()] = sha256(directory / filename)
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                check(str(directory), False, str(exc))
    check(
        "shared_sample_and_DLPRB_statistics",
        bool(samples)
        and all(x == samples[0] for x in samples)
        and all(x == preprocessors[0] for x in preprocessors),
    )
    status = "PASS" if checks and all(row["status"] == "PASS" for row in checks) else "FAIL"
    result = {
        "status": status,
        "checks": checks,
        "artifact_sha256": digests,
        "dataset_revision": data["revision"],
        "holdout_evaluated": False,
        "p1": "skipped: legacy schemas required retraining",
    }
    if write:
        reports = Path(data["reports_dir"])
        (reports / "model_quality_gate.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8", newline="\n"
        )
        lines = [
            "# Week 5 model quality gate",
            "",
            f"Status: **{status}**.",
            "Legacy W3/W4 artifacts FAILED the updated plan (44/static6); all were retrained on cloud.",
            "No inference-only patch was used to approve legacy artifacts. P1 quantiles skipped.",
            "Normalization is fitted once at the cutoff, globally per KPI; earlier fit origins are",
            "training samples, not separate deployed episodes. No holdout has been evaluated.",
            "",
        ]
        lines += [f"- {r['status']}: {r['check']} {r['detail']}" for r in checks]
        lines += [
            "",
            "Hashes and provenance: model_quality_gate.json; initial failures: model_quality_gate_initial.json.",
        ]
        (reports / "model_quality_gate.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    result = audit()
    print(result["status"])
    if result["status"] != "PASS":
        raise SystemExit(1)
