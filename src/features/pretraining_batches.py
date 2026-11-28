"""Source-balanced univariate windows from verified external training segments."""

import json
import copy
from pathlib import Path

import numpy as np
import torch

from src.data.io import sha256
from src.features.pretraining_shards import iter_sharded_segments


class SampledTargetBatches:
    """Fixed-count replacement sampling over an already fold-bounded target batcher."""

    def __init__(self, base, samples):
        if samples < 1 or not len(base.pairs):
            raise ValueError("Positive samples and eligible target pairs required")
        self.base, self.n_samples, self.device = base, samples, base.device

    def batches(self, batch_size, shuffle_seed=None):
        rng = np.random.default_rng(0 if shuffle_seed is None else shuffle_seed)
        sampled = copy.copy(self.base)
        sampled.pairs = self.base.pairs[rng.integers(len(self.base.pairs), size=self.n_samples)]
        sampled.n_samples = self.n_samples
        yield from sampled.batches(batch_size)


class PretrainingBatches:
    """Sample collections, then entities, then valid origins without crossing gaps.

    Scaling is global per source over its already isolated training prefix.
    Entity IDs affect sampling only and never enter the model input.
    """

    def __init__(self, reports, source_weights, samples=65536, device="cpu"):
        weights = np.asarray(source_weights, dtype=float)
        if len(reports) != len(weights) or not len(weights) or not np.isfinite(weights).all():
            raise ValueError("Aligned reports and finite weights required")
        if (weights <= 0).any() or samples < 1:
            raise ValueError("Positive source weights and sample count required")
        self.weights = weights / weights.sum()
        self.device, self.n_samples = device, samples
        self.sources, self.provenance = [], []
        for report_path in reports:
            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
            path = Path(report["processed_path"])
            if sha256(path) != report["processed_sha256"]:
                raise ValueError("Prepared source checksum mismatch")
            grouped, flat = {}, []
            if report.get("format") == "sharded_prefix_v1":
                for entity, values, phase in iter_sharded_segments(report):
                    grouped.setdefault(json.dumps(entity), []).append((values, phase))
                    flat.append(values)
            else:
                with np.load(path, allow_pickle=False) as archive:
                    for segment in report["segments"]:
                        name = segment["name"]
                        values = archive[name][:, 0, 0].astype(np.float64)
                        phase = archive[name + "_phase"].astype(np.int64)
                        if (len(values) < 193 or len(values) != len(phase)
                                or not np.all(np.diff(phase) == 1)
                                or not np.isfinite(values).all() or (values < 0).any()):
                            raise ValueError("Expected finite nonnegative continuous prefix segment")
                        grouped.setdefault(json.dumps(segment["entity"]), []).append((values, phase))
                        flat.append(values)
            if not flat:
                raise ValueError("No eligible external segments")
            all_values = np.concatenate(flat)
            mean, scale = float(all_values.mean()), float(all_values.std())
            scale = scale if scale >= 1e-8 else 1.0
            entities = []
            for segments in grouped.values():
                entities.append([((values-mean).astype(np.float32)/scale, phase)
                                 for values, phase in segments])
            self.sources.append(entities)
            self.provenance.append({"report": str(report_path), "sha256": sha256(path),
                                    "mean": mean, "scale": scale, "entities": len(entities),
                                    "fit_scope": "prepared_external_training_prefix_only"})

    def batches(self, batch_size, shuffle_seed=None):
        """Generate fixed-count windows reproducibly; no training occurs here."""
        if batch_size < 1:
            raise ValueError("Positive batch size required")
        rng = np.random.default_rng(0 if shuffle_seed is None else shuffle_seed)
        for start in range(0, self.n_samples, batch_size):
            n = min(batch_size, self.n_samples-start)
            history, phases, labels = [], [], []
            for source in rng.choice(len(self.sources), size=n, p=self.weights):
                entities = self.sources[source]
                segments = entities[rng.integers(len(entities))]
                # Segment choice is weighted by eligible origins within the sampled entity.
                counts = np.array([len(values)-191 for values, _ in segments])
                values, hours = segments[rng.choice(len(segments), p=counts/counts.sum())]
                origin = rng.integers(167, len(values)-24)
                observed = values[origin-167:origin+1]
                history.append(np.stack([observed, np.zeros(168, dtype=np.float32)], axis=-1))
                angle = 2*np.pi*hours[origin]/np.array([24, 168])
                phases.append(np.stack([np.sin(angle), np.cos(angle)], axis=-1).reshape(4))
                labels.append(values[origin+np.array([1, 6, 24])])
            yield (torch.as_tensor(np.asarray(history), device=self.device),
                   torch.as_tensor(np.asarray(phases), dtype=torch.float32, device=self.device),
                   torch.as_tensor(np.asarray(labels), device=self.device))
