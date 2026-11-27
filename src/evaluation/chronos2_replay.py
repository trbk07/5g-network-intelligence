"""History-only median replay and immutable per-origin inference caches."""

import hashlib
import json
from pathlib import Path

import numpy as np

from src.features.chronos2_inputs import history_tasks, native_tasks


def predict_origin(pipeline, values, origin, batch_size=256):
    """Forecast every original beam, with one target and past-only KPI covariates."""
    history = history_tasks(values, origin, np.arange(values.shape[1]))
    levels = list(pipeline.quantiles)
    if .5 not in levels:
        raise ValueError("Pinned pipeline must expose its native median")
    median_index = levels.index(.5)
    predictions = pipeline.predict(native_tasks(history), prediction_length=24,
                                   context_length=168, batch_size=batch_size, cross_learning=False)
    if len(predictions) != history.shape[0]:
        raise ValueError("Foundation forecast dropped or added beam tasks")
    medians = []
    for prediction in predictions:
        array = prediction.detach().cpu().numpy() if hasattr(prediction, "detach") else np.asarray(prediction)
        if array.shape != (1, len(levels), 24):
            raise ValueError("Expected one DLPRB target, native quantiles and 24 steps")
        medians.append(array[0, median_index])
    result = np.maximum(np.stack(medians), 0).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite forecast cannot shrink original beam coverage")
    return result


def cached_origin(pipeline, values, origin, directory, model_identity, batch_size=256):
    """Reuse only an exact model/history identity; never overwrite a different cache."""
    history = history_tasks(values, origin, np.arange(values.shape[1]))
    identity = {"model": model_identity, "origin": int(origin), "beams": values.shape[1],
                "history_sha256": hashlib.sha256(history.tobytes()).hexdigest(),
                "context_length": 168, "prediction_length": 24, "cross_learning": False}
    encoded = json.dumps(identity, sort_keys=True)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f"origin_{origin}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as cache:
            if str(cache["identity_json"]) != encoded:
                raise ValueError("Existing prediction cache has a different model/history identity")
            predicted = cache["predicted"].copy()
        if predicted.shape != (values.shape[1], 24) or not np.isfinite(predicted).all():
            raise ValueError("Invalid cached forecast population")
        return predicted
    result = predict_origin(pipeline, values, origin, batch_size)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, predicted=result, identity_json=np.array(encoded))
    temporary.replace(path)
    return result
