"""Read-only inference of a fixed robust reference on development only."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.origins import evaluation_origins
from src.evaluation.validation import validation_inputs
from src.models.seasonal_median import SeasonalMedian
from src.utils.config import load_config


def main():
    panel,_ = validation_inputs(load_config("configs/data.yaml"))
    cfg = load_config("configs/hurdle_research.yaml")
    model = SeasonalMedian()
    rows = []
    for fold in cfg["folds"]:
        if fold["score_end"] > 503:
            raise ValueError("Development-only inference required")
        for h in cfg["horizons"]:
            origins = evaluation_origins(fold["stop_end"],fold["score_end"],h)
            actual = panel.values[origins+h,:,0].reshape(-1)
            predicted = np.concatenate([model.predict(panel.values[o-167:o+1],h) for o in origins])
            mask = np.isfinite(actual)
            if not np.isfinite(predicted[mask]).all():
                raise ValueError("Missing predictions cannot shrink the observed population")
            with np.load(Path("artifacts/hurdle_research")/fold["name"]/f"h{h}/score.npz",allow_pickle=False) as reference:
                for key,value in (("actual",actual),("origins",origins),("beam_ids",panel.beam_ids)):
                    np.testing.assert_array_equal(reference[key],value)
                baseline = float(np.abs(reference["hurdle_median"][mask]-actual[mask]).mean())
            error = np.abs(predicted[mask]-actual[mask])
            rows.append({"fold":fold["name"],"horizon":h,"model":model.name,"mae":float(error.mean()),
                         "wape":float(error.sum()/np.abs(actual[mask]).sum()),"n_valid":int(mask.sum()),
                         "hurdle_mae":baseline,"relative_mae_gain_pct":100*(1-error.mean()/baseline)})
    frame = pd.DataFrame(rows)
    frame.to_csv("reports/seasonal_median_development.csv",index=False)
    Path("reports/seasonal_median_protocol.json").write_text(json.dumps({
        "formula":"median of raw target at origin+h-24*d for d=1..7",
        "history":168,"parameters_fitted_or_selected":False,"scope":"read-only analytical inference",
        "evaluation":"all original observed development targets; fail on missing forecasts",
        "validation_or_holdout_used":False,"production_promoted":False,
        "claim_scope":"reused development reference; no unseen-generalization or error-floor claim"
    },indent=2))
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
