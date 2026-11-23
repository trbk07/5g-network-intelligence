"""Review Run All must remain read-only without requiring fitting packages."""

import contextlib
import io
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("name,flag", [
    ("13_zero_inflated_forecasting.ipynb", "RUN_TRAINING"),
    ("14_zero_inflated_candidate_audit.ipynb", "RUN_TRAINING"),
    ("15_probability_calibration_study.ipynb", "RUN_CALIBRATION"),
    ("16_temporal_linear_forecasting.ipynb", "RUN_TRAINING"),
    ("17_development_forecast_blending.ipynb", "RUN_SELECTION"),
    ("18_fixed_blend_candidate_audit.ipynb", "RUN_TRAINING"),
    ("19_pretrained_zero_shot_forecasting.ipynb", "RUN_INFERENCE"),
    ("20_temporal_residual_mlp.ipynb", "RUN_TRAINING"),
    ("21_positive_quantile_resolution.ipynb", "RUN_TRAINING"),
    ("22_mlp_hurdle_forecast_blending.ipynb", "RUN_SELECTION"),
    ("23_fixed_mlp_blend_candidate_audit.ipynb", "RUN_TRAINING"),
    ("24_historical_exceedance_probability.ipynb", "RUN_TRAINING"),
])
def test_hurdle_review_is_readable_and_does_not_train(name, flag, tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root / "notebooks" / name).read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert all("base64" not in c and "%%writefile" not in c for c in code)
    monkeypatch.chdir(tmp_path)
    scope = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for cell in code:
            exec(compile(cell, "hurdle_review", "exec"), scope)
    assert scope[flag] is False
    assert not list(tmp_path.rglob("*"))
