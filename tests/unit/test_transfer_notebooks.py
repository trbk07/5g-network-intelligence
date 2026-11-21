"""Research presentation and safe default execution; no cloud or fitting calls."""

import ast
import contextlib
import io
import json
from pathlib import Path

import pytest


NAMES = [
    "09_external_data_quality_audit.ipynb",
    "10_single_source_transfer_experiments.ipynb",
    "11_multisource_transfer_ablation.ipynb",
    "12_transfer_results_and_error_analysis.ipynb",
]


@pytest.mark.parametrize("name", NAMES)
def test_named_research_notebook_run_all_is_review_only(name, tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root / "notebooks" / name).read_text(encoding="utf-8"))
    assert notebook["metadata"]["research_contract"]["default_run_all_training"] is False
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert all(not c.startswith("%%writefile") and "base64" not in c for c in code)
    for cell in code:
        ast.parse(cell)
    # No data/package in a clean review directory: the notebook must explain unavailable results.
    monkeypatch.chdir(tmp_path)
    scope = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for cell in code:
            exec(compile(cell, name, "exec"), scope)
    if name.startswith(("10_", "11_")):
        assert scope["RUN_TRAINING"] is False
        from src.features.lightgbm_features import build_features
        from src.features.relative_transfer import relative_features

        values = scope["np"].random.default_rng(17).uniform(0.1, 2, (180, 2, 4))
        origins = scope["np"].array([167])
        expected, scale = relative_features(values, origins, 6)
        actual, actual_scale = scope["relative_target_columns"](build_features(values, origins, 6))
        scope["np"].testing.assert_array_equal(expected, actual)
        scope["np"].testing.assert_array_equal(scale, actual_scale)
    assert not list(tmp_path.rglob("*"))
