"""Review Run All must remain read-only without requiring fitting packages."""

import contextlib
import io
import json
from pathlib import Path


def test_hurdle_review_is_readable_and_does_not_train(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root / "notebooks/13_zero_inflated_forecasting.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert all("base64" not in c and "%%writefile" not in c for c in code)
    monkeypatch.chdir(tmp_path)
    scope = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for cell in code:
            exec(compile(cell, "hurdle_review", "exec"), scope)
    assert scope["RUN_TRAINING"] is False
    assert not list(tmp_path.rglob("*"))
