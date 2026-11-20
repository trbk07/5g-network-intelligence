"""Regression checks for the Vietnamese text corrupted during shell-based generation."""

import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("00_cloud_tests.ipynb", "Bộ test được chạy"),
        ("00_kaggle_runtime.ipynb", "kiểm tra runtime"),
        ("04_gru_review.ipynb", "Cách đọc notebook"),
        ("00_cloud_standalone.ipynb", "Cách đọc notebook"),
        ("05_model_quality.ipynb", "Chất lượng mô hình"),
    ],
)
def test_vietnamese_notebook_text_survives_json_roundtrip(filename, expected):
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root / "notebooks" / filename).read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert expected in source
    assert "\ufffd" not in source
    assert "base64.b64decode(" not in source
