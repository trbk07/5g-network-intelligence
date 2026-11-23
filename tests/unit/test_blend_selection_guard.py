"""Selection entrypoints must reject local execution before opening inputs."""

from pathlib import Path

import pytest

from scripts.select_development_blend import run_selection
from scripts.select_mlp_blend import main


def test_shared_selection_is_cloud_only(monkeypatch, tmp_path):
    monkeypatch.setattr("src.utils.runtime.is_cloud_runtime",lambda:False)
    with pytest.raises(RuntimeError,match="forbidden"):
        run_selection({},Path("absent"),Path("absent"),tmp_path/"output","test")
    assert not (tmp_path/"output").exists()


def test_mlp_entrypoint_rejects_before_parsing_or_loading(monkeypatch):
    monkeypatch.setattr("src.utils.runtime.is_cloud_runtime",lambda:False)
    with pytest.raises(RuntimeError,match="forbidden"):
        main()
