"""Regression: execution-cell substitution must never rewrite packaged Python imports."""

import json
from pathlib import Path

from scripts import build_week7_research


def test_source_cells_remain_verbatim(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stage = tmp_path / "input"
    stage.mkdir()
    source = "%%writefile scripts/research_week7.py\nfrom scripts.run_risk import risk_records\n"
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["title"]},
            {"cell_type": "code", "source": [source]},
            {
                "cell_type": "code",
                "source": [
                    "subprocess.run([sys.executable,'-m',*['scripts.run_risk', '--segment', 'val']])\n"
                ],
            },
        ]
    }
    (stage / "week6.ipynb").write_text(json.dumps(notebook))
    (stage / "kernel-metadata.json").write_text(json.dumps({}))
    monkeypatch.setattr(build_week7_research, "build", lambda mode: stage)
    build_week7_research.main()
    output = json.loads(
        Path(".cache/5g-network-intelligence-week7-research/research.ipynb").read_text()
    )
    assert "".join(output["cells"][1]["source"]) == source
    command = "".join(output["cells"][2]["source"])
    assert "['scripts.research_week7']" in command
    assert "--segment" not in command
