"""Readable research review notebook; Run All only displays saved validation reports."""

import json
from pathlib import Path


def main():
    cells = []

    def cell(kind, source):
        value = {
            "cell_type": kind,
            "metadata": {},
            "source": source.splitlines(keepends=True),
            "id": f"week7-{len(cells):02d}",
        }
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        cells.append(value)

    cell(
        "markdown",
        "# Week 7 - Model research & historical replay product\n\n"
        "Review saved reports only. **Run All never trains, replays holdout or replaces the production champion.**\n\n"
        "Four parameter sets were trained on hosted Kaggle CPU using fit 0-503 and validation 504-671. "
        "Repeated validation research is exploratory; fresh unseen chronological data is required before promotion.\n",
    )
    cell(
        "code",
        "from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display\n\n"
        "# Find attached completed research output, or local reports when reviewing in the repo.\n"
        "roots = list(Path('/kaggle/input').rglob('week7_research_decision.json')) if Path('/kaggle/input').exists() else []\n"
        "REPORTS = roots[0].parent if roots else Path('reports')\n"
        "if not (REPORTS/'week7_research_decision.json').exists():\n    raise FileNotFoundError('Attach the completed Week 7 research outputs')\n",
    )
    cell(
        "markdown",
        "## 1. Validation decision\n\nThe best exploratory score is separate from the frozen deployment champion.\n",
    )
    cell(
        "code",
        "decision = json.loads((REPORTS/'week7_research_decision.json').read_text())\ndisplay(decision)\n",
    )
    cell(
        "markdown",
        "## 2. Point forecast quality\n\nLower MAE is better. Compare each horizon against the original C1 validation replay. No holdout metrics appear here.\n",
    )
    cell(
        "code",
        "benchmark = pd.read_csv(REPORTS/'week7_research_validation.csv')\n"
        "all_beams = benchmark[benchmark.load_tier == 'All']\n"
        "mae = all_beams.pivot(index='model',columns='horizon',values='mae')\ndisplay(mae)\n"
        "# Positive percentages mean lower validation MAE than the original C1.\n"
        "display((100*(1-mae/mae.loc['lightgbm_c1'])).round(2).rename_axis('MAE reduction vs C1 (%)'))\n",
    )
    cell(
        "markdown",
        "## 3. Paired uncertainty\n\nDifference is candidate MAE minus C1 MAE. Negative favors the candidate. BS bootstrap intervals describe this validation sample, not future performance.\n",
    )
    cell(
        "code",
        "paired = pd.read_csv(REPORTS/'week7_research_paired.csv')\ndisplay(paired[['model','horizon','difference','lower','upper','status']])\n",
    )
    cell(
        "markdown",
        "## 4. Historical exceedance trade-off\n\nInspect precision, recall and false-positive rate together. P90/P95 remain fixed from fit data and are not forecast uncertainty quantiles.\n",
    )
    cell(
        "code",
        "risk = pd.read_csv(REPORTS/'week7_research_risk.csv')\ndisplay(risk[['model','horizon','threshold','precision','recall','fpr','positive_events']])\n",
    )
    cell(
        "markdown",
        "## 5. Load-group diagnostics\n\nGroups come only from the fit segment. Low-load WAPE can be large when its denominator is small.\n",
    )
    cell(
        "code",
        "display(benchmark.loc[benchmark.load_tier != 'All', ['model','horizon','load_tier','mae','wape','active_only_mae']])\n",
    )
    cell(
        "markdown",
        "## 6. Working Week 7 product\n\n"
        "The FastAPI + Streamlit + Docker product uses the frozen Week 6 champion and 24 real historical beams. "
        "From the project root, prepare verified assets and run `docker compose up --build -d --wait`. "
        "Open the dashboard at http://127.0.0.1:8501 on the same computer.\n\n"
        "The app has no training controls. Its forecasts and risk states come from one shared API. "
        "Twenty beam/origin pairs across three horizons matched saved offline predictions at absolute tolerance 1e-9.\n\n"
        "Code: [shared predictor](https://github.com/trbk07/5g-network-intelligence/blob/main/src/inference/predictor.py), "
        "[API](https://github.com/trbk07/5g-network-intelligence/blob/main/src/api/main.py), "
        "[dashboard](https://github.com/trbk07/5g-network-intelligence/blob/main/app/dashboard.py).\n",
    )
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = Path("notebooks/07_product_and_research.ipynb")
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=True), encoding="utf-8")
    output = Path(".cache/5g-network-intelligence-week7-review")
    output.mkdir(parents=True, exist_ok=True)
    (output / "review.ipynb").write_bytes(path.read_bytes())
    metadata = {
        "id": "truongbaokien/5g-network-intelligence-week7-review",
        "title": "5G Network Intelligence Week7 Review",
        "code_file": "review.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "kernel_sources": ["truongbaokien/5g-network-intelligence-week7-research"],
        "dataset_sources": [],
        "competition_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    print(output)


if __name__ == "__main__":
    main()
