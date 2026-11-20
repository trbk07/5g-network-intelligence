"""Safe report-only review of chronological model search and the separately fixed validation audit."""

import json
from pathlib import Path


def main():
    cells = []

    def cell(kind, source):
        entry = {
            "cell_type": kind,
            "metadata": {},
            "id": f"improve-{len(cells):02d}",
            "source": source.splitlines(keepends=True),
        }
        if kind == "code":
            entry.update(execution_count=None, outputs=[])
        cells.append(entry)

    cell(
        "markdown",
        "# Forecasting model improvement - saved evidence\n\n"
        "**Run All only displays saved reports. It never trains, fits weights, calibrates or reopens holdout.**\n\n"
        "Main objective: mean normalized original-unit MAE for 1h/6h/24h. "
        "The inner folds stop and score on separate chronological windows. "
        "The outer audit is previously seen validation 504-671, not a fresh test. Production and final holdout remain unchanged.\n",
    )
    cell(
        "code",
        "from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display\n\n"
        "roots = list(Path('/kaggle/input').rglob('improvement_audit.csv')) if Path('/kaggle/input').exists() else []\n"
        "REPORTS = roots[0].parent if roots else Path('reports')\n"
        "if not (REPORTS/'improvement_audit.csv').exists():\n    raise FileNotFoundError('Attach the completed improvement audit outputs')\n",
    )
    cell(
        "markdown",
        "## 1. Development folds\n\nThe scoring period is never used for early stopping within its fold. Some LightGBM candidates are screened out after inner1.\n",
    )
    cell(
        "code",
        "lgbm = pd.read_csv(REPORTS/'improvement_lgbm_cv.csv')\n"
        "catboost = pd.read_csv(REPORTS/'improvement_catboost_cv.csv')\n"
        "cv = pd.concat([lgbm,catboost],ignore_index=True)\n"
        "display(cv.pivot_table(index=['fold','model'],columns='horizon',values='mae'))\n",
    )
    cell(
        "markdown",
        "## 2. Inner-only ensemble choice\n\nWeights and a small bias grid were calibrated on cloud using only inner scores through hour 503. They were saved before audit construction.\n",
    )
    cell(
        "code",
        "selection = json.loads((REPORTS/'improvement_selection.json').read_text())\n"
        "display(pd.DataFrame([{'horizon':h,'inner_mae':r['mae'],'bias':r['bias'],"
        "'members':' + '.join(f'{n}: {w:.1f}' for n,w in r['weights'].items())} for h,r in selection.items()]))\n",
    )
    cell(
        "markdown",
        "## 3. Reused-validation audit\n\nLower MAE is better. Compare only matching horizons and common finite pairs. This is not a new final holdout result.\n",
    )
    cell(
        "code",
        "audit = pd.read_csv(REPORTS/'improvement_audit.csv')\n"
        "pooled = audit[audit.load_tier == 'All']\n"
        "display(pooled[['model','horizon','mae','rmse','wape','n_valid']])\n"
        "mae = pooled.pivot(index='model',columns='horizon',values='mae')\n"
        "display((100*(1-mae/mae.loc['lightgbm_c1'])).round(3).rename_axis('MAE reduction vs C1 (%)'))\n",
    )
    cell(
        "markdown",
        "## 4. Paired BS uncertainty\n\nDifference is improved ensemble MAE minus reference MAE. Negative favors the ensemble on this sample; it does not establish unseen-week generalization.\n",
    )
    cell(
        "code",
        "paired = pd.read_csv(REPORTS/'improvement_paired.csv')\ndisplay(paired[['horizon','reference','difference','lower','upper','status']])\n",
    )
    cell(
        "markdown",
        "## 5. Risk and zero/active behavior\n\nP90/P95 policy is unchanged and fitted only through hour 503. Precision, recall and FPR must be read together. Strict zero-FP (`point > 0` when actual is 0) is a separate metric.\n",
    )
    cell(
        "code",
        "risk = pd.read_csv(REPORTS/'improvement_risk.csv')\ndisplay(risk[['model','horizon','threshold','precision','recall','fpr','positive_events']])\n"
        "display(pooled[['model','horizon','active_only_mae','zero_fp_rate']])\n",
    )
    cell(
        "markdown",
        "## 6. Load-group errors\n\nLoad groups use fit data only. Inspect low-load WAPE and high-load absolute error alongside pooled MAE.\n",
    )
    cell(
        "code",
        "display(audit.loc[audit.load_tier != 'All', ['model','horizon','load_tier','mae','wape']])\n",
    )
    cell(
        "markdown",
        "## Scope and next evidence\n\n"
        "These results support a research candidate, not a replacement final benchmark. The production artifact directory is untouched. "
        "Fresh unseen chronological evaluation data is required for a new frozen promotion. "
        "All training, including tiny fitting tests and blend calibration, ran on hosted Kaggle. "
        "Checkpoints remain in Kaggle Output; source code is in the private project repository.\n",
    )
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = Path("notebooks/08_model_improvement_review.ipynb")
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=True), encoding="utf-8")
    output = Path(".cache/5g-network-intelligence-improve-review")
    output.mkdir(parents=True, exist_ok=True)
    (output / "review.ipynb").write_bytes(path.read_bytes())
    meta = {
        "id": "truongbaokien/5g-network-intelligence-improve-review",
        "title": "5G Network Intelligence Improve Review",
        "code_file": "review.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "kernel_sources": ["truongbaokien/5g-network-intelligence-improve-audit"],
        "dataset_sources": [],
        "competition_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(output)


if __name__ == "__main__":
    main()
