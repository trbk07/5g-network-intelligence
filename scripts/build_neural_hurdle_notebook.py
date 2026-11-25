"""Readable guide for matched multivariate direct-versus-hurdle experiments."""

import json
from pathlib import Path


def main():
    cells = []

    def add(kind, text):
        cell = {"cell_type": kind, "id": f"neural-hurdle-{len(cells):02d}",
                "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", "# 5G Network Intelligence — Multivariate Zero-Inflated Forecasting\n\n"
        "## Research question\n\nCan an explicit zero-mass distribution improve beam-level "
        "MAE and zero false positives compared with direct regression using the same encoder? "
        "The previous multisource pilot did not consistently beat compute-matched "
        "self-pretraining. This experiment changes the output distribution and uses all four "
        "original KPIs; it does not assume that the new head is superior.\n")
    add("markdown", "## Frozen temporal protocol\n\nFit/stop/score cutoffs: 335/383/443 and "
        "395/443/503. Fit preprocessing separately within each fit prefix. Stop by point MAE "
        "on the stopping segment, not composite training loss. Score complete +1/+6/+24-hour "
        "origins across all original beams. Validation 504–671 and final holdout are excluded. "
        "Repeated development reuse limits generalization claims.\n")
    add("code", "# Review Run All never trains or calibrates a model.\nRUN_TRAINING = False\n")
    add("markdown", "## Configuration\n\nBoth branches use hidden width 64, dropout 0.1, seed "
        "17, batch size 1024 and AdamW learning rate 0.001. The maximum is 40 epochs with "
        "patience 5. This is a bounded pilot, not an architecture search.\n")
    add("code", "from pathlib import Path\nimport yaml\nimport pandas as pd\n"
        "from IPython.display import display\n\nPROJECT = Path.cwd()\n"
        "path = PROJECT / 'configs/neural_hurdle.yaml'\n"
        "if path.exists():\n    protocol = yaml.safe_load(path.read_text(encoding='utf-8'))\n"
        "    display(pd.DataFrame(protocol['folds']))\n")
    add("markdown", "## Preprocessing and encoder\n\nUse global per-KPI fit-prefix "
        "medians/means/std, observed-zero preservation and explicit missing masks. Both heads "
        "share the same input definition: 168-hour centered histories, last scaled KPI "
        "levels, causal seven-day seasonal target means and phase4. Entity IDs and fitted "
        "beam summaries are excluded. Both cases are trained from scratch.\n")
    add("markdown", "## Distribution and objective\n\nDirect regression adds a learned "
        "residual to the seasonal mean and minimizes masked scaled L1. The hurdle branch "
        "learns p=P(Y>0) and positive conditional quantiles at 0.05, 0.10, 0.25 and 0.50. "
        "Softplus increments prevent quantile crossing. Train with finite-label binary "
        "cross-entropy plus positive-only mean pinball loss, with fixed equal coefficients. "
        "There is no fitted probability threshold or calibration.\n\n"
        "If p≤0.5, zero is a mixture median. Otherwise interpolate the positive quantile "
        "at 1−0.5/p. The fixed (0,0) lower interpolation boundary is an approximation. "
        "Quantiles use target/global-scale units; exact zeros are decoded directly in "
        "physical units to avoid z-score roundoff affecting zero false-positive counts.\n")
    add("code", "# Reusable model and loss implementation; no source-file dump or fitting.\n"
        "if (PROJECT / 'src/models/neural_hurdle.py').exists():\n"
        "    from src.models.neural_hurdle import NeuralHurdleMLP\n"
        "    print(NeuralHurdleMLP.__doc__)\n")
    add("markdown", "## Training and debugging\n\nEnable execution only in hosted Colab or "
        "Kaggle CUDA. Colab checkpoints/reports must resolve to Drive; Kaggle uses Output. "
        "Per-epoch logs separate the composite training objective from stopping MAE. "
        "Inspect masks, finite losses, observed targets and stopping status. Do not treat "
        "an epoch cap as completed patience or use scoring to adjust loss coefficients.\n")
    add("code", "if RUN_TRAINING:\n    from scripts.train_neural_hurdle import main\n    main()\n"
        "else:\n    print('Review mode: fitting skipped.')\n")
    add("markdown", "## Evaluation\n\nReport original-unit DLPRB MAE, WAPE = sum absolute "
        "error / sum absolute actual, full coverage and zero false-positive rate for every "
        "case/fold/horizon. Compare with the matched direct branch, historical multivariate "
        "MLP and hurdle baselines. Saved artifact checksums, CPU parity and target axes must "
        "be independently verified before accepting results.\n")
    add("code", "# Read completed reports only. Missing reports mean no measured result.\n"
        "for case in ('direct', 'hurdle'):\n"
        "    path = PROJECT / f'reports/neural_hurdle_{case}_development.csv'\n"
        "    if path.exists():\n        print(case)\n        display(pd.read_csv(path))\n"
        "    else:\n        print(case, ': no completed result available')\n")
    add("markdown", "## Limitations\n\nOne seed and reused development windows. Sparse "
        "conditional quantiles approximate the mixture median; BCE plus pinball need not "
        "minimize the final mixture MAE. Extra last-level features differ from the older "
        "residual-MLP input, which is why a new matched direct control is required. No "
        "training gain, final-test superiority or WAPE <10% result is claimed yet.\n")
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
                "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
    Path("notebooks/28_multivariate_zero_inflated_forecasting.ipynb").write_text(
        json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
