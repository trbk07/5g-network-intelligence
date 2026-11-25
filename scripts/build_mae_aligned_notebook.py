"""Reader-facing record of a frozen point-MAE objective experiment."""

import json
from pathlib import Path


def main():
    content = [
        ("markdown", "# 5G Network Intelligence — MAE-Aligned Forecasting\n\n"
         "## Research question\n\nThe previous BCE-plus-pinball hurdle lost to matched direct "
         "regression at all six development points. Does optimizing the final point forecast "
         "rather than only its distributional surrogate improve MAE? This hypothesis is "
         "unproven. Keep encoder, head shapes, seed and temporal boundaries unchanged.\n"),
        ("markdown", "## Temporal protocol\n\nFit/stop/score: 335/383/443 and 395/443/503. "
         "All four original KPIs and all original beams. Lookback 168 hours; horizons +1/+6/+24h. "
         "Validation 504–671 and holdout are excluded, including from the cloud bundle. "
         "Reused development data cannot establish fresh-test generalization.\n"),
        ("code", "# Run All in this review notebook never fits or calibrates.\nRUN_TRAINING = False\n"),
        ("markdown", "## Configuration\n\nTwo new modes, four hosted GPU fits. Reuse verified "
         "historical direct and surrogate-hurdle controls without retraining. Fixed hidden "
         "width 64, dropout 0.1, seed 17, batch 1024, maximum 40 epochs and patience 5. "
         "No coefficient grid or scoring-window tuning.\n"),
        ("code", "from pathlib import Path\nimport yaml\nimport pandas as pd\n"
         "from IPython.display import display\nPROJECT = Path.cwd()\n"
         "config = PROJECT / 'configs/mae_aligned.yaml'\n"
         "if config.exists():\n    protocol = yaml.safe_load(config.read_text(encoding='utf-8'))\n"
         "    display(pd.DataFrame(protocol['folds']))\n"),
        ("markdown", "## Preprocessing and model\n\nReuse global per-KPI fit-prefix "
         "imputation/scaling, observed-zero preservation, missing masks and phase4. The shared "
         "encoder receives centered history, last scaled levels and causal seasonal means. "
         "No fitted per-beam summary enters. Architecture: `src/models/mae_aligned.py`.\n"),
        ("markdown", "## Objectives\n\n**direct_clipped:** apply the physical nonnegative "
         "boundary before computing both training and stopping L1, matching deployment "
         "projection. This can create flat gradients below zero; no gain is assumed.\n\n"
         "**hurdle_mae:** masked scaled mixture-median L1 + 0.1 × (finite-label BCE + "
         "positive-only mean pinball). The auxiliary signal remains when p≤0.5 makes the "
         "point median constant zero. Quantiles stay noncrossing, with fixed levels "
         "0.05/0.10/0.25/0.50. The auxiliary coefficient is fixed before fitting. Joint point "
         "optimization means probability/quantile calibration must not be claimed.\n"),
        ("markdown", "## Training and debugging\n\nHosted Colab/Kaggle CUDA only. Checkpoints "
         "remain on Drive or Kaggle Output. Stopping always uses point MAE in scaled target "
         "units. Inspect finite objectives, observed counts and completed patience. An epoch "
         "cap is not completed stopping. Historical execution snapshots stay immutable.\n"),
        ("code", "if RUN_TRAINING:\n    from scripts.train_mae_aligned import main\n    main()\n"
         "else:\n    print('Review mode: fitting skipped.')\n"),
        ("markdown", "## Evaluation\n\nReport original DLPRB-unit MAE, WAPE = sum absolute "
         "error / sum absolute actual, full coverage and zero false-positive rate for each "
         "fold/horizon. Compare the two new modes with the corresponding saved historical "
         "head, MLP-B and LightGBM hurdle. Require independent checksum, temporal-axis and "
         "history-only CPU/GPU parity verification before interpreting results.\n"),
        ("code", "# Display available completed results; never invent missing metrics.\n"
         "for mode in ('direct_clipped', 'hurdle_mae'):\n"
         "    path = PROJECT / f'reports/mae_aligned_{mode}_development.csv'\n"
         "    if path.exists():\n        print(mode)\n        display(pd.read_csv(path))\n"
         "    else:\n        print(mode, ': no verified result available')\n"),
        ("markdown", "## Limitations\n\nOne seed, reused development windows and piecewise "
         "gradients. No architecture search, independent-test superiority or WAPE <10% claim. "
         "Negative results must remain visible. Do not refit on final validation or promote "
         "a model solely because the new objective sounds more aligned.\n"),
    ]
    cells = []
    for kind, text in content:
        cell = {"cell_type": kind, "id": f"mae-aligned-{len(cells):02d}", "metadata": {},
                "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
                "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
    Path("notebooks/29_mae_aligned_forecasting.ipynb").write_text(
        json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
