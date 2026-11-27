"""Build a report-first research review; Run All never trains or selects models."""

import json
from pathlib import Path


def main():
    sections = [
        ("markdown", "# 5G Network Intelligence — Chronos-2 Adaptation Evaluation\n\n"
         "## Research question\nDoes native multivariate LoRA improve DLPRB forecasting "
         "over both the unchanged synthetic-pretrained model and the strongest saved "
         "development controls? Training loss is not a forecasting accuracy metric.\n"),
        ("markdown", "## Temporal protocol and selection\n"
         "Fit/stop/score cutoffs are 335/383/443 and 395/443/503. Context: 168 hours; "
         "forecast horizons: 1, 6 and 24 hours. All original beams are evaluated. "
         "Select steps 100/300/1000 using stopping MAE only, breaking ties toward the "
         "earlier step. Freeze both choices before scoring. Hours 504 onward are "
         "excluded. The previously observed holdout is never reused.\n"),
        ("code", "# Safe review default: no training, inference or checkpoint selection.\n"
         "RUN_EVALUATION = False\nfrom pathlib import Path\nimport json\n"
         "import pandas as pd\nfrom IPython.display import display\nPROJECT = Path.cwd()\n"),
        ("markdown", "## Artifact provenance and debugging\n"
         "The separate hosted evaluator verifies base-weight hashes, every training "
         "artifact, LoRA configuration, optimizer-step identities and runtime versions. "
         "Inference caches bind the model and exact causal history. A completed training "
         "job does not imply checkpoint selection or evaluation has completed.\n"),
        ("code", "# Inspect the recorded submission; this is not a completion claim.\n"
         "submission = PROJECT / 'reports/chronos2_lora_cloud_submission.json'\n"
         "if submission.exists():\n    display(json.loads(submission.read_text(encoding='utf-8')))\n"),
        ("markdown", "## Hosted evaluation entry point\n"
         "This stage performs inference and stopping-only checkpoint selection, never "
         "fitting. Use a verified completed training output and the pinned base model "
         "in a hosted GPU runtime. Durable outputs belong on Drive or Kaggle Output.\n"),
        ("code", "if RUN_EVALUATION:\n"
         "    import subprocess, sys\n"
         "    if 'MODEL_ROOT' not in globals() or 'TRAINED_ROOT' not in globals():\n"
         "        raise RuntimeError('Attach and verify base model and completed training artifacts first')\n"
         "    subprocess.run([sys.executable, '-u', '-m', 'scripts.evaluate_chronos2_lora',\n"
         "                    '--model-root', str(MODEL_ROOT), '--trained-root', str(TRAINED_ROOT)], check=True)\n"
         "else:\n    print('Review mode: checkpoint selection and inference skipped.')\n"),
        ("markdown", "## Stopping evidence\n"
         "MAE is measured in original DLPRB units. This table explains checkpoint choice; "
         "it is separate from scoring performance.\n"),
        ("code", "stopping = PROJECT / 'reports/chronos2_lora_stopping.csv'\n"
         "if stopping.exists():\n    display(pd.read_csv(stopping))\n"
         "else:\n    print('No verified stopping results available yet.')\n"),
        ("markdown", "## Development performance\n"
         "Compare unchanged multivariate zero-shot and selected LoRA on identical frozen "
         "origins and observed target masks. MAE uses original DLPRB units; WAPE is "
         "sum absolute error / sum absolute actual values. Coverage and zero false-positive "
         "rates expose changes hidden by aggregate error.\n"),
        ("code", "scores = PROJECT / 'reports/chronos2_lora_development.csv'\n"
         "if scores.exists():\n    results = pd.read_csv(scores)\n"
         "    results['WAPE (%)'] = 100 * results['wape']\n"
         "    results['Zero false positives (%)'] = 100 * results['zero_fp_rate']\n"
         "    display(results[['fold', 'stage', 'horizon', 'mae', 'WAPE (%)',\n"
         "                     'n_valid', 'Zero false positives (%)']])\n"
         "else:\n    print('No verified development metrics available yet; no accuracy claim.')\n"),
        ("markdown", "## Independent verification\n"
         "Before accepting a result, reconstruct saved MAE/WAPE and stopping choices, "
         "verify every cache's exact causal history, and compare CPU inference with GPU "
         "forecasts at fixed beams and origins (absolute tolerance 1e-4 DLPRB units). "
         "The separate verification report records completion; absent evidence remains "
         "unverified. Review never launches verification or fitting automatically.\n"),
        ("code", "verification = PROJECT / 'reports/chronos2_lora_verification.json'\n"
         "if verification.exists():\n    display(json.loads(verification.read_text(encoding='utf-8')))\n"
         "else:\n    print('Independent result verification is not available yet.')\n"),
        ("markdown", "## Acceptance and limitations\n"
         "Compare with the saved LightGBM hurdle and MLP-B/hurdle controls on the same "
         "masks. Require no horizon regression and at least 2% mean relative MAE "
         "improvement over the strong control before further seed and uncertainty checks. "
         "Report negative results. A single seed and reused development data cannot "
         "establish independent generalization. Synthetic-only pretraining is the "
         "publisher's claim, not an independent corpus audit. WAPE <10% is unproven. "
         "No automatic production promotion or new holdout evaluation.\n"),
    ]
    cells = []
    for kind, source in sections:
        cell = {"cell_type": kind, "id": f"chronos2-evaluation-{len(cells):02d}",
                "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
                "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
    Path("notebooks/32_chronos2_adaptation_evaluation.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
