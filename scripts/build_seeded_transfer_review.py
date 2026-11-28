"""Create a separate read-only research review; never alter execution snapshots."""

import ast
import json
from pathlib import Path


def main():
    sections = [
        ("markdown", "# 5G Network Intelligence — Seeded Architecture and Transfer Review\n\n"
         "**Question:** Do within-cell attention and external encoder pretraining materially improve DLPRB forecasting?\n\n"
         "**Hypothesis:** Same-cell past histories and transferable temporal patterns may reduce error beyond the fixed strong blend. "
         "This notebook reviews completed seed-17 experiments. Run All reads reports only; it never trains, calibrates or evaluates holdout.\n"),
        ("markdown", "## 1. Temporal protocol and units\n\n"
         "Context: 168 hours; horizons: 1h, 6h, 24h; population: all 2,880 original beams. "
         "Fit/stop/score endpoints are 335/383/443 and 395/443/503. Hours 504 onward are excluded. "
         "Both folds are reused development evidence, not independent confirmation.\n\n"
         "MAE is in original DLPRB usage units; utilization percent is unverified. "
         "WAPE = sum(abs(prediction − actual)) / sum(abs(actual)); tables convert it to percent. "
         "Every horizon uses its own full valid-origin population, with no candidate-specific masking.\n"),
        ("code", "# Resolve the repository whether launched from its root or notebooks/.\n"
         "from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display\n"
         "PROJECT = Path.cwd()\nif not (PROJECT / 'reports').is_dir():\n    PROJECT = PROJECT.parent\n"
         "REPORTS = PROJECT / 'reports'\nassert REPORTS.is_dir(), 'Open this notebook inside the repository.'\n"),
        ("markdown", "## 2. Provenance and verification\n\n"
         "Seeded cold weights, 5,120 source updates, encoder-only transfer, stopping choices and saved forecasts were independently checked. "
         "Linux uniform initialization uses FMA arithmetic; exact replay reproduces cold weights bitwise on Windows. "
         "No tolerance was added to initialization checks. Forecast parity retains atol 0.0001.\n"),
        ("code", "# Refuse to present unverified tables as accepted experiment results.\n"
         "studies = ['cell_attention_v2', 'cell_multisource_v2_multi_source_pretrain',\n"
         "           'cell_multisource_v2_target_self_pretrain']\nproofs = []\n"
         "for study in studies:\n    proof = json.loads((REPORTS / f'{study}_verification.json').read_text(encoding='utf-8'))\n"
         "    assert all(proof[key] for key in ['all_metrics_reconstructed',\n"
         "        'controlled_neural_initialization_verified', 'stopping_choices_verified'])\n"
         "    assert not proof['holdout_used'] and not proof['production_promoted']\n"
         "    assert proof['maximum_cpu_gpu_error'] <= 0.0001\n"
         "    proofs.append({'study': study, 'CPU/GPU max error': proof['maximum_cpu_gpu_error'],\n"
         "                   'beam predictions replayed': proof['history_only_beam_predictions_checked'],\n"
         "                   'epoch-capped runs': proof['epoch_capped_runs']})\n"
         "display(pd.DataFrame(proofs))\n"),
        ("markdown", "## 3. Model and data controls\n\n"
         "The local and cell-attention models have equal parameter counts and share temporal patches, masks, phase features and seasonal residuals. "
         "Only off-diagonal spatial attention differs. No beam summary statistics or ID embeddings enter the model.\n\n"
         "Nine external domains belong to four collections: three NR bands, CESNET24, four Vigo signals and AGG23. "
         "Collections receive equal aggregate sampling weight. Traffic and sessions are dimensionless source tasks, not physical PRB labels. "
         "Source heads are discarded; only temporal encoder weights transfer. Target-self pretraining has the same update budget.\n"),
        ("markdown", "## 4. Full development scores against the strong comparator\n\n"
         "Positive gain means lower MAE than the unchanged historical MLP/hurdle blend. "
         "Report both folds and every horizon; do not choose a winning subset.\n"),
        ("code", "# Read fixed, independently reconstructed scores; no model fitting.\n"
         "tables = []\nfor study in studies:\n"
         "    frame = pd.read_csv(REPORTS / f'{study}_comparison.csv')\n"
         "    frame['study'] = study\n    frame['WAPE (%)'] = 100 * frame['wape']\n    tables.append(frame)\n"
         "scores = pd.concat(tables, ignore_index=True)\n"
         "display(scores[['study', 'fold', 'mode', 'horizon', 'mae', 'WAPE (%)', 'n_valid',\n"
         "                'relative_mae_gain_vs_current_pct']].round(6))\n"),
        ("markdown", "## 5. Does external pretraining help its own cold architecture?\n\n"
         "The comparison script verified that cold target tensors match bitwise across jobs. "
         "This separates a small transfer benefit from the stronger requirement to beat the existing comparator.\n"),
        ("code", "# Paired all-beam comparison, not a search for blend weights or a best seed.\n"
         "paired = pd.read_csv(REPORTS / 'cell_multisource_v2_matched_cold_comparison.csv')\n"
         "assert len(paired) == 12\n"
         "display(paired[['case', 'fold', 'horizon', 'mae', 'mae_cold', 'gain_vs_cold_pct']].round(6))\n"),
        ("markdown", "## 6. Stopping and error diagnostics\n\n"
         "Train and stopping losses diverged before the epoch limit; more epochs alone are not supported by these runs. "
         "Strict zero false-positive rate counts any positive forecast on an actual zero. "
         "It is a diagnostic, not a substitute for full-population MAE/WAPE. Earlier error accounting found positive-load errors dominate total absolute error.\n"),
        ("code", "# Keep termination evidence separate from accuracy interpretation.\n"
         "stopping = []\nfor study in studies:\n"
         "    frame = pd.read_csv(REPORTS / f'{study}_termination.csv')\n"
         "    frame['study'] = study\n    stopping.append(frame)\n"
         "display(pd.concat(stopping, ignore_index=True)[['study', 'fold', 'mode',\n"
         "    'best_epoch', 'epochs_completed', 'termination_reason', 'stale_epochs']])\n"),
        ("markdown", "## 7. Conclusion and limitations\n\n"
         "**Negative result:** multi-corpus pretraining improves five of six scores versus its cold architecture by less than 0.65%, "
         "but loses five of six comparisons against the stronger historical blend. It fails the no-regression/material-improvement gate. "
         "No production promotion or WAPE below 10% is established.\n\n"
         "Only one seeded Kaggle comparison is complete here. The predeclared Colab seed 29 is a seed/runtime robustness check, "
         "not permission to select a lucky seed. Repeated development is not fresh generalization evidence. "
         "Vigo publisher scaling scope is unknown; subsets are not independent networks. "
         "Telecom Italia was absent from these frozen experiments and needs separate overlap and sampler audits.\n\n"
         "Research modules: `src/models/cell_attention.py`, `scripts/train_cell_multisource.py`, "
         "`scripts/summarize_cell_attention.py`, `scripts/compare_seeded_transfer.py`. "
         "Detailed evidence: `reports/cell_multisource_v2_analysis.md`.\n"),
    ]
    cells = []
    for kind, source in sections:
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            ast.parse(source)
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    output = Path("notebooks/39_seeded_architecture_and_transfer_review.ipynb")
    if output.exists():
        raise FileExistsError("Review notebook already exists; edit deliberately rather than overwrite a snapshot")
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
