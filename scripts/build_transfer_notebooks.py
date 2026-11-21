"""Generate research-oriented notebooks: clear names, business cells, safe Run All and honest results."""

import json
from pathlib import Path


def notebook(title):
    cells = []

    def add(kind, source):
        value = {
            "cell_type": kind,
            "metadata": {},
            "id": f"research-{len(cells):03d}",
            "source": source.splitlines(keepends=True),
        }
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        cells.append(value)

    add(
        "markdown",
        f"# 5G Network Intelligence — {title}\n\n"
        "Historical forecasting research. **Default Run All performs no training, calibration or holdout replay.**\n",
    )
    return cells, add


def save(name, cells):
    value = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"},
            "research_contract": {
                "local_training": False,
                "final_holdout_use": False,
                "default_run_all_training": False,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = Path("notebooks") / name
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def context(add):
    add(
        "markdown",
        "## Environment and saved evidence\n\nInput: completed worker outputs or the repository. Output: paths to saved reports. Missing files stay unavailable; they are never replaced with fabricated results.\n",
    )
    add(
        "code",
        "from pathlib import Path\nimport json\nimport pandas as pd\nimport numpy as np\nfrom IPython.display import display\n\n"
        "# Find report folders in the repo and attached Kaggle outputs.\n"
        "REPORT_DIRS = [Path('reports'), Path('../reports')]  # Repo root or notebooks/ working directory.\n"
        "if Path('/kaggle/input').exists():\n"
        "    REPORT_DIRS += sorted({p.parent for p in Path('/kaggle/input').rglob('*validation.csv')})\n\n"
        "    REPORT_DIRS += sorted({p.parent for p in Path('/kaggle/input').rglob('*manifest.json')})\n\n"
        "def report_path(filename):\n"
        "    for folder in REPORT_DIRS:\n        candidate = folder / filename\n        if candidate.exists():\n            return candidate\n"
        "    return None\n",
    )


def audit_book():
    cells, add = notebook("External Data Quality Audit")
    add(
        "markdown",
        "## Research question and admission criteria\n\nCan additional public telecom data support transferable load patterns? Admission requires legal access, pinned provenance, unambiguous entity/time keys and continuous observed history. More rows alone are not evidence of better forecasting.\n",
    )
    context(add)
    add(
        "markdown",
        "## 1. Source register\n\nKeep dataset identities, licenses and domains separate. The commercial counter partitions are not automatically independent networks.\n",
    )
    add(
        "code",
        "# Small, explicit metadata view; no raw training data is loaded.\n"
        "rows = []\n"
        "for filename in ['external_candidate_manifest.json', 'zenodo_counter_manifest.json']:\n"
        "    path = report_path(filename)\n    if path is None:\n        print(f'Unavailable: {filename}')\n        continue\n"
        "    source = json.loads(path.read_text(encoding='utf-8'))\n"
        "    rows.append({'source': source.get('repo', source.get('doi')), 'revision': source.get('revision', source.get('record')), 'license': source.get('license', 'see pinned LICENSE'), 'files': len(source['files']), 'training_performed': source['training_performed']})\n"
        "display(pd.DataFrame(rows))\n",
    )
    add(
        "markdown",
        "## 2. Schema and duration contradiction\n\nThe actual OAI workbook must be checked against its description. A short single-UE series cannot be relabeled as several weeks of hourly beam observations.\n",
    )
    add(
        "code",
        "path = report_path('oai_candidate_audit.json')\nif path:\n"
        "    audit = json.loads(path.read_text(encoding='utf-8'))\n"
        "    display({key: audit[key] for key in ['kpm_records','distinct_ues','span_if_microsecond_time_seconds','median_step_if_microseconds_seconds','hourly_beam_training_eligible','decision']})\n"
        "else:\n    print('OAI workbook audit unavailable.')\n",
    )
    add(
        "markdown",
        "## 3. Natural-key conflicts\n\nDuplicate timestamps with different values are not fixed by selecting the first row or averaging silently. Quarantine ambiguous sources.\n",
    )
    add(
        "code",
        "path = report_path('zenodo_duplicate_audit.json')\n"
        "if path:\n    display(pd.DataFrame(json.loads(path.read_text(encoding='utf-8'))))\n"
        "else:\n    print('Duplicate audit unavailable.')\n",
    )
    add(
        "markdown",
        "## 4. Continuous training prefixes\n\nAggregation accepts exactly four finite 15-minute observations per hour. Windows never cross gaps. Boundaries are chosen from timestamps before training; later external data is not used by this pilot.\n",
    )
    add(
        "code",
        "path = report_path('multisource_preparation.json')\n"
        "if path:\n    preparation = json.loads(path.read_text(encoding='utf-8'))\n"
        "    display(pd.DataFrame([{'domain': s['source_id'], 'cutoff_utc': s['cutoff_utc'], 'segments': len(s['segments']), 'entity_hours': s['training_rows'], 'future_targets_used': s['future_targets_used']} for s in preparation['sources']]))\n"
        "else:\n    print('Multi-source preparation is not yet attached.')\n",
    )
    add(
        "markdown",
        "## Findings and limits\n\nThe audit admits sources for research only. It does not prove original-network improvement. Correlated bands must be balanced and cannot be counted as independent test networks. Source code: `scripts/audit_*`, `scripts/prepare_transfer_data.py`; detailed evidence is retained in the JSON reports.\n",
    )
    return save("09_external_data_quality_audit.ipynb", cells)


def experiment_book(multi):
    title = "Multi-Source Transfer Ablation" if multi else "Single-Source Transfer Experiments"
    cells, add = notebook(title)
    add(
        "markdown",
        "## Research question and hypothesis\n\nDoes external relative load-shape data reduce original-network WAPE versus the same target-only representation? Compare matched controls; do not attribute a representation change to transfer. A negative result is valid evidence.\n",
    )
    add(
        "markdown",
        "## 1. Configuration\n\nRun All is review-only. Set RUN_TRAINING=True deliberately only in a hosted runtime with the source package and data available.\n",
    )
    add(
        "code",
        "RUN_TRAINING = False\nHORIZONS = (1, 6, 24)\nSEED = 17\nTARGET_WAPE = 0.10\n"
        + (
            "CASES = ['target_only', 'pm01_nr1800', 'pm03_nr2100', 'pm03_nr3500', 'balanced_pool']\n"
            if multi
            else "EXTERNAL_WEIGHT_RATIOS = [0.0, 0.1, 0.5, 1.0]\n"
        ),
    )
    context(add)
    add(
        "markdown",
        "## 2. Temporal protocol\n\nOriginal round selection: fit through 419, stop through 503. Refit through 503 with fixed rounds; audit reused validation 504–671. Original final holdout 672–839 stays closed. External data uses only separately prepared chronological prefixes. No fresh-test claim is allowed.\n",
    )
    add(
        "code",
        "protocol = pd.DataFrame([{'stage':'round selection fit','last_target':419}, {'stage':'early stopping','last_target':503}, {'stage':'fixed refit','last_target':503}, {'stage':'reused-validation audit','last_target':671}])\ndisplay(protocol)\n",
    )
    add(
        "markdown",
        "## 3. Preprocessing and representation\n\nThe observed 168-hour mean sets a dynamic scale. Only target-history columns are divided by it; phase columns remain deterministic, incompatible exogenous channels are missing for both domains. This compares relative shape without claiming equal physical units.\n",
    )
    add(
        "code",
        "# Review the numerical contract without fitting anything.\n"
        "def relative_target_columns(features, floor=1e-6):\n"
        "    scale = np.maximum(np.nan_to_num(features[:, 20].astype(float), nan=0.0), floor)\n"
        "    # Production research implementation resolves this column by FEATURE_NAMES.\n"
        "    transformed = features.copy()\n    transformed[:, :23] /= scale[:, None]\n"
        "    transformed[:, 25:] = np.nan\n    return transformed, scale\n",
    )
    add(
        "markdown",
        "## 4. Model, loss and source weights\n\nLightGBM L1 minimizes absolute error. Weighting normalized targets by their observed scale restores original-unit absolute loss. Stop on original-domain MAE; on its fixed actual population, minimizing MAE also minimizes pooled WAPE. The source ratio is predefined.\n",
    )
    add(
        "code",
        "# Transparent objective identity; these are arithmetic checks, not model training.\n"
        "actual = np.array([1.0, 3.0])\npoint = np.array([0.8, 2.5])\nscale = np.array([0.5, 2.0])\n"
        "np.testing.assert_allclose(np.abs(point / scale - actual / scale) * scale, np.abs(point - actual))\n"
        "display({'mae': np.abs(point-actual).mean(), 'wape': np.abs(point-actual).sum()/np.abs(actual).sum()})\n",
    )
    add(
        "markdown",
        "## 5. Hosted execution\n\nReusable implementation stays in named source modules instead of a wall of writefile cells. The command below logs progress and saves checkpoints per case/horizon. It is never executed with the default configuration. Review `scripts/train_transfer_pilot.py` or `scripts/train_multisource_pilot.py` for the explicit fit/refit loops.\n",
    )
    module = "scripts.train_multisource_pilot" if multi else "scripts.train_transfer_pilot"
    add(
        "code",
        "if RUN_TRAINING:\n"
        "    import sys\n    import subprocess\n    from src.utils.runtime import require_cloud_training\n"
        "    require_cloud_training()  # Refuses the owner's local Windows runtime.\n"
        f"    subprocess.run([sys.executable, '-u', '-m', '{module}'], check=True)\n"
        "else:\n    print('Review mode: no training, calibration or final replay executed.')\n",
    )
    add(
        "markdown",
        "## 6. Saved validation results\n\nUnits: MAE in original dataset usage, WAPE as a ratio. Multiply WAPE by 100 only for display; never confuse it with MAE. Results remain unavailable until the worker finishes.\n",
    )
    kind = "multisource_pilot" if multi else "transfer_pilot"
    add(
        "code",
        f"path = report_path('{kind}_validation.csv')\n"
        "if path:\n    results = pd.read_csv(path)\n"
        "    display(results.loc[results.load_tier == 'All', ['model','horizon','mae','wape','n_valid']])\n"
        "else:\n    print('No completed saved results attached. No scores are invented.')\n",
    )
    add(
        "markdown",
        "## Conclusions and debugging\n\nCompare each source case with its representation-matched control at every horizon. Inspect load tiers, coverage and CPU/cloud parity before selecting a candidate. If acquisition, metadata or training fails, inspect the saved log and run ledger; do not reinterpret a failed run as a model result.\n",
    )
    return save(
        "11_multisource_transfer_ablation.ipynb"
        if multi
        else "10_single_source_transfer_experiments.ipynb",
        cells,
    )


def results_book():
    cells, add = notebook("Transfer Results and Error Analysis")
    add(
        "markdown",
        "## Objective and interpretation\n\nCompare all declared horizons on the original beam population. The WAPE target is <10%, not a promise. Reused validation cannot prove unseen-week performance.\n",
    )
    context(add)
    for kind, title in [
        ("transfer_pilot", "Single-source weight ablation"),
        ("multisource_pilot", "Per-source and balanced-pool ablation"),
    ]:
        add(
            "markdown",
            f"## {title}\n\nSaved metrics only. No retraining or weight fitting occurs here.\n",
        )
        add(
            "code",
            f"path = report_path('{kind}_validation.csv')\n"
            "if path:\n    frame = pd.read_csv(path)\n    pooled = frame[frame.load_tier == 'All'].copy()\n"
            "    pooled['WAPE (%)'] = 100 * pooled.wape\n    display(pooled[['model','horizon','mae','WAPE (%)','n_valid']])\n"
            "    display(frame.loc[frame.load_tier != 'All', ['model','horizon','load_tier','mae','wape','n_valid']])\n"
            "else:\n    print('Completed results are not available yet.')\n",
        )
        add(
            "markdown",
            "### Independent inference verification\n\nA saved CSV is insufficient: checkpoint hashes and history-only CPU predictions must agree with the cloud replay.\n",
        )
        add(
            "code",
            f"path = report_path('{kind}_evidence.json')\n"
            "if path:\n    display(json.loads(path.read_text(encoding='utf-8')))\nelse:\n    print('Independent verification not completed; no verified-model claim.')\n",
        )
    add(
        "markdown",
        "## Representation-matched comparison\n\nPositive gain means lower MAE than target-only under the same preprocessing. Day blocks describe consistency; overlapping forecast targets prevent treating them as independent trials.\n",
    )
    add(
        "code",
        "path = report_path('transfer_ablation_comparison.csv')\n"
        "if path:\n    display(pd.read_csv(path))\n"
        "else:\n    print('Comparison not attached; no transfer benefit is assumed.')\n",
    )
    add(
        "markdown",
        "## Research conclusion\n\nThe completed pilot found negative transfer from NR1800 at all tested weights; the balanced pool also regressed at all horizons. NR3500 gains were only 0.005%, 0.051% and 0.206% relative MAE at 1/6/24h versus target-only. These exploratory gains do not establish generalization. WAPE remains about 41–45%, above the 10% target.\n\nCheckpoint hashes and 81 history-only CPU/cloud comparisons passed with zero difference. Separate cloud jobs reproduced both shared cases exactly. Do not promote from reused validation. Original production artifacts and final holdout remain unchanged. A new unseen chronological evaluation requires a separately frozen protocol.\n",
    )
    return save("12_transfer_results_and_error_analysis.ipynb", cells)


if __name__ == "__main__":
    for path in (audit_book(), experiment_book(False), experiment_book(True), results_book()):
        print(path)
