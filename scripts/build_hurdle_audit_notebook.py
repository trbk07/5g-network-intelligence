"""Create a clear fixed-refit guide and a separate explicitly enabled hosted job."""

import json
import shutil
import subprocess
from pathlib import Path


def main():
    commit = subprocess.check_output(["git", "-c", "safe.directory=D:/5G", "rev-parse", "HEAD"], text=True).strip()
    cells = []

    def add(kind, text):
        value = {"cell_type": kind, "metadata": {}, "id": f"hurdle-audit-{len(cells):02d}", "source": text.splitlines(keepends=True)}
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        cells.append(value)

    add("markdown", "# 5G Network Intelligence — Zero-Inflated Candidate Audit\n\n"
        "## Hypothesis and development evidence\n\nThe mixture median reduced MAE by 0.065–0.460% "
        "in all six development fold/horizon comparisons. Test whether this small effect persists "
        "on reused validation; it does not establish performance on unseen data.\n")
    add("markdown", "## Frozen audit procedure\n\nFor each component/horizon, take the rounded median of "
        "the two development early-stopping round counts. Refit on targets through 503, with no early "
        "stopping, probability calibration or gate tuning. Audit 504–671 on all original eligible beams. "
        "This procedure was defined after the development results and before this audit. "
        "The original final holdout stays closed; production is unchanged.\n")
    add("code", f"RUN_TRAINING = False\nSOURCE_COMMIT = {commit!r}\n")
    add("markdown", "## Hosted inputs and provenance\n\nAttach completed zero-inflated development output "
        "and the small MIT refit source dataset. The source module separates fitting from audit features. "
        "No encoded source bundles or full-repository source dumps are included here.\n")
    add("code", "from pathlib import Path\nimport os, sys, shutil, subprocess, tarfile\n"
        "if RUN_TRAINING:\n"
        "    if not Path('/kaggle/working').is_dir():\n        raise RuntimeError('Hosted Kaggle required')\n"
        "    roots = [p.parent.parent for p in Path('/kaggle/input').rglob('train_hurdle_research.py') "
        "if (p.parent.parent/'artifacts/hurdle_research/manifest.json').exists()]\n"
        "    if len(roots) != 1:\n        raise RuntimeError('Attach one completed development output')\n"
        "    PROJECT = Path('/kaggle/working/hurdle_audit')\n    PROJECT.mkdir(exist_ok=True)\n"
        "    for folder in ('src', 'scripts', 'configs', 'data/processed', 'artifacts/hurdle_research'):\n"
        "        shutil.copytree(roots[0]/folder, PROJECT/folder, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))\n"
        "    os.chdir(PROJECT)\n    sys.path.insert(0, str(PROJECT))\n"
        "    os.environ['SOURCE_GIT_COMMIT'] = SOURCE_COMMIT\n    Path('reports').mkdir(exist_ok=True)\n")
    add("markdown", "## Refit implementation\n\nOne small attached module orchestrates fixed-round training. "
        "Inspect `scripts/finalize_hurdle_candidate.py` in the repository and the attached source. "
        "Shared causal features and mixture-median logic come from the completed development job.\n")
    add("code", "if RUN_TRAINING:\n"
        "    sources = list(Path('/kaggle/input').rglob('finalize_hurdle_candidate.py'))\n"
        "    if len(sources) != 1:\n        raise RuntimeError('Attach exactly one refit source')\n"
        "    shutil.copyfile(sources[0], 'scripts/finalize_hurdle_candidate.py')\n")
    add("markdown", "## Fixed training and debugging\n\nEighteen component fits run on hosted CPU. "
        "Each component checkpoint is saved immediately. Failures retain logs and partial artifacts.\n")
    add("code", "if RUN_TRAINING:\n"
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', 'lightgbm>=4.6,<5', 'numpy', 'pandas', 'pyyaml'], check=True)\n"
        "    with Path('reports/hurdle_audit_training.log').open('w') as log:\n"
        "        process = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.finalize_hurdle_candidate'], "
        "stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
        "        for line in process.stdout:\n            print(line, end='', flush=True)\n            log.write(line)\n            log.flush()\n"
        "        status = process.wait()\n"
        "    with tarfile.open('/kaggle/working/hurdle_candidate_results.tar.gz', 'w:gz') as archive:\n"
        "        for folder in ('reports', 'artifacts/hurdle_candidate'):\n"
        "            if Path(folder).exists():\n                archive.add(folder, arcname=folder)\n"
        "    if status:\n        raise RuntimeError('Refit failed; inspect saved log and checkpoints')\n"
        "else:\n    print('Review mode: no training or calibration.')\n")
    add("markdown", "## Saved evaluation\n\nMAE is in original usage units; WAPE is a ratio, not percent MAE. "
        "Compare the matched direct L1 control at all horizons. Independent checkpoint and history-only "
        "parity verification is required after download. No promotion from reused validation.\n")
    add("code", "paths = [Path('reports/hurdle_candidate_validation.csv'), Path('../reports/hurdle_candidate_validation.csv')]\n"
        "if Path('/kaggle/input').is_dir():\n    paths += list(Path('/kaggle/input').rglob('hurdle_candidate_validation.csv'))\n"
        "path = next((p for p in paths if p.exists()), None)\n"
        "if path:\n    import pandas as pd\n    from IPython.display import display\n"
        "    frame = pd.read_csv(path)\n    display(frame.loc[frame.load_tier == 'All', ['model', 'horizon', 'mae', 'wape', 'n_valid']])\n"
        "else:\n    print('No candidate audit scores available yet.')\n")
    notebook = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"}},
                "nbformat": 4, "nbformat_minor": 5}
    name = "14_zero_inflated_candidate_audit.ipynb"
    Path("notebooks", name).write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    control = next(c for c in cells if c["cell_type"] == "code" and "".join(c["source"]).startswith("RUN_TRAINING = False"))
    control["source"] = [f"RUN_TRAINING = True\nSOURCE_COMMIT = {commit!r}\n"]
    stage = Path(".cache/kaggle-zero-inflated-audit")
    stage.mkdir(parents=True, exist_ok=True)
    (stage / name).write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    (stage / "kernel-metadata.json").write_text(json.dumps({
        "id": "truongbaokien/5g-network-intelligence-zero-inflated-audit",
        "title": "5G Network Intelligence Zero Inflated Audit", "code_file": name,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": False, "enable_internet": True,
        "kernel_sources": ["truongbaokien/5g-network-intelligence-zero-inflated-forecasting"],
        "dataset_sources": ["truongbaokien/5g-network-intelligence-hurdle-refit-source"], "competition_sources": []
    }, indent=2))
    source = Path(".cache/kaggle-hurdle-refit-source")
    source.mkdir(parents=True, exist_ok=True)
    for path in ("scripts/finalize_hurdle_candidate.py", "LICENSE"):
        shutil.copyfile(path, source / Path(path).name)
    (source / "dataset-metadata.json").write_text(json.dumps({
        "id": "truongbaokien/5g-network-intelligence-hurdle-refit-source",
        "title": "5G Network Intelligence Hurdle Refit Source", "licenses": [{"name": "MIT"}]
    }, indent=2))
    print(stage)


if __name__ == "__main__":
    main()
