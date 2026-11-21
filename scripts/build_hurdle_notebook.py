"""Readable zero-inflation experiment; inherit pinned source instead of dumping the repo."""

import json
import pprint
import shutil
import subprocess
from pathlib import Path

import yaml


def main():
    commit = subprocess.check_output(
        ["git", "-c", "safe.directory=D:/5G", "rev-parse", "HEAD"], text=True
    ).strip()
    cells = []

    def add(kind, text):
        cell = {"cell_type": kind, "metadata": {}, "id": f"hurdle-{len(cells):02d}",
                "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", "# 5G Network Intelligence — Zero-Inflated Forecasting\n\n"
        "## Question and hypothesis\n\nCan explicit zero probability reduce original-unit MAE relative to direct L1? "
        "This is a research hypothesis, not a promised improvement. Fixed 40 causal inputs; no external source is added.\n\n"
        "## Temporal protocol\n\nFit/stopping/scoring boundaries: 335/383/443 and 395/443/503. "
        "Stopping selects component rounds only. Score windows cannot fit thresholds, probabilities or quantiles. "
        "The windows have previously supported research, so scores remain exploratory. Validation 504–671 and final holdout are excluded.\n")
    add("markdown", "## Configuration and execution control\n\nDefault review does not train. "
        "Only an explicit hosted execution may enable training. All checkpoints remain in Kaggle Output.\n")
    add("code", f"RUN_TRAINING = False\nSOURCE_COMMIT = {commit!r}\n")
    add("markdown", "## Environment and inherited source\n\nUse the completed private multisource worker output "
        "as a reproducible source/data attachment. No credentials or network clone is needed. Only relevant new implementation appears below.\n")
    add("code", "from pathlib import Path\nimport os, sys, json, shutil, subprocess, tarfile\n"
        "if RUN_TRAINING:\n"
        "    if not Path('/kaggle/working').is_dir():\n        raise RuntimeError('Hosted Kaggle required')\n"
        "    roots = [p.parent.parent for p in Path('/kaggle/input').rglob('train_multisource_pilot.py') "
        "if (p.parent.parent / 'data/processed/panel.npz').exists()]\n"
        "    if len(roots) != 1:\n        raise RuntimeError('Attach exactly one canonical multisource output')\n"
        "    PROJECT = Path('/kaggle/working/hurdle_research')\n    PROJECT.mkdir(exist_ok=True)\n"
        "    for folder in ('src', 'scripts', 'configs', 'data/processed'):\n"
        "        shutil.copytree(roots[0]/folder, PROJECT/folder, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))\n"
        "    os.chdir(PROJECT)\n    sys.path.insert(0, str(PROJECT))\n"
        "    os.environ['SOURCE_GIT_COMMIT'] = SOURCE_COMMIT\n"
        "    Path('reports').mkdir(exist_ok=True)\n")
    add("markdown", "## Fixed protocol\n\nFour positive conditional quantiles and one active-probability classifier. "
        "No class reweighting or tuned gate: probability must retain its probabilistic interpretation.\n")
    cfg = yaml.safe_load(Path("configs/hurdle_research.yaml").read_text(encoding="utf-8"))
    add("code", "# Fixed protocol: changing these values creates a new experiment.\n"
        f"PROTOCOL = {pprint.pformat(cfg, sort_dicts=False, width=90)}\n"
        "if RUN_TRAINING:\n    import yaml\n"
        "    Path('configs/hurdle_research.yaml').write_text(yaml.safe_dump(PROTOCOL), encoding='utf-8')\n")
    add("markdown", "## Model: mixture median\n\nFor active probability p≤0.5, predict zero. Otherwise use "
        "positive conditional quantile q=1−0.5/p. This follows the mixture CDF at its median. "
        "Interpolation and sorted crossing repair approximate that quantile; probability miscalibration can hurt MAE.\n")
    add("code", Path("src/models/hurdle.py").read_text(encoding="utf-8"))
    add("markdown", "## Training implementation\n\nThe attached MIT source contains the cloud trainer, "
        "inference helper and fixed YAML. The trainer keeps fit, early stopping and score arrays separate. "
        "It saves per-component rounds, checkpoints and predictions; it never selects on scoring outcomes.\n")
    add("code", "if RUN_TRAINING:\n"
        "    patches = list(Path('/kaggle/input').rglob('train_hurdle_research.py'))\n"
        "    if len(patches) != 1:\n        raise RuntimeError('Attach exactly one hurdle source dataset')\n"
        "    patch = patches[0].parent\n"
        "    shutil.copyfile(patch/'train_hurdle_research.py', 'scripts/train_hurdle_research.py')\n"
        "    shutil.copyfile(patch/'hurdle.py', 'src/models/hurdle.py')\n")
    add("markdown", "## Hosted training and debugging\n\nOne classifier, four positive quantile models and "
        "one direct L1 control per fold/horizon. Checkpoints are saved after each completed fold/horizon. "
        "Failures preserve logs and partial output. LightGBM binary/quantile objectives follow "
        "[official parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html).\n")
    add("code", "if RUN_TRAINING:\n"
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', 'lightgbm>=4.6,<5', 'numpy', 'pandas', 'pyyaml'], check=True)\n"
        "    with Path('reports/hurdle_training.log').open('w') as log:\n"
        "        process = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.train_hurdle_research'], "
        "stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
        "        for line in process.stdout:\n            print(line, end='', flush=True)\n            log.write(line)\n            log.flush()\n"
        "        status = process.wait()\n"
        "    with tarfile.open('/kaggle/working/hurdle_results.tar.gz', 'w:gz') as archive:\n"
        "        for folder in ('reports', 'configs', 'artifacts/hurdle_research'):\n"
        "            if Path(folder).exists():\n                archive.add(folder, arcname=folder)\n"
        "    if status:\n        raise RuntimeError(f'Cloud fitting failed: {status}; see saved log')\n"
        "else:\n    print('Review mode. No fitting, calibration or holdout evaluation.')\n")
    add("markdown", "## Saved evaluation and limitations\n\nMAE uses original units; WAPE is a ratio "
        "on the full observed scoring population. Report every fold/horizon and zero false-positive rate. "
        "A lower zero false-positive rate alone does not prove improvement. No probability calibration or unseen-test claim.\n")
    add("code", "path = Path('reports/hurdle_development_scores.csv')\n"
        "if path.exists():\n    import pandas as pd\n    from IPython.display import display\n"
        "    display(pd.read_csv(path))\nelse:\n    print('No completed scores attached.')\n")
    notebook = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"}},
                "nbformat": 4, "nbformat_minor": 5}
    name = "13_zero_inflated_forecasting.ipynb"
    Path("notebooks", name).write_text(json.dumps(notebook, indent=2, ensure_ascii=True), encoding="utf-8")
    control = next(c for c in cells if c["cell_type"] == "code" and "".join(c["source"]).startswith("RUN_TRAINING = False"))
    control["source"] = [f"RUN_TRAINING = True\nSOURCE_COMMIT = {commit!r}\n"]
    stage = Path(".cache/kaggle-zero-inflated-forecasting")
    stage.mkdir(parents=True, exist_ok=True)
    (stage / name).write_text(json.dumps(notebook, indent=2, ensure_ascii=True), encoding="utf-8")
    metadata = {"id": "truongbaokien/5g-network-intelligence-zero-inflated-forecasting",
                "title": "5G Network Intelligence Zero Inflated Forecasting", "code_file": name,
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": False, "enable_internet": True,
                "kernel_sources": ["truongbaokien/5g-network-intelligence-multisource-pilot"],
                "dataset_sources": ["truongbaokien/5g-network-intelligence-hurdle-source"], "competition_sources": []}
    (stage / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    source = Path(".cache/kaggle-hurdle-source")
    source.mkdir(parents=True, exist_ok=True)
    for path in ("src/models/hurdle.py", "scripts/train_hurdle_research.py", "configs/hurdle_research.yaml", "LICENSE"):
        shutil.copyfile(path, source / Path(path).name)
    (source / "dataset-metadata.json").write_text(json.dumps({
        "id": "truongbaokien/5g-network-intelligence-hurdle-source",
        "title": "5G Network Intelligence Hurdle Source", "licenses": [{"name": "MIT"}]
    }, indent=2))
    print(stage)


if __name__ == "__main__":
    main()
