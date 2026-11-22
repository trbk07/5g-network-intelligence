"""Readable development-only probability-calibration notebook and hosted job."""

import json
import pprint
import shutil
import subprocess
from pathlib import Path

import yaml


def main():
    commit = subprocess.check_output(["git", "-c", "safe.directory=D:/5G", "rev-parse", "HEAD"], text=True).strip()
    cells = []

    def add(kind, source):
        cell = {"cell_type": kind, "metadata": {}, "id": f"calibration-{len(cells):02d}", "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", "# 5G Network Intelligence — Probability Calibration Study\n\n"
        "## Hypothesis\n\nCan an affine-logit correction improve active-load probability and the mixture median? "
        "Improved log loss need not improve MAE or zero detection. Preserve all negative results.\n\n"
        "## Temporal protocol\n\nReuse frozen development classifiers and positive quantile models. Fit one "
        "fixed-L2 logistic correction on each fold's stopping labels; evaluate only on its later scoring labels. "
        "Stopping data also selected the base model rounds: this limitation is explicit. No calibration on "
        "scoring targets, reused validation 504–671 or final holdout. These development folds supported prior "
        "research and cannot establish fresh-test generalization.\n")
    add("code", f"RUN_CALIBRATION = False\nSOURCE_COMMIT = {commit!r}\n")
    add("markdown", "## Fixed configuration\n\nOne method, C=1, no hyperparameter search. "
        "Six calibrators fit in hosted Kaggle only. The base trees and quantile models are unchanged.\n")
    cfg = yaml.safe_load(Path("configs/hurdle_calibration.yaml").read_text())
    add("code", "CALIBRATION_PROTOCOL = " + pprint.pformat(cfg, sort_dicts=False, width=90) + "\n")
    add("markdown", "## Probability transformation\n\nClip probabilities only for numerical stability, "
        "then apply sigmoid(a × logit(p) + b). The following cell defines inference arithmetic only. "
        "The two coefficients are fitted on cloud in a later guarded cell.\n")
    add("code", Path("src/models/probability_calibration.py").read_text(encoding="utf-8"))
    add("markdown", "## Hosted environment and saved checkpoints\n\nAttach the completed original "
        "zero-inflated development output and the MIT calibration source dataset. No repository source dump "
        "is embedded in this notebook. Base checkpoint hashes are checked before calibration.\n")
    add("code", "from pathlib import Path\nimport os, sys, json, shutil, subprocess, tarfile\n"
        "if RUN_CALIBRATION:\n"
        "    if not Path('/kaggle/working').is_dir():\n        raise RuntimeError('Calibration requires hosted Kaggle')\n"
        "    roots = [p.parent.parent for p in Path('/kaggle/input').rglob('train_hurdle_research.py') "
        "if (p.parent.parent/'artifacts/hurdle_research/manifest.json').exists()]\n"
        "    if len(roots) != 1:\n        raise RuntimeError('Attach exactly one development output')\n"
        "    PROJECT = Path('/kaggle/working/hurdle_calibration')\n    PROJECT.mkdir(exist_ok=True)\n"
        "    for folder in ('src','scripts','configs','data/processed','artifacts/hurdle_research'):\n"
        "        shutil.copytree(roots[0]/folder, PROJECT/folder, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))\n"
        "    os.chdir(PROJECT)\n    sys.path.insert(0,str(PROJECT))\n"
        "    os.environ['SOURCE_GIT_COMMIT'] = SOURCE_COMMIT\n    Path('reports').mkdir(exist_ok=True)\n")
    add("markdown", "## Install the focused implementation\n\nOnly the calibration trainer, inference "
        "helper and declared configuration come from the source attachment. Fitting code remains in "
        "`scripts/calibrate_hurdle_development.py` for focused review and debugging.\n")
    add("code", "if RUN_CALIBRATION:\n"
        "    patches = list(Path('/kaggle/input').rglob('calibrate_hurdle_development.py'))\n"
        "    if len(patches) != 1:\n        raise RuntimeError('Attach exactly one calibration source')\n"
        "    patch = patches[0].parent\n"
        "    for name, target in (('calibrate_hurdle_development.py','scripts'),('probability_calibration.py','src/models'),('hurdle_calibration.yaml','configs')):\n"
        "        shutil.copyfile(patch/name, Path(target)/name)\n")
    add("markdown", "## Cloud calibration and debugging\n\nConvergence failures stop the experiment. "
        "Save coefficients, scoring probabilities and predictions for independent verification. "
        "No new base-model fitting or quantile selection occurs.\n")
    add("code", "if RUN_CALIBRATION:\n"
        "    subprocess.run([sys.executable,'-m','pip','install','lightgbm>=4.6,<5','scikit-learn>=1.7,<1.10','numpy','pandas','pyyaml'], check=True)\n"
        "    with Path('reports/hurdle_calibration.log').open('w') as log:\n"
        "        process = subprocess.Popen([sys.executable,'-u','-m','scripts.calibrate_hurdle_development'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)\n"
        "        for line in process.stdout:\n            print(line,end='',flush=True)\n            log.write(line)\n            log.flush()\n"
        "        status = process.wait()\n"
        "    with tarfile.open('/kaggle/working/hurdle_calibration_results.tar.gz','w:gz') as archive:\n"
        "        for folder in ('reports','artifacts/hurdle_calibration'):\n"
        "            if Path(folder).exists():\n                archive.add(folder,arcname=folder)\n"
        "    if status:\n        raise RuntimeError('Calibration failed; inspect saved log')\n"
        "else:\n    print('Review mode: no fitting or calibration.')\n")
    add("markdown", "## Saved scoring results and limitations\n\nMAE uses original units; WAPE is a ratio. "
        "Report Brier score, log loss and zero false-positive rate with every fold/horizon. "
        "Compare with raw probabilities on exactly the same scoring targets. A calibration claim "
        "requires evidence, and these exploratory scores cannot establish no overfitting. "
        "[Method reference](https://scikit-learn.org/stable/modules/calibration.html).\n")
    add("code", "paths = [Path('reports/hurdle_calibration_development.csv'),Path('../reports/hurdle_calibration_development.csv')]\n"
        "if Path('/kaggle/input').is_dir():\n    paths += list(Path('/kaggle/input').rglob('hurdle_calibration_development.csv'))\n"
        "path = next((p for p in paths if p.exists()),None)\n"
        "if path:\n    import pandas as pd\n    from IPython.display import display\n    display(pd.read_csv(path))\n"
        "else:\n    print('No completed calibration scores attached.')\n")
    add("markdown", "## Completed v1 finding and decision\n\nThe correction worsened MAE at all three "
        "horizons in inner1 (relative changes −0.0058% to −0.0477%) and improved inner2 only "
        "slightly (+0.00075% to +0.01038%). Zero false-positive rates declined but remain 96.55–98.66%. "
        "Do not add this correction to the candidate or tune it against reused validation. "
        "All scoring probabilities, forecasts and metrics were independently reconstructed from saved "
        "checkpoints and coefficients. No production change; the 10% WAPE target remains unmet.\n")
    notebook = {"cells": cells,"metadata":{"kernelspec":{"name":"python3","language":"python","display_name":"Python 3"}},"nbformat":4,"nbformat_minor":5}
    name = "15_probability_calibration_study.ipynb"
    Path("notebooks",name).write_text(json.dumps(notebook,indent=2),encoding="utf-8")
    control = next(c for c in cells if c["cell_type"]=="code" and "".join(c["source"]).startswith("RUN_CALIBRATION = False"))
    control["source"] = [f"RUN_CALIBRATION = True\nSOURCE_COMMIT = {commit!r}\n"]
    stage = Path(".cache/kaggle-probability-calibration")
    stage.mkdir(parents=True,exist_ok=True)
    (stage/name).write_text(json.dumps(notebook,indent=2),encoding="utf-8")
    (stage/"kernel-metadata.json").write_text(json.dumps({
        "id":"truongbaokien/5g-network-intelligence-probability-calibration",
        "title":"5G Network Intelligence Probability Calibration","code_file":name,
        "language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":False,"enable_internet":True,
        "kernel_sources":["truongbaokien/5g-network-intelligence-zero-inflated-forecasting"],
        "dataset_sources":["truongbaokien/5g-network-intelligence-calibration-source"],"competition_sources":[]},indent=2))
    source = Path(".cache/kaggle-calibration-source")
    source.mkdir(parents=True,exist_ok=True)
    for path in ("scripts/calibrate_hurdle_development.py","src/models/probability_calibration.py","configs/hurdle_calibration.yaml","LICENSE"):
        shutil.copyfile(path,source/Path(path).name)
    (source/"dataset-metadata.json").write_text(json.dumps({
        "id":"truongbaokien/5g-network-intelligence-calibration-source",
        "title":"5G Network Intelligence Calibration Source","licenses":[{"name":"MIT"}]},indent=2))
    print(stage)


if __name__ == "__main__":
    main()
