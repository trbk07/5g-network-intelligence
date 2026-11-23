"""Stage a readable review notebook and an explicit hosted execution copy."""

import json
import shutil
import subprocess
from pathlib import Path


def main():
    commit = subprocess.check_output(["git","-c","safe.directory=D:/5G","rev-parse","HEAD"],text=True).strip()
    cells = []

    def add(kind,text):
        cell = {"cell_type":kind,"id":f"quantile-resolution-{len(cells):02d}","metadata":{},"source":text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None,outputs=[])
        cells.append(cell)

    add("markdown","# 5G Network Intelligence - Positive Quantile Resolution\n\n"
        "## Hypothesis and prior evidence\n\nPositive targets contribute over 92% of hurdle absolute error on development. "
        "Test whether the original sparse conditional grid loses useful resolution when interpolating the mixture median. "
        "More quantiles may add estimation noise or alter crossing repair; improvement is not assumed.\n\n"
        "## Temporal protocol\n\nKeep fit/stop/score boundaries 335/383/443 and 395/443/503. "
        "Every original observed beam target and per-horizon origin is scored. Stopping selects rounds only. "
        "Both windows are reused development evidence, not independent generalization tests. "
        "Validation 504-671, final holdout and production are excluded.\n")
    add("code",f"RUN_TRAINING = False\nSOURCE_COMMIT = {commit!r}\n")
    add("markdown","## Controlled intervention\n\nReuse the exact saved classifier and quantiles 0.05, 0.1, 0.25, 0.5. "
        "Add fixed levels 0.15, 0.2, 0.3, 0.35, 0.4, 0.45 with the original LightGBM parameters and patience. "
        "This is 36 new cloud fits; no grid selection, threshold change or base-component retraining. "
        "Sorted crossing repair is unchanged, so the intervention measures the complete finer-grid pipeline, "
        "not interpolation error in isolation. Full implementation: `scripts/train_hurdle_quantile_resolution.py`.\n")
    add("markdown","## Environment, source and canonical panel\n\nAttach the completed original hurdle worker. "
        "Verify its artifact manifest before fitting. Readable trainer/config files come from a separate private MIT source dataset.\n")
    add("code","from pathlib import Path\nimport os, sys, shutil, subprocess, tarfile\n"
        "if RUN_TRAINING:\n"
        "    if not Path('/kaggle/working').is_dir():\n        raise RuntimeError('Hosted Kaggle required')\n"
        "    roots = [p.parent.parent for p in Path('/kaggle/input').rglob('hurdle_development_scores.csv') if (p.parent.parent/'artifacts/hurdle_research/manifest.json').exists()]\n"
        "    if len(roots) != 1:\n        raise RuntimeError('Attach exactly one completed hurdle output')\n"
        "    SOURCE = roots[0]/'artifacts/hurdle_research'\n"
        "    PROJECT = Path('/kaggle/working/quantile_resolution')\n    PROJECT.mkdir(exist_ok=True)\n"
        "    for folder in ('src','scripts','configs','data/processed'):\n"
        "        shutil.copytree(roots[0]/folder,PROJECT/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))\n"
        "    os.chdir(PROJECT)\n    sys.path.insert(0,str(PROJECT))\n    os.environ['SOURCE_GIT_COMMIT']=SOURCE_COMMIT\n"
        "    patches=list(Path('/kaggle/input').rglob('train_hurdle_quantile_resolution.py'))\n"
        "    if len(patches)!=1:\n        raise RuntimeError('Attach exactly one study source dataset')\n"
        "    patch=patches[0].parent\n"
        "    shutil.copyfile(patch/'train_hurdle_quantile_resolution.py','scripts/train_hurdle_quantile_resolution.py')\n"
        "    shutil.copyfile(patch/'hurdle_quantile_resolution.yaml','configs/hurdle_quantile_resolution.yaml')\n"
        "    Path('reports').mkdir(exist_ok=True)\n")
    add("markdown","## Training and debugging\n\nOnly positive targets enter each additional quantile fit. "
        "Stopping data is separate from score data. Save each new component immediately and preserve logs on failure. "
        "Original forecasts must replay exactly before comparing the refined grid. Checkpoints remain in Kaggle Output.\n")
    add("code","if RUN_TRAINING:\n"
        "    with Path('reports/hurdle_quantile_resolution_training.log').open('w') as log:\n"
        "        process=subprocess.Popen([sys.executable,'-u','-m','scripts.train_hurdle_quantile_resolution','--source-root',str(SOURCE)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)\n"
        "        for line in process.stdout:\n            print(line,end='',flush=True)\n            log.write(line)\n            log.flush()\n"
        "        status=process.wait()\n"
        "    with tarfile.open('/kaggle/working/quantile_resolution_results.tar.gz','w:gz') as archive:\n"
        "        for folder in ('reports','artifacts/hurdle_quantile_resolution'):\n"
        "            if Path(folder).exists():\n                archive.add(folder,arcname=folder)\n"
        "    if status:\n        raise RuntimeError(f'Hosted fitting failed: {status}; inspect saved log')\n"
        "else:\n    print('Review mode: no fitting, calibration or holdout evaluation.')\n")
    add("markdown","## Evaluation and limitations\n\nReport original-unit MAE and full-population WAPE (ratio, not percent). "
        "Keep negative results and all six fold/horizon comparisons. No automatic validation refit or promotion. "
        "One seed and reused windows cannot certify unseen-data accuracy or the WAPE <10% objective.\n")
    add("code","paths=[Path('reports/hurdle_quantile_resolution_development.csv'),Path('../reports/hurdle_quantile_resolution_development.csv')]\n"
        "if Path('/kaggle/input').is_dir():\n    paths+=list(Path('/kaggle/input').rglob('hurdle_quantile_resolution_development.csv'))\n"
        "path=next((p for p in paths if p.is_file()),None)\n"
        "if path:\n    import pandas as pd\n    from IPython.display import display\n    display(pd.read_csv(path))\n"
        "else:\n    print('No completed study report attached.')\n")
    notebook={"cells":cells,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"}},"nbformat":4,"nbformat_minor":5}
    name="21_positive_quantile_resolution.ipynb"
    Path("notebooks",name).write_text(json.dumps(notebook,indent=2),encoding="utf-8")
    cells[1]["source"]=[f"RUN_TRAINING = True\nSOURCE_COMMIT = {commit!r}\n"]
    stage=Path(".cache/kaggle-quantile-resolution")
    stage.mkdir(parents=True,exist_ok=True)
    (stage/name).write_text(json.dumps(notebook,indent=2),encoding="utf-8")
    (stage/"kernel-metadata.json").write_text(json.dumps({"id":"truongbaokien/5g-network-intelligence-quantile-resolution",
        "title":"5G Network Intelligence Positive Quantile Resolution","code_file":name,"language":"python","kernel_type":"notebook",
        "is_private":True,"enable_gpu":False,"enable_internet":False,"kernel_sources":["truongbaokien/5g-network-intelligence-zero-inflated-forecasting"],
        "dataset_sources":["truongbaokien/5g-network-intelligence-quantile-source"],"competition_sources":[]},indent=2))
    package=Path(".cache/kaggle-quantile-source")
    package.mkdir(parents=True,exist_ok=True)
    for path in ("scripts/train_hurdle_quantile_resolution.py","configs/hurdle_quantile_resolution.yaml","LICENSE"):
        shutil.copyfile(path,package/Path(path).name)
    (package/"dataset-metadata.json").write_text(json.dumps({"id":"truongbaokien/5g-network-intelligence-quantile-source",
        "title":"5G Network Intelligence Quantile Resolution Source","licenses":[{"name":"MIT"}]},indent=2))


if __name__ == "__main__":
    main()
