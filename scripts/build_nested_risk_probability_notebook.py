"""Readable probability study guide plus explicit hosted training copy."""

import json
import pprint
import shutil
import subprocess
from pathlib import Path

import yaml


def main():
    commit=subprocess.check_output(["git","-c","safe.directory=D:/5G","rev-parse","HEAD"],text=True).strip()
    cells=[]

    def add(kind,text):
        cell={"cell_type":kind,"metadata":{},"id":f"nested-risk-probability-{len(cells):02d}","source":text.splitlines(keepends=True)}
        if kind=="code":
            cell.update(execution_count=None,outputs=[])
        cells.append(cell)

    add("markdown","# 5G Network Intelligence - Nested Historical Exceedance Probability\n\n"
        "## Hypothesis and prior evidence\n\nIndependent P90/P95 classifiers have useful probability signal but "
        "incoherent nested-event pairs and weak fixed-0.5 F1. Reuse the pinned P90 classifier; train only "
        "P(P95 exceedance | P90 exceedance, history). Multiply the two probabilities. This guarantees "
        "nesting by construction, but may worsen Brier, log loss or ranking. No improvement is assumed.\n\n"
        "## Temporal protocol\n\nKeep the two original fit/stop/score windows, ending by hour 503. "
        "Reuse saved fit-prefix thresholds unchanged. Observed warning labels filter conditional "
        "fit/stopping samples only; they never enter features or inference-time routing. Scoring includes "
        "all eligible targets, including non-warning events. Reused windows are not fresh-test evidence. "
        "Validation 504-671, final holdout and production remain excluded.\n")
    add("code",f"RUN_TRAINING = False\nSOURCE_COMMIT = {commit!r}\n")
    add("markdown","## Fixed model and metrics\n\nSix additional binary LightGBM fits: two folds and three "
        "horizons, with unchanged 40 causal features and original hyperparameters. No class weighting, "
        "threshold search or calibration. Compare nested P95 probabilities with the saved independent "
        "P95 model on identical targets. Report Brier, log loss, average precision and fixed-0.5 "
        "classification metrics, including regressions. Warning probabilities are unchanged. "
        "This study does not improve or replace the original MAE/WAPE objective.\n")
    cfg=yaml.safe_load(Path("configs/nested_risk_probability.yaml").read_text())
    add("code","PROTOCOL = "+pprint.pformat(cfg,sort_dicts=False,width=90)+"\n")
    add("markdown","## Environment and source\n\nAttach the completed original probability worker for "
        "canonical data, pinned probability artifacts and inherited causal feature code. A separate private MIT source dataset supplies "
        "the new trainer, probability metrics and pinned configurations. Full source remains in named modules.\n")
    add("code","from pathlib import Path\nimport os,sys,shutil,subprocess,tarfile\n"
        "if RUN_TRAINING:\n"
        "    if not Path('/kaggle/working').is_dir():\n        raise RuntimeError('Hosted Kaggle required')\n"
        "    roots=[p.parent.parent for p in Path('/kaggle/input').rglob('risk_probability_development.csv') if (p.parent.parent/'data/processed/panel.npz').exists()]\n"
        "    if len(roots)!=1:\n        raise RuntimeError('Attach exactly one completed canonical source')\n"
        "    SOURCE=roots[0]/'artifacts/risk_probability'\n    PROJECT=Path('/kaggle/working/nested_risk_probability')\n    PROJECT.mkdir(exist_ok=True)\n"
        "    for folder in ('src','scripts','configs','data/processed'):\n"
        "        shutil.copytree(roots[0]/folder,PROJECT/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))\n"
        "    os.chdir(PROJECT)\n    sys.path.insert(0,str(PROJECT))\n    os.environ['SOURCE_GIT_COMMIT']=SOURCE_COMMIT\n"
        "    patches=list(Path('/kaggle/input').rglob('train_nested_risk_probability.py'))\n"
        "    if len(patches)!=1:\n        raise RuntimeError('Attach exactly one probability study source')\n"
        "    patch=patches[0].parent\n"
        "    for name,folder in (('train_nested_risk_probability.py','scripts'),('nested_probability.py','src/risk'),('probability_metrics.py','src/risk'),('nested_risk_probability.yaml','configs'),('risk.yaml','configs')):\n"
        "        shutil.copyfile(patch/name,Path(folder)/name)\n"
        "    Path('reports').mkdir(exist_ok=True)\n")
    add("markdown","## Hosted training and debugging\n\nReuse fold thresholds unchanged; save conditional checkpoints, stopping "
        "rounds, score arrays and provenance. Preserve missing targets as missing labels; do not score only warning-positive targets. "
        "Save outputs after every classifier; a failure must retain logs rather than report success.\n")
    add("code","if RUN_TRAINING:\n"
        "    with Path('reports/nested_risk_probability_training.log').open('w') as log:\n"
        "        process=subprocess.Popen([sys.executable,'-u','-m','scripts.train_nested_risk_probability','--source-root',str(SOURCE)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)\n"
        "        for line in process.stdout:\n            print(line,end='',flush=True)\n            log.write(line)\n            log.flush()\n"
        "        status=process.wait()\n"
        "    with tarfile.open('/kaggle/working/nested_risk_probability_results.tar.gz','w:gz') as archive:\n"
        "        for folder in ('reports','artifacts/nested_risk_probability'):\n"
        "            if Path(folder).exists():\n                archive.add(folder,arcname=folder)\n"
        "    if status:\n        raise RuntimeError(f'Hosted probability study failed: {status}; inspect saved logs')\n"
        "else:\n    print('Review only: no fitting or calibration.')\n")
    add("markdown","## Results and limitations\n\nReport both thresholds at every horizon/fold, eligible "
        "counts and undefined metrics. Better ranking or Brier score does not imply calibrated deployment "
        "probabilities or a useful operating point. Historical quantiles are not operator limits or congestion labels. "
        "A later unseen-data study is required before any production change.\n")
    add("code","paths=[Path('reports/nested_risk_probability_development.csv'),Path('../reports/nested_risk_probability_development.csv')]\n"
        "if Path('/kaggle/input').is_dir():\n    paths+=list(Path('/kaggle/input').rglob('nested_risk_probability_development.csv'))\n"
        "path=next((p for p in paths if p.is_file()),None)\n"
        "if path:\n    import pandas as pd\n    from IPython.display import display\n    display(pd.read_csv(path))\n"
        "else:\n    print('No completed probability report attached.')\n")
    notebook={"cells":cells,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"}},"nbformat":4,"nbformat_minor":5}
    name="25_nested_historical_exceedance_probability.ipynb"
    Path("notebooks",name).write_text(json.dumps(notebook,indent=2),encoding="utf-8")
    cells[1]["source"]=[f"RUN_TRAINING = True\nSOURCE_COMMIT = {commit!r}\n"]
    stage=Path(".cache/kaggle-nested-risk-probability")
    stage.mkdir(parents=True,exist_ok=True)
    (stage/name).write_text(json.dumps(notebook,indent=2),encoding="utf-8")
    (stage/"kernel-metadata.json").write_text(json.dumps({"id":"truongbaokien/5g-network-intelligence-nested-risk-probability",
        "title":"5G Network Intelligence Nested Risk Probability","code_file":name,"language":"python","kernel_type":"notebook",
        "is_private":True,"enable_gpu":False,"enable_internet":False,
        "kernel_sources":["truongbaokien/5g-network-intelligence-risk-probability"],
        "dataset_sources":["truongbaokien/5g-network-intelligence-nested-risk-source"],"competition_sources":[]},indent=2))
    package=Path(".cache/kaggle-nested-risk-probability-source")
    package.mkdir(parents=True,exist_ok=True)
    for path in ("scripts/train_nested_risk_probability.py","src/risk/nested_probability.py","src/risk/probability_metrics.py","configs/nested_risk_probability.yaml","configs/risk.yaml","LICENSE"):
        shutil.copyfile(path,package/Path(path).name)
    (package/"dataset-metadata.json").write_text(json.dumps({"id":"truongbaokien/5g-network-intelligence-nested-risk-source",
        "title":"5G Network Intelligence Nested Risk Source","licenses":[{"name":"MIT"}]},indent=2))


if __name__ == "__main__":
    main()
