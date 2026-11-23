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
        cell={"cell_type":kind,"metadata":{},"id":f"risk-probability-{len(cells):02d}","source":text.splitlines(keepends=True)}
        if kind=="code":
            cell.update(execution_count=None,outputs=[])
        cells.append(cell)

    add("markdown","# 5G Network Intelligence - Historical Exceedance Probability\n\n"
        "## Hypothesis and prior evidence\n\nThe MAE-focused MLP blend has low recall for upper historical "
        "exceedances. Test direct binary probability prediction rather than treating a median-like point forecast "
        "as a probability. No improvement or operational congestion detection is assumed.\n\n"
        "## Temporal protocol\n\nFit/stop/score boundaries remain 335/383/443 and 395/443/503. "
        "Construct each fold's per-beam P90/P95 from its fit prefix only, then freeze them for stopping/scoring. "
        "The saved hour-503 validation thresholds must not label earlier development windows. "
        "All eligible observed targets and original beam axes are retained. Reused development is not fresh-test evidence. "
        "Validation 504-671, final holdout and production are excluded.\n")
    add("code",f"RUN_TRAINING = False\nSOURCE_COMMIT = {commit!r}\n")
    add("markdown","## Fixed model and metrics\n\nTwelve LightGBM binary fits: two folds, three horizons, "
        "two historical thresholds. Keep 40 causal features, no class reweighting or calibration. "
        "Thresholds define labels and scoring; fitted beam summaries are not model inputs. "
        "Primary metrics are Brier, log loss and average precision. A fit-prevalence control uses no future labels. "
        "Report precision/recall/F1 at fixed probability 0.5; do not choose a threshold from scoring results. "
        "Probabilities do not use DLPRB units and do not replace the MAE/WAPE target.\n")
    cfg=yaml.safe_load(Path("configs/risk_probability.yaml").read_text())
    add("code","PROTOCOL = "+pprint.pformat(cfg,sort_dicts=False,width=90)+"\n")
    add("markdown","## Environment and source\n\nAttach the completed original hurdle worker for "
        "canonical data and inherited causal feature code. A separate private MIT source dataset supplies "
        "the new trainer, probability metrics and pinned configurations. Full source remains in named modules.\n")
    add("code","from pathlib import Path\nimport os,sys,shutil,subprocess,tarfile\n"
        "if RUN_TRAINING:\n"
        "    if not Path('/kaggle/working').is_dir():\n        raise RuntimeError('Hosted Kaggle required')\n"
        "    roots=[p.parent.parent for p in Path('/kaggle/input').rglob('hurdle_development_scores.csv') if (p.parent.parent/'data/processed/panel.npz').exists()]\n"
        "    if len(roots)!=1:\n        raise RuntimeError('Attach exactly one completed canonical source')\n"
        "    PROJECT=Path('/kaggle/working/risk_probability')\n    PROJECT.mkdir(exist_ok=True)\n"
        "    for folder in ('src','scripts','configs','data/processed'):\n"
        "        shutil.copytree(roots[0]/folder,PROJECT/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))\n"
        "    os.chdir(PROJECT)\n    sys.path.insert(0,str(PROJECT))\n    os.environ['SOURCE_GIT_COMMIT']=SOURCE_COMMIT\n"
        "    patches=list(Path('/kaggle/input').rglob('train_risk_probability.py'))\n"
        "    if len(patches)!=1:\n        raise RuntimeError('Attach exactly one probability study source')\n"
        "    patch=patches[0].parent\n"
        "    for name,folder in (('train_risk_probability.py','scripts'),('probability_metrics.py','src/risk'),('risk_probability.yaml','configs'),('risk.yaml','configs')):\n"
        "        shutil.copyfile(patch/name,Path(folder)/name)\n"
        "    Path('reports').mkdir(exist_ok=True)\n")
    add("markdown","## Hosted training and debugging\n\nSave fold thresholds, binary checkpoints, stopping "
        "rounds, score arrays and provenance. Preserve missing targets as missing labels. "
        "Save outputs after every classifier; a failure must retain logs rather than report success.\n")
    add("code","if RUN_TRAINING:\n"
        "    with Path('reports/risk_probability_training.log').open('w') as log:\n"
        "        process=subprocess.Popen([sys.executable,'-u','-m','scripts.train_risk_probability'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)\n"
        "        for line in process.stdout:\n            print(line,end='',flush=True)\n            log.write(line)\n            log.flush()\n"
        "        status=process.wait()\n"
        "    with tarfile.open('/kaggle/working/risk_probability_results.tar.gz','w:gz') as archive:\n"
        "        for folder in ('reports','artifacts/risk_probability'):\n"
        "            if Path(folder).exists():\n                archive.add(folder,arcname=folder)\n"
        "    if status:\n        raise RuntimeError(f'Hosted probability study failed: {status}; inspect saved logs')\n"
        "else:\n    print('Review only: no fitting or calibration.')\n")
    add("markdown","## Results and limitations\n\nReport both thresholds at every horizon/fold, eligible "
        "counts and undefined metrics. Better ranking or Brier score does not imply calibrated deployment "
        "probabilities or a useful operating point. Historical quantiles are not operator limits or congestion labels. "
        "A later unseen-data study is required before any production change.\n")
    add("code","paths=[Path('reports/risk_probability_development.csv'),Path('../reports/risk_probability_development.csv')]\n"
        "if Path('/kaggle/input').is_dir():\n    paths+=list(Path('/kaggle/input').rglob('risk_probability_development.csv'))\n"
        "path=next((p for p in paths if p.is_file()),None)\n"
        "if path:\n    import pandas as pd\n    from IPython.display import display\n    display(pd.read_csv(path))\n"
        "else:\n    print('No completed probability report attached.')\n")
    notebook={"cells":cells,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"}},"nbformat":4,"nbformat_minor":5}
    name="24_historical_exceedance_probability.ipynb"
    Path("notebooks",name).write_text(json.dumps(notebook,indent=2),encoding="utf-8")
    cells[1]["source"]=[f"RUN_TRAINING = True\nSOURCE_COMMIT = {commit!r}\n"]
    stage=Path(".cache/kaggle-risk-probability")
    stage.mkdir(parents=True,exist_ok=True)
    (stage/name).write_text(json.dumps(notebook,indent=2),encoding="utf-8")
    (stage/"kernel-metadata.json").write_text(json.dumps({"id":"truongbaokien/5g-network-intelligence-risk-probability",
        "title":"5G Network Intelligence Risk Probability","code_file":name,"language":"python","kernel_type":"notebook",
        "is_private":True,"enable_gpu":False,"enable_internet":False,
        "kernel_sources":["truongbaokien/5g-network-intelligence-zero-inflated-forecasting"],
        "dataset_sources":["truongbaokien/5g-network-intelligence-risk-probability-source"],"competition_sources":[]},indent=2))
    package=Path(".cache/kaggle-risk-probability-source")
    package.mkdir(parents=True,exist_ok=True)
    for path in ("scripts/train_risk_probability.py","src/risk/probability_metrics.py","configs/risk_probability.yaml","configs/risk.yaml","LICENSE"):
        shutil.copyfile(path,package/Path(path).name)
    (package/"dataset-metadata.json").write_text(json.dumps({"id":"truongbaokien/5g-network-intelligence-risk-probability-source",
        "title":"5G Network Intelligence Risk Probability Source","licenses":[{"name":"MIT"}]},indent=2))


if __name__ == "__main__":
    main()
