"""Professional review guide plus an explicitly enabled hosted GPU worker."""

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
        cell = {"cell_type": kind, "metadata": {}, "id": f"temporal-linear-{len(cells):02d}", "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", "# 5G Network Intelligence — Temporal Linear Forecasting\n\n"
        "## Research question\n\nDoes a global linear projection of the complete 168-hour history "
        "capture useful temporal information beyond the current feature-based candidates? Compare "
        "target-only A with four-KPI B using identical temporal windows and beam populations. "
        "This is an independent adaptation of [NLinear](https://github.com/cure-lab/LTSF-Linear), "
        "not a reproduction of its published benchmark scores.\n\n"
        "## Protocol\n\nFit/stopping/scoring cutoffs are 335/383/443 and 395/443/503. "
        "Stopping selects the epoch only. Score every eligible per-horizon origin and all original beams. "
        "These reused development windows are exploratory. Validation 504–671 and final holdout are excluded.\n")
    add("code", f"RUN_TRAINING = False\nSOURCE_COMMIT = {commit!r}\n")
    add("markdown", "## Configuration\n\nFour hosted GPU fits: A/B on two folds, one seed. "
        "All beams participate in fitting; no beam identity or summary is a numerical model input.\n")
    cfg = yaml.safe_load(Path("configs/temporal_linear.yaml").read_text())
    add("code", "PROTOCOL = " + pprint.pformat(cfg,sort_dicts=False,width=90) + "\n")
    add("markdown", "## Preprocessing and architecture\n\nFit global per-KPI median/mean/std only "
        "on each training prefix; preserve missing indicators. Subtract the last imputed/scaled physical "
        "KPI value from its history; keep masks unchanged. Flatten the 168-hour window, append phase4, "
        "project linearly to 1/6/24h, then restore the target's last value. The initial weights reproduce "
        "the causal seven-day seasonal mean. Full implementation: `src/models/temporal_linear.py`. "
        "No neural fitting occurs in review mode.\n")
    add("markdown", "## Hosted source and canonical inputs\n\nAttach the completed original hurdle "
        "development output for shared code and canonical panel, plus the small MIT temporal-linear "
        "source dataset. Only the new model, trainer and protocol are copied from that source attachment.\n")
    add("code", "from pathlib import Path\nimport os, sys, shutil, subprocess, tarfile\n"
        "if RUN_TRAINING:\n"
        "    if not Path('/kaggle/working').is_dir():\n        raise RuntimeError('Hosted Kaggle required')\n"
        "    roots = [p.parent.parent for p in Path('/kaggle/input').rglob('train_hurdle_research.py') "
        "if (p.parent.parent/'data/processed/panel.npz').exists()]\n"
        "    if len(roots) != 1:\n        raise RuntimeError('Attach exactly one canonical source output')\n"
        "    PROJECT = Path('/kaggle/working/temporal_linear')\n    PROJECT.mkdir(exist_ok=True)\n"
        "    for folder in ('src','scripts','configs','data/processed'):\n"
        "        shutil.copytree(roots[0]/folder,PROJECT/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))\n"
        "    os.chdir(PROJECT)\n    sys.path.insert(0,str(PROJECT))\n"
        "    os.environ['SOURCE_GIT_COMMIT'] = SOURCE_COMMIT\n    Path('reports').mkdir(exist_ok=True)\n"
        "    patches = list(Path('/kaggle/input').rglob('train_temporal_linear.py'))\n"
        "    if len(patches) != 1:\n        raise RuntimeError('Attach exactly one temporal-linear source')\n"
        "    patch = patches[0].parent\n"
        "    for filename,folder in (('train_temporal_linear.py','scripts'),('temporal_linear.py','src/models'),('temporal_linear.yaml','configs')):\n"
        "        shutil.copyfile(patch/filename,Path(folder)/filename)\n")
    add("markdown", "## GPU verification\n\nUse the hosted image's PyTorch installation. "
        "A missing GPU stops fitting; there is no local or CPU training fallback.\n")
    add("code", "if RUN_TRAINING:\n"
        "    import torch\n    print('PyTorch:',torch.__version__)\n"
        "    if not torch.cuda.is_available():\n        raise RuntimeError('CUDA GPU is required')\n"
        "    subprocess.run(['nvidia-smi'],check=True)\n")
    add("markdown", "## Training and debugging\n\nThe shared masked-L1 trainer writes best/last "
        "checkpoint, optimizer/RNG state and per-epoch history. AdamW weight decay is fixed at its "
        "0.01 default. The stopping score is joint scaled L1; scoring reports original-unit MAE after "
        "nonnegative clipping. Failures preserve partial artifacts and a bounded progress log.\n")
    add("code", "if RUN_TRAINING:\n"
        "    with Path('reports/temporal_linear_training.log').open('w') as log:\n"
        "        process = subprocess.Popen([sys.executable,'-u','-m','scripts.train_temporal_linear'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)\n"
        "        for line in process.stdout:\n            print(line,end='',flush=True)\n            log.write(line)\n            log.flush()\n"
        "        status = process.wait()\n"
        "    with tarfile.open('/kaggle/working/temporal_linear_results.tar.gz','w:gz') as archive:\n"
        "        for folder in ('reports','artifacts/temporal_linear'):\n"
        "            if Path(folder).exists():\n                archive.add(folder,arcname=folder)\n"
        "    if status:\n        raise RuntimeError('Training failed; inspect saved log and checkpoints')\n"
        "else:\n    print('Review mode: no fitting, backward or optimizer steps.')\n")
    add("markdown", "## Scoring, negative results and limitations\n\nEvery horizon has its own complete "
        "scoring origins, so short horizons are not restricted by +24h label availability. Report MAE "
        "in original units, WAPE as a ratio, coverage and zero false-positive rate. One seed and reused "
        "development folds cannot establish generalization. No production promotion or holdout replay.\n")
    add("code", "paths = [Path('reports/temporal_linear_development.csv'),Path('../reports/temporal_linear_development.csv')]\n"
        "if Path('/kaggle/input').is_dir():\n    paths += list(Path('/kaggle/input').rglob('temporal_linear_development.csv'))\n"
        "path = next((p for p in paths if p.exists()),None)\n"
        "if path:\n    import pandas as pd\n    from IPython.display import display\n    display(pd.read_csv(path))\n"
        "else:\n    print('No completed scores attached; no improvement is assumed.')\n")
    notebook = {"cells":cells,"metadata":{"kernelspec":{"name":"python3","language":"python","display_name":"Python 3"}},"nbformat":4,"nbformat_minor":5}
    name = "16_temporal_linear_forecasting.ipynb"
    Path("notebooks",name).write_text(json.dumps(notebook,indent=2),encoding="utf-8")
    control = next(c for c in cells if c["cell_type"]=="code" and "".join(c["source"]).startswith("RUN_TRAINING = False"))
    control["source"] = [f"RUN_TRAINING = True\nSOURCE_COMMIT = {commit!r}\n"]
    stage = Path(".cache/kaggle-temporal-linear")
    stage.mkdir(parents=True,exist_ok=True)
    (stage/name).write_text(json.dumps(notebook,indent=2),encoding="utf-8")
    (stage/"kernel-metadata.json").write_text(json.dumps({
        "id":"truongbaokien/5g-network-intelligence-temporal-linear",
        "title":"5G Network Intelligence Temporal Linear","code_file":name,"language":"python","kernel_type":"notebook",
        "is_private":True,"enable_gpu":True,"enable_internet":False,
        "kernel_sources":["truongbaokien/5g-network-intelligence-zero-inflated-forecasting"],
        "dataset_sources":["truongbaokien/5g-network-intelligence-temporal-linear-source"],"competition_sources":[]},indent=2))
    source = Path(".cache/kaggle-temporal-linear-source")
    source.mkdir(parents=True,exist_ok=True)
    for path in ("scripts/train_temporal_linear.py","src/models/temporal_linear.py","configs/temporal_linear.yaml","LICENSE"):
        shutil.copyfile(path,source/Path(path).name)
    (source/"dataset-metadata.json").write_text(json.dumps({
        "id":"truongbaokien/5g-network-intelligence-temporal-linear-source",
        "title":"5G Network Intelligence Temporal Linear Source","licenses":[{"name":"MIT"}]},indent=2))
    print(stage)


if __name__ == "__main__":
    main()
