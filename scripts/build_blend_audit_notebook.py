"""Professional fixed-refit blend audit guide and explicitly enabled hosted worker."""

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
        cell={"cell_type":kind,"metadata":{},"id":f"blend-audit-{len(cells):02d}","source":text.splitlines(keepends=True)}
        if kind=="code":
            cell.update(execution_count=None,outputs=[])
        cells.append(cell)

    add("markdown", "# 5G Network Intelligence — Fixed Blend Candidate Audit\n\n"
        "## Development evidence and question\n\nThe blend chosen on inner1 reduced inner2 MAE "
        "0.318% / 0.556% / 0.216% at 1/6/24h. It is fixed at 20% B for 1/6h and 20% A for 24h. "
        "Test whether the effect persists on reused validation; do not change these coefficients.\n\n"
        "## Frozen audit procedure\n\nRefit A/B through target hour 503 for the rounded median of "
        "development best_epoch+1 (A=4, B=3). No early stopping. Reuse the previously verified hurdle "
        "validation forecasts unchanged. Only after refitting, audit 504–671 on all original eligible "
        "beams. No hyperparameter search, calibration, final holdout replay or production promotion.\n")
    add("code",f"RUN_TRAINING = False\nSOURCE_COMMIT = {commit!r}\n")
    add("markdown", "## Fixed protocol and pinned evidence\n\nThe saved selection and hurdle replay "
        "must match their pinned SHA256 values. The original hurdle is not retrained.\n")
    cfg=yaml.safe_load(Path("configs/blend_candidate_audit.yaml").read_text())
    add("code","AUDIT_PROTOCOL = "+pprint.pformat(cfg,sort_dicts=False,width=90)+"\n")
    add("markdown", "## Completed hosted source inputs\n\nAttach Temporal Linear, development blend "
        "and the original zero-inflated candidate audit outputs. A small MIT source attachment supplies "
        "the new fixed-epoch trainer and forecast arithmetic. No repository source dump is embedded.\n")
    add("code", "from pathlib import Path\nimport os,sys,shutil,subprocess,tarfile\n"
        "if RUN_TRAINING:\n"
        "    if not Path('/kaggle/working').is_dir():\n        raise RuntimeError('Hosted Kaggle required')\n"
        "    linear_roots=[p.parent.parent for p in Path('/kaggle/input').rglob('temporal_linear_development.csv') if (p.parent.parent/'artifacts/temporal_linear/manifest.json').exists()]\n"
        "    blend_roots=[p.parent.parent for p in Path('/kaggle/input').rglob('development_blend_confirmation.csv') if (p.parent.parent/'artifacts/development_blend/manifest.json').exists()]\n"
        "    hurdle_replays=list(Path('/kaggle/input').rglob('hurdle_candidate_val.npz'))\n"
        "    if len(linear_roots)!=1 or len(blend_roots)!=1 or len(hurdle_replays)!=1:\n        raise RuntimeError('Attach exactly one completed output for each required source')\n"
        "    LINEAR=linear_roots[0]/'artifacts/temporal_linear'\n    BLEND=blend_roots[0]/'artifacts/development_blend'\n    HURDLE_REPLAY=hurdle_replays[0]\n"
        "    PROJECT=Path('/kaggle/working/blend_candidate')\n    PROJECT.mkdir(exist_ok=True)\n"
        "    for folder in ('src','scripts','configs','data/processed'):\n"
        "        shutil.copytree(linear_roots[0]/folder,PROJECT/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))\n"
        "    os.chdir(PROJECT)\n    sys.path.insert(0,str(PROJECT))\n    os.environ['SOURCE_GIT_COMMIT']=SOURCE_COMMIT\n    Path('reports').mkdir(exist_ok=True)\n"
        "    patches=list(Path('/kaggle/input').rglob('finalize_blend_candidate.py'))\n"
        "    if len(patches)!=1:\n        raise RuntimeError('Attach exactly one fixed-audit source')\n"
        "    patch=patches[0].parent\n"
        "    for name,folder in (('finalize_blend_candidate.py','scripts'),('forecast_blend.py','src/models'),('blend_candidate_audit.yaml','configs')):\n"
        "        shutil.copyfile(patch/name,Path(folder)/name)\n")
    add("markdown", "## GPU and fitting controls\n\nUse the hosted image's PyTorch. Missing CUDA "
        "stops fitting. Save every epoch's final state and training loss; do not choose a best epoch "
        "from validation. The last fixed epoch is the candidate.\n")
    add("code", "if RUN_TRAINING:\n"
        "    import torch\n    print('PyTorch',torch.__version__)\n"
        "    if not torch.cuda.is_available():\n        raise RuntimeError('Hosted CUDA GPU required')\n"
        "    subprocess.run(['nvidia-smi'],check=True)\n")
    add("markdown", "## Fixed refit and audit\n\nTwo neural fits, seven epochs in total. "
        "After fitting, construct the declared mixture and retain all component/audit scores. "
        "A validation gain is exploratory evidence, not an unseen-test guarantee.\n")
    add("code", "if RUN_TRAINING:\n"
        "    command=[sys.executable,'-u','-m','scripts.finalize_blend_candidate','--linear-root',str(LINEAR),'--blend-root',str(BLEND),'--hurdle-replay',str(HURDLE_REPLAY)]\n"
        "    with Path('reports/blend_candidate_training.log').open('w') as log:\n"
        "        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)\n"
        "        for line in process.stdout:\n            print(line,end='',flush=True)\n            log.write(line)\n            log.flush()\n"
        "        status=process.wait()\n"
        "    with tarfile.open('/kaggle/working/blend_candidate_results.tar.gz','w:gz') as archive:\n"
        "        for folder in ('reports','artifacts/blend_candidate'):\n"
        "            if Path(folder).exists():\n                archive.add(folder,arcname=folder)\n"
        "    if status:\n        raise RuntimeError('Fixed audit failed; inspect saved logs/checkpoints')\n"
        "else:\n    print('Review mode: no fitting or calibration.')\n")
    add("markdown", "## Saved metrics and limitations\n\nMAE uses original units, WAPE is a ratio. "
        "Report every horizon and full original population. Keep negative results, inspect zero "
        "false-positive rates and independently verify checkpoint/forecast consistency after download. "
        "Repeated validation cannot prove no overfitting or future performance. Production is unchanged.\n")
    add("code", "paths=[Path('reports/blend_candidate_validation.csv'),Path('../reports/blend_candidate_validation.csv')]\n"
        "if Path('/kaggle/input').is_dir():\n    paths+=list(Path('/kaggle/input').rglob('blend_candidate_validation.csv'))\n"
        "path=next((p for p in paths if p.exists()),None)\n"
        "if path:\n    import pandas as pd\n    from IPython.display import display\n    frame=pd.read_csv(path)\n    display(frame.loc[frame.load_tier=='All',['model','horizon','mae','wape','n_valid']])\n"
        "else:\n    print('No completed audit scores attached.')\n")
    add("markdown", "## Completed v1 audit and trade-offs\n\nFixed blend MAE was 0.226834 / "
        "0.250427 / 0.249162 at 1/6/24h. WAPE was 40.86% / 44.42% / 44.94%. "
        "This improves hurdle at 1/6h but slightly regresses at 24h. The 24h exploratory "
        "base-station interval crosses zero; NR3500 transfer retains lower 6/24h point MAE. "
        "Zero false-positive rates also worsen. All ten artifact files, full blend arithmetic and "
        "81 history-only component predictions were verified. Keep the declared weights, record "
        "the regression, and do not promote from reused validation. The 10% WAPE target remains unmet.\n")
    notebook={"cells":cells,"metadata":{"kernelspec":{"name":"python3","language":"python","display_name":"Python 3"}},"nbformat":4,"nbformat_minor":5}
    name="18_fixed_blend_candidate_audit.ipynb"
    Path("notebooks",name).write_text(json.dumps(notebook,indent=2),encoding="utf-8")
    control=next(c for c in cells if c["cell_type"]=="code" and "".join(c["source"]).startswith("RUN_TRAINING = False"))
    control["source"]=[f"RUN_TRAINING = True\nSOURCE_COMMIT = {commit!r}\n"]
    stage=Path(".cache/kaggle-fixed-blend-audit")
    stage.mkdir(parents=True,exist_ok=True)
    (stage/name).write_text(json.dumps(notebook,indent=2),encoding="utf-8")
    (stage/"kernel-metadata.json").write_text(json.dumps({
        "id":"truongbaokien/5g-network-intelligence-fixed-blend-audit","title":"5G Network Intelligence Fixed Blend Audit",
        "code_file":name,"language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":True,"enable_internet":False,
        "kernel_sources":["truongbaokien/5g-network-intelligence-temporal-linear","truongbaokien/5g-network-intelligence-development-blend","truongbaokien/5g-network-intelligence-zero-inflated-audit"],
        "dataset_sources":["truongbaokien/5g-network-intelligence-fixed-blend-source"],"competition_sources":[]},indent=2))
    source=Path(".cache/kaggle-fixed-blend-source")
    source.mkdir(parents=True,exist_ok=True)
    for path in ("scripts/finalize_blend_candidate.py","src/models/forecast_blend.py","configs/blend_candidate_audit.yaml","LICENSE"):
        shutil.copyfile(path,source/Path(path).name)
    (source/"dataset-metadata.json").write_text(json.dumps({"id":"truongbaokien/5g-network-intelligence-fixed-blend-source",
        "title":"5G Network Intelligence Fixed Blend Source","licenses":[{"name":"MIT"}]},indent=2))
    print(stage)


if __name__=="__main__":
    main()
