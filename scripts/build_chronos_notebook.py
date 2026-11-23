"""Readable pretrained zero-shot study; no local model installation or weights."""

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
        cell={"cell_type":kind,"metadata":{},"id":f"zero-shot-{len(cells):02d}","source":text.splitlines(keepends=True)}
        if kind=="code":
            cell.update(execution_count=None,outputs=[])
        cells.append(cell)

    add("markdown", "# 5G Network Intelligence — Pretrained Zero-Shot Forecasting\n\n"
        "## Hypothesis and provenance limits\n\nCan a fixed pretrained forecaster use the 168-hour "
        "target history better than the current development references? Test one pinned Chronos-Bolt "
        "Base checkpoint and its native median; do not fine-tune or select quantiles. Model-card license: "
        "Apache-2.0. The pinned weights appear in public history before the target dataset's first "
        "visible commit, but the full pretraining corpus has not been audited. This is not a no-leakage "
        "certification or an unseen-test claim. [Author model card](https://huggingface.co/amazon/chronos-bolt-base).\n\n"
        "## Temporal protocol\n\nReuse development scoring windows after 383 through 443 and after "
        "443 through 503. Forecast all original beams from exactly 168 observed target values. "
        "No validation 504–671, original final holdout or production promotion.\n")
    add("code",f"RUN_INFERENCE = False\nSOURCE_COMMIT = {commit!r}\n")
    add("markdown", "## Fixed configuration\n\nOne checkpoint, FP32 GPU inference, batch 256 and "
        "quantile 0.5. All weights are frozen. Negative point forecasts are clipped to zero as declared "
        "before scoring. No data-dependent calibration, model search or local weight download.\n")
    cfg=yaml.safe_load(Path("configs/chronos_zero_shot.yaml").read_text())
    add("code","PROTOCOL = "+pprint.pformat(cfg,sort_dicts=False,width=90)+"\n")
    add("markdown", "## Input window and missingness\n\nThe next cell defines target-only history "
        "slicing. Observed zeros remain zeros; missing values remain NaN for the model's native mask. "
        "No future row or non-target KPI enters the context.\n")
    add("code",Path("src/features/zero_shot_context.py").read_text(encoding="utf-8"))
    add("markdown", "## Hosted setup and audited source\n\nAttach the canonical completed hurdle "
        "output and the project zero-shot source dataset. The pretrained library wheel is downloaded "
        "from its recorded public PyPI URL and verified against its SHA256 before installation. "
        "Model weights download only inside this explicitly enabled hosted worker.\n")
    add("code", "from pathlib import Path\nimport os,sys,json,shutil,subprocess,tarfile,hashlib\nfrom urllib.request import urlopen\n"
        "if RUN_INFERENCE:\n"
        "    if not Path('/kaggle/working').is_dir():\n        raise RuntimeError('Hosted Kaggle required')\n"
        "    roots=[p.parent.parent for p in Path('/kaggle/input').rglob('train_hurdle_research.py') if (p.parent.parent/'data/processed/panel.npz').exists()]\n"
        "    if len(roots)!=1:\n        raise RuntimeError('Attach exactly one canonical source output')\n"
        "    PROJECT=Path('/kaggle/working/chronos_zero_shot')\n    PROJECT.mkdir(exist_ok=True)\n"
        "    for folder in ('src','configs','data/processed'):\n"
        "        shutil.copytree(roots[0]/folder,PROJECT/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))\n"
        "    os.chdir(PROJECT)\n    sys.path.insert(0,str(PROJECT))\n    os.environ['SOURCE_GIT_COMMIT']=SOURCE_COMMIT\n"
        "    for folder in ('scripts','reports'):\n        Path(folder).mkdir(exist_ok=True)\n"
        "    Path('scripts/__init__.py').touch()\n"
        "    patches=list(Path('/kaggle/input').rglob('evaluate_chronos_zero_shot.py'))\n"
        "    if len(patches)!=1:\n        raise RuntimeError('Attach exactly one zero-shot source')\n"
        "    patch=patches[0].parent\n"
        "    for name,folder in (('evaluate_chronos_zero_shot.py','scripts'),('zero_shot_context.py','src/features'),('chronos_zero_shot.yaml','configs'),('pretrained_forecaster_source_audit.json','reports')):\n"
        "        shutil.copyfile(patch/name,Path(folder)/name)\n"
        "    audit=json.loads((patch/'pretrained_forecaster_source_audit.json').read_text())\n"
        "    wheel=audit['library_distribution']\n    wheel_path=PROJECT/wheel['filename']\n"
        "    with urlopen(wheel['url'],timeout=60) as response:\n        wheel_path.write_bytes(response.read())\n"
        "    if hashlib.sha256(wheel_path.read_bytes()).hexdigest()!=wheel['sha256']:\n        raise RuntimeError('Audited library wheel checksum mismatch')\n"
        "    os.environ['USE_TF']='0'\n    os.environ['USE_FLAX']='0'\n"
        "    subprocess.run([sys.executable,'-m','pip','install',str(wheel_path),'transformers=='+audit['transformers_version_pin']],check=True)\n")
    add("markdown", "## Frozen GPU inference and debugging\n\nThe evaluator verifies the safetensors "
        "hash, disables gradients, keeps FP32 and saves a forecast cache after each origin. "
        "No optimizer or fitting operation occurs. The source checkpoint remains separately in "
        "Kaggle Output; the result archive contains project reports and forecasts, not copied model weights.\n")
    add("code", "if RUN_INFERENCE:\n"
        "    with Path('reports/chronos_zero_shot.log').open('w') as log:\n"
        "        process=subprocess.Popen([sys.executable,'-u','-m','scripts.evaluate_chronos_zero_shot'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)\n"
        "        for line in process.stdout:\n            print(line,end='',flush=True)\n            log.write(line)\n            log.flush()\n"
        "        status=process.wait()\n"
        "    with tarfile.open('/kaggle/working/chronos_zero_shot_results.tar.gz','w:gz') as archive:\n"
        "        for folder in ('reports','artifacts/chronos_zero_shot'):\n"
        "            if Path(folder).exists():\n                archive.add(folder,arcname=folder)\n"
        "    if status:\n        raise RuntimeError('Zero-shot job failed; inspect saved log and partial cache')\n"
        "else:\n    print('Review mode: no model download, inference, fitting or calibration.')\n")
    add("markdown", "## Saved metrics and limitations\n\nMAE is in original usage units, WAPE is a "
        "ratio and every declared horizon retains its full original population. Scores remain exploratory "
        "because development has been reused and pretraining overlap is not certified absent. "
        "No improvement or calibrated uncertainty is assumed from the publisher's benchmark claims.\n")
    add("code", "paths=[Path('reports/chronos_zero_shot_development.csv'),Path('../reports/chronos_zero_shot_development.csv')]\n"
        "if Path('/kaggle/input').is_dir():\n    paths+=list(Path('/kaggle/input').rglob('chronos_zero_shot_development.csv'))\n"
        "path=next((p for p in paths if p.exists()),None)\n"
        "if path:\n    import pandas as pd\n    from IPython.display import display\n    display(pd.read_csv(path))\n"
        "else:\n    print('No completed zero-shot scores attached.')\n")
    add("markdown", "## Completed v1 finding and decision\n\nThe checkpoint improved 6h development "
        "MAE by 0.458% / 0.283% across the two folds, but 1h and 24h regressed in both. "
        "WAPE remained 40.51–47.08%, and zero false-positive rates were 99.81–99.93%. "
        "All 127 artifact files and mapping of 875,520 scored forecasts to their origin caches "
        "were checked; this did not independently execute the external model or certify its corpus. "
        "Do not promote this full-range standalone configuration or tune it on reused validation.\n")
    notebook={"cells":cells,"metadata":{"kernelspec":{"name":"python3","language":"python","display_name":"Python 3"}},"nbformat":4,"nbformat_minor":5}
    name="19_pretrained_zero_shot_forecasting.ipynb"
    Path("notebooks",name).write_text(json.dumps(notebook,indent=2),encoding="utf-8")
    control=next(c for c in cells if c["cell_type"]=="code" and "".join(c["source"]).startswith("RUN_INFERENCE = False"))
    control["source"]=[f"RUN_INFERENCE = True\nSOURCE_COMMIT = {commit!r}\n"]
    stage=Path(".cache/kaggle-chronos-zero-shot")
    stage.mkdir(parents=True,exist_ok=True)
    (stage/name).write_text(json.dumps(notebook,indent=2),encoding="utf-8")
    (stage/"kernel-metadata.json").write_text(json.dumps({
        "id":"truongbaokien/5g-network-intelligence-chronos-zero-shot","title":"5G Network Intelligence Chronos Zero Shot",
        "code_file":name,"language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":True,"enable_internet":True,
        "kernel_sources":["truongbaokien/5g-network-intelligence-zero-inflated-forecasting"],
        "dataset_sources":["truongbaokien/5g-network-intelligence-zero-shot-source"],"competition_sources":[]},indent=2))
    source=Path(".cache/kaggle-zero-shot-source")
    source.mkdir(parents=True,exist_ok=True)
    for path in ("scripts/evaluate_chronos_zero_shot.py","src/features/zero_shot_context.py","configs/chronos_zero_shot.yaml","reports/pretrained_forecaster_source_audit.json","LICENSE"):
        shutil.copyfile(path,source/Path(path).name)
    (source/'README.md').write_text('Project code: MIT. Referenced external checkpoint: Apache-2.0 per pinned author model card. No model weights or original beam series are included in this source dataset. Pretraining corpus overlap is not certified absent.\n')
    (source/"dataset-metadata.json").write_text(json.dumps({"id":"truongbaokien/5g-network-intelligence-zero-shot-source",
        "title":"5G Network Intelligence Zero Shot Source","licenses":[{"name":"MIT"}]},indent=2))
    print(stage)


if __name__=="__main__":
    main()
