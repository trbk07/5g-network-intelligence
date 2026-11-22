"""Readable hosted-only blend selection with a separate confirmation fold."""

import json
import pprint
import shutil
import subprocess
from pathlib import Path

import yaml


def main():
    commit = subprocess.check_output(["git","-c","safe.directory=D:/5G","rev-parse","HEAD"],text=True).strip()
    cells = []

    def add(kind,source):
        cell = {"cell_type":kind,"metadata":{},"id":f"blend-{len(cells):02d}","source":source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None,outputs=[])
        cells.append(cell)

    add("markdown", "# 5G Network Intelligence — Development Forecast Blending\n\n"
        "## Question and negative evidence\n\nTemporal Linear is worse as a standalone model in all "
        "12 comparisons. Signed-error correlation with hurdle is 0.91–0.95, so complementarity is limited "
        "and no benefit is assumed. Test only a small predefined convex blend, retaining weight zero.\n\n"
        "## Temporal protocol\n\nSelect variant and weight on inner1 only for each horizon. Save every "
        "choice before reading inner2 numerical forecasts/targets. Apply those choices unchanged on inner2. "
        "Do not optimize again after confirmation. Both folds are reused development evidence, not fresh "
        "unseen-test data. Validation 504–671, original final holdout and production are excluded.\n")
    add("code",f"RUN_SELECTION = False\nSOURCE_COMMIT = {commit!r}\n")
    add("markdown", "## Fixed configuration\n\nNo base-model fitting. Choosing coefficients is "
        "calibration and runs only on hosted Kaggle. Weight grid: 0, 0.05, 0.1, 0.2. "
        "Ties favor smaller weight, then A. Confirmation requires no horizon regression and at least "
        "one strict gain with nonzero weight. Selecting only zeros is not an improvement.\n")
    cfg = yaml.safe_load(Path("configs/development_blend.yaml").read_text())
    add("code","PROTOCOL = "+pprint.pformat(cfg,sort_dicts=False,width=90)+"\n")
    add("markdown", "## Prediction arithmetic\n\nThe following cell only defines the fixed-weight "
        "formula. It does not search coefficients or fit a model. Both forecasts must cover the same targets.\n")
    add("code",Path("src/models/forecast_blend.py").read_text(encoding="utf-8"))
    add("markdown", "## Completed source outputs\n\nAttach the original hurdle development job and "
        "Temporal Linear job. Their complete artifact manifests are verified before selection. "
        "Source, config and forecast provenance remain separate from fitted blend choices.\n")
    add("code", "from pathlib import Path\nimport os,sys,shutil,subprocess,tarfile\n"
        "if RUN_SELECTION:\n"
        "    if not Path('/kaggle/working').is_dir():\n        raise RuntimeError('Hosted Kaggle required for coefficient selection')\n"
        "    hurdle_roots = [p.parent.parent for p in Path('/kaggle/input').rglob('hurdle_development_scores.csv') if (p.parent.parent/'artifacts/hurdle_research/manifest.json').exists()]\n"
        "    linear_roots = [p.parent.parent for p in Path('/kaggle/input').rglob('temporal_linear_development.csv') if (p.parent.parent/'artifacts/temporal_linear/manifest.json').exists()]\n"
        "    if len(hurdle_roots)!=1 or len(linear_roots)!=1:\n        raise RuntimeError('Attach exactly one completed output for each base family')\n"
        "    HURDLE = hurdle_roots[0]/'artifacts/hurdle_research'\n"
        "    LINEAR = linear_roots[0]/'artifacts/temporal_linear'\n"
        "    PROJECT = Path('/kaggle/working/development_blend')\n    PROJECT.mkdir(exist_ok=True)\n"
        "    shutil.copytree(hurdle_roots[0]/'src',PROJECT/'src',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))\n"
        "    os.chdir(PROJECT)\n    sys.path.insert(0,str(PROJECT))\n"
        "    os.environ['SOURCE_GIT_COMMIT']=SOURCE_COMMIT\n"
        "    for folder in ('scripts','configs','reports'):\n        Path(folder).mkdir(exist_ok=True)\n"
        "    Path('scripts/__init__.py').touch()\n"
        "    patches = list(Path('/kaggle/input').rglob('select_development_blend.py'))\n"
        "    if len(patches)!=1:\n        raise RuntimeError('Attach exactly one blend source dataset')\n"
        "    patch = patches[0].parent\n"
        "    for name,folder in (('select_development_blend.py','scripts'),('forecast_blend.py','src/models'),('development_blend.yaml','configs')):\n"
        "        shutil.copyfile(patch/name,Path(folder)/name)\n")
    add("markdown", "## Selection, fixed confirmation and debugging\n\nCheckpoints remain immutable. "
        "Save the full inner1 grid, selected coefficients, inner2 forecasts and a single pass/fail decision. "
        "No search on inner2 and no automatic follow-up validation job.\n")
    add("code", "if RUN_SELECTION:\n"
        "    command=[sys.executable,'-u','-m','scripts.select_development_blend','--linear-root',str(LINEAR),'--hurdle-root',str(HURDLE)]\n"
        "    with Path('reports/development_blend.log').open('w') as log:\n"
        "        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)\n"
        "        for line in process.stdout:\n            print(line,end='',flush=True)\n            log.write(line)\n            log.flush()\n"
        "        status=process.wait()\n"
        "    with tarfile.open('/kaggle/working/development_blend_results.tar.gz','w:gz') as archive:\n"
        "        for folder in ('reports','artifacts/development_blend'):\n"
        "            if Path(folder).exists():\n                archive.add(folder,arcname=folder)\n"
        "    if status:\n        raise RuntimeError('Blend experiment failed; inspect saved log')\n"
        "else:\n    print('Review mode: no selection, fitting or calibration.')\n")
    add("markdown", "## Saved results and limitations\n\nMAE is in original units; WAPE is a ratio. "
        "Report all horizons including regressions. Reused development confirmation does not establish "
        "generalization or absence of overfitting. Zero-weight selection must be reported as no gain.\n")
    add("code", "for name in ('development_blend_selection_grid.csv','development_blend_confirmation.csv'):\n"
        "    paths=[Path('reports')/name,Path('../reports')/name]\n"
        "    if Path('/kaggle/input').is_dir():\n        paths+=list(Path('/kaggle/input').rglob(name))\n"
        "    path=next((p for p in paths if p.exists()),None)\n"
        "    if path:\n        import pandas as pd\n        from IPython.display import display\n        print(name)\n        display(pd.read_csv(path))\n"
        "    else:\n        print(f'{name}: no completed result attached')\n")
    notebook={"cells":cells,"metadata":{"kernelspec":{"name":"python3","language":"python","display_name":"Python 3"}},"nbformat":4,"nbformat_minor":5}
    name="17_development_forecast_blending.ipynb"
    Path("notebooks",name).write_text(json.dumps(notebook,indent=2),encoding="utf-8")
    control=next(c for c in cells if c["cell_type"]=="code" and "".join(c["source"]).startswith("RUN_SELECTION = False"))
    control["source"]=[f"RUN_SELECTION = True\nSOURCE_COMMIT = {commit!r}\n"]
    stage=Path(".cache/kaggle-development-blend")
    stage.mkdir(parents=True,exist_ok=True)
    (stage/name).write_text(json.dumps(notebook,indent=2),encoding="utf-8")
    (stage/"kernel-metadata.json").write_text(json.dumps({
        "id":"truongbaokien/5g-network-intelligence-development-blend","title":"5G Network Intelligence Development Blend",
        "code_file":name,"language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":False,"enable_internet":False,
        "kernel_sources":["truongbaokien/5g-network-intelligence-zero-inflated-forecasting","truongbaokien/5g-network-intelligence-temporal-linear"],
        "dataset_sources":["truongbaokien/5g-network-intelligence-blend-source"],"competition_sources":[]},indent=2))
    source=Path(".cache/kaggle-blend-source")
    source.mkdir(parents=True,exist_ok=True)
    for path in ("scripts/select_development_blend.py","src/models/forecast_blend.py","configs/development_blend.yaml","LICENSE"):
        shutil.copyfile(path,source/Path(path).name)
    (source/"dataset-metadata.json").write_text(json.dumps({"id":"truongbaokien/5g-network-intelligence-blend-source",
        "title":"5G Network Intelligence Blend Source","licenses":[{"name":"MIT"}]},indent=2))
    print(stage)


if __name__ == "__main__":
    main()
