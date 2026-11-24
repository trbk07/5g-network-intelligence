"""Reuse the verified neural worker layout with an explicit nonlinear study contract."""

import json
import pprint
import shutil
import subprocess
from pathlib import Path

import yaml


def main():
    commit=subprocess.check_output(["git","-c","safe.directory=D:/5G","rev-parse","HEAD"],text=True).strip()
    source=json.loads(Path("notebooks/16_temporal_linear_forecasting.ipynb").read_text(encoding="utf-8"))
    if "Temporal Linear Forecasting" not in "".join(source["cells"][0]["source"]):
        raise ValueError("Expected the verified temporal worker template")
    cells=[]
    cfg=yaml.safe_load(Path("configs/temporal_tcn.yaml").read_text())
    for cell in source["cells"]:
        text="".join(cell["source"])
        if cell["cell_type"]=="markdown" and text.startswith("## Completed v1 result"):
            continue  # Never copy another model's historical result into the new study.
        text=text.replace("temporal_linear","temporal_tcn").replace("temporal-linear","temporal-TCN")
        if text.startswith("RUN_TRAINING = False"):
            text=f"RUN_TRAINING = False\nSOURCE_COMMIT = {commit!r}\n"
        elif text.startswith("PROTOCOL = "):
            text="PROTOCOL = "+pprint.pformat(cfg,sort_dicts=False,width=90)+"\n"
        elif cell["cell_type"]=="markdown" and text.startswith("## Preprocessing and architecture"):
            text=("## Preprocessing and architecture\n\nFit global per-KPI preprocessing only on each fit prefix. "
                "Keep physical KPI missing masks and phase4. Center physical history by the last observed-window "
                "value, project to 32 channels, and use seven residual blocks with kernel size 3 and dilations "
                "1/2/4/8/16/32/64. Left padding only; LayerNorm is per time across channels. The nominal receptive "
                "field is 255, covering the available 168-hour window. A zero-initialized head adds a residual "
                "to the causal seven-day seasonal target mean. Weights are shared across beams; no fitted beam "
                "summaries or neighbouring beam inputs. This is a compact adaptation, not a reproduction of the "
                "original TCN paper. Full source: `src/models/temporal_tcn.py`.\n")
        if cell["cell_type"]=="code" and "for filename,folder in" in text:
            text=text.replace("(('train_temporal_tcn.py','scripts'),", "(('train_temporal_mlp.py','scripts'),('train_temporal_tcn.py','scripts'),")
        cell["source"]=text.splitlines(keepends=True)
        cell["id"]=f"temporal-tcn-{len(cells):02d}"
        cells.append(cell)
    cells[0]["source"]=["# 5G Network Intelligence - Temporal Dilated Convolution\n\n"
        "## Hypothesis and prior evidence\n\nA nonlinear MLP improves the matched linear reference, but "
        "needs hurdle blending for better MAE. Test whether temporal weight sharing and dilated convolution "
        "learn useful local/multiscale structure with a compact parameter budget. A/B inputs, causal seasonal "
        "initialization, loss and optimizer policy match earlier neural studies; improvement is not assumed. "
        "Architecture inspiration: [Bai et al., 2018](https://arxiv.org/abs/1803.01271).\n\n"
        "## Temporal protocol\n\nFit/stop/score cutoffs remain 335/383/443 and 395/443/503. "
        "Stopping chooses epochs; scoring cannot tune the architecture. Retain every original beam and "
        "eligible per-horizon scoring origin. These reused windows cannot prove independent generalization. "
        "Validation 504-671, original holdout and production are excluded. No extra dataset is added.\n"]
    source["cells"]=cells
    name="26_temporal_dilated_convolution.ipynb"
    Path("notebooks",name).write_text(json.dumps(source,indent=2),encoding="utf-8")
    control=next(c for c in cells if c["cell_type"]=="code" and "".join(c["source"]).startswith("RUN_TRAINING = False"))
    control["source"]=[f"RUN_TRAINING = True\nSOURCE_COMMIT = {commit!r}\n"]
    stage=Path(".cache/kaggle-temporal-tcn")
    stage.mkdir(parents=True,exist_ok=True)
    (stage/name).write_text(json.dumps(source,indent=2),encoding="utf-8")
    (stage/"kernel-metadata.json").write_text(json.dumps({
        "id":"truongbaokien/5g-network-intelligence-temporal-tcn","title":"5G Network Intelligence Temporal TCN",
        "code_file":name,"language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":True,"enable_internet":False,
        "kernel_sources":["truongbaokien/5g-network-intelligence-temporal-residual-mlp"],
        "dataset_sources":["truongbaokien/5g-network-intelligence-temporal-tcn-source"],"competition_sources":[]},indent=2))
    package=Path(".cache/kaggle-temporal-tcn-source")
    package.mkdir(parents=True,exist_ok=True)
    for path in ("scripts/train_temporal_mlp.py","scripts/train_temporal_tcn.py","src/models/temporal_tcn.py","configs/temporal_tcn.yaml","LICENSE"):
        shutil.copyfile(path,package/Path(path).name)
    (package/"dataset-metadata.json").write_text(json.dumps({"id":"truongbaokien/5g-network-intelligence-temporal-tcn-source",
        "title":"5G Network Intelligence Temporal MLP Source","licenses":[{"name":"MIT"}]},indent=2))
    print(stage)


if __name__=="__main__":
    main()
