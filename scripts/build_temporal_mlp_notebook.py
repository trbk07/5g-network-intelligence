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
    cfg=yaml.safe_load(Path("configs/temporal_mlp.yaml").read_text())
    for cell in source["cells"]:
        text="".join(cell["source"])
        if cell["cell_type"]=="markdown" and text.startswith("## Completed v1 result"):
            continue  # Never copy another model's historical result into the new study.
        text=text.replace("temporal_linear","temporal_mlp").replace("temporal-linear","temporal-MLP")
        if text.startswith("RUN_TRAINING = False"):
            text=f"RUN_TRAINING = False\nSOURCE_COMMIT = {commit!r}\n"
        elif text.startswith("PROTOCOL = "):
            text="PROTOCOL = "+pprint.pformat(cfg,sort_dicts=False,width=90)+"\n"
        elif cell["cell_type"]=="markdown" and text.startswith("## Preprocessing and architecture"):
            text=("## Preprocessing and architecture\n\nFit global per-KPI medians/means/std on the "
                "training prefix only. Preserve missing masks and phase4. Center physical history by its "
                "last imputed/scaled value; a 64-unit GELU network with dropout 0.1 predicts a nonlinear "
                "residual. Add the causal seven-day seasonal target mean. A zero-initialized final head "
                "starts from that analytical baseline. All weights are shared across beams; no fitted "
                "beam summary or neighbouring beam values enter. Full source: `src/models/temporal_mlp.py`.\n")
        cell["source"]=text.splitlines(keepends=True)
        cell["id"]=f"temporal-mlp-{len(cells):02d}"
        cells.append(cell)
    cells[0]["source"]=["# 5G Network Intelligence — Temporal Residual MLP\n\n"
        "## Hypothesis and prior evidence\n\nThe linear history model regressed at every tested "
        "horizon/fold. Test a small nonlinear residual network while keeping the same input policy, "
        "training budget and temporal windows. Compare one-KPI A and four-KPI B; no gain is assumed. "
        "A fit-only spatial diagnostic found weak average same-cell residual association, so graph "
        "complexity is not added in this study.\n\n"
        "## Temporal protocol\n\nFit/stopping/scoring cutoffs: 335/383/443 and 395/443/503. "
        "Stopping selects epochs, scoring cannot tune architecture. All original beams and complete "
        "per-horizon scoring origins are retained. These development folds were reused and cannot "
        "prove fresh-test generalization. Validation 504–671 and final holdout are excluded.\n"]
    source["cells"]=cells
    name="20_temporal_residual_mlp.ipynb"
    Path("notebooks",name).write_text(json.dumps(source,indent=2),encoding="utf-8")
    control=next(c for c in cells if c["cell_type"]=="code" and "".join(c["source"]).startswith("RUN_TRAINING = False"))
    control["source"]=[f"RUN_TRAINING = True\nSOURCE_COMMIT = {commit!r}\n"]
    stage=Path(".cache/kaggle-temporal-mlp")
    stage.mkdir(parents=True,exist_ok=True)
    (stage/name).write_text(json.dumps(source,indent=2),encoding="utf-8")
    (stage/"kernel-metadata.json").write_text(json.dumps({
        "id":"truongbaokien/5g-network-intelligence-temporal-residual-mlp","title":"5G Network Intelligence Temporal Residual MLP",
        "code_file":name,"language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":True,"enable_internet":False,
        "kernel_sources":["truongbaokien/5g-network-intelligence-temporal-linear"],
        "dataset_sources":["truongbaokien/5g-network-intelligence-temporal-mlp-source"],"competition_sources":[]},indent=2))
    package=Path(".cache/kaggle-temporal-mlp-source")
    package.mkdir(parents=True,exist_ok=True)
    for path in ("scripts/train_temporal_mlp.py","src/models/temporal_mlp.py","configs/temporal_mlp.yaml","LICENSE"):
        shutil.copyfile(path,package/Path(path).name)
    (package/"dataset-metadata.json").write_text(json.dumps({"id":"truongbaokien/5g-network-intelligence-temporal-mlp-source",
        "title":"5G Network Intelligence Temporal MLP Source","licenses":[{"name":"MIT"}]},indent=2))
    print(stage)


if __name__=="__main__":
    main()
