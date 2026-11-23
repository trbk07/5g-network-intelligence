"""Fit-only descriptive cross-beam residual association; no model fitting."""

import json
from pathlib import Path

import numpy as np

from src.data.schema import parse_beam_id
from src.evaluation.validation import validation_inputs
from src.utils.config import load_config


def association(residual,groups):
    values=[]
    attempted=0
    for group in groups:
        block=residual[:,group]
        # Original fit values are complete; do not invent missing-value correlations.
        complete=np.isfinite(block).all(0)
        observed=block[:,complete]
        if observed.shape[1]<2:
            continue
        attempted+=len(group)*(len(group)-1)//2
        with np.errstate(invalid="ignore",divide="ignore"):
            correlation=np.corrcoef(observed,rowvar=False)
        upper=correlation[np.triu_indices(len(correlation),1)]
        values.extend(upper[np.isfinite(upper)].tolist())
    array=np.asarray(values)
    if not len(array):
        raise ValueError("No finite paired cell correlations")
    return {"attempted_pairs":attempted,"finite_pairs":len(array),"mean":float(array.mean()),
            "median":float(np.median(array)),"p10":float(np.quantile(array,.1)),"p90":float(np.quantile(array,.9))}


def main():
    panel,_=validation_inputs(load_config("configs/data.yaml"))
    target=panel.values[:504,:,0]
    times=np.arange(168,504)
    historical=np.stack([target[times-24*day] for day in range(1,8)])
    residual=target[times]-np.nanmean(historical,axis=0)
    cells={}
    for index,beam in enumerate(panel.beam_ids):
        bs,cell,_=parse_beam_id(beam)
        cells.setdefault((bs,cell),[]).append(index)
    if len(cells)!=90 or any(len(group)!=32 for group in cells.values()):
        raise ValueError("Expected canonical 90 cells with 32 beams each")
    groups=list(cells.values())
    permutation=np.random.default_rng(17).permutation(len(panel.beam_ids))
    shuffled=[permutation[start:start+32] for start in range(0,len(permutation),32)]
    result={"scope":"descriptive fit segment 0-503 only, residual times 168-503",
        "residual":"target minus causal mean of target at same phase over previous seven days",
        "same_cell":association(residual,groups),"shuffled_group_control":association(residual,shuffled),
        "n_cells":len(groups),"group_size":32,"seed":17,"training_performed":False,
        "interpretation":"simultaneous association is not future predictability, causation or an irreducible-error bound; no model inputs or test split changed"}
    Path("reports/cell_dependence_audit.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
