# Available-target supervision: verified negative result

Both arms completed genuine patience stopping in Kaggle. The independent CPU
audit passed: exact artifact coverage, fit-only scalers, seed reconstruction,
bitwise matched cold states, target counts, optimizer exposure, checkpoint
selection, all saved metrics and forecast replay were verified. The replay
checked 2,304 history-only beam predictions; maximum CPU/GPU error was
0.00000190735, below the fixed 0.0001 tolerance. No local training or checkpoint
download was used.

| Fold | Arm | Epochs | Optimizer updates |
| --- | --- | ---: | ---: |
| inner1 | joint | 19 | 7,752 |
| inner1 | available | 19 | 8,987 |
| inner2 | joint | 23 | 13,271 |
| inner2 | available | 18 | 11,556 |

The available-label policy adds 118,080 eligible targets per fold but does not
yield a consistent accuracy gain. Its average relative MAE gain against the
new joint-label control is **-0.01031%**. Against the fixed strong comparator,
joint and available arms average **-0.49964%** and **-0.50902%**, respectively.
All declared initial/material quality gates fail. WAPE remains approximately
40–47%; the below-10% objective is unmet.

Full fold/horizon scores and paired comparisons are retained in
`partial_supervision_verified_metrics.json` and
`partial_supervision_paired_comparisons.csv`. Do not select a favorable horizon
or erase the negative rows. MAE is in physical DLPRB usage units, while WAPE is
summed absolute error divided by summed absolute observed target load.

Both arms use the same per-batch horizon-balanced training loss, unchanged
joint stopping windows, seed, architecture and scoring population. Natural
window counts and optimizer exposure differ, so this is not a pure label-count
effect. The folds are reused development, not fresh generalization evidence.
No production artifact is changed or promoted. Original hours 504 onward are
excluded. The next already-declared hypothesis is target-patch/past-covariate
attention; these negative scores do not justify a longer repeat of this recipe.
