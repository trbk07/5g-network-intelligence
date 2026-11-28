# Model quality extension: structural improvement before marginal blending

The owner authorizes changing the original plan, longer hosted training and
additional suitable datasets to pursue materially better forecasting. Preserve
the historical eight-week release and its evaluation; this extension does not
rewrite earlier experimental decisions or authorize holdout tuning.

## Current evidence

The verified Chronos-2 LoRA experiment improves its pretrained control but loses
to the current stronger blend at five of six development points. The previous
external-data MLP pretraining also failed to establish stable gains. Neither
finding justifies another small blend search or a claim that more data alone
will deliver WAPE below 10%.

## Next structural experiment

Test raw within-cell spatial information, absent from the per-beam models. Each
sample contains 168 past hours for all 32 beams of one cell, four KPIs and missing
masks. Use shared temporal patches and spatial attention, with an identical
parameter-count control that blocks off-diagonal attention. No fitted beam
statistics, identity embeddings, future covariate actuals or learned graph from
scoring targets are allowed. Topology is parsed from existing station/cell IDs.

Frozen protocol: `configs/cell_attention.yaml`. Four fits: two folds and two
modes, seed 17, maximum 100 epochs, patience 12, physical-unit masked MAE.
Checkpoint optimizer/RNG state every epoch in Kaggle Output or Drive. A cap is
not completed early stopping; inspect termination before any continuation.

Aim for at least 5% mean relative MAE improvement with no horizon regression
against the fixed strong control. The initial 2% candidate gate only qualifies
further investigation. Require additional seeds, paired time-block uncertainty,
full original-population coverage, forecast reconstruction and CPU/GPU parity.
Keep mixed and negative results; never expand a search after confirmation to
retroactively make it pass. More training does not guarantee a material gain.

## Subsequent decisions

If spatial attention improves over its matched control and the strong baseline,
test stability before any fresh-data deployment decision. If it fails, inspect
zero/positive error and temporal structure, then choose a genuinely different
representation or suitable external pretraining source. Do not optimize the
number of downloaded files or mix incompatible units as the same target.

The existing original holdout and hours 504 onward stay excluded from all new
tuning. Repeated development comparisons cannot establish fresh generalization.
No production promotion or final-validation refit is implicit in this plan.
