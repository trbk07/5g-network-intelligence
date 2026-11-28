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

The completed cell-attention screen also fails to establish a material gain:
spatial mixing improves three of six MAEs versus its matched local control,
without consistently beating the current blend. The original GPU attention
fastpath caused inference drift. A separate frozen-checkpoint math replay
reproduced original forecasts and achieved CPU/GPU parity within 1.91e-6 units.
Future training code explicitly disables that fastpath for stopping/scoring.
The already-running multi-corpus v1 uses its historical source snapshot and
requires the same backend-consistency verification before quality acceptance.

### Reproducibility correction after v1 intake

Cold-initialization reconstruction exposed a second v1 defect: the lightweight
`set_seed` helper seeded Python/NumPy only, and the new cell controller omitted
the explicit Torch seeding used by earlier neural studies. Thus v1 point scores
and strict replay are valid numerical observations, but neither the local/cell
comparison nor the source-pretraining comparison is a controlled shared-start
ablation. Do not attribute differences to architecture/data or certify seed 17
for those neural initializations. Preserve all original snapshots and this
negative finding; the original release and older seeded neural studies are not
changed by this correction.

The separate v2 studies explicitly seed CPU Torch and all CUDA generators, save
source/target cold weights, hash the pre/post-transfer states, and use strict
attention for stopping/scoring. New artifact roots, configs, private workers and
notebooks prevent overwriting v1. All 100-epoch/12-patience and 5,120-step budgets
stay fixed. Exact cold-weight and transfer checks are mandatory acceptance gates.

An additional seed-29 baseline is declared before viewing v2 scoring results.
It uses Colab when the Kaggle baseline is queued, fetching the four pinned public
GOTS-F train files directly from the publisher rather than uploading local beam
payloads. Canonical prefix hashes must match before fitting, and hours 504 onward
are redacted. Report both seeds and runtimes; this is seed/runtime robustness,
not pure seed variance or permission to cherry-pick the best seed. All checkpoints
remain on Drive. The seed-17 multi-corpus controls remain paired within Kaggle.

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

The expanded data study is now authorized in parallel with the structural
screen. `configs/cell_multisource.yaml` freezes longer source pretraining:
5,120 updates per run, with a compute-matched target-self-pretraining control.
The four Vigo domains and CESNET-AGG23 add 40,653 eligible prefix entity-hours;
they join the audited radio and CESNET24 prefixes. Transfer encoder weights
only, discard the source head, and fine-tune physical DLPRB outputs on each
original fit prefix. Publisher-scaled Vigo data requires explicit provenance
and past-context normalization; this does not certify publisher preprocessing.
The 5.12 GB Telecom Italia intake remains outside the frozen experiment until
its own acquisition, continuity and overlap checks pass.

Time and source-count limits are not the optimization objective. Use two folds
for screening, then additional seeds and valid walk-forward windows for strong
candidates. Five highly overlapping windows are not five independent tests.
Retain the original chronological exclusions when allocating additional budget.

If spatial attention improves over its matched control and the strong baseline,
test stability before any fresh-data deployment decision. If it fails, inspect
zero/positive error and temporal structure, then choose a genuinely different
representation or suitable external pretraining source. Do not optimize the
number of downloaded files or mix incompatible units as the same target.

The existing original holdout and hours 504 onward stay excluded from all new
tuning. Repeated development comparisons cannot establish fresh generalization.
No production promotion or final-validation refit is implicit in this plan.
