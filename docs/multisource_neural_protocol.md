# Multisource neural pretraining pilot

This extension tests a training recipe, not a production replacement. The
source intake and numerical input check are complete; no cloud fit or MAE gain
is established by these checks.

The first pilot uses a univariate 64-unit residual MLP to make the representation
transfer explicit. Five fixed external epochs each sample 65,536 windows. The
last epoch is transferred; the external in-prefix diagnostic is never used for
epoch selection. All cases use the same target fine-tuning budget and frozen
inner1/inner2 fit, stopping and scoring boundaries. Original validation and
holdout are excluded. See `configs/multisource_neural_pilot.yaml`.

Radio-only, CESNET-only and pooled cases have the same external optimization
budget. In the pool, CESNET receives half the sampling mass and the three radio
domains share the other half. Sampling is uniform over entities within a source,
then proportional to eligible origins across that entity's continuous segments.
Scaling is global per source using prepared training prefixes, never per entity.
All targets remain in their own source scale during pretraining.

The target-only case has no external optimization stage. Therefore its comparison
with a pretrained case measures the whole recipe, including extra optimization;
it cannot isolate the causal contribution of additional datasets on its own.
The implemented fifth case, target-self-pretraining, samples the same fixed
number of windows and epochs from each fold's original fit prefix. Its global
preprocessor is fitted at 335 or 395 respectively; maximum supervised timestamps
cannot exceed those cutoffs. It never pretrains through 503 for reuse in an
earlier fold. Fine-tuning starts a fresh optimizer and restores the pre-stage
Torch RNG stream. Actual valid-target and update counts must still be checked
after training because missing original labels can affect supervision density.

Checkpoints include optimizer and Torch CPU/CUDA RNG state. External pretraining
can resume its matching `last.pt`; a partially completed target study must be
inspected before restart because the existing target trainer rejects overwrite.
Colab checkpoint and report paths must resolve to mounted Drive. Kaggle runs
retain artifacts in Output. All fitting is guarded against local execution.

Launch review must verify source checksums, radio overlap audit, current source
commit, storage, CUDA, all control definitions and compute budget. After execution,
replay saved forecasts and check full-population coverage and inference parity.
An epoch cap must be reported separately from completed early-stopping patience.
Negative transfer must remain visible. Reused development scores do not prove
unseen-network generalization or attainment of WAPE below 10%.

Reader notebook: `notebooks/27_multisource_representation_pretraining.ipynb`.
Its default Run All displays available evidence and never fits a model.
