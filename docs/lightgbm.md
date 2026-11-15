# Week 3 - Global direct LightGBM

## Reproduce

Install the ML extra in the project environment, prepare data and run baselines first:

```powershell
uv pip install --python .venv/Scripts/python.exe -e ".[dev,ml]"
.venv/Scripts/python.exe -m scripts.prepare_data --step all
.venv/Scripts/python.exe -m scripts.run_baselines --segment val
.venv/Scripts/python.exe -m scripts.train_lgbm
.venv/Scripts/python.exe -m scripts.evaluate_models --model lightgbm --segment val
```

Training is CPU only, four threads, seed 17, deterministic column-wise histograms.
Exactly two configurations are run, each with one direct model for horizons 1/6/24.
Both use 31 leaves, learning rate 0.05, maximum 500 rounds and validation-MAE
early stopping with patience 50. C1 uses absolute-error regression; C2 squared-error
regression. No further hyperparameter search is performed. The 500-round cap is
retained even when C2 selects an iteration near that cap.

## Features and boundaries

There are exactly 44 ordered float32 features, listed in reports/feature_manifest.json:
9 origin lags, 4 target-aligned seasonal lags and their observed-only 7-day seasonal
mean, 9 target rolling mean/std/max features, 2 target phase indices, 15 auxiliary
KPI lag/rolling features and 4 fit-only beam statistics. The seasonal mean uses
all seven daily references, not just the four individually named seasonal lags.

Population rolling std uses ddof=0. NaN is preserved; missing-only windows and
fit-only statistics remain NaN. No imputation, zero replacement, scaling, clipping
or calendar timestamp inference is performed. Only missing targets are removed
from fitting/validation, with counts recorded in artifact metadata.

Features are computed in arrays ordered origin-major/beam-major. Rolling reductions
use actual per-origin windows, vectorized over beams, without materializing all
sliding windows. This also ensures exact float32 parity between batch and replay
features; prefix subtraction could otherwise introduce tiny values in zero windows.

Static mean/std/P90/active ratio are computed once from the declared fit prefix.
Training features may use those training-fitted statistics, as specified in the plan;
validation features never fit statistics again. Training stride is 2 and evaluation
stride 1. Both training and replay load a validation-truncated panel, preventing
holdout observations from being used during Week 3. No holdout runner is provided.

The evaluator supports `predict_at(history, origin, horizon)` for models needing
the absolute phase. Only history through the origin is supplied; origin is integer
time context, not future telemetry. Plain predict without time context is rejected
for LightGBM. The static beam order must match the input panel.

## Artifacts and model selection

`artifacts/lgbm/{c1|c2}/h{1|6|24}/` contains model.txt, meta.json and static.npz.
Artifacts and forecast NPZ files are local generated files excluded from Git.
Saved text models contain only the selected best iteration. Metadata records
training config, seed, source commit, dataset revision, target exclusions and
the full validation learning history. Static artifacts include beam IDs and fit_end.

The evaluation command reloads artifacts and checks dataset revision, feature schema,
beam order, fit boundary and model parameters. Batch predictions must match independent
history-window replay at three validation origins for each of the six models.
Checksums for every artifact and the forecast archive are saved in reports.

Config selection minimizes mean normalized MAE across horizons, dividing each
horizon MAE by that horizon's best validation baseline MAE. The result is stored
in configs/lightgbm.yaml. This selection is not the Week 5 champion freeze.
Re-running evaluate_models reproduces metrics without retraining; the runner
rejects holdout evaluation. All validation comparisons use the same origin sets.

See reports/lightgbm_analysis.md for observed results, load-tier/active/zero metrics,
feature gain importance, top beam errors, at least three failure examples and skill
scores. Importance is descriptive, not causal attribution. Large residuals do not
establish congestion or faults. Week 3 validation results are not final test results.
