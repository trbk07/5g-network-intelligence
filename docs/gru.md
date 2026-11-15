# Week 4 GRU experiments

GRU-A uses DLPRB plus a missingness mask (2 channels). GRU-B uses four KPIs plus
four masks (8 channels). Both consume 168 hours, two GRU layers with hidden size
64 and dropout 0.1, then concatenate a 6-dimensional static vector before a
64-unit ReLU head predicting +1/+6/+24 jointly.

The static vector contains training-normalized beam mean/std and sin/cos origin
phase for 24/168 hours. Imputer medians and per-KPI mean/population std are fitted
only on the full training prefix, not validation. Missing values are imputed with
the training median and accompanied by 1=missing masks; zero is never missing.
An entirely missing training KPI uses median/mean 0 and scale 1. No target is
imputed: masked L1 excludes and counts each missing target entry. Predictions
are inverse-scaled to original DLPRB usage, without clipping.

The same 600-beam stratified sample is used for A/B and both seeds. Sampling is
proportional over joint tertiles of fit-only mean/std/active ratio, with deterministic
largest-remainder allocation. Training joint origins use stride 3; validation
early stopping uses all beams and joint origins for all three horizons. Final
benchmark replay uses the per-horizon origin sets from Week 2, including the
additional late origins for h=1 and h=6. Holdout is not used.

## Run on GPU

```text
pip install -e ".[dev,ml,neural]"
python -m scripts.train_gru --variant B --seed 17 --pilot
python -m scripts.train_gru --variant B --seed 17 --resume
python -m scripts.train_gru --variant A --seed 17
python -m scripts.train_gru --variant A --seed 42
python -m scripts.train_gru --variant B --seed 42
python -m scripts.evaluate_models --model gru --segment val
```

Use --seconds-limit to constrain a run; the limit is checked after each completed
epoch/checkpoint. The experiment-wide GPU target is at most four hours. Do not
run full training on CPU; the CLI rejects it. CPU fixtures are intentionally tiny.
The measured pilot and frozen settings are in reports/gru_compute.md. All model
training is forbidden on the owner's local machine, including synthetic fitting
tests; those tests are marked cloud_training and skipped locally. Run the complete
test suite inside Colab/Kaggle instead. Local checks only lint and test non-fitting paths.

Artifacts live under artifacts/gru/{A|B}/seed{17|42}/ and include best.pt, last.pt,
preprocessor.npz, history.jsonl and meta.json. Put the artifact directory on Drive
or Kaggle Output to retain checkpoints. Writes use a temporary file and atomic
replace. Resume restores optimizer, epoch, best score, early-stop patience and
Python-independent Torch CPU/CUDA RNG state; batch shuffles are seed+epoch.
Changing model/training/sample/dataset identity is rejected. max_epochs may be
extended for a pilot resume. Metadata best_epoch is zero-based.

Use torch.load(weights_only=True) for project checkpoints; never load arbitrary
pickled models from third parties. Replaying a saved artifact checks fit boundary,
dataset revision, beam order and batch/history parity. Reports provide per-run
metrics, mean and sample SD across seeds, A/B ablation, comparisons with C1 and
best baselines, load-tier/active/zero analysis and descriptive failure examples.
Two seeds provide descriptive variation only. One-seed SD is undefined.

## Package a private Kaggle job from local source

Packaging performs no fitting. After restoring Kaggle authentication, use:

```powershell
python -m scripts.package_gru_job --owner YOUR_KAGGLE_USERNAME
kaggle kernels push -p .cache/kaggle-gru --accelerator NvidiaTeslaT4 -t 1800
```

The packager creates a readable notebook with separate Python source cells and
Vietnamese explanations, not an encoded source payload. This launches the four
declared experiments and validation replay on Kaggle,
writing per-epoch checkpoints to Kaggle Output and a result archive at completion.
Each individual run has a 360-second budget checked after a full epoch; inspect
meta.json for budget_exhausted and resume on cloud if further work is needed.
The job itself has a 30-minute timeout, below the plan's four-hour GPU ceiling.

## Review without retraining

`notebooks/04_gru_review.ipynb` is the readable companion notebook. Data preparation,
training and evaluation each have explicit flags, all false by default. It includes
actual recorded validation results, module source with comments, a one-beam/origin
debug cell, a single-run training cell and an explicit four-experiment switch.
The original Kaggle training/checkpoint outputs remain under notebook version 1;
subsequent versions are review-only and do not silently retrain on Run All.
