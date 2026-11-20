# Weekly log

Week: 1 — Data engineering and audit

Hours used: Not tracked; no claim against the 12-hour budget.

Tasks completed:
- Local repository, Python 3.12 environment, configuration and reproducibility utilities.
- Downloaded all 12 CSVs at revision c9ecaf153354b2cc415c64750a966c47c7f3f6ac.
- Manifest with MIT license, SHA256, bytes, row counts and source columns.
- Main panel (840, 2880, 4), separate week 6/week 11 panels, NPZ parity tests.
- Audit, figures, EDA notebook, data dictionary, conservative split decision.
- 17 local tests passed, including the offline end-to-end pipeline test; Ruff passed.

Week 1 infrastructure: complete. Kaggle, Colab CLI and Google Drive checkpoint
read/write have all been verified.

Kaggle setup completed after owner OAuth login. Private notebook version 2 completed
on two Tesla T4 GPUs; checkpoint probe, full data pipeline and 16 tests passed.
See reports/kaggle_runtime.json. No training performed. Colab and Drive were
subsequently verified as recorded below.

Decision: Retain internal chronological holdout. The card describes week 6 as
immediate future but time resets; continuity approval remains false. No model trained.

Verified observations:
- 2,880 beams covering exactly 30 BS × 3 cells × 32 beams.
- Main panel contains no missing KPI observations.
- DLPRB zero ratio 0.23770089285714285 (observed denominator).
- Actual CSV time column is unnamed; normalization is explicit in source metadata.

Validation command used on this managed Windows workspace:
`python -m pytest -q -o addopts='' -p no:cacheprovider --basetemp=.cache/tests-final-2`
returned 16 passed. A new temporary directory was used because directories created
by the sandbox account had incompatible ownership when switching execution contexts.
`ruff check .` passed. This is an environment ACL issue, not a fixture failure.

Next week's priority: Origin alignment, evaluation metrics, three baselines and leakage tests.

Week 1 follow-up: manifest now rejects missing source files as well as checksum
mismatches. Panel preparation generates evaluation.yaml from the same split decision
as split.json. The integration test validates CSV alignment, preserved missing/zero
values, separate week 11 storage, audit figures and agreement between split outputs.
The Kaggle version 2 result above covers the earlier 16-test code snapshot;
the follow-up is validated locally with 17 tests.

Colab follow-up: authenticated CLI, T4 CUDA available, full Week 1 data pipeline,
17 tests and Ruff passed. Runtime checkpoint write probe passed, reports downloaded,
and GPU runtime terminated. See reports/colab_runtime.json. No model training done.

Drive follow-up: owner completed Google consent; mount and checkpoint read/write
passed in the dedicated project folder. Evidence: reports/drive_verification.json.
CPU runtime terminated after verification. Week 1 infrastructure checks complete.

## Week 2 - Evaluation framework and baselines

Hours used: Not tracked; no claim against the 12-hour budget.

Tasks completed:
- T2.1: training, joint-training and evaluation origins; cross-horizon target
  disjointness checks; exact origin arrays saved in processed split.json.
- T2.2: all eight planned metrics/load-tier functions, explicit exclusion
  counts and flags for undefined results, hand-calculated metric tests.
- T2.3: history-only Forecaster protocol, fixed-model replay evaluator,
  pickle-free forecast records, coverage-aware benchmark rows.
- T2.4: Persistence, Seasonal Naive 24h and Seasonal Naive 168h, with rejection
  of future/out-of-history seasonal references and +24h equivalence test.
- T2.5: future perturbation canaries, deliberately leaky predictor detection,
  fit-only load-tier canary and validation-runner holdout perturbation test.
- T2.6: validation benchmark on all 2,880 real beams, overall/load-tier metrics,
  active/zero analysis, per-beam errors and top-10 report, replay NPZ with digest,
  provenance, notebook and evaluation protocol documentation.

Validation: 45 tests passed locally; Ruff passed; baseline notebook schema and
Python cell syntax validated. Cloud hardware and Week 1 cloud results remain
documented separately; Week 2 benchmark was run locally on CPU.

Validation winners (DLPRB usage MAE, not final holdout performance):
- +1h: Persistence, approximately 0.297747.
- +6h: Seasonal Naive 24h, approximately 0.332699.
- +24h: Persistence and Seasonal Naive 24h tie, approximately 0.332065.

Tasks unfinished: None in Week 2 scope.
Blocking issue: None.
Decision: Gate G2 passed. Holdout was not evaluated; the baseline CLI refuses it.
Next week's priority: Week 3 causal features, LightGBM and validation benchmark.

## Week 3 - LightGBM forecasting

Hours used: Not tracked; no claim against the 12-hour budget.

Tasks completed:
- T3.1: 44 ordered features with fit-only static stats, vectorization over beams,
  native NaN preservation, 19 hand-calculated feature checks and future canaries.
- T3.2: six global direct models (C1/C2 for +1/+6/+24), training stride 2,
  early stopping on validation MAE, saved text models/static stats/provenance,
  independent artifact load and prediction parity checks.
- T3.3: both planned configs evaluated; C1 selected by mean normalized validation
  MAE over the three horizons and recorded in configs/lightgbm.yaml.
- T3.4: validation benchmark, load-tier/active/zero analysis, feature importance,
  skill versus baselines, top beam errors, 18 failure examples, analysis notebook.
- T3.5: 52 local tests passed; Ruff passed; deterministic training fixture,
  save/load and batch/history replay parity, holdout perturbation test, and
  exact CSV metric equality after reloading all six real artifacts and replaying.

Observed C1 validation MAE / skill versus the best baseline:
- +1h: approximately 0.230489 / 22.59% improvement.
- +6h: approximately 0.253698 / 23.75% improvement.
- +24h: approximately 0.251002 / 24.41% improvement.

Best iteration: C1 344/275/235; C2 499/499/491 for +1/+6/+24 respectively.
C2 reached near the planned round cap; the cap was retained, with no extra search.
Real-data training and validation replay ran locally on CPU with four threads.
Artifacts and forecast NPZ files remain local generated files; reports/checksums
and source are committed. The full benchmark can be reproduced from configuration.

Tasks unfinished: None in Week 3 scope.
Blocking issue: None.
Decision: Gate G3 passed. No holdout evaluation, final champion freeze or GRU training.
Next week's priority: Week 4 GRU data pipeline, GPU pilot and limited neural benchmarks.

## Week 4 - GRU and comparison

Hours used: Not tracked; no claim against the active-work budget.

Tasks completed:
- T4.1: shared stratified 600-beam sample, training-prefix median imputer/scaler,
  missingness masks, static6, causal batches and three aligned targets.
- T4.2: GRU-A/B, AdamW/masked L1, gradient clipping, early stopping, per-epoch
  best/last checkpoint, optimizer/RNG/patience resume; synthetic overfit and exact
  CPU-fixture resume tests are included in the hosted-cloud suite.
- T4.3: measured Colab T4 pilot, 8.43 seconds/epoch, about 445 MiB peak allocated
  GPU memory. Retained batch256, stride3, 600 beams and seeds17/42.
- T4.4: all four real-data runs completed on Kaggle GPU, checkpoints in version1
  Output. A17/A42/B17/B42 completed 22/22/25/15 epochs; zero-based best epochs
  were 16/16/19/9. No run exhausted its budget. Total train/replay about 11.5 minutes.
- T4.5: all-beam validation, A/B ablation, comparisons with LightGBM/baselines,
  load-tier and active/zero metrics, failure examples, 12 replay records and manifests.

Validation: 56 tests passed on Kaggle, including fitting tests; Ruff passed.
Local checks after the owner's prohibition performed no model fitting: 52 passed,
4 fitting tests skipped. AGENTS.md and runtime guards enforce cloud-only training.

Findings: GRUs improve over best baselines but do not beat LightGBM C1 on any
horizon. GRU-B mean MAE is about 0.242076/0.273128/0.263339 for +1/+6/+24.
The B-vs-A benefit is small/mixed across horizons, and two-seed SD is descriptive.
Strict actual-zero/prediction>0 FPR is high (about 97.7-99.7% across variant/horizon
means), so these neural forecasts frequently produce positive values on zeros.
No extra tuning or clipping was applied after seeing validation results.

Notebook presentation: replaced encoded-source Kaggle pages with commented,
sectioned Python notebooks. Review versions do not retrain on Run All. The original
GPU training/checkpoint results remain in Kaggle GRU version1 and local artifacts.
Fixed Vietnamese encoding lost by the initial PowerShell-generated utility notebooks.
Generators now write ASCII-safe JSON Unicode escapes; four text regression checks
passed and the downloaded Kaggle source retained Vietnamese headings/comments.

Tasks unfinished: None in Week 4 scope.
Blocking issue: None.
Decision: Gate G4 passed with both variants and both seeds evaluated. No holdout used.
Next week's priority: Week 5 risk engine, validation champion selection and freeze.

## Week 5 — Updated quality/research plan

Date: 2026-11-20 (Asia/Bangkok). Hours not tracked; no claim against the active-hour budget.

The updated plan supersedes the old Week 5 risk/freeze schedule. Holdout stays
closed this week; risk, final freeze and refit belong to the revised Week 6.

T5.1: initial quality FAIL for all legacy ML artifacts (44/static6, A preprocessing
shape4). Retrained six LightGBM40 models on hosted cloud CPU and four phase4 GRUs
on Kaggle GPU. New artifacts isolated in lgbm_w5/gru_w5; old artifacts/reports
preserved. Current gate PASS, 38 artifact files hashed. A/B global DLPRB parameters
and training sample match exactly; A parameters shape1, B shape4.

T5.2: locked shared finite actual/forecast masks, 100% coverage across all 2,880 beams.
Paired 1,000-draw bootstrap on 30 verified BS clusters, same draws across horizons
for normalized-family comparisons. Reports separate raw missingness/comparison exclusions.

T5.3: deterministic atlas: top5 error beams, 5 representatives per Low/Mid/High tier,
2 miss-peak cases and 1 zero-to-active case; three requested figures rendered and inspected.

T5.4: A/B paired-origin/seed/mask comparison and two-seed descriptive ablation.
GRU family candidates use the declared two-seed forecast mean, not the best seed.
LightGBM C1 remains the best validation candidate; causal40 MAE is approximately
0.230635 / 0.253473 / 0.251246 at +1/+6/+24h. Explanations for neural differences
remain hypotheses; no causal/operational incident claim is made.

T5.5: baseline champions allowed and cheaper-CI-fallback tested. Candidate C1 is
recorded with frozen:false, holdout_allowed:false and artifact/forecast/gate hashes.
No pre-holdout tag, final refit or holdout evaluation was performed.

T5.6 P1: skipped, as explicitly required when mandatory retraining is needed.

Validation: hosted worker 65 tests passed (including all fitting tests); five
notebook text checks are separately validated locally. Local suite 66 passed,
4 fitting tests skipped. Ruff passed. No local model training.

Deliverables: quality gate, common-mask comparison, paired CIs, ablation/model story,
error atlas and champion-candidate registry. A readable review notebook is ready;
the owner explicitly approved its new Kaggle review destination after the
automatic egress review requested authorization. It was uploaded with training
disabled and a CPU-only review execution, which completed successfully at
https://www.kaggle.com/code/truongbaokien/5g-network-intelligence-week5-review.

Gate G5: research P0 complete. Next priority: revised Week 6 risk engine/validation
trade-off, final freeze, full-development cloud refit and one guarded holdout run.
