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
