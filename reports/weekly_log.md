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
