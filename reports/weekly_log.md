# Weekly log

Week: 1 — Data engineering and audit

Hours used: Not tracked; no claim against the 12-hour budget.

Tasks completed:
- Local repository, Python 3.12 environment, configuration and reproducibility utilities.
- Downloaded all 12 CSVs at revision c9ecaf153354b2cc415c64750a966c47c7f3f6ac.
- Manifest with MIT license, SHA256, bytes, row counts and source columns.
- Main panel (840, 2880, 4), separate week 6/week 11 panels, NPZ parity tests.
- Audit, figures, EDA notebook, data dictionary, conservative split decision.
- 16 unit tests passed and Ruff passed.

Tasks unfinished:
- GitHub remote: authentication restored; private repository created for trbk07.
- Colab/Drive/Kaggle account setup and GPU verification: not performed.

Blocking issue: Cloud GPU allocation and Drive authorization require an interactive
session owned by the user. Runtime bootstrap notebook is prepared and syntax checked.

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
