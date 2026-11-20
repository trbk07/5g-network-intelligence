# 5G Network Intelligence

**5G Network Intelligence: KPI Forecasting & Capacity Risk Monitoring**

A reproducible 5G beam-level KPI forecasting and historical capacity-risk monitoring
system using LightGBM, GRU, FastAPI, Streamlit, and Docker.

Week 1 implements dataset ingestion, provenance, panel construction, descriptive audit,
and chronological split preparation. Models and application components are scheduled
for subsequent weeks in [the execution plan](plan5G.md).

Week 2 adds chronological origins, metrics with coverage flags, a history-only
evaluator, three baselines and leakage canaries. Run the validation benchmark:

```powershell
.venv/Scripts/python.exe -m scripts.run_baselines --segment val
```

See [evaluation protocol](docs/evaluation.md), [baseline analysis](reports/baseline_analysis.md)
and `notebooks/02_baselines.ipynb`. Validation results are not final holdout results.

LightGBM uses the updated **40-feature** causal contract. Legacy Week 3 44-feature
and Week 4 static6 artifacts were rejected and retrained on cloud in Week 5. Run training
inside hosted Colab/Kaggle only (local training is forbidden by the owner):

```powershell
python -m scripts.train_lgbm
python -m scripts.evaluate_models --model lightgbm --segment val
```

See [LightGBM protocol](docs/lightgbm.md), [validation analysis](reports/lightgbm_analysis.md)
and `notebooks/03_lightgbm.ipynb`. The observed selected configuration is C1;
Week 5 produces a champion-ready candidate only; Week 6 performs the final freeze
and holdout evaluation under the updated [plan](plan5G.md).

Week 4 adds GRU-A/B input preprocessing, beam sampling, resumable GPU training
and validation replay. See [GRU protocol](docs/gru.md) and the
[GPU pilot measurements](reports/gru_compute.md). Full neural training requires CUDA.
All training, including synthetic fitting tests, runs only on hosted Colab/Kaggle.
Local test runs automatically skip tests marked `cloud_training`.
Use [the commented GRU review notebook](notebooks/04_gru_review.ipynb) to inspect
the code, debug one beam and view the measured comparison. It does not retrain
on Run All. All four GRU runs finished on Kaggle; C1 remains better on validation.

Week 5 quality/research evidence is in [the quality gate](reports/model_quality_gate.md),
[common-mask analysis](reports/model_analysis.md) and [error atlas](reports/error_atlas.md).
Candidate metadata is in `configs/champion_candidate.yaml`: **not frozen**, holdout disabled.
P1 quantile bands were skipped because all legacy models required cloud retraining.
Use [the Week 5 review notebook](notebooks/05_model_quality.ipynb) for code, coverage,
paired intervals and the atlas; Run All does not train or reopen holdout.

```powershell
# Read existing artifacts/forecasts and perform statistics only; no fitting occurs.
.venv/Scripts/python.exe -m scripts.quality_audit
.venv/Scripts/python.exe -m scripts.evaluate_models --model lightgbm --segment val --diagnostics
```

Week 6 is complete: LightGBM C1 was frozen at tag `pre-holdout`, refitted on
hosted Kaggle CPU with 284/457/414 fixed rounds and evaluated once on the internal
holdout. No local training or holdout tuning occurred. The deployment configuration
is `configs/champion.yaml`; Week 5 candidate metadata remains a historical record.

| Horizon | Holdout MAE | Best validation baseline MAE | MAE reduction |
| --- | ---: | ---: | ---: |
| 1h | 0.228237 | 0.298215 | 23.47% |
| 6h | 0.250562 | 0.327570 | 23.51% |
| 24h | 0.249446 | 0.326987 | 23.71% |

All three paired BS-bootstrap 95% MAE-difference intervals favor C1. Historical
P90 recall is only 10-16%, so the alerting limitation remains explicit.
See [final evidence and limitations](reports/week6_results.md),
[the Week 6 protocol](docs/week6.md) and
[the readable review notebook](notebooks/06_champion_risk_holdout.ipynb).
The [private Kaggle review](https://www.kaggle.com/code/truongbaokien/5g-network-intelligence-week6-review)
displays saved results and never retrains on Run All. Checkpoints remain in
the [completed private execution Output](https://www.kaggle.com/code/truongbaokien/5g-network-intelligence-week6-final/output).

## Run Week 1

Requires Python 3.12+ and internet access for the public dataset.
Run commands from the repository root (PowerShell):

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev,ml,neural]"
.venv/Scripts/python.exe -m scripts.prepare_data --step all
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/ruff.exe check .
```

Or activate a standard Python virtual environment and use `pip install -e ".[dev,ml,neural]"`.
Individual pipeline stages: `--step download`, `--step panel`, `--step audit`.
Configuration lives in `configs/data.yaml`; paths resolve from the working directory.
`requirements-week1.lock.txt` records the validated environment.
The panel stage also writes `configs/evaluation.yaml` from the resolved split
decision, keeping it consistent with `data/processed/split.json`. Source manifests
reject missing or unexpected CSVs in addition to checksum mismatches.

For cloud setup, use [the Colab/Kaggle bootstrap notebook](notebooks/00_runtime_setup.ipynb)
and [infrastructure instructions](docs/infra.md). It checks the actual GPU and
checkpoint storage; cloud access and GPU quota must be verified in your own session.

## Data and methodology

Dataset: [NETOP GOTS-F](https://huggingface.co/datasets/netop/gotsf-ds), MIT license.
The revision is pinned in configuration; the downloaded card and file checksums
are retained. Please cite Fechete et al., “Goal-Oriented Time-Series Forecasting:
Foundation Framework Design,” as requested by the dataset authors.

Four hourly KPIs are aligned on numeric time and `(BS, cell, beam)` identifiers.
NaN remains missing; zero remains observed. Missing keys are outer-aligned as NaN;
duplicate keys, nonnumeric values and infinities raise errors. Never silently drop rows.
DLPRB is described as **usage**, since its physical unit is unverified.

The conservative default uses train indices 0–503 for fit, 504–671 for validation,
and 672–839 for internal holdout. Week 6 and week 11 remain separate. Primary
splitting requires a documented continuity review and matching source indices.
Do not train before reviewing `reports/split_decision.md` and `split.json`.

## Outputs

| Path | Content |
| --- | --- |
| `data/raw/` | Revision-pinned source CSVs, card and provenance |
| `data/processed/panel.npz` | Main time × beam × KPI panel |
| `data/processed/panel_meta.json` | Panel schema and provenance |
| `data/processed/week6.npz`, `week11.npz` | Separate target weeks |
| `data/processed/split.json` | Inclusive chronological split bounds |
| `reports/data_manifest.json` | SHA256, size, rows, columns, revision, license |
| `reports/data_audit.md`, `data_audit.json` | Audit summary and machine-readable detail |
| `reports/figures/data_audit.png` | Distribution, zeros, sample series, phase profiles |
| `reports/run_metadata.json` | Config, hash, seed, package versions, Git commit |

Audit figures include holdout observations for Week 1 data verification only;
do not use them to tune models. Final internal holdout and historical risk results
are recorded in [the Week 6 report](reports/week6_results.md).
This project is historical/offline replay, not live network telemetry. Statistical
threshold exceedance is not verified congestion or incident detection.

GitHub repository name: `5g-network-intelligence`. Dashboard name: **5G Network
Intelligence**. CV title: **5G Network Intelligence — KPI Forecasting & Capacity
Risk Monitoring**.
