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

Week 3 implements 44 causal features and global direct LightGBM models. Run training
inside hosted Colab/Kaggle only (local training is forbidden by the owner):

```powershell
python -m scripts.train_lgbm
python -m scripts.evaluate_models --model lightgbm --segment val
```

See [LightGBM protocol](docs/lightgbm.md), [validation analysis](reports/lightgbm_analysis.md)
and `notebooks/03_lightgbm.ipynb`. The observed selected configuration is C1;
final champion selection and holdout evaluation remain scheduled for Week 5.

Week 4 adds GRU-A/B input preprocessing, beam sampling, resumable GPU training
and validation replay. See [GRU protocol](docs/gru.md) and the
[GPU pilot measurements](reports/gru_compute.md). Full neural training requires CUDA.
All training, including synthetic fitting tests, runs only on hosted Colab/Kaggle.
Local test runs automatically skip tests marked `cloud_training`.
Use [the commented GRU review notebook](notebooks/04_gru_review.ipynb) to inspect
the code, debug one beam and view the measured comparison. It does not retrain
on Run All. All four GRU runs finished on Kaggle; C1 remains better on validation.

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
do not use them to tune models. Forecasting results are validation-only; final
holdout and risk results have not been measured.
This project is historical/offline replay, not live network telemetry. Statistical
threshold exceedance is not verified congestion or incident detection.

GitHub repository name: `5g-network-intelligence`. Dashboard name: **5G Network
Intelligence**. CV title: **5G Network Intelligence — KPI Forecasting & Capacity
Risk Monitoring**.
