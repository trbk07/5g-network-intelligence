# Week 7 - Historical replay product and isolated model research

The product serves the frozen Week 6 LightGBM C1 champion. Week 7 research uses
the original fit/validation ranges only and never overwrites those artifacts.
The already observed final holdout cannot validate a new candidate. Fresh unseen
chronological data is required before a future promotion/generalization claim.

## Start the product

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-app.txt
# Use existing verified local Week 6 artifacts + the canonical panel:
.venv/Scripts/python.exe -m scripts.fetch_demo_assets
docker compose up --build -d --wait
```

Dashboard: http://127.0.0.1:8501. API documentation: http://127.0.0.1:8000/docs.
The API binds only to localhost through Compose. Stop with `docker compose down`.

For an authorized clean clone without artifacts, install/authenticate Kaggle CLI
and replace asset setup with `python -m scripts.fetch_demo_assets --download`.
It fetches the existing private Week 5/Week 6 outputs and checks the pinned archive
and panel SHA256. Private notebook access is required. No public release URLs or
public access are claimed. See [asset details](../demo_assets/README.md).

Without Docker, start two terminals from the project root:

```powershell
.venv/Scripts/python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
.venv/Scripts/python.exe -m streamlit run app/dashboard.py
```

`ASSET_DIR` configures the API's verified local assets. `API_URL` configures the
dashboard's API (default `http://127.0.0.1:8000`, Compose `http://api:8000`). There
are no local model-fitting commands in startup or requests. Containers run as a
non-root user, with read-only filesystems and a read-only asset mount.

## Data and inference contract

The demo loads 24 actual beams: BS 0/5/15/29, three cells, two beams per cell.
The original global models retain the full training beam metadata; they accept
the selected beams because none of the 40 numerical inputs is a beam statistic.
P90/P95 are selected from the frozen full-development threshold table by beam ID.
All artifacts, panel identity and beam order are checked before serving.

`Predictor` loads models once at startup, serializes native LightGBM calls and
uses the same causal feature builder as offline evaluation. The immutable store
returns copies of observed history. Predictions use only the last 168 steps,
ending at the requested origin. Known target actuals are attached after inference
for historical display; they are never prediction inputs. Missing numeric values
become JSON `null`. Entirely missing target history excludes risk scoring, even
if a model can emit a finite fallback prediction.

| Endpoint | Contract |
| --- | --- |
| `GET /health` | Verified model/data readiness; 503 if assets are missing/corrupt |
| `GET /model-info` | Champion version, horizons, limitations, revision, saved benchmark |
| `GET /replay/meta` | Exactly the loaded beams, verified topology, origin range and data scope |
| `POST /forecast` | `{origin, beam_ids}`; last 168 points, three forecasts/target indices and known actuals |
| `POST /risk` | Same forecasts plus historical P90/P95, status, validity and exclusion reason |

Requests accept 1-32 distinct loaded beams and an integer origin 671-839. Unknown
beam returns 404; invalid origin, bool/fractional origin, duplicate/oversized batch
or unsupported extra horizon returns 422. Internal failures never expose a stack
trace in HTTP responses. Targets after hour 839 have no known actual (`null`).

Historical P90/P95 use `>=`. Equal thresholds have no Warning band. They are not
forecast q90, a congestion probability or verified incidents. No forecast band
is drawn. The benchmark panel reports the saved full-network holdout, not a new
score from the demo subset. P90 recall of about 10-16% remains visible in the UI.

## Verification

```powershell
$env:FIVEG_API_URL = 'http://127.0.0.1:8000'
.venv/Scripts/python.exe -m pytest -q --basetemp=.cache/pytest-week7 -p no:cacheprovider
.venv/Scripts/python.exe -m scripts.check_inference_parity --url $env:FIVEG_API_URL
```

Training tests are automatically skipped locally. Product smoke explicitly
tests real HTTP and Streamlit AppTest: beam A, beam B, then a new origin.
The parity report checks 20 beam/origin pairs across three horizons against the
saved offline forecasts. Declared LightGBM absolute tolerance is 1e-9; observed
maximum error is 0, including the actual Linux Docker API and dashboard transport.
The dashboard has also been rendered in headless Edge, with no JavaScript errors.

The shipped Docker profile is LightGBM/baseline only. Neither Docker image
installs Torch or CUDA packages. The optional Python registry also routes GRU
ensembles on CPU when torch is installed; an initialized-state fixture checks
phase4 and subset inference without fitting. A future GRU Docker deployment
still needs a separately built CPU-only dependency profile and image benchmark.

Implementation references: [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/),
[Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest),
[Compose services](https://docs.docker.com/reference/compose-file/services/).

## Model research

`configs/research_week7.yaml` predeclares four LightGBM variants: deeper L1,
regularized L1, shallow L1 and squared error. All retain the causal40 schema.
Training/early stopping use fit 0-503 and validation 504-671; every model trains
only in hosted Kaggle CPU. P90/P95 policy is unchanged. Report validation MAE,
load tiers, paired BS intervals against the original C1 validation forecasts and
historical risk precision/recall/FPR. Repeated validation search is exploratory.

The first notebook version failed before fitting because a packaging substitution
rewrote an import. Version 2 preserves source cells verbatim and reruns validation
research only. It does not invoke the final holdout runner. The submitted training
notebook shows readable Python, separate configuration/source/execution cells and
useful comments. Review notebooks display saved reports without fitting.

Completed research: version 2 trained all 12 models. Deeper L1 improved mean
normalized validation MAE by 0.337%; gains by horizon were 0.393% / 0.339% /
0.279%. Historical P90 recall remains only about 11-17%. The squared-error
alternative increased recall but worsened MAE/FPR, so it was not chosen for
point forecasting. See [the full result](../reports/week7_results.md) and
`configs/research_candidate_week7.yaml`. This candidate is research-only and
requires fresh unseen data before promotion.
