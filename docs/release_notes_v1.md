# v1.0.0 release notes

5G Network Intelligence delivers historical beam-level KPI forecasts at
1/6/24 hours, historical P90/P95 exceedance scenarios and an inspectable CPU
replay product. This release does not claim live telemetry, production SLAs,
verified congestion detection or state-of-the-art forecasting.

- Frozen C1 internal holdout MAE: 0.228237 / 0.250562 / 0.249446, improving
  23.47-23.71% over validation-selected baselines.
- LightGBM causal40 and GRU phase4 experiments; all fitting on hosted cloud.
- Five read-only FastAPI endpoints, shared CPU predictor and Streamlit dashboard.
- Checksummed 24-beam real-data demo package, read-only non-root Docker services.
- Remote clean-clone Docker QA, offline/API parity and measured CPU latency/RSS.
- Model card, experiment report, limitations, real screenshots and 120s video.
- Research-only ensemble: 0.878% lower reused-validation MAE vs original C1;
  no production promotion or new final-test claim.

Prepared assets: `5g-demo-assets.tar.gz`, its checksum metadata and the champion
artifact manifest. The exact checksum is committed in
`reports/demo_asset_package.json`. The package contains real NETOP GOTS-F subset
data with dataset attribution; no credentials or full raw source data.

After obtaining the pinned package from an authorized source, install it with:

```powershell
python -m scripts.fetch_demo_assets --demo-archive PATH_TO_5g-demo-assets.tar.gz
docker compose up --build -d --wait
```

Open http://127.0.0.1:8501 or http://127.0.0.1:8000/docs on the same machine.
Download from the actual [v1.0.0 release](https://github.com/trbk07/5g-network-intelligence/releases/tag/v1.0.0).
Repository access is required because the repository remains private. The
published asset has been downloaded, checksum-verified and installed separately.

Known limits: a single internal holdout, uncertain physical usage units, weak
P90 recall (about 10-16%), high strict actual-zero FP rate and no predictive
bands. See `docs/limitations.md` and `reports/experiment_report.md`.

Code is licensed under MIT, copyright 2026 Truong Bao Kien, as selected by the owner.
The draft release asset has been downloaded from GitHub, checksum-verified and
installed in a fresh directory. Dataset license/citation are preserved independently.
