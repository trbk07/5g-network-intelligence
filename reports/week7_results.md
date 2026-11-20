# Week 7 results

Gate G7: PASS - CPU historical replay product.

## Product

- Dashboard: http://127.0.0.1:8501; API docs: http://127.0.0.1:8000/docs.
- Frozen Week 6 LightGBM C1; 24 real beams; five read-only endpoints; 168-step history and three forecasts.
- API and dashboard are healthy in Docker. Images: about 142 MB / 196 MB; no Torch/CUDA packages.
- 88 local tests passed, 4 fitting tests skipped. Ruff passed. Actual dashboard rendered with no JavaScript errors.
- 20 beam/origin pairs x 3 horizons match offline/shared/API/dashboard transport, maximum absolute error 0, tolerance 1e-9.
- Historical thresholds are labeled distinctly; no invented forecast bands or live telemetry.

## Cloud-only model improvement

Four predeclared configurations x three horizons trained on hosted Kaggle CPU, only original fit/validation. All 12 checkpoints saved and independently reloaded.

| Horizon | Original C1 validation MAE | Deeper L1 validation MAE | MAE reduction | Paired 95% difference CI |
| --- | ---: | ---: | ---: | --- |
| 1h | 0.230635 | 0.229728 | 0.393% | [-0.001119, -0.000733] |
| 6h | 0.253473 | 0.252613 | 0.339% | [-0.001089, -0.000670] |
| 24h | 0.251246 | 0.250547 | 0.279% | [-0.000847, -0.000556] |

Mean normalized validation MAE reduction: 0.337%. This is a small exploratory gain, not a new final holdout result.
All paired intervals favor deeper L1 on this validation set. They are not adjusted for repeated search/multiple comparisons and do not establish unseen-week performance.
Deeper L1 P90 recall remains around 11-17%, close to C1. This does not resolve the peak-alerting limitation.
Squared-error training raises P90 recall (about 16-23%) but worsens average normalized MAE by about 3.99% and roughly doubles P90 FPR. It was not selected for point forecasting.

## Freeze and provenance

Production champion, threshold policy and Week 6 final scores are unchanged. No holdout rerun, recalibration or model promotion occurred.
Candidate: `configs/research_candidate_week7.yaml`. Fresh unseen chronological data is required for a separate promotion protocol.
Research notebook: https://www.kaggle.com/code/truongbaokien/5g-network-intelligence-week7-research (version 2, COMPLETE).
Review notebook: https://www.kaggle.com/code/truongbaokien/5g-network-intelligence-week7-review (saved reports, no training).
Version 1 failed before fitting due to an import rewritten by the packager. Version 2 preserves source cells; a regression test covers this bug.
Frozen Week 6 source remains at tag `pre-holdout`; product code is intentionally a later commit.
Checkpoints/forecast NPZs stay in private Kaggle Output and Git-ignored local artifacts; source and small reports are committed.
