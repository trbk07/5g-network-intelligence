# Portfolio summary

5G Network Intelligence is a reproducible historical beam-level forecasting
and capacity-risk monitoring project. It forecasts DLPRB usage at one, six and
24 hours and compares predictions with development-fitted historical P90/P95
thresholds. The interface explicitly presents offline replay, rather than
claiming live network telemetry or verified congestion detection.

The project aligns four hourly KPIs across 2,880 beams, preserves missing values
and observed zeros, and pins dataset provenance and chronological splits.
Global LightGBM models use 40 causal inputs; GRU A/B experiments use global
per-KPI preprocessing and four phase features. Model fitting runs only on
hosted Kaggle/Colab, while local execution supports verification and CPU inference.

The frozen LightGBM champion reduced internal holdout MAE by 23.47–23.71%
against validation-selected baselines. Its alerting limitations remain visible:
P90 recall is approximately 10–16%. A later research ensemble improved reused
validation MAE by 0.878% against the original C1, without changing the final
holdout or deploying an unvalidated replacement.

A shared predictor powers FastAPI and Streamlit, packaged with read-only Docker
assets. Twenty beam/origin pairs across three horizons matched saved offline
forecasts. The project emphasizes inspectable evidence, reproducibility and
honest limits, rather than production or state-of-the-art claims.

## CV bullets

- Built a reproducible 5G KPI forecasting pipeline for 2,880 beams with pinned
  provenance, chronological validation and causal feature/leakage checks.
- Compared global LightGBM and GRU A/B models; achieved 23.47–23.71% lower
  internal holdout MAE than validation-selected baselines at 1/6/24 hours.
- Implemented historical P90/P95 exceedance monitoring with explicit guards,
  confusion metrics and documented low-recall limitations.
- Delivered shared CPU inference, a five-endpoint FastAPI service and Streamlit
  replay dashboard; verified 60 offline/API prediction comparisons and Docker
  deployment with checksum-verified demo assets.

Ownership note: these bullets describe the repository's delivered engineering
work. Review and adapt the wording to the parts you can personally explain and
demonstrate; do not claim unaided authorship or external production deployment.
