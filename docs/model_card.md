# Model card - frozen LightGBM C1

The deployed model is a historical beam-level DLPRB usage forecaster, not a live
network controller. Intended users are reviewers, researchers and analysts
exploring offline KPI forecasting and statistical capacity-risk scenarios.

Three independent global LightGBM models forecast +1/+6/+24 hours from 168
observed hours and 40 causal inputs. Native missing values are retained. Beam
IDs and fit-beam summaries are metadata, not numerical model inputs. The model
version is the annotated `pre-holdout` commit recorded in the artifact manifest.

Data: NETOP GOTS-F (MIT), revision
`c9ecaf153354b2cc415c64750a966c47c7f3f6ac`; 2880 beams and four hourly KPIs.
Physical DLPRB units and continuity with external test weeks are unverified.
Fallback internal split, validation-only selection and one final holdout are
documented in [the experiment report](../reports/experiment_report.md).

All training occurred on hosted Kaggle, never the owner's computer. Final
development fit covers hours 0-671, seed 17, fixed rounds 284/457/414 without
holdout early stopping. Final MAE is 0.228237/0.250562/0.249446, reducing error
23.47%/23.51%/23.71% relative to validation-selected baselines. See CSV evidence
for exact values, masks, paired intervals and exclusions.

Outputs are point usage forecasts. Normal/Warning/Critical statuses compare
points with development-fitted historical P90/P95. These are not forecast
quantiles, congestion probabilities, verified incidents or guaranteed coverage.
No predictive bands are supplied. Missing values use JSON null; exclusions have
explicit reasons. All-missing target history is excluded from risk scoring.

Do not use this demo for automated resource allocation, customer-impacting
alarms, live operational decisions or claims of cross-network generalization.
P90 recall is only about 10-16%. Low-load WAPE and actual-zero errors are material
weaknesses. Inspect [limitations](limitations.md) before interpreting results.

The later improved ensemble is a separate research-only candidate, evaluated
on reused validation. It is not silently promoted to this frozen model. A new
promotion requires unseen chronological data and a frozen evaluation protocol.

Artifacts have SHA256 manifests and independent load/predict parity checks.
The app profile uses CPU LightGBM and has no Torch/CUDA dependency. Dataset
attribution is in `dataset_card.md`; artifact access currently requires the
authorized private project outputs.
