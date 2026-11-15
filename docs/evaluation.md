# Evaluation protocol - Week 2

All indices and split endpoints are inclusive hourly relative indices. Calendar
timestamps and operational utilization units are not inferred.

## Frozen split and origins

The current fallback split is fit 0-503, validation 504-671 and holdout 672-839.
Lookback is 168, horizons are 1, 6 and 24. Fit statistics and load tiers use 0-503
only. The model remains fixed during validation, but history is updated with
actual observations through each origin. There is no retraining at each step.

| Segment | Horizon | First origin | Last origin | Count |
| --- | --- | --- | --- | --- |
| Training | 1 | 167 | 502 | 336 |
| Training | 6 | 167 | 497 | 331 |
| Training | 24 | 167 | 479 | 313 |
| Validation | 1 | 503 | 670 | 168 |
| Validation | 6 | 503 | 665 | 163 |
| Validation | 24 | 503 | 647 | 145 |

Joint training origins are 167-479 for all three horizons. All models share the
same origin set per horizon. Targets across train/validation/holdout are checked
for overlap across all horizons, not just matching horizons. Primary split
boundary counts from plan section 2 are separately covered by tests.
`data/processed/split.json` stores the exact arrays for every horizon and segment.
Generating holdout origin indices does not run a holdout evaluation.

## Predictor contract

`Forecaster.predict(history, horizon)` receives a read-only copy with shape
`(168, n_beams, 4)`, ending at the current origin, and returns one DLPRB prediction
per beam in panel order. It receives no future observations. Predictions may be
NaN when required history is missing; infinities and incorrect shapes are rejected.
The evaluator saves origin/target indices, beam IDs, actuals and predictions.
Inputs must use contiguous hourly relative indices starting at zero.

Persistence uses the last target. Seasonal naive references `origin+h-period`;
out-of-history and future references raise errors. Seasonal Naive 24h and
Persistence are identical at h=24 and count as tied results, not independent evidence.

## Metrics and exclusions

Every metric returns `MetricResult(value, n_valid, n_excluded, flags)`. Nonfinite
actual/prediction pairs are excluded and counted; zero remains an observation.
Undefined metrics return NaN with a flag, never an artificial zero score.

| Metric | Definition |
| --- | --- |
| MAE | Mean absolute error over valid pairs |
| RMSE | Square root of mean squared error |
| WAPE | Sum absolute error / sum absolute actual, ratio units |
| Macro-WAPE | Equal-weight average of defined per-beam WAPE |
| Macro-MAE | Equal-weight average of defined per-beam MAE |
| Active-only MAE | MAE among valid pairs with actual > 0 |
| Zero-observation FPR | Fraction of valid actual-zero pairs with prediction > 0 |

WAPE with zero denominator is undefined; macro metrics flag excluded beams.
Active/zero subset metrics count other pairs as excluded. The benchmark CSV also
reports missing target/prediction counts, active/zero counts and beams without
any valid pairs. Model comparisons must consider coverage if missing predictions
differ. This dataset's current validation benchmark has full paired coverage.
Zero-observation FPR is not threshold-exceedance risk FPR and is not an incident metric.

Load tiers are tertiles of per-beam training-only means. Identical means stay
together, so ties may leave a tier empty. All-missing fit beams are Unknown.

## Reproduce and inspect

```powershell
.venv/Scripts/python.exe -m scripts.run_baselines --segment val
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/ruff.exe check .
```

The runner validates the frozen split against evaluation config and truncates
the panel at the validation endpoint before creating model inputs. It refuses
`--segment holdout`; the final guarded runner is scheduled for Week 5.

The NPZ replay archive is generated locally and excluded from Git. Load it with
`np.load(path, allow_pickle=False)` and use its JSON `metadata` index to locate
`record_i_origins`, `targets`, `beam_ids`, `actual` and `predicted` arrays. Horizon
records have different numbers of origins. Run provenance is saved separately
in `reports/baselines_run_metadata.json`.
The archive digest, size and source revision are tracked in
`reports/baseline_forecast_manifest.json` for integrity checking.

Canary tests perturb every observation after the origin and compare histories
and baseline outputs. A deliberately leaky future reader must fail the same
canary. Training-only load tiers and the full validation runner also have
future/holdout perturbation tests. Scaler, imputer and learned feature tests will
be added with the modules that introduce those transformations in later weeks.
