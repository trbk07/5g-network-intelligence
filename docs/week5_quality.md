# Week 5 - Model quality and research evidence

The updated plan takes precedence over the original schedule. Week 5 does not
open holdout, freeze a final champion or fit a final refit model. P1 quantiles were
skipped because the mandatory quality audit required retraining.

## Schema correction and artifact isolation

Legacy W3/W4 had 44 LightGBM features and a GRU static6 vector containing fit-beam
statistics. The new gate rejects these inputs. Initial failures are retained in
reports/model_quality_gate_initial.json, with superseded reports in
reports/legacy_w3_w4/. No legacy model was made eligible by an inference-only patch.

All six direct LightGBM models were retrained on hosted cloud CPU with exactly
40 causal features, and all four GRU runs on Kaggle GPU with phase4 and head
input hidden_size+4. New models live in artifacts/lgbm_w5 and artifacts/gru_w5;
old artifacts remain separate. Checkpoints are retained in Kaggle Output.

GRU imputer/scaler are global per KPI: A has shape (1,), B shape (4,). DLPRB
statistics match exactly. Beam IDs are metadata, not fitted beam parameters.
Normalization is fitted once at the fit cutoff; earlier training origins are not
independent deployed episodes. Beam summaries remain valid for sampling and
diagnostic tiers only. Preprocessing save/load parity and future perturbation are tested.

## Common masks and paired uncertainty

Forecasts must have identical beam order, origins, revision and actual observations.
Each horizon's comparison mask requires finite actual and finite predictions for
all compared runs. It does not require all 168 history rows to be nonmissing.
Every comparison reports origin/pair/beam counts, coverage and exclusions. Raw
missing labels are reported separately from pairs excluded by other candidates.

Cloud actuals are compared bit-for-bit between candidates. Cross-platform CSV
parser differences of a few float64 ulps are permitted only against the local
diagnostic panel at rtol/atol 1e-12, never for mismatched origin/beam keys.

Per-horizon paired CIs use MAE(A)-MAE(B). Verified topology gives 30 BS clusters;
bootstrap resamples complete BS clusters with replacement, preserving within-BS
beam/time dependence. Seed 17, 1,000 draws, percentile 95% intervals. Unverified
mapping falls back to beam clusters with an explicit dependency limitation.
With only 30 clusters and one validation window, CIs should be interpreted cautiously.
The intervals are exploratory and not adjusted for multiple comparisons; they do
not replace the later frozen independent holdout evaluation.

Candidate scores average MAE/best-baseline-MAE over the three horizons, giving
horizons equal weight. The multi-horizon CI uses the same resampled BS weights
across all horizons and normalizes within each horizon before averaging.

## Candidate rule declared before comparison

Order: Persistence, Seasonal24, Seasonal168, LGBM-C1, LGBM-C2, GRU-A, GRU-B.
This is a simplicity/cost assumption, not measured latency. Select the nominal
lowest mean normalized MAE; a cheaper family can replace it if the paired
normalized difference CI includes zero (no statistically resolved disadvantage).
Baseline champions are permitted. A family must provide all three horizons.

GRU family candidates use the arithmetic mean of both declared seed forecasts;
the best seed is not cherry-picked. Per-seed A/B comparisons use identical masks,
samples and origins. Mean/sample SD across two seeds is descriptive only.
Candidate metadata records artifact references, hashes and frozen:false, not a
final deployment decision. No pre-holdout tag is created.

## Error atlas rules

Rank top 5 beams by equal-weight mean horizon MAE of the recommended reference.
Pick 5 representatives per load tier from equally spaced training-mean ranks.
For h=24 pick the two largest distinct underpredictions among actual values at
or above validation P90, and the largest zero-to-active underprediction. Ties use
original numeric beam/time ordering. The plotted best baseline is selected per
horizon; plots use the same target indices/window for all compared models.

The atlas shows abrupt peaks and zero-to-active spikes that point forecasts miss;
a persistence-like baseline can echo a past peak 24 hours late. These are observed
retrospective forecast failures, not validated congestion or incident labels.
Short history, beam sampling and bias/variance are hypotheses for neural shortfall,
not demonstrated causal explanations. Load tiers and event cases are diagnostic,
not additional model-selection tuning signals.

## Reproduce

Use scripts/build_week5_job.py to package readable source cells and retrain on a
private Kaggle GPU notebook. The execution notebook deliberately enables training;
all review notebooks default to no retraining. Run the hosted validation worker
with scripts/build_week5_validation.py to attach completed model outputs, run fitting
tests on cloud and produce the research evidence. Local code only reads checkpoints,
computes metrics/statistics, lints and runs non-training tests.
