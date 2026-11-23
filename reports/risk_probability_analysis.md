# Direct exceedance probabilities: useful ranking, weak fixed operating point

Hosted Kaggle v1 completed 12 binary LightGBM fits: two development folds, three
horizons, and historical P90/P95 labels. Each fold's thresholds use its own fit
prefix (335 or 395), not the later hour-503 validation thresholds. The original
40 causal feature inputs are unchanged; thresholds define labels and evaluation,
not additional fitted beam inputs. Class weighting and calibration were absent.

## Probability quality

All 12 Brier scores are lower than the corresponding constant fit-prevalence
control. Relative Brier reduction is 6.12-15.14%. Average precision ranges from
0.3006 to 0.3913 for warning events, whose scoring base rates are 0.1148-0.1323;
critical AP ranges from 0.1907 to 0.2767, against base rates 0.0615-0.0708.
These describe probability error and ranking on reused development windows,
not evidence of calibrated deployment probabilities or new-network performance.

## Fixed decision threshold and matched controls

At the predeclared probability threshold 0.5, warning recall is 2.40-10.78% and
critical recall is 0.15-3.12%. F1 is lower than matched hurdle point-exceedance
decisions in all 12 cases. The comparison uses the same fold-specific historical
thresholds, eligible targets, origins and beam axes, rather than comparing
earlier development labels with a later validation risk policy. Persistence and
seasonal-naive decisions are also retained in `risk_probability_operating_points.csv`.
Hard decisions are reported as classification metrics, not proper probability
quality benchmarks.

The 0.5 operating point prioritizes a different false-alarm/miss trade-off than
one might choose for rare events. This study did not select another operating
point, and results do not justify inventing an operator cost or alert budget.
Do not retune a threshold on these scoring labels.

## Nested-event inconsistency

Independent classifiers predict P(P95 exceedance) above P(P90 exceedance) on
0.0347-0.8715% of scored pairs, at tolerance 1e-12. Maximum excess is 0.23694.
Because P95 exceedance is a subset of P90 exceedance under these saved cutoffs,
these are incoherent probability pairs. The audit reports them without applying
posthoc clipping or calibration. Preserve raw outputs and the historical worker
snapshot; a coherent modeling alternative needs a separately declared study.

## Verification and decision

All 38 artifact files passed checksum verification. Fit-only threshold quantiles,
eligibility/counts, every label and scoring axis, fit-prevalence constants, and
all saved metrics were reconstructed. All classifiers have the original 40
features and saved stopping-selected rounds. Seventy-two history-only predictions
match hosted outputs exactly. No local model fitting, new threshold calibration,
validation 504-671 evaluation, original holdout replay or production promotion
occurred.

Retain the probability/ranking evidence, but reject replacing the existing risk
policy with these independent classifiers at 0.5. This does not improve the
original target MAE/WAPE; WAPE <10% remains unmet. One seed, reused development
windows and historical quantile labels do not prove generalization, absence of
overfitting, or operational congestion detection.
