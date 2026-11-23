# Development error accounting: positive-load errors dominate

This report audits fixed forecasts only. It neither fits models nor searches
thresholds, coefficients or subpopulations for deployment. All original observed
targets are retained; model, actual, origin and beam axes match exactly across
the hurdle, Temporal Linear A/B and residual MLP A/B forecasts. The two reused
development folds end at 443 and 503. No validation or holdout is accessed.

Across the six hurdle fold/horizon cases, zero targets account for 21.6-24.4% of
observations but only 6.52-7.15% of total absolute error. More than 92% of the
absolute error is on positive targets. Strict zero false-positive rate alone
therefore exaggerates the importance of zero predictions for the MAE objective:
it counts every positive forecast equally regardless of error magnitude.

Among positive targets, 60.3-64.1% are underpredicted and mean signed error ranges
from -0.1064 to -0.0798 original units. These are target-conditioned diagnostics,
not available inference-time labels. In a zero-inflated asymmetric distribution,
an unconditional median can legitimately be below the positive-target mean;
these statistics do not justify adding a bias correction or changing the loss.

The CSV also reports the hypothetical WAPE if zero-target errors alone were
eliminated while positive-target errors stayed unchanged. This uses an oracle
target label and is not an achievable forecast or an error-floor claim. It
clarifies why zero-only correction cannot close the observed gap to 10%.

Next research hypothesis: test whether the coarse conditional positive-quantile
grid in the hurdle model loses useful resolution when recovering the mixture
median. Keep the saved probability classifier and original positive quantile
models fixed; add a predefined finer positive grid trained only on the same
cloud development prefixes. Compare full-population forecasts with the original
hurdle, without threshold tuning or a validation/holdout refit. This is a proposed
controlled experiment, not a claimed gain. Both folds remain reused development
evidence; an eventual candidate still needs independent generalization evidence.

Reproduce accounting with `python -m scripts.audit_development_errors`. Numerical
tests cover missing targets, observed-target forecast rejection, negative-target
rejection and exact zero/positive error reconciliation. No local fitting occurs.
