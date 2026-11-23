# Positive quantile resolution: mixed, negligible development changes

Hosted Kaggle version 1 completed 36 additional positive-quantile fits. The
original probability classifier and four positive quantile models were reused
unchanged from the pinned manifest. Six new levels (0.15, 0.2, 0.3, 0.35, 0.4,
0.45) used the original 40 causal features, hyperparameters and stopping rule.
No calibration, coefficient selection or local fitting occurred.

| Fold | Horizon | Original MAE | Refined MAE | Relative MAE gain |
| --- | --- | --- | --- | --- |
| inner1 | 1h | 0.228831 | 0.228869 | -0.01698% |
| inner1 | 6h | 0.258466 | 0.258434 | +0.01228% |
| inner1 | 24h | 0.245016 | 0.244955 | +0.02479% |
| inner2 | 1h | 0.238425 | 0.238496 | -0.03005% |
| inner2 | 6h | 0.255061 | 0.255080 | -0.00744% |
| inner2 | 24h | 0.260410 | 0.260386 | +0.00905% |

MAE uses original target units; positive relative gain means lower MAE. Refined
WAPE remains 40.46-46.97%, far above the 10% goal. All original eligible beam
targets and per-horizon origins remain in the comparison.

Decision: reject the finer-grid configuration as a replacement. Both 1h results
regress, the 6h sign changes between folds, and the 24h gains are negligible.
All improvement intervals cross zero. The inner2 1h spatial interval is entirely
positive (worse); the other five cross zero. Intervals resample 30 base stations
with 1,000 paired draws and seed 17. They are conditional spatial descriptions
of reused windows, not fresh-test confidence intervals or corrections for
multiple experiments. No production model or validation-refit protocol changes.

Archive integrity verified 48 artifact files. Every actual/origin/beam axis and
reported MAE/WAPE/coverage was reconstructed. Original coarse forecasts replay
within 1e-8; 72 original/refined history-only predictions match the saved cloud
forecasts exactly. Source manifest SHA256:
`593d524646ed96e1b2c2d62eb67892f8698c7b23d08dff6a7420b3bbd918efb0`.

The intervention changes interpolation resolution and sorted crossing repair
together; it does not isolate pure interpolation error. One seed and reused
development windows cannot establish generalization or an irreducible error
floor. Nonetheless, this controlled result gives no evidence that densifying
this conditional grid alone materially closes the accuracy gap. Keep the
negative result rather than selecting only the better horizons. Validation
504-671 and original final holdout were excluded throughout the experiment.
