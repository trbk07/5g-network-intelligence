# Fixed seasonal median: no improvement over hurdle

This analytical reference predicts the median of seven raw target values at
`origin + horizon - 24*d`, for `d=1..7`. It uses the same 168-hour history and
original horizons 1/6/24h. No coefficients, periods, thresholds or parameters
are fitted or selected. Execution is read-only inference, not local training.
It is an additional research reference; the original release baselines remain
unchanged.

The reference tests whether a robust daily phase statistic is already sufficient
to improve MAE, without adding model capacity. It is worse than the saved hurdle
forecasts in all six development fold/horizon cases: MAE regression ranges from
3.42% to 18.63%. WAPE ranges from 44.46% to 50.43%. Full figures and valid-target
counts are in `seasonal_median_development.csv`.

Every origin, actual target and beam axis matches the original hurdle score
population. Missing historical values are omitted from the seven-day median;
an entirely missing seasonal history produces a missing forecast, which causes
the audit to fail if the target is observed. No missing prediction silently
shrinks scoring coverage. Five analytical tests cover exact causal references,
outlier/missing handling and invalid horizons; lint passes.

Decision: reject this fixed formula as a replacement. Do not tune its period or
combine weights using these scores. There is no new validation or holdout replay,
and no production promotion. These reused development windows cannot prove an
irreducible error floor or a conclusion about all robust forecasting methods.
