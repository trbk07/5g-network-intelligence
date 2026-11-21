# Development signal diagnostics

Scope: original fit segment 0–503 only. This is descriptive preprocessing and
statistics, with no model fit, calibration, validation scoring or final holdout
access. Reproduce with `python -m scripts.audit_development_signal`.

24.12% of observed fit targets are exactly zero. Differences between beam means
account for 52.65% of the pooled target sum of squares. Beam means are diagnostic
only and are not exported as forecasting inputs.

For the current-load feature, pooled versus within-beam-centered Pearson
correlation with the future target is:

| Horizon | Pooled correlation | Within-beam-centered correlation |
| --- | ---: | ---: |
| 1h | 0.8545 | 0.6840 |
| 6h | 0.5344 | -0.0141 |
| 24h | 0.8107 | 0.5889 |

At 6h, the existing causal seven-day target-seasonal-mean feature has correlation
0.8731 pooled and 0.7001 within-beam-centered. This supports inspecting seasonal
behavior separately from short-lag behavior when interpreting development
errors. It does not establish causality, sufficient predictive information or
an irreducible-error bound, and does not justify replacing model evaluation.

Centering uses only the same descriptive fit population; correlations are not
independent samples or statistical significance claims. The 120 feature/horizon
associations are saved in `development_feature_associations.csv`, with pair
coverage. No feature is added, removed or tuned by this audit. The hurdle
experiment remains governed by its predeclared protocol and 40 causal inputs.
