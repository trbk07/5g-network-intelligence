# Fixed blend validation audit

Hosted GPU v1 completed the two fixed neural refits (A=4, B=3 epochs). The hurdle
was not retrained. Pinned selection and hurdle-replay hashes match; the last
checkpoint epoch and global preprocessing cutoff are correct. Ten artifact files
matched their manifest. All 1,370,880 mixture forecasts and reported metrics were
reconstructed; 81 history-only component/blend predictions passed CPU/GPU parity
(maximum error 1.23e-7, below the declared neural tolerance of 1e-5).

| Horizon | Hurdle MAE | Fixed blend MAE | Blend WAPE |
| --- | ---: | ---: | ---: |
| 1h | 0.227494 | 0.226834 | 40.86% |
| 6h | 0.251004 | 0.250427 | 44.42% |
| 24h | 0.249159 | 0.249162 | 44.94% |

The original-unit MAE reductions at 1h/6h are small; 24h slightly regresses by
about 0.000003 MAE. Keep this regression visible and do not change the 24h
weight or variant after the audit. NR3500 transfer still has lower 6h/24h point
MAE; this is not a single winner across all horizons and criteria.

Exploratory paired 95% intervals resample 30 base-station clusters, preserving
all observed origins and common draws across horizons. Against hurdle, the 24h
MAE-difference interval is approximately [-0.000249, 0.000275], crossing zero.
The mean C1-normalized MAE difference is -0.001710, with interval
[-0.002598, -0.000720]. These intervals describe spatial variability conditional
on this repeatedly observed week; they are unadjusted for model search and do
not establish future-week generalization or absence of overfitting.

Zero false-positive rates rise to 99.77% / 99.65% / 99.94%, worse than hurdle at
all horizons. With nonnegative components and positive blend weights, a positive
forecast from either component can make the mixture positive; better aggregate
MAE must not be presented as better zero-event detection or capacity monitoring.

Decision: retain the fixed mixture as a research artifact with a mixed audit
result. No production promotion, new coefficient search, final holdout replay or
claim of WAPE below 10%. Further model work must use development evidence and a
separately frozen independent evaluation before any generalization claim.
