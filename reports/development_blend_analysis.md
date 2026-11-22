# Forecast blend: fixed development confirmation

Hosted Kaggle v1 selected all coefficients using inner1 MAE only and saved them
before loading numerical confirmation arrays. The fixed choices were B/0.2 for
1h and 6h, A/0.2 for 24h. The standalone linear configuration remains rejected;
this separate blend study tests limited complementary information.

| Horizon | Inner2 hurdle MAE | Fixed blend MAE | Relative MAE reduction |
| --- | ---: | ---: | ---: |
| 1h | 0.238425 | 0.237667 | 0.318% |
| 6h | 0.255061 | 0.253643 | 0.556% |
| 24h | 0.260410 | 0.259847 | 0.216% |

The declared development screen passes: no horizon regresses and at least one
strictly improves with nonzero linear weight. All three happen to improve.
Five artifact files matched their hashes; all 437,760 confirmation forecasts and
metrics were reconstructed exactly. Cloud choices were checked for consistency
with the saved selection grid and tie rule; no coefficient was fitted locally.

The selected weight is at the grid's upper boundary. This does not justify
expanding the grid after observing confirmation. Keep the 20% weights and
variants fixed. Both folds supported previous research and are not fresh
independent tests. This is not evidence of generalization or no overfitting.

Next separate protocol, defined before any mixture validation audit: refit the
needed linear A/B models through target hour 503 for a fixed epoch count derived
from the two original stopping runs (rounded median of best_epoch+1: A=4, B=3).
Reuse the already verified fixed-refit hurdle forecasts; do not refit or modify
them. Audit the fixed 80% hurdle / 20% selected-linear mixture on reused
validation 504–671 only. No coefficient search, early stopping, calibration or
production promotion is permitted in that audit; original final holdout remains
closed. The 10% WAPE goal is not achieved by the current development results.
