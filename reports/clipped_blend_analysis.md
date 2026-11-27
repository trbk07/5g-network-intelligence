# Clipped forecast blending: verified strict-gate failure

Decision: reject this recipe and preserve the previously confirmed MLP-B/hurdle blend. No final-validation refit, holdout replay or production promotion is authorized. Do not modify weights after observing inner2.

## Frozen protocol

Kaggle private CPU worker `truongbaokien/5g-ni-clipped-forecast-blending`, version 1, source commit `516dc90`. No base model was retrained. Cloud calibration selected each horizon's weight from 0/0.05/0.10/0.20 on inner1; ties favored smaller weights. Every choice was persisted before reading inner2 numerical arrays. All three selected weights were 0.20 and remained unchanged during confirmation.

Pinned source manifests and every declared artifact were checked. Predictions retain identical actual/origin/beam axes. The stricter gate requires no regression against hurdle or the current MLP-B/hurdle blend, plus at least 0.1% relative gain at one horizon with nonzero weight. Both folds are repeatedly reused development evidence, not unseen-test data.

## Inner2 confirmation

MAE is in original DLPRB dataset units. Positive gain means lower MAE. WAPE is sum absolute error / sum absolute actual, shown as a percentage.

| Horizon | Clipped weight | New MAE | Current blend MAE | Gain vs current | New WAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1h | 0.20 | 0.237017 | 0.237189 | +0.0721% | 40.21% |
| 6h | 0.20 | 0.253359 | 0.253525 | +0.0653% | 43.69% |
| 24h | 0.20 | 0.259377 | 0.259279 | -0.0377% | 42.46% |

## Interpretation and verification

- The older hurdle-only confirmation gate passes, with gains of about 0.590%/0.667%/0.397%. That is insufficient because an already confirmed, stronger blend exists.
- Against that current control, +1h/+6h gain only about 0.072%/0.065%; +24h regresses about 0.038%. The no-regression requirement and the 0.1% material-gain requirement both fail.
- 437760 saved forecasts were exactly recomputed; maximum discrepancy 0.0. All confirmation MAE/WAPE, coverage and strict-control decision were reconstructed without local coefficient search.
- Result archive SHA256 `c01671d69b45de0efc96281894bb23dd60dcfaf8af1bfd9b52d75b028453d611`; five artifact files verified; frozen artifacts untouched.
- The preceding error decomposition remains useful: clipped training lowers zero-target error but often raises positive-target error. Its residuals correlate around 0.93–0.95 with hurdle and 0.98 with MLP-B. This does not prove an irreducible error floor or rule out a different model/data design.
- One-seed base models and reused development folds. WAPE <10% remains unmet. Stop this predefined blend recipe; retain all negative and mixed outcomes rather than expanding the grid after confirmation.
