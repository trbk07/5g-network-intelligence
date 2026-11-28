# Cell spatial-temporal attention: verified mixed result

Decision: do not promote. Spatial mixing improves three of six MAEs versus the matched local architecture, but does not consistently beat the existing fixed MLP-B/hurdle blend. More epochs alone are not justified by this result. All four fits completed patience; no epoch cap.

## Numerical consistency resolution

The original fused GPU attention inference path reproduced its saved forecasts, but differed from CPU above the predefined 1e-4 DLPRB tolerance. Disabling that inference fastpath achieved strict CPU/GPU parity without changing weights or selected epochs. The same discrepancy occurred on CPU Torch 2.11 and 2.15; CPU historical/refactored model outputs were identical. Keep original artifacts and strict replay separate. Future cell-study stopping/scoring explicitly disable the fastpath.

## Inner2 strict-math scores

MAE: original DLPRB units. WAPE: total absolute error / total absolute actual, shown as percent. Reused development only.

| Mode | Horizon | MAE | Current blend MAE | Gain vs current | WAPE |
| --- | --- | ---: | ---: | ---: | ---: |
| local | 1h | 0.238225 | 0.237189 | -0.437% | 40.42% |
| local | 6h | 0.252126 | 0.253525 | +0.551% | 43.48% |
| local | 24h | 0.261190 | 0.259279 | -0.737% | 42.76% |
| cell | 1h | 0.237807 | 0.237189 | -0.261% | 40.35% |
| cell | 6h | 0.253295 | 0.253525 | +0.091% | 43.68% |
| cell | 24h | 0.261646 | 0.259279 | -0.913% | 42.83% |

## Evidence and limitations

- 2304 history-only beam predictions verified; maximum local CPU/GPU discrepancy 1.9073486e-06 DLPRB units.
- Every MAE/WAPE, target mask, origin and beam axis was independently reconstructed. Preprocessing was recomputed from each fit prefix; 1e-12 float64 reduction roundoff tolerance.
- Source training: private Kaggle `truongbaokien/5g-ni-cell-spatial-temporal-attention`, v1, source `a575f3f`. Inference-only strict replay: `truongbaokien/5g-ni-cell-attention-precision-replay`, v1, source `6871a8a`.
- Original archive SHA256: `5a0289377c58b8a20d7ff9a5a0643a951a2ca9e8efe7dad326496d20890dea86`; strict replay archive: `b29eaa92058f8acc9f37997e7e95ad1bf1d75b46981bdb13a03c6661a2125f9e`.
- One seed, repeatedly reused development windows, no fresh-test superiority claim. No holdout or hours 504 onward used. WAPE below 10% remains unmet.
- The multi-corpus study is a separate running experiment with matched target-self pretraining; its source gains remain unproven.
