# MAE-aligned objectives: verified development result

Decision: retain the objective-alignment finding, but do not promote either standalone model or refit on final validation. The mixture-MAE objective improves the old surrogate hurdle consistently, without surpassing the stronger historical controls. Nonnegative direct training trades fewer zero false positives for mixed MAE effects.

## Protocol and execution

Kaggle private `truongbaokien/5g-ni-mae-aligned-forecasting`, version 1, source `50b92367a9796a0db569ea3b6fe9a0e810c3f675`. Four fits completed on Tesla T4 / Torch 2.11.0+cu130, matching the historical control environment. The Colab fallback was prepared but did not fit; its assignment was later reclaimed. No control was retrained.

Encoder/head shapes, seed 17, four KPI inputs, batch size 1024 and maximum 40 epochs/patience 5 are unchanged. Fit/stop/score cutoffs: 335/383/443 and 395/443/503. The auxiliary coefficient 0.1 was fixed before training; no coefficient search used scoring. Original values from hour 504 onward were redacted.

## Inner2 point estimates

MAE is in original DLPRB dataset units. WAPE is sum absolute error / sum absolute actual, displayed as a percentage. These are reused development scores, not final validation or holdout.

| Mode | 1h MAE | 6h MAE | 24h MAE | 1h WAPE | 6h WAPE | 24h WAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| direct_clipped | 0.238706 | 0.255242 | 0.261757 | 40.50% | 44.02% | 42.85% |
| hurdle_mae | 0.240210 | 0.257603 | 0.262369 | 40.75% | 44.43% | 42.95% |

## Findings and limitations

- Mixture-MAE plus auxiliary losses improves all six MAEs versus the earlier BCE-plus-pinball hurdle: 1.085–2.500% on inner1 and 0.389–0.919% on inner2. This supports investigating objective alignment; it does not establish general superiority.
- The point-optimized hurdle still loses to the historical LightGBM hurdle at every point, and to MLP-B at five of six points. Its zero false-positive rate is approximately 99.5–100%; a distributional head does not guarantee useful zero-mass discrimination. No probability/quantile calibration claim is permitted.
- Direct-clipped improves three of six MAEs against matched old direct regression. Its +6h gain on inner1 reverses on inner2. Zero false-positive rate improves at all six points by 10.43–29.45 percentage points, but this alone cannot justify replacing the stronger point forecaster.
- 72 history-only beam predictions checked; maximum CPU/GPU discrepancy 1.19e-07 original units (tolerance 1e-5). All four fits completed patience; no epoch cap.
- Result archive SHA256 `28dc528915bee92784f52692d2ec7152056618b9d5d0ed48d82d294e4885df49`; 34 artifact files verified; frozen artifacts untouched.
- One seed and repeatedly reused development windows. WAPE <10% remains unmet. No holdout evaluation, production promotion, independent-test superiority or model-family error-floor claim.
