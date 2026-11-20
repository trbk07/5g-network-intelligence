# Week 6 final result

Gate G6: PASS for the selected LightGBM C1 champion, internal fallback holdout.
Training and all fitting tests ran only on hosted Kaggle CPU. One final holdout attempt; no reruns or tuning.

## Point forecast benchmark

| Horizon | Champion MAE | Validation-selected baseline | Baseline MAE | MAE reduction | 95% paired MAE-difference CI |
| --- | ---: | --- | ---: | ---: | --- |
| 1h | 0.228237 | persistence | 0.298215 | 23.47% | [-0.074471, -0.065295] |
| 6h | 0.250562 | seasonal_naive_24 | 0.327570 | 23.51% | [-0.084647, -0.068236] |
| 24h | 0.249446 | persistence | 0.326987 | 23.71% | [-0.086264, -0.067386] |

Each comparison uses 100% finite-pair coverage, 2880 beams and 168/163/145 origins. All three intervals are below zero.
The bootstrap resamples 30 verified BS clusters, 1000 repetitions, seed 17. It preserves paired errors and complete time histories within each BS.
MAE reduction is relative to the validation-selected baseline, not relative to the Week 5 validation score.

## Risk trade-off and limitations

C1 produces fewer false alarms than baseline but misses many exceedances. Point-forecast quality does not establish strong alerting performance.
| Horizon | Scenario | Precision | Recall | FPR | Actual positive pairs |
| --- | --- | ---: | ---: | ---: | ---: |
| 1h | historical_warning_p90 | 64.31% | 16.19% | 1.13% | 54002 |
| 1h | historical_critical_p95 | 62.64% | 8.56% | 0.31% | 28038 |
| 6h | historical_warning_p90 | 62.52% | 10.43% | 0.80% | 53302 |
| 6h | historical_critical_p95 | 61.95% | 4.42% | 0.17% | 27709 |
| 24h | historical_warning_p90 | 60.81% | 10.67% | 0.86% | 46556 |
| 24h | historical_critical_p95 | 59.32% | 4.74% | 0.20% | 24164 |

- P90/P95 were fitted on development hours 0-671. All 2880 beams were eligible; no near-zero/low-activity exclusions occurred in this dataset.
- Positive events count beam-hour pairs, not distinct incidents. Historical exceedance is not ground-truth congestion.
- Strict actual-zero false-positive rate (`prediction > 0`) is 99.57-99.87% for C1: small positive predictions remain common when actual usage is zero. This is separate from threshold-risk FPR.
- Low-load WAPE is large because its actual-value denominator is small; inspect load-tier and active/zero metrics alongside pooled MAE.
- Only one 168-hour internal holdout was evaluated. Generalization to a different network, future weeks or verified incidents is unproven; BS-bootstrap intervals do not establish it.
- DLPRB physical units are unverified; reported absolute errors use dataset usage units.
- No quantile model, holdout calibration or post-holdout parameter changes were introduced.
