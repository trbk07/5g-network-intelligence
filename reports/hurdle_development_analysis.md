# Zero-inflated development results

Hosted Kaggle version 1 completed all 36 component fits. All 48 checkpoint,
metadata and prediction files matched their manifest hashes; 72 history-only
beam predictions matched cloud output exactly. Original validation 504–671 and
the final holdout were not used in this experiment.

| Fold | Horizon | Direct L1 MAE | Mixture median MAE | Relative MAE reduction |
| --- | ---: | ---: | ---: | ---: |
| inner1 | 1h | 0.229888 | 0.228831 | 0.460% |
| inner1 | 6h | 0.259182 | 0.258466 | 0.276% |
| inner1 | 24h | 0.245175 | 0.245016 | 0.065% |
| inner2 | 1h | 0.239192 | 0.238425 | 0.321% |
| inner2 | 6h | 0.255399 | 0.255061 | 0.132% |
| inner2 | 24h | 0.260673 | 0.260410 | 0.101% |

WAPE remains 40.45–46.98%, above the 10% target. Zero false-positive rates fall
slightly but remain 97.90–99.35%. Neither statistic supports a claim of reliable
zero detection or capacity-event detection. The effect is small and these
development windows have been used for prior research; no significance,
generalization or absence-of-overfitting claim is made.

Next fixed audit procedure, defined after development results and before audit:
take the rounded median of the two component early-stopping round counts, refit
through target hour 503, then inspect reused validation 504–671. Keep a matched
direct L1 control. No calibration, fitted thresholds, validation early stopping,
final holdout replay or production promotion is permitted. This adds 18 hosted
component fits, not a new hyperparameter search or fresh independent test.

Hosted candidate audit: private Kaggle
`truongbaokien/5g-network-intelligence-zero-inflated-audit`, version 1.
Its output archive is `hurdle_candidate_results.tar.gz`. After completion:

```powershell
python -m scripts.import_hurdle_results .cache/hurdle-candidate-results/hurdle_candidate_results.tar.gz --kind candidate
python -m scripts.summarize_hurdle_candidate
```

The summarizer checks component round counts against the two development
metadata files, checkpoint hashes, original validation targets, every expected
origin and the full beam axis. It reconstructs MAE/WAPE and zero false-positive
rate, then verifies 72 history-only predictions. It compares both candidate
cases with original C1 and the earlier improved ensemble. Those reference
forecasts are saved validation outputs; no final holdout is opened. A positive
result against the matched L1 control alone is insufficient to establish an
advance over the best existing research candidate.
