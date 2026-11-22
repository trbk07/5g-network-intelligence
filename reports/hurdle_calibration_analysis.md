# Probability calibration: negative development result

Private Kaggle v1 completed all six affine-logit calibrator fits. Twelve artifact
files matched their manifest hashes. All saved scoring probabilities,
mixture-median predictions and reported metrics were reconstructed without
fitting locally; see `hurdle_calibration_verification.json` and the parity CSV.
The original base checkpoints and positive-quantile models were unchanged.

| Fold | Horizon | Raw MAE | Corrected MAE | Relative MAE reduction |
| --- | ---: | ---: | ---: | ---: |
| inner1 | 1h | 0.228831 | 0.228844 | -0.005842% |
| inner1 | 6h | 0.258466 | 0.258527 | -0.023494% |
| inner1 | 24h | 0.245016 | 0.245133 | -0.047728% |
| inner2 | 1h | 0.238425 | 0.238423 | 0.000750% |
| inner2 | 6h | 0.255061 | 0.255034 | 0.010380% |
| inner2 | 24h | 0.260410 | 0.260390 | 0.007416% |

The first fold regresses at every horizon. The second improves by tiny amounts;
Brier/log loss also show mixed fold-dependent effects. Zero false-positive rates
fall in every comparison but remain 96.55–98.66%. This is not reliable zero
detection, and it does not offset the inconsistent MAE result under the project
objective.

Decision: do not add this correction to the research candidate; do not refit it
on reused validation or try further calibration settings against that audit.
Keep the uncalibrated hurdle candidate and earlier ensemble as separate research
references. Frozen production remains unchanged. The WAPE <10% target is unmet.

The calibration stopping labels had also selected base-model round counts; the
later scoring labels did not fit calibration. These reused development folds do
not establish independent-test generalization. No validation 504–671 or final
holdout was accessed in this calibration experiment. All regressions are retained.
