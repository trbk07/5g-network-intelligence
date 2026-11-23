# Pretrained zero-shot study: mixed development result

The pinned Chronos-Bolt Base job completed on hosted GPU without target-domain
fitting or calibration. All 127 project artifact files matched their manifest;
875,520 scored predictions matched the per-origin caches and all metrics were
reconstructed on the original scoring origins, targets and beams.

| Fold | Horizon | Chronos MAE | Hurdle MAE | Relative MAE reduction |
| --- | ---: | ---: | ---: | ---: |
| inner1 | 1h | 0.231555 | 0.228831 | -1.191% |
| inner1 | 6h | 0.257282 | 0.258466 | 0.458% |
| inner1 | 24h | 0.245517 | 0.245016 | -0.205% |
| inner2 | 1h | 0.238804 | 0.238425 | -0.159% |
| inner2 | 6h | 0.254339 | 0.255061 | 0.283% |
| inner2 | 24h | 0.263505 | 0.260410 | -1.189% |

Only 6h improves in both folds. The 1h and 24h results regress consistently.
WAPE is 40.51–47.08%; zero false-positive rates are 99.81–99.93%. This does not
bridge the gap to the 10% WAPE goal, demonstrate reliable zero detection or
justify a full-range standalone replacement. The previously confirmed fixed
linear/hurdle blend also has lower inner2 6h MAE than this checkpoint.

Decision: retain this as an exploratory pretrained reference; do not spend a
reused-validation audit, tune quantiles or promote the checkpoint. Record every
regression. Original production and final holdout remain unchanged.

The local verifier checked cache/score mapping and reconstructed metrics; it did
not independently execute the external model. No model weights were downloaded
locally. Public weight/dataset chronology is documented but the complete
pretraining corpus is not audited, so absence of overlap is not certified. A
better score alone would not establish leakage-free zero-shot generalization.
