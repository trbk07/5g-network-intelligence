# Nested exceedance probabilities: coherent, but not an alert replacement

Hosted Kaggle v1 completed six conditional critical-event classifiers. Original
warning classifiers and fold-fit P90/P95 thresholds were reused unchanged.
Critical probabilities are the product of warning probability and conditional
critical-given-warning probability. Warning labels filter fit/stopping samples
only; they never enter features or prediction-time routing. All eligible scoring
targets, including non-warning events, remain in evaluation.

| Fold | Horizon | Relative Brier improvement | Independent AP | Nested AP |
| --- | --- | --- | --- | --- |
| inner1 | 1h | +0.0516% | 0.275435 | 0.278407 |
| inner1 | 6h | +0.2075% | 0.198381 | 0.202991 |
| inner1 | 24h | -0.0459% | 0.190736 | 0.193847 |
| inner2 | 1h | +0.0524% | 0.276719 | 0.279344 |
| inner2 | 6h | +0.0941% | 0.225705 | 0.230647 |
| inner2 | 24h | -0.0617% | 0.235395 | 0.237278 |

Positive Brier improvement means lower error. AP and log loss improve in all six
cases; Brier improves slightly at 1h/6h and regresses at 24h in both folds. The
architecture eliminates all nesting violations without posthoc correction.
Warning probabilities are exactly unchanged.

At the fixed probability threshold 0.5, recall and F1 decline in all six cases.
Inner1 24h emits no positive alerts: precision is undefined and explicitly
flagged, while recall/F1 are zero. Other nested critical recall values are only
0.073-2.163%. Retain the fixed operating-point negative result; do not select
another cutoff from scoring labels.

All 18 artifact files matched their hashes. Source probabilities, fold-fit
threshold statistics, conditional population counts, labels, beam/time axes,
feature schema/rounds, product arithmetic and all reported metrics were
reconstructed. All 108 history-only component predictions match hosted outputs
exactly. No local fitting, threshold calibration, validation 504-671 evaluation
or original holdout replay occurred.

Retain the coherent method and measured trade-offs, but reject this fixed-0.5
configuration as an alert-policy replacement. Coherence is not proof of useful
recall, calibration or operational reliability. One seed and reused development
windows cannot establish unseen-time generalization. Historical quantiles are
not operator congestion labels. Production remains frozen and the original
target MAE/WAPE objective is unchanged; WAPE <10% remains unmet.
