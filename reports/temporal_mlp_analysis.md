# Temporal residual MLP: stronger neural reference, no standalone promotion

Hosted Kaggle GPU version 1 completed four fits using the fixed protocol at
source commit 654a516d5424dc96a27fc0efe45c59a844a18726. The 64-unit GELU residual
network has 22,019 parameters for A and 86,531 for B. Both share weights across
all original beams, use prefix-only global per-KPI preprocessing and masks, and
add a causal seven-day seasonal mean. No neighbouring beam summaries enter.

Compared with matched Temporal Linear checkpoints, all 12 fold/variant/horizon
MAEs improve by 1.02-3.84%. This establishes a useful negative control against
insufficient nonlinear capacity in that specific linear model. It does not
establish that increasing capacity indefinitely helps.

The stronger hurdle reference remains better in 10 of 12 comparisons. Inner1 B
at 6h improves by only 0.0133%; inner2 A at 6h improves by 0.2042%. Remaining
regressions range from 0.244% to 1.789%. WAPE is 40.61-47.58%, and strict zero
false-positive rates remain 93.03-97.77%. The full table preserves every case.

Decision: reject this fixed configuration as a standalone replacement. Do not
spend a validation refit or final holdout on it. This result alone does not show
whether its forecast errors could complement another model in an independently
specified ensemble; no such gain is claimed or selected locally.

All 32 artifact checksums matched; every scoring origin, actual and beam axis
matched the canonical development panel and matched baselines. All saved metrics
were reconstructed. Seventy-two history-only CPU predictions matched hosted GPU
predictions within 1e-5 original units (maximum 1.80e-7). The shared loader change
also reproduced the earlier Temporal Linear verification unchanged. Locally only
inference, archive validation and initialized-state tests ran.

Best epochs (zero-based): inner1 A=3/B=8, inner2 A=10/B=10. Actual epochs completed
were 9/14/16/16. All stopped under the fixed patience rule, none exhausted the
30-epoch budget. Epoch selection uses stopping data, not scoring data.

Limitations: one seed; reused development windows; joint neural training/stopping
origins versus per-horizon hurdle components. The matched linear comparison uses
the same neural training/evaluation policy. These windows cannot provide fresh
independent generalization evidence. Validation 504-671 and final holdout were
excluded. Frozen production artifacts remain unchanged and WAPE <10% is unmet.
