# Dilated temporal convolution: stronger neural reference, incomplete patience

Hosted Kaggle GPU v1 completed four TCN runs after its queue cleared. The
Colab fallback had no verified model fitting: one runtime was reclaimed and
the replacement required fresh Drive authorization. The actual imported
checkpoints identify Kaggle Tesla T4, PyTorch 2.11.0+cu130, source commit
`5b6c2c87d2a03ed095ac7efd575c31612430fea1`. Do not attribute these fits to Colab
or repeat the experiment solely to change platform presentation.

The model uses 32 channels and seven left-padded, kernel-3 residual blocks with
dyadic dilation, per-time LayerNorm and dropout 0.1. It has 22,383 parameters
for A and 22,575 for B, shares weights across all original beams and retains
global per-KPI preprocessing, masks, phase4 and seasonal-mean initialization.
This is a compact adaptation inspired by [Bai et al.](https://arxiv.org/abs/1803.01271),
not a reproduction of their original architecture or results.

Compared on identical development targets, TCN lowers MAE versus Linear in
12/12 cases and versus matched MLP in 10/12. Variant B lowers MAE versus MLP-B
in all six cases by 0.156-0.757%. Versus hurdle it improves only 5/12 cases:
it remains worse at 1h in every fold/variant. Full MAE, WAPE, coverage and
zero false-positive rates remain in `temporal_tcn_comparison.csv`.

| Fold / B variant | 1h MAE | 6h MAE | 24h MAE |
| --- | --- | --- | --- |
| inner1 B | 0.230047 | 0.256650 | 0.246777 |
| inner2 B | 0.238673 | 0.254090 | 0.260083 |

## Stopping limit discovered during verification

Best epochs are zero-based: inner1 A=15/B=26, inner2 A=14/B=16. Completed
epochs are 21/30/20/22. Inner1 B reached the 30-epoch cap with only three stale
epochs, below patience five. Its stopping procedure is not complete. The
historical `budget_exhausted=false` field refers to the wall-time limit; it
must not be read as proof that an epoch cap did not bind. Preserve the original
metadata unchanged and report derived termination status separately.

The trainer now records an explicit termination reason for future runs.
Fixed-refit epoch derivation now rejects development metadata whose declared
protocol has not completed patience, even when wall-time budget is not exhausted.
The prior Linear and MLP stopping counts must still reproduce before reusing
that shared policy. Do not derive a new fixed refit from the capped TCN run or
expand training after inspecting validation/holdout. Any continuation needs a
separately declared stopping-data-only budget policy and preserved source states.

All 32 artifact files match their hashes. Full observed-target scoring axes,
metrics and fit-prefix preprocessing were verified. Seventy-two history-only
CPU predictions agree with GPU output within 1e-5 original units (maximum
1.80e-7). No local training occurred. These are one-seed reused development
comparisons, not independent generalization evidence or an error-floor claim.
No validation 504-671 audit, original holdout replay or production promotion
occurred. WAPE <10% remains unmet.
