# GRU compute pilot

Colab primary pilot completed on Tesla T4, PyTorch 2.14.1+cu130.
Variant B, seed 17, hidden size 64, two layers, dropout 0.1.

| Item | Observed / frozen setting |
| --- | --- |
| Training beams | 600, stratified using fit-only mean/std/active ratio |
| Training origins | 105 joint origins, stride 3 |
| Training samples | 63,000 |
| Validation beams | All 2,880 |
| Validation joint origins | 145, stride 1 |
| Validation samples | 417,600 |
| Batch size | 256 |
| One epoch incl. validation/checkpoint | 8.434536489 seconds |
| Peak allocated GPU memory | 466,961,408 bytes (about 445 MiB) |
| Epoch-1 training scaled L1 | 0.3088957179 |
| Epoch-1 validation scaled L1 | 0.2756678847 |

Decision: retain A/B, seeds 17/42, batch 256, 600 beams, stride 3 and maximum
25 epochs with patience 5. The estimated 100-epoch upper bound from this pilot
is about 14 minutes, well within the four-hour GPU budget, excluding setup and
final replay. This is an estimate, not a guarantee. Per-run time limits and
per-epoch checkpoints constrain execution. Checkpoints will be written to the
project Drive folder after owner authorization for this runtime, or Kaggle Output
if Drive cannot be authorized. No full CPU GRU training is permitted.
