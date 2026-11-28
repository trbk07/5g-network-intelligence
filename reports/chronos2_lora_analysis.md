# Chronos-2 multivariate LoRA: verified development result

Decision: do not promote this adapter or extend its budget automatically. Fine-tuning
improves the unchanged synthetic-pretrained control at all six development points,
but loses to the stronger fixed MLP-B/hurdle control at five of six points.

## Evidence

Both folds completed 1,000 hosted Kaggle T4 optimizer steps. Each selected step is
1,000, determined from stopping MAE among steps 100/300/1000 before scoring. The
adapter contains 1,205,888 trainable parameters. Original data from hour 504 onward
was redacted; the previously observed holdout was not used.

The independent verifier reconstructed all MAE/WAPE values, stopping choices,
frozen axes and exact causal cache identities. Eight CPU/GPU origin comparisons
passed; maximum discrepancy was 3.5762786865234375e-7 DLPRB units. The initial
research gate failed. No production artifacts changed.

## Inner2 results

MAE uses original DLPRB units; WAPE is total absolute error divided by total
absolute actual values. These are reused development results, not a fresh test.

| Horizon | Selected LoRA MAE | Current blend MAE | Relative MAE gain | LoRA WAPE |
| --- | ---: | ---: | ---: | ---: |
| 1h | 0.241755 | 0.237189 | -1.9251% | 41.0151% |
| 6h | 0.251273 | 0.253525 | +0.8880% | 43.3347% |
| 24h | 0.260395 | 0.259279 | -0.4306% | 42.6260% |

The selected adapter predicts a positive value for every zero target at all six
points. This is a useful error diagnostic, not proof of an irreducible error floor.
More compute or more datasets cannot be assumed to fix it without a controlled
experiment. Preserve this negative result when comparing future architectures.

Archive SHA256: `199149ece40a911b6d7f2dadec5857ffb72cfccc3401adc5c5ee585c27cc5083`.
409 artifact files were verified. Source: private Kaggle
`truongbaokien/5g-ni-chronos2-adaptation-evaluation`, version 1, code source
`0bcd01c`. Detailed reconstruction is in `chronos2_lora_verification.json` and
`chronos2_lora_control_comparison.csv`.
