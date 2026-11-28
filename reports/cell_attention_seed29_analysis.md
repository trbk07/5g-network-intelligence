# Predeclared Colab seed/runtime robustness: verified negative result

Source `96806fe4ab4334a41b7b7a189753c209676477b9`, hosted Colab T4,
PyTorch 2.14.1+cu130. The Kaggle seed-17 comparator used Torch 2.11.0+cu130;
this is joint seed/runtime robustness, not an isolated estimate of seed variance.
All four fits completed early stopping: 17/17/18/18 epochs and 12 stale epochs.
No fitting occurred locally. Checkpoints remain on Drive.

The in-place cloud verifier reconstructed all twelve metric rows, checked
manifest coverage/hashes, fit-only preprocessing, exact seed-29 cold weights,
stopping choices and all original forecast axes. Across 2,304 history-only beam
predictions, maximum CPU/GPU error was 7.152557373046875e-7, below 1e-4.
Only aggregate verification/metric JSON was downloaded. An archive checkpoint
download was rejected by automatic review and was not executed.

## Confirmation-fold spatial model

| Horizon | MAE (DLPRB usage units) | WAPE | MAE gain vs fixed strong comparator |
|---|---:|---:|---:|
| 1h | 0.239574 | 40.6451% | -1.0056% |
| 6h | 0.253087 | 43.6474% | +0.1728% |
| 24h | 0.259796 | 42.5279% | -0.1995% |

Across local/spatial modes, folds and horizons, seed 29 loses ten of twelve
comparisons with the historical fixed MLP/hurdle blend. It does not establish
material improvement or WAPE below 10%. No seed is selected as a new champion.
Index 504 onward remained redacted; original holdout was neither opened nor
used for tuning. Reused development cannot establish fresh generalization.

## Engineering correction

The original launcher archives symlinked Drive folders using tarfile's default
link behavior. Its training results are intact on Drive, but that archive is
not a valid regular-file result intake. Future launcher exports dereference
these links. A separate export command verified all hashes and packaged regular
files without retraining. Cloud verification then avoided checkpoint transfer
entirely, preserving the required storage boundary.

Evidence: `cell_attention_seed29_aggregate_evidence.json` and
`cell_attention_seed29_comparison.csv`. Reproduce aggregate comparisons with
`python -m scripts.summarize_seed29_aggregates`; cloud verification source is
`scripts/verify_colab_seed29.py`.
