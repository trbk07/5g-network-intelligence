# Temporal linear forecasting: negative development result

Hosted GPU Kaggle version 1 completed four fits: A/B on both development folds.
All 32 artifact files matched their hashes. Metrics were reconstructed on every
eligible per-horizon scoring origin and all original beams. Seventy-two
history-only CPU/GPU predictions agreed within the declared 1e-5 original-unit
tolerance; maximum difference was 1.20e-7. No local training occurred.

Both variants have higher MAE than the saved hurdle development forecasts at
every horizon in both folds. Relative MAE regression ranges from 1.30% to 4.71%.
Full scores, including WAPE, coverage and zero false-positive rate, are retained
in `temporal_linear_comparison.csv`; no easier population or horizon is selected.

Best epochs are zero-based: inner1 A=3/B=1, inner2 A=2/B=2. Runs completed
9/7/8/8 epochs respectively and stopped under the fixed patience rule. None
reported exhausted training budget. This does not prove global optimality of
the architecture or make another learning-rate search justified by itself.

Decision: reject this configuration as a standalone candidate; do not spend a
reused-validation audit or final holdout on it. Preserve its checkpoints and
negative results. A lower zero false-positive rate does not outweigh higher MAE
under the project objective. No production model changed; WAPE <10% remains unmet.

The comparison is exploratory: one seed, reused development folds, joint neural
training/stopping origins versus per-horizon hurdle components. Scoring origins,
targets and beam axes match exactly. These results do not establish an
irreducible-error bound or a conclusion about all neural architectures.
