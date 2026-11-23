# Pretrained zero-shot development study

Question: can a frozen forecaster provide useful target-history predictions
without new target-domain fitting? One pinned checkpoint, native median 0.5,
168-hour context, horizon 24 with 1/6/24h scoring, float32 GPU, batch 256, seed 17.
No fine-tuning, coefficient selection, quantile search or production promotion.
The model uses target-only context; this is not a multivariate model comparison
under identical input information.

The author [model card](https://huggingface.co/amazon/chronos-bolt-base) declares
Apache-2.0. There is no separate LICENSE file in the model repository at the
pinned revision; that fact is recorded rather than inventing a license file.
The package wheel has its own Apache license. Model revision, safetensors SHA,
metadata hashes, exact library/dependency pins and wheel SHA are recorded in
`pretrained_forecaster_source_audit.json`.

The same weight blob is visible from 2024-11-13; the target repository's oldest
visible commit is 2025-04-22. These are public repository observations, not the
actual pretraining cutoff or a complete training-corpus audit. Pretraining
overlap is not certified absent; the study cannot certify a leakage-free
independent test, even if a score improves. A model is not counted as a newly
acquired independent dataset. Further overlap evidence would be necessary before
any generalization or promotion claim.

Only development windows after 383 through 443 and after 443 through 503 are
scored. Preserve all original beams and separate per-horizon origin coverage.
Every input ends at its origin and contains exactly 168 observed target rows.
Zeros are observations, NaNs remain native missing inputs, infinities are invalid.
Clip negative median forecasts to zero as declared before scores; no smoothing,
denominator changes or hard-beam exclusions. Model outputs beyond each window's
eligible target boundary are not scored or retained as target evidence.

Download only metadata and a source wheel locally for audit, never model weights
or a runtime model. The hosted GPU worker downloads pinned safetensors, verifies
hashes, freezes/evaluates the network and uses inference mode. Cache forecasts
after each origin. The original checkpoint stays separately in private Kaggle
Output; the result archive contains project forecasts/reports only. Failed jobs
retain partial cache and logs and cannot be represented as completed results.

This benchmark is exploratory because development was reused. Validation
504–671 and original final holdout are excluded. Report every fold/horizon,
original-unit MAE, WAPE ratio, coverage and zero false-positive rate. Compare with
the saved hurdle development forecasts before further work. Publisher benchmark
claims are not evidence that this checkpoint improves the original network.
