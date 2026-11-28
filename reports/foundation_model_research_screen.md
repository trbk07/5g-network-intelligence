# Additional foundation-model screen — 2026-11-28

Official sources were checked after the seeded architecture/transfer failures.
This is candidate screening, not an experiment result or a quality claim.

## TimesFM 2.5

The [official model card](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
declares Apache-2.0 weights, 200M parameters and pretraining on GiftEvalPretrain,
Wikimedia, Google Trends, synthetic and augmented data. Its inference API
provides point and quantile forecasts. A median forecast is more aligned with
MAE than an unconditional mean, but target-domain benefit remains unproven.
The [official fine-tuning example](https://github.com/google-research/timesfm/blob/master/timesfm-forecasting/examples/finetuning/README.md)
documents Transformers/PEFT LoRA support.

Before any run: pin model/code revisions and hashes; audit available pretraining
membership against the original and acquired corpora; preserve explicit unknowns;
declare full-population two-fold screening with no target future covariates.
Do not claim a contamination-free test from an incomplete corpus description.
No weights downloaded or fitting launched by this screen.

## TimesFM 3.0

The [official repository license notice](https://github.com/google-research/timesfm#license-notice-for-pretrained-weights)
restricts downloaded 3.0 weights to non-commercial, non-production use; source
code and weights through 2.5 use different terms. Consequently 3.0 downloaded
weights are not selected as a production candidate for this project. No paid
Google Cloud service is provisioned by this screen.

The immediate scientific conclusion remains unchanged: nine-domain transfer
has not demonstrated material target-domain gains. Additional downloaded bytes
are not evidence of forecast quality; every future candidate must beat the
fixed strong comparator under the same causal protocol.
