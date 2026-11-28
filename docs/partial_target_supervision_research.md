# Retaining available short-horizon supervision

## Evidence and hypothesis

`CellBatches` intentionally requires all 1h/6h/24h labels. This is a valid joint
training protocol, but it excludes short-horizon labels near the end of each
fit prefix. Additional external bytes cannot recover those unused original
labels. A separate research batcher now retains available labels while masking
targets beyond an explicit cutoff. Historical experiments remain unchanged.

For 2,880 beams, assuming observed labels as in the audited prefix:

| Fit endpoint | Joint labels | Available short-horizon labels | Additional labels |
|---|---:|---:|---:|
| 335 | 1,252,800 | 1,370,880 | 118,080 (9.43%) |
| 395 | 1,771,200 | 1,889,280 | 118,080 (6.67%) |

The input tensor and target backing tensor are cropped before transformation
or gathering. Labels whose origin+horizon exceeds the cutoff stay NaN; no
future values are fetched and no target is imputed. Source code is
`src/features/partial_cell_inputs.py`.

Nine array-only checks verify exact parity with historical joint batches,
complete label counts, missing-label masks, cutoff poisoning invariance and
invalid-boundary rejection. No fitting or optimizer operation ran locally.

## Required experiment controls

This implementation has not been trained or shown an accuracy gain. A hosted
controller and two matched configurations are now implemented, with the same
batch horizon-balanced training objective for both new joint-label control and
available-label candidate. The old historical control is not silently reused
as an objective-matched arm. Stopping remains complete joint windows and full
scoring populations remain unchanged. Finish
the already running Telecom data/compute experiment before allocating a new
study. If tested, use the same architecture, seed, global fit-prefix
preprocessing, target cutoffs, original scoring populations and joint stopping
criterion. Explicitly disclose changes in horizon weighting: pooling all finite
labels weights shorter horizons more heavily. A matched training objective
across control and candidate is required. Natural window counts and therefore
optimizer updates per epoch also differ; report both. This two-arm study cannot
by itself attribute a gain exclusively to label availability rather than the
associated extra optimizer steps. A pure data-effect claim needs an additional
step-matched control.

Record horizon-specific supervision counts and checkpoint/optimizer/RNG state
on hosted Colab/Kaggle. Verify full-population MAE/WAPE and numerical parity
before any acceptance. Do not infer a gain from additional label counts,
alter original observations from index 504 onward, reuse holdout, or promote a
model from this implementation-only result.
