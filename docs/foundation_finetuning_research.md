# Research priority: foundation-model fine-tuning

## Decision

Prioritize native multivariate **Chronos-2-Synth + LoRA**, with a matched
multivariate zero-shot control from the same pinned checkpoint. Defer further
small blend searches and the pending TCN patience-maintenance job. Existing
production artifacts and their release gates remain unchanged.

This is a method choice, not an accuracy claim. The main question is whether
target-domain adaptation of a substantially pretrained temporal model beats the
existing strong controls, rather than whether another small coefficient grid
beats a weak reference.

## Why this differs from completed experiments

- Earlier Chronos-Bolt was univariate **zero-shot**, without fine-tuning; it
  regressed at 1h/24h. That result does not test native multivariate adaptation.
- The external-data MLP pilot used a small residual network and short fixed
  pretraining, with inconsistent gains over self-pretraining. It does not prove
  that large-scale representation learning or other datasets cannot help.
- Recent neural heads and blends produced mixed or very small gains. The latest
  clipped blend failed the stricter current-control gate; do not reopen its grid
  after inspecting inner2.

## Primary-source review

| Method | Evidence relevant to this project | Decision |
| --- | --- | --- |
| Chronos-2-Synth + LoRA | Publisher declares synthetic-only univariate/multivariate pretraining; native grouped multivariate forecasting; released API supports LoRA | First experiment |
| Chronos-2 standard | Native multivariate/covariate support, but mixed real/synthetic pretraining corpus needs additional overlap assessment | Not the initial checkpoint |
| PatchTST masked pretraining | Shared channel-independent patch encoder and masked reconstruction support own-corpus pretraining and downstream fine-tuning | Next independent architecture if needed |
| MOMENT | General masked-pretraining framework with limited-supervision evidence, including forecasting probes | Secondary option; not assumed superior for this beam task |

References: [Chronos-2-Synth model card](https://huggingface.co/autogluon/chronos-2-synth),
[official Chronos repository](https://github.com/amazon-science/chronos-forecasting),
[released package](https://pypi.org/project/chronos-forecasting/2.3.2/),
[PatchTST paper](https://arxiv.org/abs/2211.14730),
[official PatchTST implementation](https://github.com/yuqinie98/PatchTST),
[MOMENT paper](https://proceedings.mlr.press/v235/goswami24a.html).

The synthetic-only statement is publisher provenance, not an independent audit
of the entire pretraining process. Do not turn it into an unconditional
no-leakage guarantee. Independently enforce our original-data boundaries.

## Frozen first experiment

Model and runtime pins are in `configs/chronos2_synth.yaml` and
`configs/chronos2_lora.yaml`. The downloaded 475,963,368-byte safetensors payload
matches the publisher SHA256. Configuration is separately pinned. The PyPI
2.3.2 wheel was inspected without importing or fitting the external model.

1. Use DLPRB as the target and the other three KPIs as native past-only
   covariates within each beam's multivariate task. Only DLPRB future labels
   contribute supervision; future covariate actuals are never supplied.
   Preserve zeros and NaNs; disable cross-learning between different beams.
   Never supply future actual covariates.
2. Fine-tune only each fold's fit prefix, ending at 335 or 395. Native window
   sampling uses context 168, minimum past 168 and prediction length 24; every
   supervised timestamp remains inside that prefix.
3. Use LoRA rank 8, alpha 16, learning rate 1e-5, maximum 1,000 steps, seed 17.
   Preserve optimizer/RNG checkpoints every 100 steps on Drive/Kaggle Output.
   Fail before optimization if PEFT is absent or the trainable fraction exceeds
   5%; the upstream API otherwise silently falls back to full fine-tuning.
4. Checkpoints 100/300/1000 are eligible for selection by original-unit DLPRB
   stopping MAE, with earlier-step ties. Native quantile-loss checkpoint selection
   is disabled. Training completion alone is not model-selection completion.
5. Score the selected adapter and unchanged synthetic-pretrained control on all
   original beams and frozen development origins at 1h/6h/24h. Compare with
   verified LightGBM hurdle and MLP-B/hurdle controls. Reconstruct MAE/WAPE,
   coverage, zero error and saved inference parity before accepting any gain.
6. A research candidate needs no horizon regression and at least 2% mean relative
   improvement versus the strong current control, followed by seed/uncertainty
   checks. This is not an automatic production-promotion or validation-refit gate.

Full evaluation and checkpoint selection remain required work. There is no new
MAE/WAPE result yet. Reused development cannot prove unseen-test generalization;
the previously observed holdout stays excluded throughout.

## Additional-data strategy

Use the already audited radio prefixes and CESNET for a **separate** domain
adaptation stage only after establishing that target fine-tuning can learn.
Match additional target-only updates to isolate the value of external data.
Investigate more continuous telecom sources when domain coverage is inadequate;
do not optimize dataset count or concatenate bytes, throughput and PRB as one
physical label. Check license, units, overlap, continuity and timestamp cutoffs
before use. Pretrained synthetic knowledge is not counted as a newly downloaded
raw telecom dataset.

No local fit/calibration is permitted. Acquisition, source inspection and input
boundary tests are allowed locally; all adaptation runs on hosted GPUs.
