# Literature-driven forecasting experiments

Research reviewed: 2026-11-28. Objective: improve physical DLPRB MAE and
WAPE, not simply increase dataset count or reproduce benchmark rankings.

| Primary source | Relevant idea | Decision for this project |
| --- | --- | --- |
| [TimeXer, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html) / [official implementation](https://github.com/thuml/TimeXer) | Separate endogenous temporal patches from exogenous variable histories; bridge them with attention. | Implement a paired past-covariate ablation first. DLPRB is the target; three other KPIs provide history only. |
| [iTransformer, ICLR 2024](https://arxiv.org/abs/2310.06625) / [official repository](https://github.com/thuml/iTransformer) | Treat individual variables as tokens to model their relationships. | Supports testing explicit KPI relationships; defer a separate implementation until the first ablation is assessed. |
| [PatchTST, ICLR 2023](https://arxiv.org/abs/2211.14730) / [official repository](https://github.com/yuqinie98/PatchTST) | Temporal patches, channel-independent shared encoding and masked self-supervised learning. | Next hypothesis if covariate attention fails: masked target-domain pretraining. This differs from our earlier source future-forecasting pretraining. |
| [Goal-Oriented Time-Series Forecasting](https://arxiv.org/abs/2504.17493) / [authors' repository](https://github.com/netop-team/gotsf) | Forecast quality focused on application-specific value ranges, including wireless data. | Relevant to capacity-risk monitoring. Range-weighted objectives are not automatically appropriate for the current global MAE/WAPE objective; evaluate them separately. |

These are architectural hypotheses, not evidence that their published gains
transfer to this beam dataset. No published test scores determine our settings.

## Reference code and attribution

The Time-Series-Library reference was inspected at revision
`4e938a1767106324dd753b2a44832bf870a0252e`:
[TimeXer implementation](https://github.com/thuml/Time-Series-Library/blob/4e938a1767106324dd753b2a44832bf870a0252e/models/TimeXer.py),
[MIT license](https://github.com/thuml/Time-Series-Library/blob/4e938a1767106324dd753b2a44832bf870a0252e/LICENSE).
The standalone TimeXer repository's license was not established during review.
Our model is independently implemented; no upstream code is vendored.
iTransformer reference revision: `c2426e68ca13f74aaec08045c5c724d8ad328124`;
PatchTST reference revision: `204c21efe0b39603ad6e2ca640ef5896646ab1a9`.

## Implemented experiment: target patches and past KPI attention

Code: `src/models/past_covariate_attention.py`; hosted controller:
`scripts/train_past_covariate_attention.py`; configuration:
`configs/past_covariate_attention.yaml`.

The target history is split into 12-hour patches. A global target token attends
to three whole-history KPI tokens. Two arms have the same parameters and cold
initialization: one receives past supporting KPIs, the other receives constant
tokens. This tests whether those inputs help, rather than comparing unequal
model capacities. Both arms retain DLPRB history, masks and phase4.

This is a TimeXer-inspired adaptation, not a faithful paper reproduction. We
use fit-prefix global KPI preprocessing, missing masks, seasonal residuals,
three direct horizons and MAE. The block's feed-forward/cross-attention order
also differs from the reference implementation. No future covariates are used.

Use context 168 hours and horizons 1/6/24 hours. Fit/stop/score endpoints remain
335/383/443 and 395/443/503. Exclude original hours 504 onward. Training seed
17, at most 80 epochs, patience 12. Checkpoints remain in hosted storage.
Stopping MAE chooses checkpoints; scoring cannot change training settings.
MAE is in physical DLPRB usage units; WAPE is summed absolute error divided
by summed absolute observed load. Standardized MAE is proportional to physical
MAE within each fold because its target scale is a positive global constant.

Assess every fold/horizon against the paired control and fixed strong blend.
A 2% mean point improvement with no regression is an initial screen; a 5%
improvement still requires robustness, uncertainty and independent artifact
verification. Reused development folds do not establish fresh generalization.
No holdout evaluation or production promotion is authorized by this experiment.

## Why changing the hypothesis is warranted

The completed Telecom data/budget study passed the independent integrity and
replay audit but failed all declared quality gates. Mean relative MAE gains
against the fixed strong control were -0.03936% (Telecom matched budget),
+0.01117% (existing sources extended budget), and -0.08743% (Telecom extended
budget). These are negligible or negative, not a breakthrough.

Evidence is preserved in `reports/telecom_transfer_verified_metrics.json`,
`reports/telecom_transfer_paired_comparisons.csv`, and
`reports/telecom_transfer_quality_assessment.json`. More eligible data hours
do not mean all observations were visited: source epochs sampled fixed-size
windows with replacement. The next experiment therefore changes information
representation instead of assuming another increase in source volume suffices.

## Current verification status

The architecture passes lint and six local forward/guard tests, including
future poisoning at the actual batch interface and physical/standardized MAE
equivalence. Seven metadata-only verifier tests cover checkpoint identity and
artifact coverage/checksums. These checks do not train or establish forecast
quality. Hosted training has started on Colab after the queued Kaggle duplicate
was authoritatively cancelled. The live progress record is
`reports/past_covariate_attention_progress.json`.

Independent verification is implemented in
`scripts/verify_colab_past_covariates.py`. Run it in Colab only after all four
declared fits complete. It verifies the pinned source manifest, canonical
development prefix, artifact hashes, fitted preprocessing, matched seeded cold
states, checkpoint training identities, genuine patience stopping, supervision
counts and optimizer updates. It reconstructs all saved metrics and replays
history-only forecasts on CPU with the predeclared 0.0001 tolerance. Weights stay
on Drive; only aggregate evidence is exported. Implementation or passing unit
tests must not be described as a completed cloud audit.

`scripts/assess_past_covariate_attention.py` accepts only the complete verified
study and compares both cases to the fixed strong comparator, as well as the
past-KPI case to its new control. Seven aggregate-only gate tests cover missing
cases, population mismatch, source/parity verification and regressions hidden by
a positive mean. The reader review is
`notebooks/46_past_kpi_attention_results_and_error_analysis.ipynb`; its default
Run All displays available evidence and never trains or loads checkpoints.
