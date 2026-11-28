# Project completion audit

The active goal covers all required gates in `plan5G.md` plus the owner's later
dataset/model-quality research extension. MIT v1.0.0 is published; the additional
WAPE research remains active and is not claimed complete.
Scientific, application and clean-clone checks below have direct saved evidence.

| Gate / requirement | Current evidence | Status / remaining work |
| --- | --- | --- |
| G1 data provenance / split | data audit, pinned revision, frozen split reports; release_evidence_audit.json | Pinned raw/panel checksums reverified |
| G2 baselines / metrics / causal evaluator | release science subset: 30 tests passed | Baselines, metric flags, origins, masks and leakage rechecked without fitting |
| G3 LightGBM causal40 | release_model_quality_recheck.json; cloud metadata | Read-only schema/provenance recheck PASS |
| G4 GRU A/B phase4 | four W5 runs / ablation; release model recheck; CPU initialized-state fixture | Schema/provenance and optional CPU routing PASS; no local fitting |
| G5 common mask / CI / atlas | model_quality_gate, model_analysis, error_atlas; generated experiment_report | Reconciled into generated report with limitations |
| G6 freeze / once-only holdout / risk | pre-holdout tag, completion hashes, risk reports | Frozen evidence verified during improvement; preserve unchanged |
| G7 API / dashboard / Compose / parity | clean_clone_qa.json, 60 parity comparisons | Fresh remote clone + assets + Docker PASS at release commit c976d6c; this is not a clean-clone test of later research commits |
| Model improvement | 53 search fits, fixed ensemble refit, reused-validation audit | Research candidate verified; no unseen-data promotion claim |
| External transfer extension | transfer_pilot_evidence.json, multisource_pilot_evidence.json, transfer_ablation_decision.md | 54 artifact files verified, 81 parity checks passed; negative NR1800/pool transfer; NR3500 gains <=0.207% are exploratory |
| Research notebook presentation | notebook_standards.md; notebooks 09–13; transfer_review_execution_evidence.json | Hosted review loaded both complete report tables; reader Run All performs no fitting |
| Zero-inflated forecasting extension | hurdle_development_analysis.md, hurdle_verification.json | Cloud v1 complete: 48 files verified, 72 history-only parity predictions exact; MAE gains 0.065–0.460% on exploratory development folds |
| Zero-inflated candidate refit | hurdle_candidate_analysis.md, hurdle_candidate_verification.json, hurdle_candidate_cluster_intervals.csv | Hosted v1 complete; 21 files and 72 parity predictions verified; MAE 0.227494/0.251004/0.249159, small gains vs earlier ensemble; 24h spatial interval crosses zero; no promotion |
| Probability calibration study | hurdle_calibration_analysis.md, hurdle_calibration_verification.json | Hosted six-calibrator study complete; all forecasts/metrics reconstructed; mixed negative result, correction rejected; validation and holdout excluded |
| Temporal linear study | temporal_linear_analysis.md, temporal_linear_verification.json | Four hosted GPU fits complete; 32 files and 72 history-only CPU/GPU predictions verified; all 12 comparisons regress, standalone candidate rejected; validation and holdout excluded |
| Fixed development blend | development_blend_analysis.md, development_blend_verification.json | Inner1-selected 20% linear weights improve all three inner2 MAEs; 437,760 fixed forecasts reconstructed; small reused-development gains only, no promotion |
| Fixed blend validation audit | blend_candidate_analysis.md, blend_candidate_verification.json | Two fixed GPU refits complete; 10 files, full replay arithmetic and 81 history-only predictions verified; 1h/6h MAE improves, 24h slightly regresses and zero false-positive rates worsen; no promotion |
| Pretrained zero-shot reference | chronos_zero_shot_analysis.md, chronos_zero_shot_verification.json | Hosted inference complete; 127 files and cache mapping for 875,520 scored predictions checked; only 6h improves, 1/24h regress; corpus overlap not certified, no independent local model execution or promotion |
| Temporal residual MLP | temporal_mlp_analysis.md, temporal_mlp_verification.json | Four hosted GPU fits verified: 32 files, 72 history-only parity predictions; beats matched linear in 12/12 cases but loses to hurdle in 10/12; standalone replacement rejected; validation and holdout excluded |
| Positive quantile resolution | hurdle_quantile_resolution_analysis.md, hurdle_quantile_resolution_verification.json | 36 additional hosted fits; 48 files and 72 exact history-only predictions verified; mixed MAE changes below 0.031%, replacement rejected; validation and holdout excluded |
| Fixed MLP development blend | mlp_blend_analysis.md, mlp_blend_verification.json | Hosted selection complete; 5 files and 437,760 forecasts verified; fixed 20% B improves all inner2 MAEs by 0.434-0.602%; reused development only, no production promotion |
| Fixed MLP blend validation audit | mlp_blend_candidate_analysis.md, mlp_blend_candidate_verification.json | Ten-epoch hosted B refit complete; 6 files, full replay and 81 history-only predictions verified; MAE improves all horizons, WAPE 40.80/44.28/44.79%, zero FP worsens; reused validation only, no promotion |
| Direct exceedance probability study | risk_probability_analysis.md, risk_probability_verification.json | 12 hosted fits verified: 38 files and 72 exact history-only predictions; Brier improves fit-prior, fixed-0.5 F1 loses all 12 matched cases; nesting violations retained, no promotion |
| Nested exceedance probability study | nested_risk_probability_analysis.md, nested_risk_probability_verification.json | 6 hosted fits verified: 18 files, 108 exact history-only predictions; zero nesting violations; AP/log loss improve, 24h Brier and all fixed-0.5 recall/F1 regress; no promotion |
| Dilated temporal convolution | temporal_tcn_analysis.md, temporal_tcn_verification.json | Four Kaggle T4 runs verified: 32 files and 72 history-only predictions; B improves matched MLP in 6/6 cases; one epoch-capped run lacks completed patience, no refit/promotion |
| T8.1 clean clone / artifact I/O / routing | clean_clone_qa.json; dedicated artifact/E2E tests | PASS at checked snapshot; final source recheck before release |
| T8.2 latency / RSS / image size | inference_benchmark.json/.md | Measured and inspected; no production SLA claim |
| T8.3 generated experiment report | scripts/build_report.py, experiment_report.md | Generated from CSV; figure inspected |
| T8.4 first-screen README / cards / architecture / limits | README and dedicated model_card/architecture/limitations docs | Present and reviewed |
| T8.5 real screenshots / 90-150s video / portfolio / interview | docs/screenshots, 120s video, portfolio summary, 12 Q&A | Completed and visual frames inspected |
| T8.6 v1.0.0 release / assets / attribution / QA | release_qa.md, published_release_evidence.json, release_evidence_audit.json | Published MIT release/tag; published package download/install PASS; private access required |

Training and calibration remain cloud-only. Final holdout is not reused for
tuning or new model selection. Local inference/verification is permitted.
No gate is declared complete solely because a test command passed; its coverage
must match the requirement and the actual artifact/runtime behavior.

Latest read-only evidence audit passed 96 checks (release_evidence_audit.json).
The pinned demo archive can be installed directly without original raw data;
checksum-denial and archive-traversal rejection tests passed. A separate target
directory installation loaded the expected frozen champion and 24 beams.
The published GitHub asset has also been downloaded, checksum-verified and
installed into a fresh directory. Further dataset/WAPE research is separate
open work and is not represented as a completed quality improvement.

## Research extension checkpoint: 2026-11-28

The read-only release evidence verifier was rerun after the new research work:
96/96 checks pass. This confirms protected evidence integrity, not a fresh
clean-clone test of every later research commit or a new holdout evaluation.

| Extension | Evidence | Decision / remaining work |
| --- | --- | --- |
| Telecom data and pretraining budget | telecom_transfer_verified_metrics.json; telecom_transfer_quality_assessment.json | Independent audit passed; all quality gates failed. No promotion. |
| Available target supervision | partial_supervision_verified_metrics.json; partial_supervision_analysis.md | Four hosted target fits completed; audit verified 2,304 history-only predictions with maximum error 0.00000190735. Added labels fail quality gates. |
| Literature-driven target/covariate attention | literature_driven_training.md; past_covariate_attention_submission.json | Model/controller implemented; Kaggle v1 queued at latest observation. No training or accuracy gain established for this model. |
| Colab attention failover | past_covariate_colab_preparation.json | T4/Drive prepared; canonical development prefix rebuilt from pinned public sources and hours 504 onward redacted. Fitting awaits authoritative cancellation of the queued duplicate. |
| Causal and integrity checks | test_past_covariate_attention.py; test_partial_supervision_audit.py | 15 forward/metadata tests passed locally, with no model fit/backward. Includes future poisoning at the batch interface and standardized/physical MAE equivalence. |

The broader quality objective remains unmet. Repeated development comparisons
and extra data are not substitutes for material improvement or fresh evidence.
