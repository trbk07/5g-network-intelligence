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
