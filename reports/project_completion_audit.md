# Project completion audit

The active goal covers all required gates in `plan5G.md`. Existing evidence is
not equivalent to a release-ready clean-clone verification. This checklist stays
open until every required item is directly inspected or executed.

| Gate / requirement | Current evidence | Status / remaining work |
| --- | --- | --- |
| G1 data provenance / split | data audit, pinned revision, frozen split reports | Recheck manifests and split consistency for release |
| G2 baselines / metrics / causal evaluator | baseline reports, evaluator tests | Recheck oracle, NaN and leakage tests without local fitting |
| G3 LightGBM causal40 | W5 checkpoints / quality gate | Verify archived evidence and environment pins |
| G4 GRU A/B phase4 | four W5 runs / ablation | Verify schemas and CPU routing fixtures; no local fitting |
| G5 common mask / CI / atlas | model_quality_gate, model_analysis, error_atlas | Reconcile into experiment report |
| G6 freeze / once-only holdout / risk | pre-holdout tag, completion hashes, risk reports | Frozen evidence verified during improvement; preserve unchanged |
| G7 API / dashboard / Compose / parity | week7 product evidence, 60 parity comparisons | Repeat from clean clone with assets |
| Model improvement | 53 search fits, fixed ensemble refit, reused-validation audit | Research candidate verified; no unseen-data promotion claim |
| T8.1 clean clone / artifact I/O / routing | Partial existing API tests | Missing dedicated clean-clone and all-family fixture evidence |
| T8.2 latency / RSS / image size | scripts/bench_inference.py | Execute benchmark and inspect output |
| T8.3 generated experiment report | Existing separate weekly reports | Missing scripts/build_report.py and consolidated evidence |
| T8.4 first-screen README / cards / architecture / limits | README partly covers product | Missing final model card, architecture and limitations docs |
| T8.5 real screenshots / 90-150s video / portfolio / interview | Week7 dashboard screenshot | Missing API screenshot, video, portfolio summary and >=8 Q&A |
| T8.6 v1.0.0 release / assets / attribution / QA | No final release verified | Missing release QA, asset distribution, tag/release and link checks |

Training and calibration remain cloud-only. Final holdout is not reused for
tuning or new model selection. Local inference/verification is permitted.
No gate is declared complete solely because a test command passed; its coverage
must match the requirement and the actual artifact/runtime behavior.
