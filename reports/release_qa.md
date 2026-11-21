# Release QA - v1.0.0

Release v1.0.0 is published in the private repository at source c976d6c.
The MIT license was explicitly selected by the owner. Further WAPE research is
separate from this frozen portfolio release.

| Requirement | Evidence | Result |
| --- | --- | --- |
| Remote clean clone + Docker build/start | clean_clone_qa.json, clean_clone_build.log, commit ec9372e | PASS for latest runtime/packaging source with explicit verified assets |
| Five API endpoints / 404 / 422 / health | clean-clone runtime requests | PASS |
| Demo asset checksum / 24 real beams | demo_asset_package.json | PASS, package 1,788,914 bytes |
| End-to-end / ordered batches / load once | 12 targeted artifact/predictor/E2E tests | PASS, no model fitting |
| Baseline routing | three baseline fixture routes | PASS |
| LightGBM routing | actual frozen artifacts with synthetic request ordering | PASS |
| Optional GRU CPU routing | initialized-state fixture, phase4, no training | PASS; GRU Docker profile not shipped |
| Frozen model / holdout provenance | existing artifact/output checksum verification | PASS; final targets not replayed |
| Original LightGBM/GRU artifact quality | release_model_quality_recheck.json | PASS, read-only schema/provenance/preprocessing checks |
| Baseline/metric/common-mask/origin/leakage science subset | 30 focused tests | PASS, no fitting |
| CPU latency / RSS / image footprint | inference_benchmark.json/.md | PASS, single-process measurement, no SLA claim |
| Experiment report generated from CSV | scripts/build_report.py | PASS; rendered plot inspected |
| Model card / architecture / limitations | docs/model_card.md, architecture.md, limitations.md | Present |
| Portfolio / interview preparation | portfolio_summary.md, interview_questions.md | Present, 12 Q&A; no invented production claims |
| Real screenshots and 90-150s video | capture_demo.py, docs/screenshots, demo_walkthrough.webm | PASS: actual 120s video; frames at 35s/80s and API screenshot inspected |
| Safe test suite / lint | release_nontraining_tests.log; Ruff | PASS: 103 passed, 1 smoke skipped, 5 fitting tests deselected; real HTTP parity separately rerun |
| Dataset attribution and code license | LICENSE; dataset_card.md | PASS: owner-selected MIT code license; original dataset attribution retained |
| Release download + checksum cold start | published v1.0.0 asset downloaded and installed in a fresh directory | PASS: exact pinned checksum and expected model/24-beam identity |
| v1.0.0 tag and GitHub Release | annotated v1.0.0 tag at c976d6c; GitHub release isDraft=false | PASS, private repository visibility preserved |

The clean-clone test used independently downloaded GitHub source plus an
explicit asset archive, not hidden local source or training data. It shut down
its isolated QA Compose project after checking the running service contracts.
Later source changes must be rechecked before the final release tag.
