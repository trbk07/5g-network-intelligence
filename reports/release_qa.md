# Release QA - in progress

Release v1.0.0 has not yet been declared ready. Outstanding items are explicit.

| Requirement | Evidence | Result |
| --- | --- | --- |
| Remote clean clone + Docker build/start | clean_clone_qa.json, clean_clone_build.log, commit f403602 | PASS for that source snapshot with explicit verified assets |
| Five API endpoints / 404 / 422 / health | clean-clone runtime requests | PASS |
| Demo asset checksum / 24 real beams | demo_asset_package.json | PASS, package 1,788,914 bytes |
| End-to-end / ordered batches / load once | 12 targeted artifact/predictor/E2E tests | PASS, no model fitting |
| Baseline routing | three baseline fixture routes | PASS |
| LightGBM routing | actual frozen artifacts with synthetic request ordering | PASS |
| Optional GRU CPU routing | initialized-state fixture, phase4, no training | PASS; GRU Docker profile not shipped |
| Frozen model / holdout provenance | existing artifact/output checksum verification | PASS; final targets not replayed |
| CPU latency / RSS / image footprint | inference_benchmark.json/.md | PASS, single-process measurement, no SLA claim |
| Experiment report generated from CSV | scripts/build_report.py | PASS; rendered plot inspected |
| Model card / architecture / limitations | docs/model_card.md, architecture.md, limitations.md | Present |
| Portfolio / interview preparation | portfolio_summary.md, interview_questions.md | Present, 12 Q&A; no invented production claims |
| Real screenshots and 90-150s video | capture_demo.py, docs/screenshots, demo_walkthrough.webm | PASS: actual 120s video; frames at 35s/80s and API screenshot inspected |
| Safe test suite / lint | pytest -m 'not cloud_training'; Ruff | PASS: 101 passed, 1 smoke skipped, 5 fitting tests deselected; real HTTP parity separately rerun |
| Dataset attribution and code license | dataset_card.md names dataset MIT/citation | Code license awaits owner choice |
| Release download + checksum cold start | verified local archive package exists | Release URL/download path pending |
| v1.0.0 tag and GitHub Release | No final release created yet | Pending final QA and license |

The clean-clone test used independently downloaded GitHub source plus an
explicit asset archive, not hidden local source or training data. It shut down
its isolated QA Compose project after checking the running service contracts.
Later source changes must be rechecked before the final release tag.
