# Project execution constraints

- The owner forbids all local model training. Run training only in hosted Google
  Colab or Kaggle runtimes, including LightGBM CPU training and synthetic tests
  that call model.fit/backward/optimizer.step. Preprocessing/statistics checks are allowed locally.
- Local work may edit code, lint, validate inputs, perform non-training tests,
  inspect metrics and orchestrate cloud jobs/download artifacts.
- Keep training checkpoints on Drive or Kaggle Output. Do not use holdout for
  tuning or selection; follow plan5G.md and the frozen split configuration.
- Commit as the owner's configured Git identity; never replace it with Codex.
- User-facing Kaggle/Colab notebooks must contain readable Python cells, section
  explanations and useful comments. Never display base64/encoded source bundles.
  Separate configuration, preprocessing, model, training, debugging and evaluation;
  default review notebooks must not retrain on Run All.
- Updated Week 5 contract: LightGBM has 40 causal inputs; GRU has phase4 and
  global per-KPI preprocessing (A:1, B:4). Beam summaries may support sampling and
  diagnostics only. Week 5 prepares a candidate; no final freeze or holdout evaluation.
- Reader-facing research notebooks follow docs/notebook_standards.md: descriptive
  filenames/titles, hypotheses and temporal protocol, one business purpose per
  cell, useful comments, explicit metrics/units, negative results and limitations.
  Do not dump all source files into their main reading flow. Historical execution
  snapshots remain immutable; presentation updates must not trigger repeat fitting.
