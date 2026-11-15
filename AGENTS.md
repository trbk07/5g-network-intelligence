# Project execution constraints

- The owner forbids all local model training. Run training only in hosted Google
  Colab or Kaggle runtimes, including LightGBM CPU training and synthetic tests
  that call fit/backward/optimizer.step.
- Local work may edit code, lint, validate inputs, perform non-training tests,
  inspect metrics and orchestrate cloud jobs/download artifacts.
- Keep training checkpoints on Drive or Kaggle Output. Do not use holdout for
  tuning or selection; follow plan5G.md and the frozen split configuration.
- Commit as the owner's configured Git identity; never replace it with Codex.
