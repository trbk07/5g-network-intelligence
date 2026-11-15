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
