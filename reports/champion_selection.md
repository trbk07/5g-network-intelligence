# Frozen champion selection

Family: **lightgbm_c1**. Config hash: `7f11d8e4209b4a7e49a5a67e9f754e5f9c2607d130fe1d60f7fc2dbb83d5593e`.
Applied the predeclared Week 5 normalized-MAE/paired-CI rule; baseline fallback was eligible.
Risk policy was fixed from fit-segment units/sampling before validation risk metrics; it did not change the point champion.
Final fit bounds: [0, 671]; holdout bounds: [672, 839].
Refit rounds/epochs are copied from validation artifacts. No holdout early stopping or tuning.
Create the annotated pre-holdout tag before any final evaluation, then generate freeze_manifest.json.

```yaml
1:
  family: lightgbm
  iterations: 284
  parameters: &id001
    objective: regression_l1
    num_leaves: 31
    learning_rate: 0.05
    n_estimators: 500
    early_stopping_rounds: 50
  seed: 17
  feature_schema: lgbm-causal40-v2
6:
  family: lightgbm
  iterations: 457
  parameters: *id001
  seed: 17
  feature_schema: lgbm-causal40-v2
24:
  family: lightgbm
  iterations: 414
  parameters: *id001
  seed: 17
  feature_schema: lgbm-causal40-v2
```