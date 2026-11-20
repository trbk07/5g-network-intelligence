# Week 5 model quality gate

Status: **PASS**.
Legacy W3/W4 artifacts FAILED the updated plan (44/static6); all were retrained on cloud.
No inference-only patch was used to approve legacy artifacts. P1 quantiles skipped.
Normalization is fitted once at the cutoff, globally per KPI; earlier fit origins are
training samples, not separate deployed episodes. No holdout has been evaluated.

- PASS: feature_schema 
- PASS: artifacts/lgbm_w5/c1/h1 
- PASS: artifacts/lgbm_w5/c1/h1/train_samples 
- PASS: artifacts/lgbm_w5/c1/h1/validation_samples 
- PASS: artifacts/lgbm_w5/c1/h6 
- PASS: artifacts/lgbm_w5/c1/h6/train_samples 
- PASS: artifacts/lgbm_w5/c1/h6/validation_samples 
- PASS: artifacts/lgbm_w5/c1/h24 
- PASS: artifacts/lgbm_w5/c1/h24/train_samples 
- PASS: artifacts/lgbm_w5/c1/h24/validation_samples 
- PASS: artifacts/lgbm_w5/c2/h1 
- PASS: artifacts/lgbm_w5/c2/h1/train_samples 
- PASS: artifacts/lgbm_w5/c2/h1/validation_samples 
- PASS: artifacts/lgbm_w5/c2/h6 
- PASS: artifacts/lgbm_w5/c2/h6/train_samples 
- PASS: artifacts/lgbm_w5/c2/h6/validation_samples 
- PASS: artifacts/lgbm_w5/c2/h24 
- PASS: artifacts/lgbm_w5/c2/h24/train_samples 
- PASS: artifacts/lgbm_w5/c2/h24/validation_samples 
- PASS: artifacts/gru_w5/A/seed17 
- PASS: artifacts/gru_w5/A/seed17/samples 
- PASS: artifacts/gru_w5/A/seed42 
- PASS: artifacts/gru_w5/A/seed42/samples 
- PASS: artifacts/gru_w5/B/seed17 
- PASS: artifacts/gru_w5/B/seed17/samples 
- PASS: artifacts/gru_w5/B/seed42 
- PASS: artifacts/gru_w5/B/seed42/samples 
- PASS: shared_sample_and_DLPRB_statistics 

Hashes and provenance: model_quality_gate.json; initial failures: model_quality_gate_initial.json.