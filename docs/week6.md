# Week 6 - Champion, historical risk and final holdout

The selected Week 5 candidate is LightGBM C1 (40 causal features). Final refit
uses development hours 0-671 and fixed validation iterations 284/457/414 for
1/6/24 hours, seed 17. The final internal holdout is 672-839. No holdout tuning,
early stopping or model selection is allowed. Quantile models are out of scope.

Training runs only on hosted Kaggle CPU. This uses the already authenticated
Kaggle runtime and durable Output instead of requiring another Colab Drive mount.
The final refit implementation supports the selected LightGBM and baseline
fallback; a different GRU champion would require a separate fixed-epoch refit.

1. Run `python -m scripts.run_risk --segment val` for development-fit P90/P95
   comparisons. Use the canonical Week 5 cloud panel, whose checksum is frozen.
2. Run `python -m scripts.freeze_champion`; review `configs/champion.yaml`.
3. Commit source/configs with the owner's Git identity, then create and push the
   annotated `pre-holdout` tag. Generate `reports/freeze_manifest.json` with
   `src.inference.freeze.create_attestation()` from that exact HEAD.
4. Package `python -m scripts.build_week6_job --mode final`; upload the private
   execution notebook. It attaches completed Week 5 Output, checks all source
   hashes, runs cloud tests, refits and saves independent load-verified models.
5. The guarded holdout runner checks tag attestation, executing commit, frozen
   config, panel checksum, artifact checksums and a single-attempt marker before
   accessing final targets. It compares against validation-selected baselines
   on common masks, with paired 30-BS bootstrap confidence intervals.
6. Download durable reports/checkpoints. `python -m scripts.run_risk --segment
   holdout` verifies saved outputs without repeating the replay. Package the
   default `--mode review` notebook to display saved reports without training.

Cloud provenance trusts the packaged local Git attestation and compares every
protected source/config hash; it does not fetch GitHub or independently verify
a tag signature. The private job submission must use the tagged source package.
The runner refuses duplicate attempts inside the persisted workspace. The job
submission ledger must also prevent launching a fresh workspace with the same
holdout configuration. A failed attempt requires a recorded reason and diff,
never silent retries or selection based on final metrics.

Historical thresholds use observed-only linear P90/P95. Eligibility requires
168 observations, P90 above 1e-8 and active ratio at least 0.05. These constants
were fixed from fit units and hourly coverage, before validation risk metrics.
Actual and predicted events both use `>=`. P90=P95 has no Warning band.
Undefined precision/recall have NaN plus explicit flags. Statistical exceedance
is not verified congestion or an incident. Diagnostic load tiers use only the
final development observations. FP/FN cases are deterministic first examples.

Artifacts: `artifacts/champion/` contains models, context, metadata, feature
schema, historical thresholds and a SHA256 manifest. Reports include the final
benchmark, paired comparison, risk metrics by horizon/threshold/load group,
threshold eligibility, FP/FN cases and completion provenance.
