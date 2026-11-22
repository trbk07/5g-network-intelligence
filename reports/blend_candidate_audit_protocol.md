# Fixed blend candidate audit

The preceding inner1 selection and fixed inner2 confirmation passed its declared
development screen. This separate protocol was recorded before the new audit:

- Keep weights and variants unchanged: 80% hurdle plus 20% linear B at 1/6h,
  80% hurdle plus 20% linear A at 24h. Do not expand the original weight grid.
- Refit the linear components through target hour 503, on all beams and joint
  fit origins. Fit global per-KPI preprocessing only on that prefix.
- Use rounded median development best_epoch+1, yielding A=4 and B=3 epochs.
  Preserve seed, architecture, optimizer and other fixed training parameters.
  Save the last specified epoch, not a validation-selected checkpoint.
- Reuse the existing verified hurdle validation replay unchanged. The selection
  JSON and hurdle replay have pinned SHA256 values in the config.
- Only after both neural fits finish, audit reused validation 504–671 on every
  eligible per-horizon origin and original beam. Retain selected-linear,
  hurdle-reference and mixture forecasts plus load-tier metrics.

No early stopping, probability calibration, coefficient changes, hurdle refit,
original final holdout access or production promotion is permitted. Validation
is repeatedly used exploratory evidence, not a fresh independent evaluation.
A new point-MAE gain would not prove future generalization or no overfitting.
WAPE remains a ratio on the same original population, with no denominator change.

Hosted implementation: `scripts/finalize_blend_candidate.py`. Reader notebook:
`18_fixed_blend_candidate_audit.ipynb`, default training off. Two fixed GPU fits
run in a separately enabled private Kaggle worker, saving epoch checkpoints,
optimizer/RNG states and logs to Kaggle Output. Local checks exercise arithmetic
on epoch metadata and review execution only; no local neural fitting occurs.

Once the hosted job completes, download its archive and verify without fitting:

```powershell
python -m scripts.import_hurdle_results .cache/blend-candidate-results/blend_candidate_results.tar.gz --kind blend_candidate
python -m scripts.summarize_blend_candidate
```

Intake is restricted to the new research directory and named reports/forecast
file. Verification checks the pinned selection/replay, original stopping metadata,
fixed last-epoch identity, global preprocessing cutoff, all origins and beams,
complete blend arithmetic, reconstructed metrics and history-only component
predictions. Hurdle CPU parity uses 1e-8 tolerance; neural/blend CPU/GPU parity
uses 1e-5 in original units to account for float32 operations. These software
checks do not establish a quality gain or authorize promotion.
