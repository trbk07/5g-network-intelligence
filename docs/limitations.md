# Limitations and interpretation

- Historical offline replay, not live telemetry or deployed network automation.
- One network dataset and one 168-hour internal final holdout; no evidence of
  cross-network or unseen future-week generalization.
- External weeks are excluded because chronological continuity is not verified.
- DLPRB physical units are uncertain; absolute errors use dataset usage units.
- P90/P95 are historical per-beam thresholds. No ground-truth congestion/incident
  labels or calibrated congestion probabilities are available.
- Point-error gains do not solve alerting: P90 recall is about 10-16%, and many
  exceedances are missed. Small positive predictions at actual zero are common.
- Low-load WAPE has a small denominator; compare absolute, active-only and
  load-group errors rather than relying on pooled WAPE alone.
- Two GRU seeds per variant and a bounded compute budget limit architecture
  conclusions. Negative results are specific to this protocol.
- Paired BS-bootstrap intervals reflect the sampled network and time period,
  not uncertainty over arbitrary future deployments.
- Post-freeze model research reused validation. Its search intervals are not
  multiple-comparison corrected. No new final-test or global-optimality claim.
- The demo contains 24 real beams. Full-network benchmark values shown in the
  UI are saved results, not scores computed from the selected demo window.
- Missing observations remain missing. Entirely missing target history is not
  treated as a valid risk alarm, even if the model emits a finite fallback.
- No predictive uncertainty bands, production SLA, automated remediation or
  continuous ingestion are implemented.
- Clean-clone/release requirements are tracked in the completion audit; do not
  infer release readiness from an individual weekly test result.
