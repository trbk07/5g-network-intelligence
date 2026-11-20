# Interview questions and evidence-based answers

1. **Why persistence and seasonal baselines?** They test whether the model adds
   value beyond the current observation or a prior daily/weekly observation.
   The best comparator per horizon was chosen on validation, before holdout.
2. **How did you prevent leakage?** Features consume observations only through
   the origin. Split cutoffs constrain targets, preprocessing uses fit data,
   future-perturbation tests check invariance, and final selection precedes the
   annotated pre-holdout tag. Fit-beam summary inputs were rejected and retrained.
3. **What is rolling-origin replay?** A fixed model forecasts at successive
   origins. Newly observed historical values may enter later histories, but the
   model is not refitted during evaluation and never sees its future targets.
4. **Why a global model?** Sharing patterns across beams provides more training
   examples and simplifies artifact management. This model uses no beam-specific
   numerical fit summaries; beam IDs remain ordering/threshold metadata.
5. **What did GRU A/B show?** A used target history, B added the other KPIs.
   Two seeds per variant were measured. C1 remained preferable under validation;
   this is not a universal claim that GRU or multivariate inputs are inferior.
6. **Are P90/P95 predictive confidence intervals?** No. They are per-beam
   historical usage thresholds fitted on development data. Exceedance status
   is not a calibrated congestion probability or an incident label.
7. **What happens when P90 equals P95?** There is no Warning interval. Values
   at or above the common threshold are Critical; invalid beams are Excluded.
8. **How do you interpret false alarms and misses?** Compare predicted and
   actual threshold exceedance on common finite pairs. Precision, recall, FPR,
   event base rate and counts must be read together. Low recall remains a weakness.
9. **How did you check app parity?** Twenty real beam/origin pairs over all
   three horizons were compared with saved offline forecasts, the shared
   predictor, API and dashboard transport. Maximum observed error was zero.
10. **Why no real-time claim?** The project has historical relative indices,
    no live ingestion or operational congestion labels. Offline replay makes
    the demonstrated behavior reproducible without pretending to run a network.
11. **Why keep the original champion after improvement research?** Later search
    reused validation, and the original final holdout had already been seen.
    Fresh chronological data is needed before a new frozen promotion claim.
12. **What would you improve next?** Acquire unseen periods and verified event
    labels, test drift and alerting objectives, and quantify performance beyond
    this network. Avoid tuning thresholds against the already observed holdout.
