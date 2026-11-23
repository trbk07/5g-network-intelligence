# Fixed-threshold risk audit: better hurdle recall, many missed events

All results use unchanged saved per-beam P90/P95 thresholds fitted through hour
503. The threshold archive SHA256 is pinned in `research_risk_verification.json`.
No thresholds, classifier cutoffs, weights or forecasts were fitted or selected
in this audit. The original baseline and C1 rows reproduce the existing risk
evaluation exactly. All confusion counts reconcile on identical original beam,
origin and observed-target axes; 42 rows cover seven models and all horizons.

## MLP blend compared with hurdle

| Horizon | P90 recall, hurdle to MLP blend | P90 F1, hurdle to MLP blend | P95 recall, hurdle to MLP blend | P95 F1, hurdle to MLP blend |
| --- | --- | --- | --- | --- |
| 1h | 17.83% to 19.00% | 0.2790 to 0.2920 | 10.54% to 11.94% | 0.1807 to 0.2001 |
| 6h | 11.08% to 11.33% | 0.1874 to 0.1911 | 5.33% to 5.38% | 0.0981 to 0.0989 |
| 24h | 11.09% to 11.76% | 0.1874 to 0.1965 | 5.18% to 5.56% | 0.0953 to 0.1016 |

The MLP blend improves recall/F1 versus hurdle in all six threshold/horizon
cases, but most actual exceedances are still missed. P90 precision is
59.72-63.07%; P95 precision is 58.38-61.86%. At 1h, P90 false alarms increase
from 5,509 to 6,110 while misses fall from 45,131 to 44,493. P95 false alarms
increase from 1,779 to 2,142 while misses fall from 26,038 to 25,632.

Persistence is an important counterexample to a universal model-quality claim:
it has higher recall and F1 at 1h and 24h for both thresholds, with much lower
precision and many more false alarms. MLP blend has higher F1 at 6h. These are
different operating trade-offs, not proof of one best risk model. Full precision,
recall, F1, false-positive rate, counts, exclusions and undefined-metric flags
remain in `research_risk_validation.csv`.

## Interpretation and next research question

The current risk policy compares point forecasts optimized for MAE with upper
historical quantiles. A median-like point forecast is not an exceedance
probability. Improved average absolute error need not capture rare upper-tail
events, and simply increasing positive forecasts at actual zero targets is not
the same as crossing P90/P95. Keep these diagnostics distinct.

A separately declared probability-forecasting study could test direct historical
exceedance labels, probability quality and precision-recall trade-offs. Any such
study must construct its thresholds from each development fold's fit prefix,
not from the saved hour-503 thresholds when scoring earlier development folds.
Thresholds are label/scoring policy; they must not become prohibited fitted
per-beam input summaries. This report does not train such a classifier or select
a decision threshold. It also does not change the existing MAE/WAPE objective.

P90/P95 are statistical historical scenarios, not operator capacity limits or
observed congestion labels. Validation has been reused, intervals across future
weeks are unavailable, and no original holdout replay occurs. Keep production
frozen; this descriptive audit cannot authorize a deployment or a claim of
operational reliability. The WAPE <10% goal remains unmet.
