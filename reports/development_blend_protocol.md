# Limited development forecast blending

Temporal Linear remains rejected as a standalone candidate. A descriptive
development audit found signed-error correlation with hurdle of 0.91–0.95 and
opposite-sign errors in about 7–9% of observations. These associations are not
evidence that a blend improves MAE; they motivate only a small predefined test.
The audit did not fit or select weights locally.

On hosted Kaggle, choose the Temporal Linear variant (A/B) and its convex weight
from {0, 0.05, 0.1, 0.2} per horizon using inner1 original-unit MAE only. Ties
prefer smaller weight and then A. Save all selections before loading inner2
numerical target/forecast arrays. Manifest byte-level checksum verification may
read both folds first; no inner2 values influence selection.

Apply all choices unchanged to inner2. Pass the declared development screen only
if no horizon regresses and at least one improves with nonzero linear weight.
All-zero selection is no improvement. Do not change coefficients or variant
after inspecting confirmation. Keep the complete inner1 grid, inner2 forecasts,
source hashes and pass/fail decision. No new base-model fit is involved, but
coefficient selection is calibration and is forbidden locally.

This is reused development evidence: inner2 had already supported earlier
research and is not a fresh independent test. No validation 504–671, original
final holdout or production artifact may be used. The job cannot automatically
launch a validation audit or promote a candidate. Report every horizon, including
regressions and zero weights. A passed development screen is not proof of future
generalization, statistical significance or absence of overfitting.

Reader notebook: `17_development_forecast_blending.ipynb`, default selection off.
Focused source: `scripts/select_development_blend.py`; inference arithmetic:
`src/models/forecast_blend.py`. Checkpoints remain in their original hosted
outputs. New choices, confirmation forecasts and manifests go to Kaggle Output.
