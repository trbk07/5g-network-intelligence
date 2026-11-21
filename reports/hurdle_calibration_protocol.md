# Affine-logit development calibration

Question: does a fixed sigmoid correction of active-load probabilities improve
the mixture-median forecast? Better probability loss need not improve MAE or
zero detection. The raw frozen classifier and positive-quantile checkpoints
remain unchanged.

Fit one L2 logistic regression (C=1, LBFGS, maximum 200 iterations, tolerance
1e-6, seed 17) on clipped classifier logits for each original development
stopping window. Clipping epsilon is fixed at 1e-6. No class reweighting or
hyperparameter search. Evaluate raw and corrected probabilities on the later
scoring window for the same fold/horizon. Six calibrators fit on hosted Kaggle;
local tests exercise only inference arithmetic and safe default notebook review.

Stopping labels also selected the original base-model round count. This makes
the calibration window reused development evidence; the scoring window remains
separate from calibrator fitting. Both development score windows have supported
prior research, so neither is a new independent test. Calibration cannot consume
validation 504–671, final holdout labels or production artifacts.

Save coefficients, calibration target boundary, scoring probabilities and
predictions, source manifest hash, complete metrics and convergence failures.
Reproduce the raw hurdle scores to detect drift in inherited checkpoints or
feature computation. Report all six comparisons using original-unit MAE, WAPE,
zero false-positive rate, Brier score and binary log loss. Do not select on one
metric while hiding regressions in another. No generalization or no-overfitting
claim is authorized by this exploratory experiment.

Implementation: `scripts/calibrate_hurdle_development.py`. Reader notebook:
`15_probability_calibration_study.ipynb`, default RUN_CALIBRATION=False.
Hosted execution explicitly enables calibration and writes durable Kaggle Output.

The [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html)
describes sigmoid calibration and its assumptions. This project uses a fixed
L2 logistic fit on logits, not a claim of automatic perfect calibration or the
exact `CalibratedClassifierCV` training procedure.
