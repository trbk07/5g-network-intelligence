# Model improvement protocol

Completed: both search workers and the fixed ensemble audit finished on Kaggle.
The new candidate reduces mean normalized reused-validation MAE by 0.878% vs C1
and 0.543% vs Week 7 deeper. All three horizon MAEs improve. See
[the full result](../reports/model_improvement_results.md) and
`configs/improved_research_candidate.yaml`. It is research-only; no new final
holdout score or production promotion was performed.

Objective: minimize mean normalized original-unit MAE across +1/+6/+24 hours.
Risk precision/recall/FPR, active/zero errors and load groups are secondary
diagnostics. Do not silently change the objective to higher peak recall when
MAE worsens. No claim of global optimality can follow a finite experiment.

The frozen Week 6 production champion, its artifacts and final holdout remain
unchanged. The already observed final holdout 672-839 cannot be reused for
selection. All search, calibration, ensemble-weight fitting and model fitting
run on hosted Kaggle only. Local work may inspect saved metrics, lint, test
non-training logic and run CPU inference.

## Development validation

| Fold | Fit cutoff | Early-stop target interval for horizon h | Score target interval for horizon h |
| --- | ---: | --- | --- |
| inner1 | 335 | 335+h through 383 | 383+h through 443 |
| inner2 | 395 | 395+h through 443 | 443+h through 503 |

Inputs always have 168 observed hours. Training targets never exceed the fit
cutoff. The stopping and scoring targets do not overlap within a fold. Earlier
observations become available in later folds, as in chronological deployment.
Fold scores are development selection evidence, not a new test set.

Screen nine LightGBM recipes on inner1. Promote its three best recipes per
horizon plus the C1/deeper references to inner2. CatBoost depth 6/8 runs on both
folds with GPU training. Both families receive the same 40 causal inputs; no
per-beam fit summaries become numerical inputs. Save checkpoints and score
arrays after every fit, with independent load/predict checks.

Recipes include direct targets, larger trees, every training origin, recent
sample weights, residuals anchored on observed last/seasonal usage, and target
normalization using the observed rolling-168 mean. The floor 0.1 is predeclared.
Normalized target loss is weighted by that dynamic scale to restore original-unit
absolute error before clipping. All stopping metrics are original-unit MAE.
Clipping to nonnegative usage is predeclared for all new predictions.

## Ensemble and final development refit

Rank the recipes confirmed on both folds. The shortlist contains the best four
single models plus a complementary CatBoost, C1 and the best actual baseline
when absent. Try convex blends with at most three members and weights in 0.1
steps. Then calibrate a small fixed bias grid on the same inner scores. These
are model-calibration operations and therefore also run on cloud, not locally.

Write `improvement_selection.json` and its SHA256 before constructing audit
features or targets. For each selected member, use fit cutoff 419 and stopping
targets 419+h..503 to select rounds; refit all development targets through 503
with those fixed rounds. Do not early-stop on the audit period. Save the target
recipe/weights/bias with a separate research-only schema so production's ordinary
LightGBM loader cannot silently interpret normalized predictions as usage.

Audit once on the **reused original validation** targets 503+h..671, with paired
BS bootstrap intervals and historical risk metrics. This period was already
seen in previous project work: it is not a fresh test or proof of future-week
generalization. No promotion/new final result is justified without unseen
chronological evaluation data and a separately frozen protocol.

Compare against the existing C1 and Week 7 deeper validation replays on the same
finite pairs. Expose all horizon results and failures rather than reporting only
the winning horizon. Keep all new artifacts outside `artifacts/champion`.

## Cloud entry points

- `scripts/build_improvement_job.py --family lgbm`: CPU worker.
- `scripts/build_improvement_job.py --family catboost`: GPU complementary worker.
- `scripts/build_improvement_audit.py`: attach completed workers, canonical Week 5
  data and saved Week 7 validation references; cloud-select, refit and audit.
- `src/models/research_ensemble.py::ResearchEnsemble`: independent CPU inference
  from the saved research candidate; no fitting in `predict_at`.
- `python -m scripts.predict_research --origin 503 --beam-ids 0_0_0`: local
  inference only from downloaded artifacts; the command restricts targets to
  development/validation and never exposes final holdout actuals.
- `python -m scripts.summarize_improvement`: checks saved artifact hashes,
  selection identity and 60 CPU/cloud predictions, then summarizes existing
  scores without fitting or choosing new weights.

The execution notebooks show plain Python files in separate cells, with explicit
cloud training sections and progress logs. A review notebook must only display
saved reports. Checkpoints persist in private Kaggle Output.

Stop after the declared screening, confirmation, complementary-family and blend
audit; report if gains do not replicate. Any new search round needs a new
predeclared development hypothesis, not adjustments chosen from final holdout.

Implementation references: [LightGBM custom evaluation](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.train.html)
and [CatBoost regression losses](https://catboost.ai/docs/en/concepts/loss-functions-regression).
