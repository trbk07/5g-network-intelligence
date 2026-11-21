# Zero-inflated forecasting experiment

Hypothesis: a separate active-load probability and positive conditional
distribution may reduce original-unit MAE where a direct L1 model predicts
small positive values for zero targets. This is not evidence of improvement.

The mixture median is zero when active probability p <= 0.5; otherwise it is
the positive conditional quantile at 1 - 0.5/p. Four fixed quantiles
(0.05, 0.1, 0.25, 0.5), nonnegative clipping, crossing repair and interpolation
approximate this value. Probability error and quantile approximation can worsen
MAE. No class reweighting or probability calibration is applied.

The direct L1 control has the same 40 causal inputs and shared tree parameters.
Two chronological fit/stopping/scoring folds end at 335/383/443 and 395/443/503.
Early stopping uses the component's loss, not the scoring population. Scoring
cannot tune thresholds or quantiles. These development windows supported prior
research: they are exploratory evidence, not a fresh independent test.

No targets from reused validation 504–671 or the original final holdout are
available to this experiment. Frozen production is unchanged. Six components
per fold/horizon yield 36 cloud fits. Local checks exercise only inference
mathematics and the default review notebook, with no fitting calls.

Hosted execution: private Kaggle
`truongbaokien/5g-network-intelligence-zero-inflated-forecasting`, version 1.
Source dependency: completed multisource worker output plus a private MIT
dataset containing only the new trainer, inference helper, configuration and
license. The source notebook is explicit training; the repository reader copy
defaults to RUN_TRAINING=False. Metrics are pending; no gain is claimed.

Report every fold and horizon: MAE (original units), WAPE (ratio), full observed
population count and zero false-positive rate. The last metric alone cannot
justify selecting the model. Checkpoints, predictions and logs remain in Kaggle
Output, including a partial archive if the trainer fails.

After the job completes, download `hurdle_results.tar.gz` and run:

```powershell
python -m scripts.import_hurdle_results .cache/hurdle-results/hurdle_results.tar.gz
python -m scripts.summarize_hurdle_research
```

The importer accepts only `artifacts/hurdle_research/` and the two named hurdle
reports. It validates archive paths, member types, manifest coverage and hashes
before installation. Archived production artifacts and configs are excluded;
different existing research files cause failure rather than replacement. The
summarizer separately checks temporal coverage, recomputes metrics and compares
history-only inference against saved cloud forecasts. Archive checks alone are
not evidence of model quality or successful completion of the experiment.

Implementation reference: [official LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html)
for binary and quantile objectives. The mixture interpretation is a project
derivation, not a claim made by the library documentation.
