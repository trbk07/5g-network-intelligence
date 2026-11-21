# Research notebook standard

Notebook names describe the research question or task, not temporary run state.
Use a numbered snake_case filename and a corresponding descriptive title. Keep
run IDs, commits and versions in metadata/provenance cells.

Use `NN_research_task.ipynb` for filenames and
`5G Network Intelligence — Research Task` for the first heading. The hosted
Kaggle/Colab display title must describe the same task. Avoid vague names such as
`final`, `new`, `test2`, `best_model` or `notebook_v3`; keep execution versions
in provenance instead. Use UTF-8 for titles, Markdown and comments.

The reader-facing order is: research question, hypothesis, protocol and metric,
configuration, source/provenance, temporal split, preprocessing, model/loss,
training controls, evaluation, error analysis, conclusions and limitations.
Every code cell has one business purpose; short Markdown explains its input,
output and reason. Comments explain important choices and debugging boundaries.

Use modular source files for reusable implementation. Do not dump every source
file into the reading flow, display encoded bundles or mix environment setup,
training and result interpretation in one cell. Link to exact research modules
and show relevant model/metric logic where it helps review. Training-only workers
may retain historical source snapshots as evidence; new reader-facing notebooks
must not reproduce those snapshots as a wall of writefile cells.

Default Run All is safe: review never fits; experiment guides set RUN_TRAINING
to False. Enabling training requires an explicit configuration change and a
hosted Colab/Kaggle guard. No local fit/calibration, no holdout tuning or replay.

Tables show dataset/domain, horizon, population, units, metric denominator and
coverage. Report negative results and every declared horizon. Keep tracebacks
useful, progress logs bounded and checkpoints identifiable. No invented results.

Names for the current extension:

- 09_external_data_quality_audit.ipynb
- 10_single_source_transfer_experiments.ipynb
- 11_multisource_transfer_ablation.ipynb
- 12_transfer_results_and_error_analysis.ipynb
- 13_zero_inflated_forecasting.ipynb
- 14_zero_inflated_candidate_audit.ipynb
- 15_probability_calibration_study.ipynb

Historical execution notebooks remain attached to their original job versions;
renaming or restyling must not silently trigger repeated fitting or erase run
provenance. New presentation notebooks are separate from those execution records.
