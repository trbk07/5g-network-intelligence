# Research extension - WAPE below 10%

The owner selected pooled WAPE: sum of absolute error divided by sum of absolute
actual usage. The target is strictly below 0.10 for each original 1/6/24-hour
horizon, with full eligible-beam coverage and explicit exclusions. No renaming
MAE 0.1 as 10%, changing the denominator, smoothing the evaluation target or
dropping difficult/zero beams to manufacture success is permitted.

Current improved candidate WAPE on reused validation is approximately
41.05% / 44.62% / 45.00%. The target therefore requires a large gain. It is an
experimental objective, not an achieved result or a promise. Absolute absence of
overfitting cannot be proven; chronological out-of-sample checks, gap/group
guards, train-only preprocessing, source deduplication and frozen selection
reduce and expose that risk.

The original holdout remains closed for tuning. Existing frozen production
artifacts and final results remain unchanged. Additional independent periods
must be assigned train/validation/test roles before target-driven search.
Previously observed validation results cannot be relabeled as a fresh test.

External candidate acquisition is authorized, but merging is conditional on
verified license, units, sampling, entity meaning and distribution compatibility.
The OAI KPM candidate is emulated RAN data with PRB allocation and throughput,
not the same commercial beam population. Its README describes controlled
traffic and roughly five weeks; this is not evidence that it supplies useful
pretraining or independent testing for the original network. Audit first.

Primary sources:
- https://github.com/CTTCADAPTRU/OAI-KPM-dataset
- https://huggingface.co/datasets/netop/gotsf-ds
- https://github.com/netop-team/gotsf

All model fitting, fine-tuning, transfer learning and ensemble calibration must
run on hosted Kaggle/Colab, including synthetic fitting tests. Local acquisition,
schema/statistical auditing and inference are allowed. See wape_research.yaml
for the machine-readable objective and guard policy.

## Acquisition findings

- OAI KPM revision `328ab90c9faabbba6a036255ac503683c227fa34` was downloaded
  with README/license and SHA256 records. The actual KPM sheet has 1138 rows,
  one UE and roughly 19 minutes at near-one-second spacing, assuming the
  timestamp-like `Latency` column is microseconds. It is inconsistent with
  the five-week description and rejected for direct hourly beam training.
- Zenodo record `17815388`, CC BY 4.0, partitions 01/02 were downloaded and
  verified against upstream MD5 plus recorded SHA256. They describe commercial
  sector counters every 15 minutes. Data attribution remains separate from the
  MIT code license and original NETOP data.
- Partition 01 has no duplicate entity/time keys but long temporal gaps for
  some entities. Forecast windows must never bridge missing intervals.
- Partition 02 NR has 1,800,920 rows and 722,137 conflicting site/sector/time
  keys after exact-row deduplication (there are no exact duplicate rows).
  Selecting the first row or averaging without authoritative semantics would
  silently change the target. Quarantine it until the natural key is resolved.

See the external manifests and audits in reports/. No new model quality or
external test score is claimed from acquisition/structural evidence alone.

## Multi-source strategy

There is no fixed small limit on dataset count. `configs/dataset_registry.yaml`
tracks separate sources and partitions, including rejected/gated candidates,
without counting multiple bands as independent networks. Prioritize measured
original-domain gains over raw data volume. Compare each new source individually,
then source-balanced combinations, against a target-only control. Keep native
test sets and source identities isolated. No score from an easier external task
can substitute for original 1/6/24-hour beam forecasting quality.

The first pilot uses target-history relative-shape features with exogenous
columns missing for both domains. It compares external weight ratios
0/0.1/0.5/1.0. Only original hours through 503 select rounds; fixed refit then
audits reused validation 504-671. External chronological future data and the
original final holdout remain unused. Scaling invariance and future-perturbation
tests pass locally without fitting.

The initial external training fraction of 60% yielded no continuous segment
long enough for the required 168-hour history plus targets. It was changed to
80% based only on timestamp/coverage structure, before model scores, retaining
the final 20% for future independent evaluation. This structural revision is
explicit, not a performance-driven change to a test split.
