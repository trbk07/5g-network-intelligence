# Telecom Italia prefix readiness

The acquired 5,117,552,328-byte publisher HDF5 passed MD5 and SHA256 checks.
Streaming preprocessing audited 29,677 entity ranges partitioning 39,793,555
rows. Metric reads were restricted to the first 1,190 of 1,488 hourly grid
positions. Gaps were preserved; no interpolation or model fitting occurred.

Preparation yielded 14,581 continuous segments with 13,598,826 entity-hours.
The source-only quality audit excluded 3,278 whole constant segments and 306
exact whole-segment duplicates. It retained **10,997 segments, 10,694 entities
and 12,029,560 entity-hours**. Zeros inside varying segments remain observed
values. Original-domain observations and evaluation masks were not changed.

The audited manifest binds every shard hash and explicitly lists excluded
segment identities. The sampler checks all shard hashes before yielding any
data, validates continuous hours and missing non-source channels, and treats
the entire collection as one sampling domain. A real-data sampler smoke check
constructed 128 finite windows of shape 168×2 and three-horizon labels without
fitting. Source normalization is not a fitted forecasting model.

## Boundaries

The ODbL-1.0 attribution is “from BigDataChallenge contest”; derived-database
distribution obligations remain applicable. No raw/sharded database is pushed
to the public code repository. Metric units are not verified as PRB; this is
eligible only for context-normalized representation pretraining, not physical
target pooling. No original-target quality improvement has been established.

Duplicate screening compares exact entire aligned segments within this
collection and against nine prepared external domains. It does not prove that
all partial intervals differ or establish the original GOTS-F calendar/network
identity. No independent-network count or fresh-generalization claim follows.

Evidence: `telecom_italia_preparation.json`. The larger per-segment prefix report
and audited manifest remain local data artifacts. Reproduce with
`python -m scripts.prepare_telecom_prefix`, then
`python -m scripts.audit_telecom_shards`; both refuse to overwrite existing output.
