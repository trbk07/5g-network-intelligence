# Data dictionary

Source: `netop/gotsf-ds`, pinned revision in `configs/data.yaml`;
see the archived `dataset_card.md` for authoritative wording.

Verified release difference: CSVs encode time as an **unnamed first column**.
The loader explicitly renames this index to `Time` and records the normalization
in panel metadata. It does not infer timestamps from row positions.

| Field | Meaning | Policy |
| --- | --- | --- |
| Time | Hourly relative index | No calendar timestamp or timezone inferred |
| beam ID | `bs_cell_beam` | Numeric tuple sort; documented bounds 0–29, 0–2, 0–31 |
| DLPRB | Downlink physical resource block usage | Target; physical unit unverified |
| DLThpVol | Downlink throughput volume | Physical unit unverified |
| DLThpTime | Throughput transmission time | Physical unit unverified |
| MR_number | Measurement report count; card also says user count | Do not assume unique users |

Canonical array axes: `(time, beam, KPI)`. KPI order: DLPRB, DLThpVol,
DLThpTime, MR_number. Float64 preserves source precision. Raw NaN and zero are
retained. Infinite values and malformed numeric fields are rejected. Negative
observations are counted and retained for review, not silently removed.

Missing target samples must later be excluded from training/evaluation with
exclusion counts. LightGBM uses native missing handling. GRU imputation will
fit training-only medians and provide missingness masks in Week 4.

Active ratio means positive observations / nonmissing observations. Zero ratio
uses the same denominator. All-missing ranges and ratios are JSON null.
Phase indices modulo 24 and 168 do not represent known calendar hours.
