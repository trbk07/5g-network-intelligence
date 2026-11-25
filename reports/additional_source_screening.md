# Additional source screening after the quantile experiment

The original-network WAPE objective remains unmet. Broader acquisition continues,
but a new repository name does not establish additional independent observations.
No training, calibration or target-domain evaluation occurred in this screen.

## VERGE OAI: exact duplicate observations

Pinned repository: [VERGE-PROJECT/OAI_RAN_KPM_dataset at c7939ae](https://github.com/VERGE-PROJECT/OAI_RAN_KPM_dataset/tree/c7939ae2884e07c62b6909b797b6fa9e07496b18).
The public workbook, README and LICENSE were downloaded separately. Workbook:
156,026 bytes, SHA256
`d570d8c0567b7ec85fd5e3ca865d1cc109dc741583b01d3ee4f1ff964892d30c`.

The workbook and README are byte-identical to the previously audited
CTTCADAPTRU/OAI-KPM-dataset files, despite the workbook filename difference.
The LICENSE differs: this pinned repository contains GPL-3.0 text, whereas the
prior source carried CC0. Keep both provenance records and do not treat either
as a new independent network, fresh test set or additional training sample.
No downloaded files are redistributed by this audit; only hashes and metadata
are committed. Existing structural rejection of the identical workbook still
applies. See `verge_oai_candidate_manifest.json` and `oai_candidate_audit.json`.

## Glasgow: incompatible measurement task

The [publisher record](https://zenodo.org/records/20465872) describes CC BY 4.0
data: 720 speed-test measurements across devices/providers/locations on three
days in April 2025. Its listed targets concern speed, latency and signal strength.
This does not establish continuous hourly beam-level PRB histories of 168 hours
plus forecast targets. Reject direct use in the current PRB transfer pipeline;
do not substitute speed-test prediction accuracy for original-domain WAPE.
Only publisher metadata was inspected; no row-level audit or download is claimed.

## 5G-STARDUST: acquisition not yet established

The [official D4.1 deliverable](https://www.5g-stardust.eu/wp-content/uploads/2024/09/5G-STARDUST-D4.1-v1.0.F_bis.pdf)
describes cellular counters, including downlink PRB load, and hourly/15-minute
observations in rural and railway scenarios. It is a public document; that alone
does not establish a downloadable dataset, adequate continuous duration or a
dataset reuse license. Keep this candidate unverified, not declared unavailable
or licensed for training. No contact with authors or access bypass was attempted.

These earlier screens did not establish additional usable training data. The
subsequent CESNET intake below adds a distinct pretraining candidate. Preserve
the frozen production model and do not tune against the observed holdout.

## CESNET-TimeSeries24: intake and subsequent neural pilot

The [publisher distribution](https://zenodo.org/records/23296272) provides
institution-level hourly IP traffic under CC BY 4.0. Acquisition verified both
the record MD5 and publisher SHA256SUMS. The hourly HDF5 contains 1,880,182 rows,
283 institution IDs and a regular 6,718-hour global index with no duplicate
institution/hour keys. Institution IDs do not constitute independent networks.

Only the first 5,374 hours are eligible for pretraining. No future-prefix target
values were inspected. The eligible prefix contains 636 uninterrupted segments
of at least 193 hours, totaling 1,490,456 entity-hours. Prepared arrays preserve
timestamps and gaps; no interpolation or PRB-label substitution is performed.
See `cesnet_acquisition.json`, `cesnet_prefix_audit.json` and
`cesnet_preparation.json` for checksums, coverage and segment provenance.

This is traffic volume, not PRB utilization. Its use is representation
pretraining followed by target-domain fine-tuning, with matched target-only,
target-self-pretraining, radio-only and pooled-source controls. The completed
Kaggle v3 pilot was verified but did not establish consistent external-data gains;
see `multisource_neural_analysis.md`. Intake JSON reports remain historical
snapshots of preprocessing without training. The experiment outline is
`configs/multisource_neural_pretraining.yaml`; architecture and compute budgets
remain launch gates, not implicit permission to tune against final holdout.
