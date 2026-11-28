# Additional target-data screen — 2026-11-28

This screen inspected publisher metadata only. No new observations were
downloaded, no authentication conditions bypassed, and no model fitted.

The inspected [NETOP dataset catalog](https://huggingface.co/netop/datasets) and
[GOTS-F card](https://huggingface.co/datasets/netop/gotsf-ds) did not identify an
additional beam-level target extension beyond the existing five training weeks
and two target weeks. This is a bounded search, not proof that no suitable data
exists. Already observed target weeks cannot be relabeled as a fresh holdout.

The [5G Network Energy Consumption card](https://huggingface.co/datasets/netop/5G-Network-Energy-Consumption)
declares MIT, hourly cell load/energy measurements and eight days of coverage.
File access requires agreeing to share Hugging Face profile information.
No acceptance or file download was attempted: eight days do not provide the
193-hour minimum continuous sequence required by the current 168-hour-context,
24-hour-horizon source task, even before reserving an external suffix.
Its cell-level energy/load observations must not be called extra beam-level
DLPRB labels or an independent GOTS-F test.

The current Telecom study therefore remains the active data-expansion test.
Do not shorten evaluation windows, interpolate missing target days, concatenate
different entities in time, or change units merely to admit a larger file count.
Changing a source pretraining task would require a separate declared protocol;
this screen does not change any frozen target split or acceptance gate.
