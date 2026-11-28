# Reproducibility correction for the cell v1 research studies

The new cell-attention controller called the project's lightweight `set_seed`,
which seeds Python and NumPy only. Unlike older neural controllers, it omitted
explicit `torch.manual_seed` and `torch.cuda.manual_seed_all` before model
initialization. Source-pretraining initialization had the same omission.

Affected experiments:

- `5g-ni-cell-spatial-temporal-attention`, v1, source `a575f3f`.
- `5g-ni-multi-corpus-spatial-encoder-pretraining`, v1, source `61ba235`.

The point forecasts, raw MAE/WAPE and artifact checksums are real. Strict
inference replay resolves the separately identified GPU fastpath drift. However,
the two v1 studies are not controlled shared-initialization ablations and must
not be described as verified neural seed-17 runs. The cold-initialization audit
detected this issue; no source/architecture superiority or production promotion
is accepted from these studies. Historical notebook claims of identical starts
are superseded by this correction, without editing their execution snapshots.

Earlier release artifacts and older controllers with explicit Torch seeding are
outside this correction. The original holdout remains excluded.

Seeded v2 studies use separate configs, artifact roots, private workers and
notebooks 37/38. They explicitly seed CPU Torch and every CUDA generator, save
source and target cold states before optimization, hash post-transfer states,
and require exact seed/transfer reconstruction. GPU attention fastpath is
disabled consistently for stopping and scoring. Budgets and data membership are
frozen before rerun; negative v1 results and all original snapshots are retained.
