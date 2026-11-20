# Verified historical demo assets

Run `python -m scripts.fetch_demo_assets` from the project root after the frozen
Week 6 artifacts and canonical panel are available. It creates 24 real beams
across BS 0/5/15/29, three cells and two beams per cell, plus the original global
LightGBM models and historical thresholds. It does not train or select a model.

For a clean authorized clone, install/authenticate Kaggle CLI and run
`python -m scripts.fetch_demo_assets --download`. This downloads the existing
private Week 5/Week 6 execution outputs and verifies the pinned final archive
and panel SHA256 before preparing assets. Access to those private notebooks is
required; no public GitHub release asset has been published. Offline use can
copy the frozen artifacts and canonical panel from an authorized machine.

The generated manifest includes all asset checksums, beam ordering, dataset
revision, fit cutoff, frozen configuration identity and benchmark provenance.
The API refuses unavailable/corrupt assets with readiness 503. Restart after
asset setup. Assets are mounted read-only and excluded from Git and Docker's
build context. Never insert credentials into images or manifests.

This is actual NETOP GOTS-F data under MIT, not a synthetic success example.
Cite the authors as specified by `dataset_card.md`. Relative hourly indices
are used; no wall-clock timestamps or physical DLPRB units are invented.
