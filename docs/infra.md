# Infrastructure status

Validated locally on Windows, Python 3.12.13 using `.venv` and uv.
The default system Python is an MSYS Python without pip; use the project venv.
Dependencies are recorded in `requirements-week1.lock.txt`.

CPU preprocessing only in Week 1. Kaggle has now been verified; Colab and Google
Drive remain unverified because browser automation is unavailable.
Before Week 4: create the Colab runtime, mount the user's Drive for checkpoints,
measure one GPU epoch, and record GPU model, VRAM, epoch time and quota;
use Kaggle only if Colab is unavailable. Never run full GRU training on CPU.

GitHub CLI authentication is now valid for `trbk07`. The private repository is
https://github.com/trbk07/5g-network-intelligence.

## Colab / Kaggle bootstrap

Open `notebooks/00_runtime_setup.ipynb` in Colab (upload notebook), select a GPU
runtime, and upload the source ZIP from GitHub → Code → Download ZIP. Run the
cells in order. Drive authorization is interactive and must be granted by the
account owner. The notebook verifies checkpoint storage and records the actual
GPU name, VRAM and PyTorch version to `checkpoints/runtime.json`.

For Kaggle, create a private notebook, upload the setup notebook, attach the source
ZIP as a private input, enable Internet and a GPU accelerator, then set SOURCE_ZIP
to the attached file path. Save a version with outputs to retain checkpoints.
The Week 4 training pilot remains unverified until actually run.
No full CPU training is started by this notebook.

## Fallback when browser automation is unavailable

`notebooks/00_cloud_standalone.ipynb` embeds an allowlisted source bundle with a
SHA256 integrity check. Upload this one file to Colab or Kaggle; no source ZIP,
GitHub token, or private repository access is needed inside the cloud runtime.
Rebuild after code changes with `python -m scripts.build_cloud_notebook`.
Credentials, raw data, environment folders and Git history are excluded.

An isolated Kaggle CLI environment is available locally at `.kaggle-venv`.
The account owner completed Kaggle OAuth login for `truongbaokien`.
Never send tokens through chat or commit credential files.

## Verified Kaggle run - 2026-11-15

Private notebook: https://www.kaggle.com/code/truongbaokien/5g-network-intelligence-runtime

Version 2 completed successfully with Internet enabled and a 1,200-second run limit.
Actual devices: two Tesla T4 GPUs, each reporting 15,360 MiB via nvidia-smi.
Python 3.13.16; PyTorch 2.11.0+cu130; CUDA available. Checkpoint write/read probe
passed at `/kaggle/working/checkpoints`. Outputs were saved by the completed run
and the small verification reports downloaded locally.

All 12 pinned dataset CSVs were downloaded in the cloud; the main panel was
rebuilt as `(840, 2880, 4)` and all 16 tests passed. Evidence is recorded in
`reports/kaggle_runtime.json`. No GRU training or epoch-time measurement was done.
Version 1 failed because the image exposed a Colab environment variable;
runtime detection now prioritizes Kaggle and avoids mounting Google Drive there.

## Colab CLI fallback

Official `google-colab-cli` 0.8.0 is installed in `.colab-venv`. On this Windows
machine its help and OAuth authorization prompt work. The upstream documentation
supports Linux/macOS only, so VM provisioning on Windows is still unverified.
Docker is installed but its Linux engine was not running during inspection.

Owner login command (run locally, never paste the resulting code into chat):

```powershell
D:\5G\.colab-venv\Scripts\colab.exe --auth oauth2 sessions
```

Open the authorization URL printed by the command, complete Google consent, and
paste the code back into that same terminal. After authentication, the agent can
test session provisioning and remote execution. No Colab GPU allocation has yet
been verified, and no subscription or paid credits have been purchased.
