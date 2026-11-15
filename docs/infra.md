# Infrastructure status

Validated locally on Windows, Python 3.12.13 using `.venv` and uv.
The default system Python is an MSYS Python without pip; use the project venv.
Dependencies are recorded in `requirements-week1.lock.txt`.

CPU preprocessing only in Week 1. GPU, Colab, Google Drive and Kaggle accounts
have **not been configured or verified**. No GPU availability or quota is claimed.
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
GPU quota and the Week 4 training pilot remain unverified until actually run.
No full CPU training is started by this notebook.
