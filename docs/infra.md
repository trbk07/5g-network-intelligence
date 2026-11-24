# Infrastructure status

Validated locally on Windows, Python 3.12.13 using `.venv` and uv.
The default system Python is an MSYS Python without pip; use the project venv.
Dependencies are recorded in `requirements-week1.lock.txt`.

CPU preprocessing only in Week 1. Kaggle and Colab CLI execution have now been verified. Google Drive mounting and checkpoint read/write have also passed.
Before Week 4: measure one GPU training epoch and record GPU model, VRAM,
epoch time and quota; runtime provisioning and Drive checkpoint storage are verified.
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

`notebooks/00_cloud_standalone.ipynb` now contains readable module source cells
and Vietnamese explanations, replacing the earlier encoded-source bootstrap at
the owner's request. It needs no GitHub token or source ZIP. Rebuild after code
changes with `python -m scripts.build_cloud_notebook`. All training flags default
to false; explicit cloud training mode is described in docs/gru.md.

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
supports Linux/macOS only; despite that limitation, CPU and T4 provisioning,
execution, file download and Drive mounting have passed on this Windows machine.
Docker is installed but its Linux engine was not running during inspection.

Owner login command (run locally, never paste the resulting code into chat):

```powershell
D:\5G\.colab-venv\Scripts\colab.exe --auth oauth2 sessions
```

Open the authorization URL printed by the command, complete Google consent, and
paste the code back into that same terminal. After authentication, the agent can
test session provisioning and remote execution. Colab CPU execution and one Tesla T4 GPU were subsequently verified after owner
OAuth login. Both test runtimes were terminated. No paid plan was purchased.


## Verified Colab connection - 2026-11-15

Official CLI OAuth authenticated successfully. A CPU runtime executed Python,
then a T4 runtime reported CUDA available and PyTorch 2.14.1+cu130. Hardware
evidence is in reports/colab_runtime.json. Downloading files also succeeded.
After owner consent, the allowlisted source bundle was uploaded and the full
Week 1 pipeline completed on Colab T4. All 17 tests and Ruff passed; the main panel
was (840, 2880, 4). Local runtime checkpoint write/read succeeded, reports were
downloaded and the GPU runtime was terminated. Drive mounting was verified in a subsequent CPU session (see below). Local runtime
disk is ephemeral and must not be treated as persistent Drive storage.


## Verified Google Drive checkpoint storage - 2026-11-15

After owner consent through Google, the official CLI mounted Drive at
`/content/drive`. Created the project checkpoint directory at
`/content/drive/MyDrive/5g-network-intelligence/checkpoints`, wrote a timestamped
setup-check JSON file and verified its contents by reading it back. No unrelated
Drive files were inspected or modified. The report was downloaded to
`reports/drive_verification.json`; the CPU runtime was terminated afterwards.
Future sessions may require fresh Google authorization before mounting Drive.

## TCN Colab fallback when Kaggle is queued

The same `configs/temporal_tcn.yaml` and frozen development protocol can run on
hosted Colab T4. Packaging is local; all fitting remains remote. The launcher
requires mounted Drive, verifies the complete source/data transport and refuses
an existing execution-state file. Checkpoints and reports are symlinked into a
dedicated project run directory on Drive, with a result archive saved at completion.

1. Run `python -m scripts.package_colab_tcn` from the repository. Record the
   printed bundle SHA256 and source commit; the archive is in `.cache/colab-tcn`.
2. Provision T4 with `.colab-venv/Scripts/colab.exe new --session tcn-colab --gpu T4`.
3. Upload the archive to `/content/tcn_source.zip` using the CLI `upload` command.
4. Run `colab drivemount --session tcn-colab /content/drive` in an interactive
   terminal. Complete Google's consent and only then press Enter in that terminal.
   Noninteractive stdin can return prematurely and cause credential propagation
   to fail. A replacement runtime may require a new Google confirmation.
5. Execute `scripts/run_colab_tcn.py` remotely with CLI `exec --session tcn-colab
   --file scripts/run_colab_tcn.py --timeout 0`, supplying `--env
   TCN_BUNDLE_SHA256=<recorded-sha>` and `--env
   TCN_RUN_NAME=temporal_tcn_colab_<unique-run-name>`.

The launcher checks hosted Colab/CUDA before preparation and refuses local
execution. It does not train when imported or tested. Runtime/source/storage
evidence and a PID/state record are written before training begins. Inspect that
state and the actual cloud session after a connection loss; never resubmit solely
because the local observation timed out. A queued Kaggle run is not proof of a
Colab fit; report each platform's actual state separately. Do not claim the queued
Kaggle version was cancelled unless its cancellation has been verified.
