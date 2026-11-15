# Infrastructure status

Validated locally on Windows, Python 3.12.13 using `.venv` and uv.
The default system Python is an MSYS Python without pip; use the project venv.
Dependencies are recorded in `requirements-week1.lock.txt`.

CPU preprocessing only in Week 1. GPU, Colab, Google Drive and Kaggle accounts
have **not been configured or verified**. No GPU availability or quota is claimed.
Before Week 4: create the Colab runtime, mount the user's Drive for checkpoints,
measure one GPU epoch, and record GPU model, VRAM, epoch time and quota;
use Kaggle only if Colab is unavailable. Never run full GRU training on CPU.

Local Git initialization and commit are possible. GitHub CLI currently reports
an invalid token for its configured account. Remote repository creation/push
requires the user to restore authentication using `gh auth login`.
