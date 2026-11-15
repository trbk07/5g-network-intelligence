"""Owner policy: all model fitting must happen in hosted Colab or Kaggle."""
import os
from pathlib import Path


def is_cloud_runtime() -> bool:
    """Recognize hosted notebook runtime markers; ordinary local runs are rejected."""
    if os.name == "nt":
        return False
    return (Path("/kaggle/working").is_dir()
            or ("COLAB_RELEASE_TAG" in os.environ and Path("/content").is_dir()))


def require_cloud_training() -> None:
    """Refuse model training outside the owner's allowed hosted notebook runtimes."""
    if not is_cloud_runtime():
        raise RuntimeError("Local model training is forbidden by the owner. Use Colab or Kaggle.")
