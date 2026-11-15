"""Inspect the actual notebook runtime without starting model training."""
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def inspect_runtime() -> dict:
    """Return CPU/GPU availability and environment versions observed at execution."""
    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": False,
        "gpu_devices": [],
    }
    try:
        import torch
    except ImportError:
        result["torch"] = "not installed"
    else:
        result["torch"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()
        result["gpu_devices"] = [
            {"name": torch.cuda.get_device_name(i),
             "vram_bytes": torch.cuda.get_device_properties(i).total_memory}
            for i in range(torch.cuda.device_count())
        ]
    if shutil.which("nvidia-smi"):
        process = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
            capture_output=True, text=True, check=False, timeout=20,
        )
        result["nvidia_smi"] = process.stdout.strip() or process.stderr.strip()
    return result


if __name__ == "__main__":
    report = inspect_runtime()
    Path("reports").mkdir(exist_ok=True)
    Path("reports/runtime.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
