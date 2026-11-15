"""Write UTF-8-safe runtime/test review notebooks without shell text transcoding."""
import hashlib
import json
from pathlib import Path


def build() -> None:
    """Generate readable notebook pages and stage private Kaggle metadata."""
    root = Path(__file__).resolve().parents[1]

    def cell(kind: str, source: str) -> dict:
        result = {"cell_type": kind, "metadata": {},
                  "id": hashlib.sha256(source.encode("utf-8")).hexdigest()[:12],
                  "source": source.splitlines(keepends=True)}
        if kind == "code":
            result.update(execution_count=None, outputs=[])
        return result

    def write(slug: str, title: str, filename: str, cells: list[dict]) -> None:
        notebook = {"cells": cells, "metadata": {"kernelspec": {
            "display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 5}
        # JSON Unicode escapes keep transport ASCII-safe; JSON readers restore Vietnamese exactly.
        text = json.dumps(notebook, indent=2, ensure_ascii=True)
        (root / "notebooks" / filename).write_text(text, encoding="utf-8")
        folder = root / ".cache" / slug
        folder.mkdir(parents=True, exist_ok=True)
        (folder / filename).write_text(text, encoding="utf-8")
        metadata = {"id": f"truongbaokien/{slug}", "title": title, "code_file": filename,
                    "language": "python", "kernel_type": "notebook", "is_private": True,
                    "enable_gpu": False, "enable_internet": False, "dataset_sources": [],
                    "competition_sources": [], "kernel_sources": [], "model_sources": []}
        (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    runtime = json.loads((root / "reports/kaggle_runtime.json").read_text(encoding="utf-8"))
    write("5g-network-intelligence-runtime", "5G Network Intelligence Runtime", "00_kaggle_runtime.ipynb", [
        cell("markdown", """# 5G Network Intelligence — kiểm tra runtime

Notebook này chỉ giải thích và kiểm tra hạ tầng, không train model.
Số đo bên dưới thuộc lần kiểm tra GPU đã hoàn tất. Bản review chạy CPU để không tốn quota.
Code GRU chi tiết có tại [notebook review](https://www.kaggle.com/code/truongbaokien/5g-network-intelligence-gru).
"""),
        cell("code", "# Số đo thật của lần xác minh GPU trước đó; không tạo một GPU job mới.\n"
             "import pandas as pd\nfrom IPython.display import display\n"
             f"verified_devices = {runtime['runtime']['gpu_devices']!r}\n"
             "display(pd.DataFrame(verified_devices))\n"
             "print('Pipeline Week 1: passed; tests: 16 passed; checkpoint write/read: passed')\n"),
        cell("markdown", """## Kiểm tra live khi cần

Chọn GPU trong Settings rồi bật `RUN_LIVE_CHECK` để xem GPU của phiên hiện tại.
Cell này chỉ đọc thông tin phần cứng, không training.
"""),
        cell("code", """RUN_LIVE_CHECK = False
if RUN_LIVE_CHECK:
    import torch
    print('PyTorch:', torch.__version__)
    print('CUDA:', torch.cuda.is_available())
    for index in range(torch.cuda.device_count()):
        print(index, torch.cuda.get_device_name(index))
else:
    print('Review mode: bỏ qua live GPU check.')
"""),
    ])
    write("5g-network-intelligence-tests", "5G Network Intelligence Tests", "00_cloud_tests.ipynb", [
        cell("markdown", """# 5G Network Intelligence — cloud tests

Bộ test được chạy trên hosted Kaggle; không chạy fitting tests trên máy cá nhân.
**56 tests passed** ở run trước, gồm synthetic overfit, resume, feature canaries và validation runner.
Bản review này không chạy lại test/training. Source tests có trong GitHub `tests/`;
báo cáo lưu ở `reports/cloud_test_results_week4.json`.
"""),
        cell("code", """# Kết quả đã đo, không phải một lần chạy test mới.
import pandas as pd
from IPython.display import display
display(pd.DataFrame([{
    'environment': 'Kaggle hosted CPU',
    'tests_passed': 56,
    'skipped': 0,
    'ruff': 'passed',
}]))
"""),
        cell("markdown", """## Chạy lại có chủ đích trong notebook cloud

Upload source ZIP của repo và giải nén trước. Đặt `PROJECT` tới thư mục có `pyproject.toml`.
`RUN_TESTS` mặc định False. Fitting tests chỉ được phép chạy trên hosted Colab/Kaggle.
"""),
        cell("code", """from pathlib import Path
import subprocess
import sys

RUN_TESTS = False
PROJECT = Path('/kaggle/working/5g-network-intelligence')
if RUN_TESTS:
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('Chỉ chạy fitting tests trên hosted cloud.')
    if not (PROJECT / 'pyproject.toml').exists():
        raise FileNotFoundError('Upload source và đặt PROJECT đúng thư mục.')
    subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=PROJECT, check=True)
else:
    print('Review mode: không chạy lại tests hoặc training.')
"""),
    ])
    print("Generated Unicode-safe runtime and cloud-test notebooks")


if __name__ == "__main__":
    build()
