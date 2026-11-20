"""Build a readable, commented Kaggle notebook with real source cells, never encoded blobs."""

import hashlib
import json
import pprint
import subprocess
from pathlib import Path

import pandas as pd


def build() -> Path:
    """Create review-first notebook; all training is explicitly disabled by default."""
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    cells = []

    def add(kind: str, source: str, output: str | None = None) -> None:
        cell = {
            "cell_type": kind,
            "metadata": {},
            "id": hashlib.sha256(source.encode()).hexdigest()[:12],
            "source": source.splitlines(keepends=True),
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
            if output:
                cell["outputs"] = [
                    {"output_type": "display_data", "metadata": {}, "data": {"text/html": output}}
                ]
        cells.append(cell)

    add(
        "markdown",
        """# 5G Network Intelligence — Week 4: GRU-A và GRU-B

## Cách đọc notebook

Notebook này dùng **code Python thật, có comment**, chia theo từng module.
Không có khối source base64. Mỗi cell `%%writefile` ghi đúng file được hiển thị
để bạn có thể sửa trực tiếp, chạy lại cell và debug module đó.

**Mặc định Run All chỉ review, không train lại.** Toàn bộ training chỉ được phép
trên hosted Colab/Kaggle. Kết quả hiện tại theo schema phase4 được retrain ở
[Week 5 GPU job](https://www.kaggle.com/code/truongbaokien/5g-network-intelligence-week5),
Version 1 → Output. Artifact static6 của Week 4 chỉ còn là evidence lịch sử.

### Mục lục
1. Kết quả đã đo và cấu hình chạy
2. Dữ liệu, schema và temporal split
3. Sampling, scaler/imputer và missingness masks
4. Kiến trúc GRU và vòng lặp training/resume
5. CLI orchestration và validation replay
6. Chạy từng bước, debug một beam và xem learning curves

**Luồng:** CSV → panel → training-only preprocessing → sequence 168h →
GRU + phase4 → forecast +1/+6/+24h → validation metrics.
""",
    )
    comparison = pd.read_csv(root / "reports/gru_comparison_val.csv")
    compact = comparison[
        [
            "variant",
            "horizon",
            "mae_mean",
            "mae_sd",
            "skill_vs_baseline",
            "lightgbm_mae",
            "skill_vs_lightgbm",
        ]
    ]
    add(
        "markdown",
        """## 1. Kết quả validation đã đo

MAE dùng đơn vị **DLPRB usage**. `skill > 0` nghĩa là cải thiện so với mô hình tham chiếu.
Mean ± sample SD của **2 seeds** chỉ mô tả biến động, không chứng minh statistical stability.
GRU tốt hơn best baseline nhưng **chưa vượt LightGBM C1**. Holdout chưa được đánh giá.
""",
    )
    add(
        "code",
        "# Bảng số đo thật từ reports/gru_comparison_val.csv; không chạy training ở cell này.\n"
        "import pandas as pd\nfrom IPython.display import display\n"
        "MEASURED_RESULTS = "
        + pprint.pformat(compact.to_dict(orient="records"), width=88, sort_dicts=False)
        + "\n"
        "overview = pd.DataFrame(MEASURED_RESULTS)\n"
        "display(overview.round(6))\n",
        compact.to_html(index=False, float_format=lambda v: f"{v:.6f}"),
    )
    add(
        "markdown",
        """### Cấu hình và các cờ thực thi

Để review/debug source: giữ các cờ bên dưới là `False`. Khi muốn chạy lại, bật từng
bước và chọn GPU trong Settings. **Không bật RUN_TRAINING khi chỉ muốn đọc notebook.**
Chọn một variant/seed để debug trước khi chạy bốn experiments. Artifact có sẵn cần
`RESUME=True`; chương trình không ghi đè một run đã có checkpoint.
""",
    )
    add(
        "code",
        f"BUILD_SOURCE_COMMIT = {revision!r}\n"
        + """from pathlib import Path
import os
import sys
import subprocess

# Đây là runtime cloud, không phải máy cá nhân. Không có training ở cell khởi tạo.
IS_KAGGLE = Path('/kaggle/working').is_dir()
IS_COLAB = not IS_KAGGLE and 'COLAB_RELEASE_TAG' in os.environ
if not (IS_KAGGLE or IS_COLAB):
    raise RuntimeError('Chỉ chạy notebook này trong hosted Kaggle hoặc Colab.')
PROJECT = Path('/kaggle/working/5g-review' if IS_KAGGLE else '/content/5g-review')
PROJECT.mkdir(exist_ok=True)
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
os.environ['SOURCE_GIT_COMMIT'] = BUILD_SOURCE_COMMIT
for folder in ('src/utils', 'src/data', 'src/evaluation', 'src/features', 'src/models', 'scripts', 'configs', 'reports'):
    path = PROJECT / folder
    path.mkdir(parents=True, exist_ok=True)
    if not folder.startswith(('configs', 'reports')):
        (path / '__init__.py').touch()
(PROJECT / 'src/__init__.py').touch()

RUN_INSTALL = False
RUN_PREPARE_DATA = False
RUN_TRAINING = False
RUN_ALL_EXPERIMENTS = False
RUN_EVALUATION = False
DEBUG_DATA = False
VARIANT = 'B'  # A: DLPRB + mask; B: 4 KPI + 4 masks
SEED = 17
RESUME = False
DEBUG_ORIGIN = 503  # first fallback validation origin
DEBUG_BEAM = 0
print('Project:', PROJECT)
""",
    )
    groups = [
        (
            "2. Dữ liệu và temporal split",
            "Time là relative hourly index; split fallback: fit 0–503, validation 504–671.\nKhông dùng holdout 672–839 để tune. Các file dưới đây kiểm tra schema, checksum và boundary.",
            [
                "src/utils/config.py",
                "src/utils/seed.py",
                "src/utils/metadata.py",
                "src/utils/runtime.py",
                "src/data/schema.py",
                "src/data/io.py",
                "src/data/panel.py",
                "src/data/audit.py",
                "src/evaluation/origins.py",
                "src/data/splits.py",
                "src/evaluation/validation.py",
            ],
        ),
        (
            "3. Sampling và GRU inputs",
            "`compute_static`: chỉ fit prefix training. `sample_beams`: cùng 600 beams cho A/B và seeds.\n`GRUPreprocessor`: median imputer + scaler training-only; mask 1 là missing, zero vẫn là observation.\n`GRUBatches`: lấy đúng t−167..t, target t+[1,6,24]; không materialize toàn bộ sliding windows.",
            [
                "src/features/static_stats.py",
                "src/features/beam_sampling.py",
                "src/features/gru_inputs.py",
            ],
        ),
        (
            "4. GRU architecture và training loop",
            "Input A có 2 channels, B có 8. GRU hidden=64, layers=2, dropout=0.1.\nPhase4 = sin/cos phase 24h/168h; no beam statistics enter model inputs. Head trả 3 targets.\nLoss chỉ lấy finite labels; AdamW + gradient clipping; validation early stopping.\n`last.pt` phục vụ resume (optimizer/RNG/patience); `best.pt` phục vụ benchmark.",
            ["src/models/gru.py", "src/models/gru_train.py", "src/models/gru_forecaster.py"],
        ),
        (
            "5. Orchestration và validation metrics",
            "CLI dưới đây nối các module nhỏ với config. `train_gru` từ chối local training.\nValidation replay dùng toàn bộ beam và cùng origin sets với baselines/LightGBM.\nWAPE zero denominator trả NaN có cờ; zero FP là prediction > 0 khi actual == 0, không phải incident FPR.",
            [
                "src/evaluation/metrics.py",
                "src/evaluation/evaluator.py",
                "scripts/prepare_data.py",
                "scripts/train_gru.py",
                "scripts/evaluate_gru.py",
            ],
        ),
    ]
    for title, explanation, paths in groups:
        add("markdown", f"## {title}\n\n{explanation}\n")
        for name in paths:
            add(
                "markdown",
                f"### `{name}`\n\nSửa cell này rồi chạy lại trước các cell sử dụng module. "
                "Nếu đã import module trong phiên, restart kernel hoặc reload module để dùng code mới.\n",
            )
            add("code", f"%%writefile {name}\n" + (root / name).read_text(encoding="utf-8"))
    add(
        "markdown",
        "## 6. Cấu hình có thể review trước khi chạy\n\nChỉ hai variants A/B và seeds 17/42; không có grid search ẩn.\n",
    )
    for name in (
        "pyproject.toml",
        "configs/data.yaml",
        "configs/evaluation.yaml",
        "configs/gru.yaml",
        "configs/lightgbm.yaml",
    ):
        add("code", f"%%writefile {name}\n" + (root / name).read_text(encoding="utf-8"))
    # Comparison inputs stay short and transparent CSV cells, rather than encoded payloads.
    add(
        "markdown",
        "### Mốc baseline và LightGBM đã đo\n\nCác CSV rút gọn này chỉ dùng so sánh metric khi replay GRU; không được dùng làm training targets.\n",
    )
    for name, columns in (
        ("baseline_benchmark_val.csv", ["model", "horizon", "load_tier", "mae"]),
        ("lightgbm_skill_val.csv", ["model", "horizon", "mae"]),
    ):
        frame = pd.read_csv(root / "reports" / name)
        if "load_tier" in columns:
            frame = frame[frame.load_tier == "All"]
        add("code", f"%%writefile reports/{name}\n" + frame[columns].to_csv(index=False))
    add(
        "markdown",
        "### Bước 1 — Dependencies và dữ liệu\n\nBật riêng từng cờ khi cần. Tải dữ liệu public ở revision đã pin; bước này không train.\n",
    )
    add(
        "code",
        """if RUN_INSTALL:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.[dev,ml,neural]'], check=True)
if RUN_PREPARE_DATA:
    subprocess.run([sys.executable, '-m', 'scripts.prepare_data', '--step', 'all'], check=True)
else:
    print('Review mode: bỏ qua download/prepare data.')
""",
    )
    add(
        "markdown",
        "### Bước 2 — Debug một beam/origin trước khi train\n\nKiểm tra shape, missing masks và target alignment. Preprocessor chỉ dùng fit segment.\n",
    )
    add(
        "code",
        """if DEBUG_DATA:
    import numpy as np
    from src.utils.config import load_config
    from src.evaluation.validation import validation_inputs
    from src.features.gru_inputs import GRUPreprocessor, GRUBatches
    panel, split = validation_inputs(load_config('configs/data.yaml'))
    prep = GRUPreprocessor.fit(panel, split['fit'][1], VARIANT)
    batch = GRUBatches(panel, prep, VARIANT, np.array([DEBUG_ORIGIN]), np.array([DEBUG_BEAM]))
    x, phase4, target3 = next(batch.batches(1))
    print('sequence:', tuple(x.shape), 'static:', tuple(phase4.shape), 'target:', tuple(target3.shape))
    print('target indices:', DEBUG_ORIGIN + np.array([1, 6, 24]))
    print('targets scaled:', target3)
else:
    print('Bật RUN_PREPARE_DATA và DEBUG_DATA để inspect dữ liệu.')
""",
    )
    add(
        "markdown",
        "### Bước 3 — Training có checkpoint\n\nChỉ chạy nếu chủ động bật RUN_TRAINING và runtime có CUDA.\nGiữ RUN_ALL_EXPERIMENTS=False để debug một run; bật True để chạy A/B × seeds 17/42.\nĐặt artifact_dir in configs/gru.yaml trên Drive hoặc dùng Kaggle Output để checkpoint bền vững.\n",
    )
    add(
        "code",
        """if RUN_TRAINING:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Chọn GPU runtime trước khi training.')
    # Danh sách experiments hiển thị rõ; không có vòng hyperparameter search ẩn.
    experiments = [('A', 17), ('A', 42), ('B', 17), ('B', 42)] if RUN_ALL_EXPERIMENTS else [(VARIANT, SEED)]
    for variant, seed in experiments:
        command = [sys.executable, '-m', 'scripts.train_gru', '--variant', variant,
                   '--seed', str(seed), '--seconds-limit', '360']
        if RESUME:
            command.append('--resume')
        print('Running:', ' '.join(command))
        subprocess.run(command, check=True)
else:
    print('Review mode: không train lại. Các kết quả đầu notebook là runs đã hoàn tất.')
""",
    )
    add(
        "markdown",
        "### Bước 4 — Learning curves và replay\n\nĐọc loss theo epoch để debug divergence/early stopping. Replay cần đủ artifacts đã khai báo trong configs/gru.yaml.\n",
    )
    add(
        "code",
        """from src.utils.config import load_config
history_path = Path(load_config('configs/gru.yaml')['artifact_dir']) / VARIANT / f'seed{SEED}' / 'history.jsonl'
if history_path.exists():
    history = pd.read_json(history_path, lines=True)
    display(history)
    history.plot(x='epoch', y=['train_l1_scaled', 'val_l1_scaled'], title=f'GRU-{VARIANT}, seed {SEED}')
else:
    print('Checkpoint phase4: mở Week 5 GPU job, Version 1 → Output; review không train lại.')

if RUN_EVALUATION:
    from scripts.evaluate_gru import evaluate_gru
    evaluate_gru()
    display(pd.read_csv('reports/gru_comparison_val.csv'))
""",
    )
    output = root / "notebooks/04_gru_review.ipynb"
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output.write_text(json.dumps(notebook, indent=2, ensure_ascii=True), encoding="utf-8")
    assert "base64.b64decode(" not in output.read_text(encoding="utf-8")
    print(f"Built readable notebook with {len(cells)} cells: {output}")
    return output


if __name__ == "__main__":
    build()
