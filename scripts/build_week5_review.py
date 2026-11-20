"""Readable Week 5 research review; no training and no automatic model selection rerun."""

import hashlib
import json
from pathlib import Path


def build() -> Path:
    """Attach completed research outputs; expose key source and safe debug examples."""
    root = Path(__file__).resolve().parents[1]
    cells = []

    def add(kind: str, source: str) -> None:
        value = {
            "cell_type": kind,
            "metadata": {},
            "id": hashlib.sha256(source.encode()).hexdigest()[:12],
            "source": source.splitlines(keepends=True),
        }
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        cells.append(value)

    add(
        "markdown",
        """# Week 5 — Chất lượng mô hình và bằng chứng nghiên cứu

## Đọc notebook theo thứ tự
1. Quality gate: vì sao artifact cũ FAIL và phải retrain trên cloud.
2. Common mask, coverage và paired BS-cluster bootstrap.
3. A/B ablation, bảng candidate và error atlas.
4. Code thật của các module chính; ví dụ debug mask.

**Run All chỉ review, không training, không mở holdout, không freeze champion.**
Notebook lấy kết quả thật từ hosted validation worker đã hoàn tất.
Source dùng Python đọc được trực tiếp, có comment; không có encoded source blob.
""",
    )
    add(
        "code",
        """from pathlib import Path
import json
import os
import shutil
import sys
import numpy as np
import pandas as pd
import yaml
from IPython.display import display, Markdown, Image

# Input là output của worker đã chạy xong; chỉ đọc kết quả, không retrain.
matches=list(Path('/kaggle/input').rglob('week5_comparison_manifest.json'))
if not matches:
    raise FileNotFoundError('Attach completed Week 5 validation notebook outputs as Kaggle Input.')
REPORTS=matches[0].parent
SOURCE=REPORTS.parent
WORK=Path('/kaggle/working/week5-review')
WORK.mkdir(exist_ok=True)
for folder in ('src','scripts','configs'):
    shutil.copytree(SOURCE/folder,WORK/folder,dirs_exist_ok=True)
os.chdir(WORK)
sys.path.insert(0,str(WORK))
RUN_TRAINING=False  # review notebook không có đường thực thi training
print('Reading completed reports:',REPORTS)
""",
    )
    add(
        "markdown",
        """## 1. Quality gate và thay đổi bắt buộc

Artifact cũ dùng LightGBM44 và GRU static6, không đạt plan mới.
Đã retrain **6 LightGBM40** và **4 GRU phase4** trên cloud; không chỉ sửa inference.
Imputer/scaler là global per-KPI: A có 1 phần tử, B có 4; DLPRB statistics chung.
Beam summary chỉ dùng cho sampling/load-tier diagnostics, không đi vào model input.
""",
    )
    add(
        "code",
        """gate=json.loads((REPORTS/'model_quality_gate.json').read_text())
print('Quality gate:',gate['status'])
display(pd.DataFrame(gate['checks']))
""",
    )
    add(
        "markdown",
        """## 2. Benchmark công bằng và CI

Mỗi horizon dùng đúng cùng `(origin, beam)` và actual cho mọi candidate.
Common mask chỉ yêu cầu actual/forecasts hữu hạn, không loại observation vì history có NaN.
Bootstrap resample nguyên BS cluster; 1.000 draws, seed17, 95% percentile CI.
CI cho MAE(A)−MAE(B): âm nghiêng về A. Có 30 BS clusters, nên diễn giải thận trọng.
""",
    )
    add(
        "code",
        """comparison=pd.read_csv(REPORTS/'model_comparison_val.csv')
columns=['model','horizon','mae','n_origins','n_pairs','n_beams','coverage_pct','samples_excluded_pct']
display(comparison[comparison.load_tier=='All'][columns])
paired=pd.read_csv(REPORTS/'paired_model_comparison_val.csv')
display(paired[(paired.model_a=='lightgbm_c1') & paired.model_b.isin(['lightgbm_c2','gru_A','gru_B'])])
""",
    )
    add(
        "markdown",
        """## 3. A/B ablation và candidate-ready

Không chọn seed đẹp nhất. GRU family candidate là trung bình forecast của cả seeds17/42;
mean ± SD theo seed trong ablation chỉ mang tính mô tả.
Quy tắc: mean normalized MAE, ưu tiên family rẻ hơn khi paired CI không phân biệt.
Baseline được phép làm champion candidate. Đây chưa phải quyết định freeze cuối.
""",
    )
    add(
        "code",
        """display(pd.read_csv(REPORTS/'ablation_study.csv'))
display(pd.read_csv(REPORTS/'champion_candidate_table.csv'))
candidate=yaml.safe_load((SOURCE/'configs/champion_candidate.yaml').read_text())
print('Candidate:',candidate['family'],'| frozen:',candidate['frozen'],'| holdout:',candidate['holdout_allowed'])
""",
    )
    add(
        "markdown",
        """## 4. Error atlas theo quy tắc, không cherry-pick

Top 5 beam theo mean horizon MAE; 5 đại diện mỗi tier bằng các rank cách đều trên fit mean.
Ba case: 2 peak underprediction lớn nhất và 1 zero-to-active miss lớn nhất tại h=24.
Cùng target hours/window cho LightGBM, GRU và best baseline mỗi horizon.
Đây là lỗi forecast retrospective, không phải nhãn congestion/incident.
""",
    )
    add(
        "code",
        """display(pd.read_csv(REPORTS/'error_atlas_beams.csv'))
display(pd.read_csv(REPORTS/'error_atlas_cases.csv'))
for name in ('actual_vs_forecast.png','mae_by_horizon.png','mae_by_load_tier.png'):
    display(Image(filename=str(REPORTS/'figures'/name)))
""",
    )
    add(
        "markdown",
        "## 5. Code để review và debug\n\nMỗi cell bên dưới ghi đúng module được hiển thị; không chạy training.\n",
    )
    for name, description in (
        (
            "src/evaluation/common_mask.py",
            "Khóa common finite pairs và phân biệt raw missingness với comparison exclusions.",
        ),
        ("src/evaluation/bootstrap.py", "Resample nguyên cluster và giữ cùng draws qua horizons."),
        (
            "src/inference/registry.py",
            "Chọn family đủ 3 horizons; cheaper fallback phải qua paired CI.",
        ),
        (
            "src/evaluation/diagnostics.py",
            "Quy tắc chọn beam/event và vẽ chart cùng target indices.",
        ),
        (
            "scripts/quality_audit.py",
            "Chỉ đọc artifact metadata/weights để xác minh schema, seed và preprocessing.",
        ),
    ):
        add("markdown", f"### `{name}`\n\n{description}\n")
        add("code", f"%%writefile {name}\n" + (root / name).read_text(encoding="utf-8"))
    add(
        "markdown",
        """### Ví dụ debug common mask (không training)

Actual vẫn hữu hạn nhưng một candidate thiếu forecast: cặp đó bị loại khỏi tất cả,
không được báo nhầm là actual bị missing. History NaN không nằm trong điều kiện mask này.
""",
    )
    add(
        "code",
        """from src.evaluation.common_mask import comparable_mask, coverage_report
actual=np.array([[0.,1.,2.],[3.,4.,5.]])
forecast_a=actual.copy()
forecast_b=actual+0.5
forecast_b[0,1]=np.nan
mask=comparable_mask(actual,{'a':forecast_a,'b':forecast_b})
print('Common mask:')
print(mask)
print(coverage_report(actual,forecast_a,mask))
""",
    )
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = root / "notebooks/05_model_quality.ipynb"
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=True), encoding="utf-8")
    output = root / ".cache/kaggle-week5-review"
    output.mkdir(parents=True, exist_ok=True)
    shutil_file = output / "review.ipynb"
    shutil_file.write_bytes(path.read_bytes())
    meta = {
        "id": "truongbaokien/5g-network-intelligence-week5-review",
        "title": "5G Network Intelligence Week5 Review",
        "code_file": "review.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": ["truongbaokien/5g-network-intelligence-week5-validation"],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return output


if __name__ == "__main__":
    print(build())
