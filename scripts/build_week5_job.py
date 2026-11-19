"""Package the mandatory Week 5 cloud rerun as readable, commented notebook cells."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def build(owner: str) -> Path:
    """Write source cells and explicit cloud-only execution; no local fitting occurs."""
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    cells = []

    def cell(kind: str, source: str) -> None:
        value = {
            "cell_type": kind,
            "metadata": {},
            "id": hashlib.sha256(source.encode()).hexdigest()[:12],
            "source": source.splitlines(keepends=True),
        }
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        cells.append(value)

    cell(
        "markdown",
        """# Week 5 — Causal schema correction and cloud rerun

The updated plan rejects legacy LightGBM44 / GRU static6 artifacts. This notebook
re-trains the declared models with **LightGBM40 / GRU phase4**. No holdout is opened.

Source is readable Python, one file per cell. Sections: configuration, source,
cloud tests, data, six LightGBM runs, four GRU runs, quality gate and archive.
Training is enabled deliberately in this execution notebook; review notebooks
remain separate and default to no training. Checkpoints are kept in Kaggle Output.
""",
    )
    cell(
        "code",
        f"SOURCE_COMMIT = {revision!r}\n"
        + """from pathlib import Path
import os
import sys
import subprocess

if not Path('/kaggle/working').is_dir():
    raise RuntimeError('Run this training notebook on hosted Kaggle GPU only.')
PROJECT = Path('/kaggle/working/week5')
PROJECT.mkdir(exist_ok=True)
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
os.environ['SOURCE_GIT_COMMIT'] = SOURCE_COMMIT
print('Cloud source commit:', SOURCE_COMMIT)
""",
    )
    paths = [root / "pyproject.toml"]
    for folder in ("src", "scripts", "configs", "tests"):
        paths.extend(
            p
            for p in (root / folder).rglob("*")
            if p.is_file()
            and p.suffix in (".py", ".yaml")
            and "__pycache__" not in p.parts
            and not p.name.startswith("build_")
            and p.name != "package_gru_job.py"
        )
    parents = sorted({str(p.relative_to(root).parent).replace("\\", "/") for p in paths})
    cell(
        "code",
        "# Prepare package directories before writing the visible source files.\n"
        f"for folder in {parents!r}:\n    Path(folder).mkdir(parents=True, exist_ok=True)\n"
        "Path('reports').mkdir(exist_ok=True)\n",
    )
    for folder in ("src", "scripts", "configs", "tests"):
        cell(
            "markdown",
            f"## Source: `{folder}/`\n\nEach cell writes the exact code shown. No encoded source payloads.\n",
        )
        for path in sorted(p for p in paths if p.relative_to(root).parts[0] == folder):
            name = path.relative_to(root).as_posix()
            cell("markdown", f"### `{name}`\n")
            cell("code", f"%%writefile {name}\n" + path.read_text(encoding="utf-8"))
    cell(
        "code",
        "%%writefile pyproject.toml\n" + (root / "pyproject.toml").read_text(encoding="utf-8"),
    )
    cell(
        "markdown",
        "## 1. Cloud dependencies and fitting tests\n\nAll fitting tests execute here on hosted cloud, never on the owner's PC.\n",
    )
    cell(
        "code",
        """subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.[dev,ml,neural]'], check=True)
result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '--ignore=tests/unit/test_notebook_encoding.py'], capture_output=True, text=True, timeout=300)
Path('reports/week5_cloud_tests.log').write_text(result.stdout + result.stderr)
print(result.stdout)
result.check_returncode()
""",
    )
    cell(
        "markdown",
        "## 2. Pinned data and validation baselines\n\nData preparation records the split. Every subsequent model input is truncated before holdout.\n",
    )
    cell(
        "code",
        """subprocess.run([sys.executable, '-m', 'scripts.prepare_data', '--step', 'all'], check=True, timeout=600)
subprocess.run([sys.executable, '-m', 'scripts.run_baselines', '--segment', 'val'], check=True, timeout=180)
""",
    )
    cell(
        "markdown",
        "## 3. Retrain six LightGBM40 models on cloud CPU\n\nC1/C2 × horizons 1/6/24; unchanged planned hyperparameters. No beam summaries in the model inputs.\n",
    )
    cell(
        "code",
        """subprocess.run([sys.executable, '-m', 'scripts.train_lgbm'], check=True, timeout=900)
subprocess.run([sys.executable, '-m', 'scripts.evaluate_models', '--model', 'lightgbm', '--segment', 'val'], check=True, timeout=300)
""",
    )
    cell(
        "markdown",
        "## 4. Retrain GRU phase4 on GPU\n\nSame beam sample, seeds and architecture budget. A fits 1 global KPI parameter; B fits 4.\n",
    )
    cell(
        "code",
        """import time
began = time.monotonic()
for variant in ('A', 'B'):
    for seed in (17, 42):
        command = [sys.executable, '-m', 'scripts.train_gru', '--variant', variant, '--seed', str(seed), '--seconds-limit', '600']
        print('Training on Kaggle GPU:', ' '.join(command), flush=True)
        subprocess.run(command, check=True, timeout=720)
subprocess.run([sys.executable, '-m', 'scripts.evaluate_models', '--model', 'gru', '--segment', 'val'], check=True, timeout=300)
print('GRU training/replay seconds:', time.monotonic()-began)
""",
    )
    cell(
        "markdown",
        "## 5. Quality gate and durable result archive\n\nDo not select a candidate if this gate fails. P1 is skipped because retraining was required.\n",
    )
    cell(
        "code",
        """import json
import tarfile
try:
    subprocess.run([sys.executable, '-m', 'scripts.quality_audit'], check=True, timeout=180)
finally:
    with tarfile.open('/kaggle/working/week5_results.tar.gz','w:gz') as archive:
        archive.add('artifacts', arcname='artifacts')
        archive.add('reports', arcname='reports')
        archive.add('configs', arcname='configs')
print('WEEK5_RETRAIN_AND_GATE_COMPLETE', flush=True)
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
    output = root / ".cache/kaggle-week5"
    output.mkdir(parents=True, exist_ok=True)
    (output / "week5.ipynb").write_text(
        json.dumps(notebook, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    metadata = {
        "id": f"{owner}/5g-network-intelligence-week5",
        "title": "5G Network Intelligence Week5",
        "code_file": "week5.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    print(build(parser.parse_args().owner))
