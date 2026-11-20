"""Cloud-only improvement workers with readable files, progress logs and durable checkpoints."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def build(family):
    root = Path.cwd()
    commit = subprocess.check_output(
        ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"], text=True
    ).strip()
    cells = []

    def cell(kind, source):
        result = {
            "cell_type": kind,
            "metadata": {},
            "id": hashlib.sha256(source.encode()).hexdigest()[:12],
            "source": source.splitlines(keepends=True),
        }
        if kind == "code":
            result.update(execution_count=None, outputs=[])
        cells.append(result)

    cell(
        "markdown",
        f"# Model improvement - {family}\n\n"
        "Explicit cloud training worker. No local fitting. Fit, stopping and scoring windows are separate. "
        "Search cannot index observations after hour 503; the final holdout is never evaluated. "
        "Inputs remain the original 40 causal features; target recipes use observed columns only. "
        "Checkpoints and metrics are written after each model into Kaggle Output.\n",
    )
    cell(
        "code",
        f"SOURCE_COMMIT = {commit!r}\n"
        + """from pathlib import Path
import os
import sys
import subprocess
import shutil
import tarfile
if not Path('/kaggle/working').is_dir():
    raise RuntimeError('Hosted Kaggle required')
PROJECT = Path('/kaggle/working/improvement')
PROJECT.mkdir(exist_ok=True)
os.chdir(PROJECT)
sys.path.insert(0,str(PROJECT))
os.environ['SOURCE_GIT_COMMIT'] = SOURCE_COMMIT
roots = [p.parent.parent for p in Path('/kaggle/input').rglob('model_quality_gate.json') if (p.parent.parent/'data/processed/panel.npz').exists()]
if not roots:
    raise FileNotFoundError('Attach completed canonical Week 5 output')
shutil.copytree(roots[0]/'data/processed',PROJECT/'data/processed',dirs_exist_ok=True)
Path('reports').mkdir(exist_ok=True)
""",
    )
    paths = [Path("pyproject.toml")]
    for folder in ("src", "scripts", "configs"):
        paths.extend(
            p
            for p in Path(folder).rglob("*")
            if p.suffix in (".py", ".yaml") and p.is_file() and "__pycache__" not in p.parts
        )
    paths.extend(
        [
            Path("tests/__init__.py"),
            Path("tests/conftest.py"),
            Path("tests/unit/test_target_recipe.py"),
        ]
    )
    cell(
        "markdown",
        "## Configuration and source\n\nOne visible file per cell. Configuration, causal target recipes and cloud training orchestration are separate.\n",
    )
    parents = sorted({p.parent.as_posix() for p in paths})
    cell(
        "code", f"for folder in {parents!r}:\n    Path(folder).mkdir(parents=True,exist_ok=True)\n"
    )
    for p in sorted(paths):
        cell("markdown", f"### `{p.as_posix()}`\n")
        cell("code", f"%%writefile {p.as_posix()}\n" + p.read_text(encoding="utf-8"))
    cell(
        "markdown",
        "## Cloud dependencies and focused tests\n\nThe marked tiny fitting test also runs here only.\n",
    )
    extra = ",'catboost>=1.2,<2'" if family == "catboost" else ""
    cell(
        "code",
        f"subprocess.run([sys.executable,'-m','pip','install','-e','.[ml,dev]'{extra}],check=True)\n"
        "subprocess.run([sys.executable,'-m','pytest','-q','tests/unit/test_target_recipe.py','--basetemp=/tmp/recipe-tests'],check=True)\n"
        + ("subprocess.run(['nvidia-smi'],check=True)\n" if family == "catboost" else ""),
    )
    cell(
        "markdown",
        "## Training and independent chronological scoring\n\nEarly stopping never uses the scoring window. No final holdout command is called.\n",
    )
    cell(
        "code",
        f"command = [sys.executable,'-u','-m','scripts.optimize_models','--family',{family!r}]\n"
        + """# Stream progress and preserve partial checkpoints even if the worker fails.
with Path('reports/training_progress.log').open('w') as log:
    process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in process.stdout:
        print(line,end='',flush=True)
        log.write(line)
        log.flush()
    returncode = process.wait()
with tarfile.open('/kaggle/working/improvement_results.tar.gz','w:gz') as archive:
    for folder in ('reports','configs','artifacts/improvement'):
        if Path(folder).exists(): archive.add(folder,arcname=folder)
if returncode:
    raise RuntimeError(f'Cloud worker failed with {returncode}; inspect saved progress log/checkpoints')
print('IMPROVEMENT_STAGE_COMPLETE',flush=True)
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
    slug = "5g-network-intelligence-improve-" + family
    out = Path(".cache") / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.ipynb").write_text(
        json.dumps(notebook, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    metadata = {
        "id": "truongbaokien/" + slug,
        "title": "5G Network Intelligence Improve " + family.title(),
        "code_file": "train.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": family == "catboost",
        "enable_internet": True,
        "kernel_sources": ["truongbaokien/5g-network-intelligence-week5"],
        "dataset_sources": [],
        "competition_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["lgbm", "catboost"], required=True)
    print(build(parser.parse_args().family))
