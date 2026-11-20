"""Readable Week 6 notebooks; review is the safe default, final execution is explicit."""

import argparse
import hashlib
import json
from pathlib import Path

from src.inference.freeze import git, verify_frozen


def build(mode="review"):
    """Package visible source and pinned completed Week 5 outputs, without local training."""
    cells = []

    def cell(kind, source):
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
        "# 5G Network Intelligence - Week 6\n\n"
        + (
            "Review only: Run All displays saved reports and never trains.\n"
            if mode == "review"
            else "Hosted CPU execution. Fixed development refit, then one guarded final holdout replay.\n"
        )
        + "P90/P95 describe historical capacity-risk scenarios, not confirmed network incidents.\n",
    )
    if mode == "review":
        cell(
            "code",
            "from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display\n"
            "roots = list(Path('/kaggle/input').rglob('holdout_completion.json'))\n"
            "if not roots: raise FileNotFoundError('Attach completed Week 6 outputs')\n"
            "REPORTS = roots[0].parent\n",
        )
        for name, title, explanation, expression in [
            (
                "champion_benchmark_holdout.csv",
                "Forecast metrics",
                "Lower MAE/RMSE/WAPE is better. Each horizon has a validation-selected baseline.",
                "table.loc[table.load_tier == 'All', ['model','horizon','mae','rmse','wape','active_only_mae']]",
            ),
            (
                "champion_benchmark_holdout.csv",
                "Errors by development load tier",
                "Load groups were fitted on development data. Compare errors within each group.",
                "table.loc[table.load_tier != 'All', ['model','horizon','load_tier','mae','wape','n_valid']]",
            ),
            (
                "paired_comparison_holdout.csv",
                "Paired BS bootstrap",
                "Difference is MAE(champion) minus MAE(baseline). A 95% interval below zero favors champion.",
                "table[['horizon','model_b','difference','lower','upper','n_clusters']]",
            ),
            (
                "risk_holdout.csv",
                "Historical exceedance metrics",
                "Precision measures alarm reliability. Recall measures the fraction of actual exceedances found. Low recall remains a limitation.",
                "table.loc[table.load_tier == 'All', ['model','horizon','threshold','precision','recall','fpr','positive_events']]",
            ),
            (
                "risk_cases_holdout.csv",
                "False alarms and misses",
                "FP is a false alarm; FN is a missed exceedance. First available examples, not inferred incidents.",
                "table.head(12)",
            ),
        ]:
            cell(
                "markdown",
                f"## {title}\n\n{explanation}\n\nSaved results; no training or new replay.\n",
            )
            cell(
                "code",
                f"# Show the columns needed for this question.\ntable = pd.read_csv(REPORTS / {name!r})\ndisplay({expression})\n",
            )
    else:
        if mode == "final":
            _, manifest = verify_frozen()
            commit = manifest["commit"]
        else:
            commit = git("rev-parse", "HEAD")
        cell(
            "markdown",
            "## 1. Cloud setup and configuration\n\nTraining is enabled only in final mode on hosted Kaggle.\n",
        )
        cell(
            "code",
            f"SOURCE_COMMIT = {commit!r}\n"
            + """from pathlib import Path
import os
import subprocess
import sys
import shutil
import tarfile
if not Path('/kaggle/working').is_dir():
    raise RuntimeError('Hosted Kaggle required')
PROJECT = Path('/kaggle/working/week6')
PROJECT.mkdir(exist_ok=True)
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
os.environ['SOURCE_GIT_COMMIT'] = SOURCE_COMMIT
"""
            + "\n# Completed Week 5 outputs provide the original cloud panel and validation evidence.\n"
            + "roots = [p.parent.parent for p in Path('/kaggle/input').rglob('model_quality_gate.json') if (p.parent.parent/'artifacts/lgbm_w5').is_dir()]\n"
            + "if not roots: raise FileNotFoundError('Attach pinned Week 5 outputs')\n"
            + "previous = roots[0]\n"
            + "for folder in ('artifacts','reports','data/processed'):\n    shutil.copytree(previous/folder, PROJECT/folder, dirs_exist_ok=True)\n",
        )
        paths = [Path("pyproject.toml"), Path("plan5G.md")]
        for folder in ("src", "scripts", "configs", "tests"):
            paths.extend(
                p
                for p in Path(folder).rglob("*")
                if p.is_file() and p.suffix in (".py", ".yaml") and "__pycache__" not in p.parts
            )
        if mode == "final":
            paths.append(Path("reports/freeze_manifest.json"))
            paths.extend(
                Path("reports") / n
                for n in ("risk_validation.csv", "risk_tradeoff_val.md", "champion_selection.md")
            )
        parents = sorted({p.parent.as_posix() for p in paths})
        cell(
            "code",
            f"for folder in {parents!r}:\n    Path(folder).mkdir(parents=True, exist_ok=True)\n",
        )
        cell(
            "markdown",
            "## 2. Readable source\n\nOne file per cell; configurations, preprocessing, model and evaluation are separate modules.\n",
        )
        for path in sorted(paths):
            cell("markdown", f"### `{path.as_posix()}`\n")
            cell("code", f"%%writefile {path.as_posix()}\n" + path.read_text(encoding="utf-8"))
        cell(
            "markdown",
            "## 3. Dependencies and debugging checks\n\nTests that fit tiny models run only on this hosted runtime.\n",
        )
        cell(
            "code",
            "subprocess.run([sys.executable,'-m','pip','install','-e','.[dev,ml,neural]'],check=True)\n"
            "result = subprocess.run([sys.executable,'-m','pytest','-q','--ignore=tests/unit/test_notebook_encoding.py'],capture_output=True,text=True,timeout=600)\n"
            "Path('reports/week6_cloud_tests.log').write_text(result.stdout+result.stderr)\nprint(result.stdout)\nresult.check_returncode()\n",
        )
        if mode == "preflight":
            cell(
                "markdown",
                "## 4. Validation risk audit\n\nNo real-data retraining or final holdout evaluation.\n",
            )
            commands = ["scripts.run_risk --segment val"]
        else:
            cell(
                "markdown",
                "## 4. Final development refit\n\nFrozen rounds from validation, no early stopping, checkpoints in Kaggle Output.\n",
            )
            cell(
                "code",
                "subprocess.run([sys.executable,'-m','scripts.refit_champion'],check=True,timeout=7200)\n",
            )
            cell(
                "markdown",
                "## 5. One guarded holdout replay\n\nNo parameter updates after final targets are opened. Duplicate attempts are refused.\n",
            )
            commands = ["scripts.run_holdout"]
        for command in commands:
            cell(
                "code",
                f"result = subprocess.run([sys.executable,'-m',*{command.split()!r}],capture_output=True,text=True,timeout=7200)\n"
                "Path('reports/holdout_run.log').write_text(result.stdout+result.stderr)\nprint(result.stdout)\nresult.check_returncode()\n",
            )
        cell(
            "markdown",
            "## 6. Durable output archive\n\nAll reports and final checkpoints remain in Kaggle Output.\n",
        )
        cell(
            "code",
            "with tarfile.open('/kaggle/working/week6_results.tar.gz','w:gz') as archive:\n"
            "    for folder in ('reports','configs','artifacts/champion'):\n"
            "        if Path(folder).exists(): archive.add(folder,arcname=folder)\n"
            "print('WEEK6_COMPLETE',flush=True)\n",
        )
    slug = "5g-network-intelligence-week6-" + mode
    output = Path(".cache") / slug
    output.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (output / "week6.ipynb").write_text(
        json.dumps(notebook, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    metadata = {
        "id": "truongbaokien/" + slug,
        "title": "5G Network Intelligence Week6 " + mode.title(),
        "code_file": "week6.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": mode != "review",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [
            "truongbaokien/5g-network-intelligence-week6-final"
            if mode == "review"
            else "truongbaokien/5g-network-intelligence-week5"
        ],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["review", "preflight", "final"], default="review")
    print(build(parser.parse_args().mode))
