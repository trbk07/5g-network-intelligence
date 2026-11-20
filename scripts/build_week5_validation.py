"""Package hosted quality tests and research analysis without retraining experiments."""

import json
from pathlib import Path

from scripts.build_week5_job import build


def package(owner: str = "truongbaokien") -> Path:
    """Attach completed cloud outputs; run fitting fixtures and comparison on hosted CPU."""
    source = build(owner)
    notebook = json.loads((source / "week5.ipynb").read_text())
    cutoff = next(
        i
        for i, c in enumerate(notebook["cells"])
        if c["cell_type"] == "markdown"
        and "".join(c["source"]).startswith("## 1. Cloud dependencies")
    )
    notebook["cells"] = notebook["cells"][:cutoff]
    notebook["cells"][0]["source"] = [
        "# Week 5 — Hosted validation and research evidence\n",
        "Uses completed Kaggle checkpoints as input. No real-data model retraining.\n",
        "Synthetic fitting tests run only here on hosted cloud; holdout stays closed.\n",
    ]
    code = """import shutil
import tarfile

# Mount the completed training notebook outputs as read-only Kaggle Input.
roots = [p.parent.parent for p in Path('/kaggle/input').rglob('model_quality_gate.json')
         if (p.parent.parent/'artifacts/lgbm_w5').is_dir()]
if not roots:
    raise FileNotFoundError('Completed Week 5 training outputs were not attached.')
previous = roots[0]
shutil.copytree(previous/'artifacts', PROJECT/'artifacts', dirs_exist_ok=True)
shutil.copytree(previous/'reports', PROJECT/'reports', dirs_exist_ok=True)
shutil.copytree(previous/'configs', PROJECT/'configs', dirs_exist_ok=True)
if (previous/'data/processed/panel.npz').exists():
    shutil.copytree(previous/'data/processed', PROJECT/'data/processed', dirs_exist_ok=True)
else:
    subprocess.run([sys.executable,'-m','scripts.prepare_data','--step','all'],check=True,timeout=600)

subprocess.run([sys.executable,'-m','pip','install','-e','.[dev,ml,neural]'],check=True)
test = subprocess.run([sys.executable,'-m','pytest','-q','--ignore=tests/unit/test_notebook_encoding.py'],
                      capture_output=True,text=True,timeout=300)
Path('reports/week5_validation_tests.log').write_text(test.stdout+test.stderr)
print(test.stdout)
test.check_returncode()
subprocess.run([sys.executable,'-m','ruff','check','src','scripts','tests'],check=True,timeout=60)
subprocess.run([sys.executable,'-m','scripts.evaluate_models','--model','lightgbm','--segment','val','--diagnostics'],
               check=True,timeout=600)
with tarfile.open('/kaggle/working/week5_validation_results.tar.gz','w:gz') as archive:
    for path in Path('reports').iterdir():
        if path.is_file() or path.name=='figures':
            archive.add(path,arcname=path.as_posix())
    archive.add('configs',arcname='configs')
print('WEEK5_RESEARCH_GATE_COMPLETE',flush=True)
"""
    notebook["cells"].append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "id": "research-analysis",
            "source": [
                "## Research gate\n",
                "Copy durable checkpoints, run tests, lock common masks and build paired comparisons/atlas.\n",
            ],
        }
    )
    notebook["cells"].append(
        {
            "cell_type": "code",
            "metadata": {},
            "id": "research-execution",
            "source": code.splitlines(keepends=True),
            "execution_count": None,
            "outputs": [],
        }
    )
    output = Path(".cache/kaggle-week5-validation")
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation.ipynb").write_text(
        json.dumps(notebook, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    metadata = {
        "id": f"{owner}/5g-network-intelligence-week5-validation",
        "title": "5G Network Intelligence Week5 Validation",
        "code_file": "validation.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [f"{owner}/5g-network-intelligence-week5"],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    return output


if __name__ == "__main__":
    print(package())
