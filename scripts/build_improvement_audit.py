"""Freeze ensemble from cloud inner folds and refit/audit without reopening final holdout."""

import json
import subprocess
from pathlib import Path


def main():
    notebook = json.loads(
        Path(".cache/5g-network-intelligence-improve-lgbm/train.ipynb").read_text()
    )
    cells = notebook["cells"]
    dependencies = next(
        i
        for i, c in enumerate(cells)
        if c["cell_type"] == "markdown" and "".join(c["source"]).startswith("## Cloud dependencies")
    )
    extra = []
    for name in (
        "scripts/finalize_improvement.py",
        "src/models/research_ensemble.py",
        "tests/unit/test_research_ensemble.py",
    ):
        extra.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "id": "audit-" + str(len(extra)),
                "source": [f"### `{name}`\n"],
            }
        )
        extra.append(
            {
                "cell_type": "code",
                "metadata": {},
                "id": "audit-" + str(len(extra)),
                "execution_count": None,
                "outputs": [],
                "source": (
                    f"%%writefile {name}\n" + Path(name).read_text(encoding="utf-8")
                ).splitlines(keepends=True),
            }
        )
    setup = """# Merge completed workers, preserving their families and progress logs.
for filename, family in [('improvement_lgbm_cv.csv','lgbm'),('improvement_catboost_cv.csv','catboost')]:
    matches = list(Path('/kaggle/input').rglob(filename))
    if not matches:
        raise FileNotFoundError(f'Completed {family} worker must be attached')
    previous = matches[0].parent.parent
    shutil.copytree(previous/'artifacts/improvement',PROJECT/'artifacts/improvement',dirs_exist_ok=True)
    for report in (previous/'reports').iterdir():
        if report.is_file():
            destination = PROJECT/'reports'/ (f'improvement_{family}_training.log' if report.name=='training_progress.log' else report.name)
            shutil.copyfile(report,destination)
# References are already measured validation replays, never the final holdout.
for filename in ('lightgbm_val.npz','week7_research_val.npz'):
    matches = list(Path('/kaggle/input').rglob(filename))
    if not matches:
        raise FileNotFoundError(f'Missing saved reference {filename}')
    (PROJECT/'reports/forecasts').mkdir(exist_ok=True)
    shutil.copyfile(matches[0],PROJECT/'reports/forecasts'/filename)
"""
    extra.append(
        {
            "cell_type": "code",
            "metadata": {},
            "id": "audit-merge",
            "execution_count": None,
            "outputs": [],
            "source": setup.splitlines(keepends=True),
        }
    )
    cells[dependencies:dependencies] = extra
    commit = subprocess.check_output(
        ["git", "-c", f"safe.directory={Path.cwd().as_posix()}", "rev-parse", "HEAD"], text=True
    ).strip()
    for cell in cells:
        source = "".join(cell["source"])
        if source.startswith("%%writefile "):
            continue
        if source.startswith("SOURCE_COMMIT = "):
            source = source.split("\n", 1)[1]
            source = f"SOURCE_COMMIT = {commit!r}\n" + source
        source = source.replace(
            "'scripts.optimize_models', '--family', 'lgbm'", "'scripts.finalize_improvement'"
        )
        # Original worker serializes its command across ordinary Python list representation.
        if source.startswith("command = "):
            source = source.split("\n", 1)[1]
            source = (
                "command = [sys.executable,'-u','-m','scripts.finalize_improvement']\n" + source
            )
            source = source.replace("'artifacts/improvement'", "'artifacts/improved_candidate'")
            source = source.replace(
                "improvement_results.tar.gz", "improvement_audit_results.tar.gz"
            )
        if "subprocess.run([sys.executable,'-m','pip'" in source:
            source = source.replace("'.[ml,dev]'", "'.[ml,dev]','catboost>=1.2,<2'")
            source = source.replace(
                "'tests/unit/test_target_recipe.py'",
                "'tests/unit/test_target_recipe.py','tests/unit/test_research_ensemble.py'",
            )
        cell["source"] = source.splitlines(keepends=True)
    cells[0]["source"] = [
        "# Improvement - Inner-selected ensemble and reused-validation audit\n\nWeights/recipes come only from inner chronological score windows through hour 503. "
        "Stopping for final rounds uses 420-503; fixed refit uses 0-503. Audit 504-671 is previously seen validation, not a fresh test. "
        "Final holdout 672-839 stays closed and the production champion is unchanged.\n"
    ]
    output = Path(".cache/5g-network-intelligence-improve-audit")
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.ipynb").write_text(
        json.dumps(notebook, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    metadata = {
        "id": "truongbaokien/5g-network-intelligence-improve-audit",
        "title": "5G Network Intelligence Improve Audit",
        "code_file": "audit.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "kernel_sources": [
            "truongbaokien/5g-network-intelligence-week5",
            "truongbaokien/5g-network-intelligence-week7-research",
            "truongbaokien/5g-network-intelligence-improve-lgbm",
            "truongbaokien/5g-network-intelligence-improve-catboost",
        ],
        "dataset_sources": [],
        "competition_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    print(output)


if __name__ == "__main__":
    main()
