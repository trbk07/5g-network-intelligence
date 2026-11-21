"""Visible cloud notebook for a first external-source transfer ablation."""

import json
from pathlib import Path

from scripts.build_improvement_job import build


def main():
    base = build("lgbm")
    notebook = json.loads((base / "train.ipynb").read_text())
    index = next(
        i
        for i, c in enumerate(notebook["cells"])
        if c["cell_type"] == "markdown" and "".join(c["source"]).startswith("## Cloud dependencies")
    )
    path = Path("tests/unit/test_transfer_preparation.py")
    notebook["cells"].insert(
        index,
        {
            "cell_type": "code",
            "metadata": {},
            "id": "transfer-portability-test",
            "execution_count": None,
            "outputs": [],
            "source": ("%%writefile " + path.as_posix() + "\n" + path.read_text()).splitlines(
                keepends=True
            ),
        },
    )
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if source.startswith("%%writefile "):
            continue
        if source.startswith("command = "):
            source = """# Acquisition/structural preprocessing only; fitting is confined to this hosted worker.
subprocess.run([sys.executable,'-u','-m','scripts.acquire_zenodo_counters'],check=True)
subprocess.run([sys.executable,'-u','-m','scripts.prepare_transfer_data'],check=True)
command = [sys.executable,'-u','-m','scripts.train_transfer_pilot']
""" + source.split("\n", 1)[1]
            source = source.replace(
                "improvement_results.tar.gz", "transfer_pilot_results.tar.gz"
            ).replace("'artifacts/improvement'", "'artifacts/transfer_pilot'")
            source = source.replace("IMPROVEMENT_STAGE_COMPLETE", "TRANSFER_PILOT_COMPLETE")
        if source.startswith("SOURCE_COMMIT = "):
            source += "\n# Saved original C1 validation replay is only a reference; final holdout is never opened.\n"
            source += "refs=list(Path('/kaggle/input').rglob('lightgbm_val.npz'))\nif not refs: raise FileNotFoundError('Missing C1 validation reference')\nPath('reports/forecasts').mkdir(exist_ok=True)\nshutil.copyfile(refs[0],'reports/forecasts/lightgbm_val.npz')\n"
        if "'tests/unit/test_target_recipe.py'" in source:
            source = source.replace(
                "'tests/unit/test_target_recipe.py'",
                "'tests/unit/test_target_recipe.py','tests/unit/test_transfer_preparation.py'",
            )
        cell["source"] = source.splitlines(keepends=True)
    notebook["cells"][0]["source"] = [
        "# Relative-shape external transfer pilot\n\nHosted CPU fitting only. Original-network control and external weight ratios 0/0.1/0.5/1.0. "
        "Only the chronological 80% training prefix of clean Zenodo partition 01 is eligible; gaps are never bridged. "
        "Normalized target shape does not assert physical-unit equivalence. Final original holdout and the external future prefix remain unused.\n"
    ]
    output = Path(".cache/5g-network-intelligence-transfer-pilot")
    output.mkdir(parents=True, exist_ok=True)
    (output / "transfer.ipynb").write_text(
        json.dumps(notebook, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    meta = json.loads((base / "kernel-metadata.json").read_text())
    meta.update(
        id="truongbaokien/5g-network-intelligence-transfer-pilot",
        title="5G Network Intelligence Transfer Pilot",
        code_file="transfer.ipynb",
    )
    (output / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(output)


if __name__ == "__main__":
    main()
