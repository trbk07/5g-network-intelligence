"""Readable hosted multi-source ablation worker; preserve source cells verbatim."""

import json
from pathlib import Path

from scripts.build_transfer_pilot import main as build_single


def main():
    build_single()
    base = Path(".cache/5g-network-intelligence-transfer-pilot")
    notebook = json.loads((base / "transfer.ipynb").read_text())
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if source.startswith("%%writefile "):
            continue
        if "scripts.acquire_zenodo_counters" in source:
            source = source.replace(
                "'scripts.acquire_zenodo_counters']",
                "'scripts.acquire_zenodo_counters','--include-large']",
            )
            source = source.replace(
                "'scripts.prepare_transfer_data'", "'scripts.prepare_transfer_sources'"
            )
            source = source.replace(
                "command = [",
                "subprocess.run([sys.executable,'-u','-m','scripts.audit_transfer_overlap'],check=True)\ncommand = [",
                1,
            )
            source = source.replace(
                "'scripts.train_transfer_pilot'", "'scripts.train_multisource_pilot'"
            )
            source = source.replace(
                "transfer_pilot_results.tar.gz", "multisource_pilot_results.tar.gz"
            )
            source = source.replace("'artifacts/transfer_pilot'", "'artifacts/multisource_pilot'")
            source = source.replace("TRANSFER_PILOT_COMPLETE", "MULTISOURCE_PILOT_COMPLETE")
        cell["source"] = source.splitlines(keepends=True)
    notebook["cells"][0]["source"] = [
        "# Balanced multi-source transfer ablation\n\nHosted CPU only. Three audited commercial frequency domains, target-only control and balanced pool. "
        "Source prefixes, exact complete-hour coverage, gaps and overlap are checked before fitting. "
        "Overlapping bands are not independent networks. Source sampling caps and weights are predefined. "
        "Original final holdout stays closed; only reused validation is audited.\n"
    ]
    output = Path(".cache/5g-network-intelligence-multisource-pilot")
    output.mkdir(parents=True, exist_ok=True)
    (output / "multisource.ipynb").write_text(
        json.dumps(notebook, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    meta = json.loads((base / "kernel-metadata.json").read_text())
    meta.update(
        id="truongbaokien/5g-network-intelligence-multisource-pilot",
        title="5G Network Intelligence Multisource Pilot",
        code_file="multisource.ipynb",
    )
    (output / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(output)


if __name__ == "__main__":
    main()
