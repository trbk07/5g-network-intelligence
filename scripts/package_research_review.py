"""Stage public-source audit evidence privately and a named, report-only Kaggle notebook."""

import json
import shutil
from pathlib import Path


def main():
    data = Path(".cache/kaggle-research-evidence")
    data.mkdir(parents=True, exist_ok=True)
    names = [
        "external_candidate_manifest.json",
        "oai_candidate_audit.json",
        "zenodo_counter_manifest.json",
        "zenodo_duplicate_audit.json",
        "zenodo_large_key_audit.json",
        "multisource_preparation.json",
        "transfer_overlap_audit.json",
        "transfer_pilot_evidence.json",
        "multisource_pilot_evidence.json",
    ]
    for name in names:
        shutil.copyfile(Path("reports") / name, data / name)
    for kind in ("transfer_pilot", "multisource_pilot"):
        for suffix in ("validation.csv", "cpu_parity.csv", "analysis.md"):
            name = f"{kind}_{suffix}"
            shutil.copyfile(Path("reports") / name, data / name)
    for name in ("transfer_ablation_comparison.csv", "transfer_ablation_decision.md"):
        shutil.copyfile(Path("reports") / name, data / name)
    (data / "README.md").write_text(
        "# External research audit evidence\n\n"
        "Derived source/provenance and structural audit metadata only. No raw target series, credentials or training checkpoints.\n"
        "Sources: CTTCADAPTRU/OAI-KPM-dataset (CC0-1.0); Zenodo DOI 10.5281/zenodo.17815388 (CC BY 4.0), "
        "Peter Lehoczký, Matúš Turcsány, Laura Krajčovičová, Filip Zatroch, Marcel Kajan and Marek Galinski. "
        "See included manifests for exact revisions/checksums. Audits are project-derived changes, not publisher statements.\n",
        encoding="utf-8",
    )
    (data / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": "truongbaokien/5g-network-intelligence-research-evidence",
                "title": "5G Network Intelligence Research Evidence",
                "licenses": [{"name": "CC-BY-4.0"}],
            },
            indent=2,
        )
    )
    stage = Path(".cache/kaggle-external-data-quality-audit")
    stage.mkdir(parents=True, exist_ok=True)
    name = "09_external_data_quality_audit.ipynb"
    shutil.copyfile(Path("notebooks") / name, stage / name)
    (stage / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "truongbaokien/5g-network-intelligence-external-data-audit",
                "title": "5G Network Intelligence External Data Audit",
                "code_file": name,
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": False,
                "enable_internet": False,
                "dataset_sources": ["truongbaokien/5g-network-intelligence-research-evidence"],
                "kernel_sources": [],
                "competition_sources": [],
            },
            indent=2,
        )
    )
    review = Path(".cache/kaggle-transfer-results-review")
    review.mkdir(parents=True, exist_ok=True)
    review_name = "12_transfer_results_and_error_analysis.ipynb"
    shutil.copyfile(Path("notebooks") / review_name, review / review_name)
    metadata = json.loads((stage / "kernel-metadata.json").read_text())
    metadata.update(
        id="truongbaokien/5g-network-intelligence-transfer-results-review",
        title="5G Network Intelligence Transfer Results Review",
        code_file=review_name,
    )
    (review / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    print("Private evidence dataset:", data, "; report-only notebook:", stage)


if __name__ == "__main__":
    main()
