"""Prepare inference-only verification of a completed Kaggle training kernel."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    commit = subprocess.check_output(["git", "-c", "safe.directory=D:/5G", "rev-parse", "HEAD"], text=True).strip()
    relative = "scripts/audit_telecom_cloud.py"
    committed = subprocess.check_output(["git", "-c", "safe.directory=D:/5G", "show", f"{commit}:{relative}"])
    digest = hashlib.sha256(committed).hexdigest()
    reference = json.loads(Path("notebooks/41_telecom_data_and_pretraining_budget_ablation.ipynb").read_text(encoding="utf-8"))
    setup = "".join(reference["cells"][3]["source"])
    ingress = "".join(reference["cells"][5]["source"]).replace("RUN_TRAINING", "RUN_AUDIT").replace(
        "/kaggle/working/telecom_transfer", "/kaggle/working/telecom_transfer_audit")
    url = f"https://raw.githubusercontent.com/trbk07/5g-network-intelligence/{commit}/{relative}"
    run = ("if RUN_AUDIT:\n    import urllib.request, subprocess\n"
        "    # Public code only: immutable commit plus exact content hash; no credentials.\n"
        f"    with urllib.request.urlopen({url!r}, timeout=60) as response:\n        code = response.read()\n"
        f"    assert hashlib.sha256(code).hexdigest() == {digest!r}, 'Audit source checksum differs'\n"
        "    audit_script = PROJECT/'audit_telecom_cloud.py'\n    audit_script.write_bytes(code)\n"
        "    matches = [p.parent for p in Path('/kaggle/input').rglob('manifest.json')\n"
        "               if p.parent.name == 'telecom_transfer' and all((p.parent/arm).is_dir()\n"
        "               for arm in ['telecom_matched', 'existing_extended', 'telecom_extended'])]\n"
        "    assert len(matches) == 1, 'Completed training artifact tree must be uniquely attached'\n"
        "    subprocess.run([sys.executable, str(audit_script), '--artifacts', str(matches[0]),\n"
        "        '--source-commit', '63187481ba9c7dfe2f3f56375aca0e30993298ee'], check=True)\n"
        "else:\n    print('Review mode: no checkpoint access, fitting or inference.')\n")
    sections = [
        ("markdown", "# 5G Network Intelligence — Telecom Experiment Integrity and Replay\n\n"
         "## Verification question\nDo the three completed data/compute arms satisfy their declared temporal, seed, "
         "transfer, compute and scoring contracts? This notebook performs inference and verification only. "
         "It never trains, calibrates, selects new weights or opens original hours 504 onward.\n"),
        ("code", "# Default Run All is a report review.\nRUN_AUDIT = False\n"),
        ("markdown", "## Verified input provenance\nUse the original hosted source prefix and audited Telecom supplement. "
         "Validate every manifest member before importing code. Completed checkpoints remain in attached Kaggle Input; "
         "only aggregate verification/metrics are exported. The training source stays pinned to its original commit.\n"),
        ("code", setup),
        ("code", ingress),
        ("markdown", "## Independent checks\nVerify exact artifact coverage/hashes; reconstruct fit-prefix preprocessing; "
         "reproduce source and target cold weights from seed 17; compare target cold states bitwise across arms. "
         "Check 5,120/15,360 actual optimizer updates and encoder-only transfer, unchanged target heads/scales, "
         "completed stopping, all original target/origin/beam axes, full-population MAE/WAPE, and CPU/GPU replay at atol 0.0001. "
         "No tolerance is selected from the current scores.\n"),
        ("code", run),
        ("markdown", "## Metric review and limitations\nMAE uses original DLPRB usage units; WAPE is total absolute error "
         "over total absolute observed load. Numerical parity is an engineering check, not a generalization claim. "
         "All arms use repeatedly reused development. Partial source overlap and original calendar identity remain uncertified. "
         "Quality gates and paired comparisons with the fixed strong comparator follow verification; a successful audit "
         "alone does not establish improvement, WAPE below 10% or production readiness.\n"),
        ("code", "# Read only completed aggregate evidence.\nimport pandas as pd\nfrom IPython.display import display\n"
         "path = Path('/kaggle/working/telecom_transfer_verified_metrics.json') if RUN_AUDIT else PROJECT/'reports/telecom_transfer_verified_metrics.json'\n"
         "if path.exists():\n    result = json.loads(path.read_text(encoding='utf-8'))\n"
         "    assert result['all_metrics_reconstructed'] and not result['holdout_used']\n"
         "    frame = pd.DataFrame(result['metrics'])\n    frame['WAPE (%)'] = 100*frame['wape']\n"
         "    display(frame)\n    display({k: v for k, v in result.items() if k != 'metrics'})\n"
         "else:\n    print('Verification pending; no accepted quality result yet.')\n"),
    ]
    cells = []
    for kind, source in sections:
        cell = {"cell_type": kind, "id": f"telecom-audit-{len(cells):02d}", "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            ast.parse(source)
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "nbformat": 4, "nbformat_minor": 5}
    name = "42_telecom_experiment_integrity_and_replay.ipynb"
    review = Path("notebooks")/name
    if review.exists():
        raise FileExistsError("Preserve existing verification notebook")
    review.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    notebook["cells"][1]["source"] = ["# Dedicated hosted inference-only worker.\nRUN_AUDIT = True\n"]
    stage = Path(".cache/kaggle-telecom-audit")
    stage.mkdir(parents=True, exist_ok=True)
    (stage/name).write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    (stage/"kernel-metadata.json").write_text(json.dumps({"id": "truongbaokien/5g-ni-telecom-experiment-integrity-and-replay",
        "title": "5G NI Telecom Experiment Integrity and Replay", "code_file": name, "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
        "dataset_sources": ["truongbaokien/5g-ni-cell-multisource-v2-inputs", "truongbaokien/5g-ni-telecom-prefix-transfer-inputs"],
        "kernel_sources": ["truongbaokien/5g-ni-telecom-data-and-pretraining-budget-ablation"], "competition_sources": []}, indent=2))
    print(json.dumps({"audit_source_commit": commit, "audit_source_sha256": digest, "training_performed": False,
                      "submission_condition": "training kernel COMPLETE with all three declared arms"}, indent=2))


if __name__ == "__main__":
    main()
