"""Readable seeded/strict-backend research guides with non-training review defaults."""

import json
from pathlib import Path


def main():
    studies = [
        ("cell_attention_v2", "37_seeded_cell_spatial_temporal_attention.ipynb",
         "Seeded Cell Spatial–Temporal Attention", "Two matched local/cell modes across two folds."),
        ("cell_multisource_v2", "38_reproducible_multi_corpus_encoder_transfer.ipynb",
         "Reproducible Multi-Corpus Encoder Transfer", "Equal-compute target-self and nine-domain external pretraining, then full-cell fine-tuning across two folds."),
    ]
    for study, filename, title, design in studies:
        sections = [
            ("markdown", f"# 5G Network Intelligence — {title}\n\n## Research question\n"
             f"{design} Does the candidate materially reduce original DLPRB MAE/WAPE "
             "against the fixed strong control? Improvement remains an experimental question.\n"),
            ("markdown", "## Temporal and reproducibility contract\nFit/stop/score cutoffs: "
             "335/383/443 and 395/443/503. Context 168 hours, four KPIs/masks, horizons "
             "1/6/24h, all 2,880 original beams in 90 cells. Exclude hours 504 onward "
             "and the observed holdout. Historical v1 omitted explicit Torch seeding; "
             "its point scores are exploratory, not controlled ablations. This protocol "
             "seeds Python, NumPy, CPU Torch and all CUDA generators, saves cold weights "
             "before any optimization/transfer, and disables the GPU attention fastpath "
             "for consistent stopping/scoring. Historical snapshots are not overwritten.\n"),
            ("code", "# Review Run All never trains.\nRUN_TRAINING = False\n"),
            ("markdown", "## Configuration\nSeed 17, maximum 100 fine-tuning epochs, patience "
             "12. Only stopping MAE selects weights. Record best/last weights, optimizer, "
             "RNG, cold initialization hashes and exact compute counts in durable cloud storage.\n"),
            ("code", "from pathlib import Path\nimport json\nimport yaml\nimport pandas as pd\n"
             "from IPython.display import display\nPROJECT = Path.cwd()\n"
             f"path = PROJECT / 'configs/{study}.yaml'\n"
             "if path.exists():\n    protocol = yaml.safe_load(path.read_text(encoding='utf-8'))\n    display(protocol)\n"),
            ("markdown", "## Inputs and model\nUse global per-KPI fit-prefix medians/means/scales, "
             "observed zeros and missing masks. Raw same-cell beam histories end at the "
             "origin; no fitted beam statistics or identity embeddings. Shared temporal "
             "patch encoder, spatial attention, causal seasonal residual and nonnegative "
             "physical DLPRB output. Local control masks off-diagonal attention.\n"),
        ]
        if study == "cell_multisource_v2":
            sections += [("markdown", "## External prefix pretraining\nNine domains belong to "
                "four collections: three NR bands, CESNET24, four Vigo KPIs and AGG23. "
                "Each collection has equal aggregate sampling weight. Use 20 epochs × "
                "65,536 windows / batch 256 = 5,120 updates per pretraining run. "
                "Target-self controls receive the same compute and only their original "
                "fit prefix. Transfer temporal encoder weights only; discard the "
                "dimensionless source head. Normalize source windows/labels from past "
                "context. Vigo publisher scaling scope is unknown; affine invariance "
                "of nondegenerate normalized windows is not a publisher preprocessing "
                "audit. Traffic bytes/sessions are not PRB labels. Telecom Italia is "
                "still outside this frozen study pending its own audit.\n"),
                ("markdown", "## Transfer checks\nSave source cold initialization, completed "
                 "pretraining checkpoint, target cold initialization and post-transfer "
                 "initialization. Verify encoder equality and unchanged target head, "
                 "spatial weights and physical scaling. Restore the target RNG stream "
                 "and start a fresh optimizer for fine-tuning.\n")]
        sections += [
            ("markdown", "## Hosted training and debugging\nColab/Kaggle only. Epoch logs "
             "show valid target counts and loss. Fine-tuning uses original-unit MAE; "
             "source pretraining uses dimensionless MAE. Interpret historical `scaled` "
             "trainer field names by stage. An epoch cap is not completed patience.\n"),
            ("code", "if RUN_TRAINING:\n    import subprocess, sys\n"
             "    (PROJECT/'reports').mkdir(exist_ok=True)\n"
             f"    with (PROJECT/'reports/{study}_training.log').open('w') as log:\n"
             f"        process = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.train_{study}'],\n"
             "            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
             "        for line in process.stdout:\n            print(line, end='', flush=True)\n"
             "            log.write(line)\n            log.flush()\n        TRAIN_EXIT = process.wait()\n"
             "else:\n    print('Review mode: no fitting.')\n"),
            ("markdown", "## Verified evaluation\nReconstruct full-population MAE/WAPE, "
             "coverage, zero error and CPU/GPU parity before acceptance. Report every "
             "horizon and negative results. Default review does not run selection or "
             "recompute fits. One seed and reused development cannot prove fresh generalization.\n"),
            ("code", f"prefixes = {[study] if study == 'cell_attention_v2' else [study+'_target_self_pretrain',study+'_multi_source_pretrain']!r}\n"
             "for prefix in prefixes:\n    verification = PROJECT/'reports'/f'{prefix}_verification.json'\n"
             "    scores = PROJECT/'reports'/f'{prefix}_development.csv'\n"
             "    if verification.exists() and scores.exists():\n"
             "        evidence = json.loads(verification.read_text())\n"
             "        if evidence.get('all_metrics_reconstructed'):\n"
             "            results = pd.read_csv(scores)\n            results['WAPE (%)'] = 100*results['wape']\n"
             "            display(results)\n            display(evidence)\n"
             "    else:\n        print(prefix, ': independent verification pending')\n"),
            ("markdown", "## Acceptance\nTarget at least 5% mean MAE improvement without "
             "horizon regression. A 2% initial research gate only qualifies additional "
             "seeds and paired time-block uncertainty checks. WAPE below 10% is not "
             "established. No holdout reuse, final refit or automatic production promotion.\n"),
        ]
        cells = []
        for kind, source in sections:
            cell = {"cell_type": kind, "id": f"seeded-{len(cells):02d}", "metadata": {}, "source": source.splitlines(keepends=True)}
            if kind == "code":
                cell.update(execution_count=None, outputs=[])
            cells.append(cell)
        notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
        Path("notebooks", filename).write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
