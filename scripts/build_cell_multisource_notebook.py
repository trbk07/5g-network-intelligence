"""Create a report-first guide for expanded multi-corpus encoder pretraining."""

import json
from pathlib import Path


def main():
    notebook = json.loads(Path("notebooks/33_cell_spatial_temporal_attention.ipynb").read_text(encoding="utf-8"))
    notebook["cells"][0]["source"] = ("# 5G Network Intelligence — Multi-Corpus Spatial Encoder Pretraining\n\n"
        "## Hypothesis\nCan longer representation pretraining on audited radio and network "
        "traffic prefixes improve original DLPRB forecasting beyond an equal-compute "
        "target-self-pretraining control? Transfer only the temporal encoder; learn the "
        "cell interaction and physical DLPRB output on the original fit prefix.\n").splitlines(keepends=True)
    notebook["cells"][3]["source"] = ("## Fixed compute and controls\nTwo cases across two folds: "
        "target-self-pretraining and pooled external pretraining, followed by identical "
        "cell fine-tuning. Each source pretraining run uses 20 epochs × 65,536 sampled "
        "windows / batch 256 = 5,120 updates. Use the last pretraining checkpoint, never "
        "choose it from scoring. Fine-tuning: maximum 100 epochs, patience 12, seed 17. "
        "External pretraining is shared between folds; target-self pretraining respects "
        "each fold's fit cutoff.\n").splitlines(keepends=True)
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        text = text.replace("configs/cell_attention.yaml", "configs/cell_multisource.yaml")
        text = text.replace("scripts.train_cell_attention", "scripts.train_cell_multisource")
        text = text.replace("reports/cell_attention_training.log", "reports/cell_multisource_training.log")
        if "path = PROJECT / 'reports/cell_attention_development.csv'" in text:
            text = ("# Review completed results only; no fitting or coefficient search.\n"
                "for case in ('target_self_pretrain', 'multi_source_pretrain'):\n"
                "    path = PROJECT / f'reports/cell_multisource_{case}_development.csv'\n"
                "    if path.exists():\n        results = pd.read_csv(path)\n"
                "        results['WAPE (%)'] = 100 * results['wape']\n"
                "        print(case)\n        display(results)\n")
        cell["source"] = text.splitlines(keepends=True)
    notebook["cells"].insert(6, {"cell_type": "markdown", "id": "expanded-source-provenance", "metadata": {},
        "source": ("## Additional data and provenance\nNine external domains: NR1800/2100/3500, "
        "CESNET-TimeSeries24, four Vigo KPIs, and CESNET-AGG23. Radio, CESNET24, Vigo "
        "and AGG23 each receive equal aggregate sampling weight, followed by entity "
        "and continuous-window sampling. They are not nine independent networks. Vigo "
        "and AGG23 add 40,653 eligible prefix entity-hours. All payload and processed "
        "checksums are pinned in preparation reports. Telecom Italia is being acquired "
        "separately and is not part of this frozen experiment.\n\n"
        "Vigo values were scaled by the original publisher per series; its scaling fit "
        "scope is unknown. Our source task normalizes using past context only, making "
        "the numerical task invariant to positive affine scaling; this is not an audit "
        "of publisher quantization or preprocessing. Network traffic and sessions are "
        "not PRB labels. The dimensionless source head is discarded. Only temporal "
        "encoder parameters transfer, with checkpoint hashes recorded at initialization. "
        "Never use source anomaly labels to tune the target forecaster.\n").splitlines(keepends=True)})
    Path("notebooks/34_multi_corpus_spatial_encoder_pretraining.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
