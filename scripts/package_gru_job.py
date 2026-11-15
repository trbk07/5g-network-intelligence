"""Package a readable private Kaggle notebook; no local model fitting or upload."""
import argparse
import json
from pathlib import Path

from scripts.build_gru_review_notebook import build


def main() -> None:
    """Enable cloud training in the commented review notebook, without encoding source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--output", default=".cache/kaggle-gru")
    args = parser.parse_args()
    notebook = json.loads(build().read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            for flag in ("RUN_INSTALL", "RUN_PREPARE_DATA", "RUN_TRAINING", "RUN_ALL_EXPERIMENTS", "RUN_EVALUATION"):
                source = source.replace(f"{flag} = False", f"{flag} = True")
            cell["source"] = source.splitlines(keepends=True)
            cell["outputs"] = []
            cell["execution_count"] = None
    notebook["cells"][0]["source"] = [
        "# Week 4 ? Cloud training job\n",
        "Training flags are enabled in this packaged version. Runs A/B x seeds 17/42 on Kaggle GPU.\n",
        "All source is readable Python with comments; checkpoints go to Kaggle Output.\n",
    ]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "week4.ipynb").write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata = {"id": f"{args.owner}/5g-network-intelligence-gru",
                "title": "5G Network Intelligence GRU", "code_file": "week4.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_internet": True, "dataset_sources": [],
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Prepared readable GPU notebook at {output}. No local training or upload performed.")


if __name__ == "__main__":
    main()
