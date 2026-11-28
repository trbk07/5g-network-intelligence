"""Create a readable TimesFM protocol/review guide with inference disabled."""

import ast
import json
from pathlib import Path


def main():
    sections = [
        ("markdown", "# 5G Network Intelligence — Pretrained Median Forecasting\n\n"
         "## Research question and hypothesis\n"
         "Can a different pretrained temporal representation beat the fixed strong DLPRB comparator? "
         "TimesFM 2.5 is screened with its fixed median (index 5), aligned with MAE. "
         "No quantile, bias, ensemble weight or seed is selected from scoring results. "
         "Prior cell attention and multi-corpus encoder transfer failed the material-improvement gate.\n"),
        ("markdown", "## Temporal protocol and metric\n"
         "Use only 168 observed hours ending at each origin, all 2,880 beams and horizons 1/6/24h. "
         "Two reused development folds: fit/stop/score endpoints 335/383/443 and 395/443/503. "
         "Zero-shot inference performs no fitting; reserved original hours 504 onward remain redacted. "
         "MAE is in physical DLPRB usage units, not utilization percent. WAPE is total absolute error "
         "divided by total absolute observed load. Its denominator and valid population must match the comparator.\n"),
        ("code", "# Review Run All never runs a model, downloads weights, or trains.\n"
         "RUN_SCREEN = False\nfrom pathlib import Path\nimport json\nimport yaml\n"
         "import pandas as pd\nfrom IPython.display import display\n"
         "PROJECT = Path.cwd()\nif not (PROJECT / 'configs').is_dir():\n    PROJECT = PROJECT.parent\n"
         "protocol = yaml.safe_load((PROJECT / 'configs/timesfm_development.yaml').read_text(encoding='utf-8'))\n"
         "display(protocol)\n"),
        ("markdown", "## Pinned source and preprocessing\n"
         "The protocol pins Apache-2.0 model weights, model revision, safetensors SHA256, TimesFM 2.0.2 "
         "and its exact wheel checksum. Source: [official model card](https://huggingface.co/google/timesfm-2.5-200m-pytorch). "
         "The installed wheel returns the fixed median as point forecast; the worker asserts equality with quantile index 5. "
         "Only past DLPRB is supplied. Input normalization, positive output constraint and flip invariance are fixed before scoring. "
         "Missing input histories fail explicitly rather than silently inventing an imputation policy.\n"),
        ("markdown", "## Hosted execution and debugging\n"
         "Google Colab T4 only for this worker. Requires the canonical public-source prefix prepared at "
         "`/content/cell_seed29`, mounted Drive, and the pinned YAML at `/content/timesfm_development.yaml`. "
         "Preparation is documented in `scripts/build_colab_seeded_cell_code.py` and "
         "`scripts/run_colab_seeded_cell.py` with `CELL_PREPARE_ONLY=true`; it must not repeat seed-29 fitting. "
         "Pretrained weights and atomic per-origin caches remain on Drive. Heartbeats and completed-origin logs "
         "distinguish a live job from a stalled observer. A restart must first inspect the live job and saved protocol.\n"),
        ("code", "if RUN_SCREEN:\n    import os\n    import subprocess\n    import sys\n"
         "    assert 'COLAB_RELEASE_TAG' in os.environ, 'Hosted Colab only.'\n"
         "    assert Path('/content/drive/MyDrive').is_dir(), 'Mount Drive first.'\n"
         "    assert os.environ.get('TIMESFM_SOURCE_COMMIT'), 'Record the exact source commit.'\n"
         "    subprocess.run([sys.executable, str(PROJECT / 'scripts/run_colab_timesfm_development.py')], check=True)\n"
         "else:\n    print('Review mode: no inference, downloads or training.')\n"),
        ("markdown", "## Aggregate result review\n"
         "Only verified aggregate reports should be exported from the runtime. Per-origin caches and pretrained "
         "weights stay on Drive. No finished or accepted result is inferred merely from submission or an active heartbeat.\n"),
        ("code", "# This review remains useful while the cloud experiment is pending.\n"
         "path = PROJECT / 'reports/timesfm_development_aggregate.json'\n"
         "if path.exists():\n    result = json.loads(path.read_text(encoding='utf-8'))\n"
         "    assert result['protocol'] == protocol\n    assert not result['holdout_used']\n"
         "    frame = pd.DataFrame(result['metrics'])\n    frame['WAPE (%)'] = 100 * frame['wape']\n"
         "    display(frame)\nelse:\n    print('Aggregate result not imported. No accuracy claim yet.')\n"),
        ("markdown", "## Decision rule, negative results and limitations\n"
         "Require no horizon regression and at least 2% mean MAE gain versus the unchanged strong comparator "
         "before additional adaptation is justified. Do not select a best fold or quantile. A failed screen stays a negative result. "
         "Pretraining includes GiftEvalPretrain, Wikimedia, Google Trends and synthetic/augmented data; "
         "absence of source overlap is not certified. This is exploratory reused development, not independent generalization. "
         "No production promotion or WAPE below 10% is established by this notebook. "
         "Downloaded TimesFM 3.0 weights have different, non-production license terms and are outside this protocol.\n"),
    ]
    cells = []
    for kind, source in sections:
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            ast.parse(source)
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"},
                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    path = Path("notebooks/40_pretrained_median_forecasting.ipynb")
    if path.exists():
        raise FileExistsError("Preserve existing research notebook")
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
