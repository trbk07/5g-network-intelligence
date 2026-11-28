"""Build a readable review notebook; no fitting or data export."""

import ast
import json
from pathlib import Path


def main():
    sections = [
        ("markdown", "# 5G Network Intelligence — Target Patches and Past KPI Attention\n\n"
         "Research question: can explicit target/covariate attention improve DLPRB MAE and WAPE? "
         "This is an independently implemented TimeXer-inspired adaptation, not a reproduction. "
         "See [research notes](../docs/literature_driven_training.md) for primary sources and limitations."),
        ("code", "# Review Run All does not train. Training belongs in a dedicated hosted worker.\nRUN_TRAINING = False\n"),
        ("markdown", "## Temporal protocol and metrics\nContext: 168 observed hours. Horizons: 1/6/24h. "
         "Fit/stop/score endpoints: 335/383/443 and 395/443/503. All original hours 504 onward are excluded. "
         "MAE uses physical DLPRB usage units; WAPE = sum absolute error / sum absolute observed load. "
         "These are reused development folds, not an untouched test."),
        ("code", "from pathlib import Path\nimport yaml\nimport pandas as pd\nfrom IPython.display import display\n"
         "PROJECT = Path.cwd()\nif not (PROJECT/'configs').is_dir() and (PROJECT.parent/'configs').is_dir():\n"
         "    PROJECT = PROJECT.parent\nCONFIG = PROJECT/'configs/past_covariate_attention.yaml'\n"
         "cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8')) if CONFIG.exists() else None\n"
         "print(cfg if cfg else 'Open from the project checkout to inspect configuration.')\n"),
        ("markdown", "## Preprocessing and architecture\nFit global per-KPI medians/scales on the fit prefix only. "
         "Keep missing masks and phase4. Encode DLPRB into 12-hour patches; a global target token attends "
         "to three past-KPI tokens. Predict a correction to a causal seasonal baseline. "
         "The control receives constant covariate tokens with identical parameter capacity. "
         "No future covariates, beam identity embeddings or fitted beam summaries are inputs."),
        ("markdown", "## Training controls\nSeed 17, maximum 80 epochs, patience 12. "
         "Select checkpoints on stopping MAE only; score every declared horizon on the full population. "
         "Save cold states, scaler, best/last checkpoints, RNG and provenance in hosted storage. "
         "Implementation: `src/models/past_covariate_attention.py`; controller: "
         "`scripts/train_past_covariate_attention.py`. Review does not repeat fitting."),
        ("code", "if RUN_TRAINING:\n    raise RuntimeError('Use a separately pinned hosted training worker; review never launches fitting.')\n"
         "print('Review mode: no fitting.')\n"),
        ("markdown", "## Evaluation and debugging\nInspect both arms and all fold/horizon rows. "
         "Do not call label availability or parameter count an accuracy gain. "
         "Before accepting results, independently reconstruct metrics, verify preprocessing/cold states "
         "and replay forecasts with a fixed numerical tolerance."),
        ("code", "# Read completed reports only. Missing output means the experiment is not yet evaluated.\n"
         "for case in ['no_covariates', 'past_covariates']:\n"
         "    report = PROJECT/'reports'/f'past_covariate_attention_{case}_development.csv'\n"
         "    if report.exists():\n        frame = pd.read_csv(report)\n"
         "        frame['WAPE (%)'] = frame['wape']*100\n        print(case)\n        display(frame)\n"
         "    else:\n        print(case, ': no completed report available')\n"),
        ("markdown", "## Decision and limitations\nInitial screen: at least 2% mean MAE improvement against "
         "the fixed strong control without fold/horizon regression. A 5% material gain still needs uncertainty "
         "and robustness evidence. WAPE below 10% remains unproven. The prior Telecom data/budget study "
         "failed its quality gates; this new architectural hypothesis may also fail. "
         "No holdout evaluation or production promotion follows from this notebook."),
    ]
    cells = []
    for kind, source in sections:
        cell = dict(cell_type=kind, id=f'past-kpi-{len(cells):02d}', metadata={},
                    source=source.splitlines(keepends=True))
        if kind == 'code':
            ast.parse(source)
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    target = Path('notebooks/44_target_patches_and_past_kpi_attention.ipynb')
    if target.exists():
        raise FileExistsError('Preserve the existing guide')
    target.write_text(json.dumps(dict(cells=cells, nbformat=4, nbformat_minor=5,
        metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}}),
        ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
