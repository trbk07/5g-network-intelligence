"""Prepare a hosted inference-only audit; submit only after training completes."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    commit = subprocess.check_output(['git', '-c', 'safe.directory=D:/5G', 'rev-parse', 'HEAD'], text=True).strip()
    relative = 'scripts/audit_partial_supervision_cloud.py'
    source = subprocess.check_output(['git', '-c', 'safe.directory=D:/5G', 'show', f'{commit}:{relative}'])
    digest = hashlib.sha256(source).hexdigest()
    reference = json.loads(Path('notebooks/43_available_target_supervision.ipynb').read_text(encoding='utf-8'))
    setup = ''.join(reference['cells'][3]['source'])
    ingress = ''.join(reference['cells'][5]['source']).replace('RUN_TRAINING', 'RUN_AUDIT').replace(
        '/kaggle/working/partial_supervision_study', '/kaggle/working/partial_supervision_audit')
    run = f"""if RUN_AUDIT:
    import urllib.request, subprocess
    # Retrieve public verifier code by immutable commit and exact content hash.
    url = 'https://raw.githubusercontent.com/trbk07/5g-network-intelligence/{commit}/{relative}'
    with urllib.request.urlopen(url, timeout=60) as response:
        code = response.read()
    assert hashlib.sha256(code).hexdigest() == '{digest}'
    script = PROJECT/'audit_partial_supervision_cloud.py'
    script.write_bytes(code)
    roots = [p.parent for p in Path('/kaggle/input').rglob('manifest.json')
             if p.parent.name == 'partial_supervision'
             and all((p.parent/arm).is_dir() for arm in ['joint', 'available'])]
    assert len(roots) == 1, 'Attach exactly one completed paired experiment'
    subprocess.run([sys.executable, str(script), '--artifacts', str(roots[0]),
        '--source-commit', 'a2d7ff0896d897942770473871289a66c612e8db'], check=True)
else:
    print('Review mode: no fitting or checkpoint access.')
"""
    sections = [
        ('markdown', '# 5G Network Intelligence — Available Supervision Integrity and Replay\n\n'
         'Verify the completed joint/available-label experiment independently. This audit never trains or calibrates.'),
        ('code', '# Default review does not access checkpoints or run inference.\nRUN_AUDIT = False\n'),
        ('markdown', '## Provenance and temporal boundary\nVerify all source/input checksums. '
         'Keep checkpoints in attached Kaggle Input; output aggregate evidence only. '
         'Use original fit/stop/score endpoints 335/383/443 and 395/443/503, context 168h, '
         'horizons 1/6/24h. Original hours 504 onward remain redacted.'),
        ('code', setup), ('code', ingress),
        ('markdown', '## Independent acceptance checks\nVerify exact artifact coverage, fit-only preprocessing, '
         'seed-reconstructed cold weights and bitwise equality across arms. Recount observed labels per horizon, '
         'check epoch training/stopping counts and actual optimizer steps. Reconstruct all saved MAE/WAPE '
         'from the full population and replay history-only forecasts at fixed atol 0.0001. '
         'Require genuine patience stopping; an epoch cap does not pass this audit.'),
        ('code', run),
        ('markdown', '## Metrics and limitations\nMAE uses physical DLPRB units; WAPE divides summed absolute '
         'error by summed absolute observed load. Natural window counts and optimizer exposure differ between '
         'arms, so this is not a pure label-count effect. Reused development does not establish fresh '
         'generalization. Integrity PASS alone is not a quality gain or permission to promote a model.'),
        ('code', "import pandas as pd\nfrom IPython.display import display\n"
         "result_path = Path('/kaggle/working/partial_supervision_verified_metrics.json') if RUN_AUDIT else PROJECT/'reports/partial_supervision_verified_metrics.json'\n"
         "if result_path.exists():\n    result = json.loads(result_path.read_text(encoding='utf-8'))\n"
         "    frame = pd.DataFrame(result['metrics'])\n    frame['WAPE (%)'] = 100*frame['wape']\n"
         "    display(frame)\n    display(pd.DataFrame(result['exposure']))\n"
         "else:\n    print('Independent verification pending.')\n"),
    ]
    cells = []
    for kind, source in sections:
        cell = dict(cell_type=kind, id=f'partial-audit-{len(cells):02d}', metadata={}, source=source.splitlines(keepends=True))
        if kind == 'code':
            ast.parse(source)
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = dict(cells=cells, nbformat=4, nbformat_minor=5,
        metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}})
    name = '45_available_supervision_integrity_and_replay.ipynb'
    review = Path('notebooks')/name
    if review.exists():
        raise FileExistsError('Preserve existing review guide')
    review.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
    notebook['cells'][1]['source'] = ['# Dedicated hosted inference-only worker.\nRUN_AUDIT = True\n']
    stage = Path('.cache/kaggle-partial-audit')
    stage.mkdir(parents=True, exist_ok=True)
    (stage/name).write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
    (stage/'kernel-metadata.json').write_text(json.dumps({
        'id': 'truongbaokien/5g-ni-available-supervision-integrity-and-replay',
        'title': '5G NI Available Supervision Integrity and Replay', 'code_file': name,
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
        'enable_gpu': False, 'enable_internet': True,
        'dataset_sources': ['truongbaokien/5g-ni-cell-multisource-v2-inputs', 'truongbaokien/5g-ni-partial-supervision-code'],
        'kernel_sources': ['truongbaokien/5g-ni-available-target-supervision'], 'competition_sources': []}, indent=2))
    print(json.dumps({'audit_source_commit': commit, 'audit_source_sha256': digest,
                      'training_performed': False, 'submission_condition': 'training COMPLETE'}, indent=2))


if __name__ == '__main__':
    main()
