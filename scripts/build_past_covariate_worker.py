"""Package only committed code and reuse the existing approved hosted data input."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    stage = Path('.cache/past-covariate-code')
    worker = Path('.cache/kaggle-past-covariate')
    stage.mkdir(parents=True, exist_ok=True)
    worker.mkdir(parents=True, exist_ok=True)
    files = {p.as_posix(): p for p in Path('src').rglob('*.py')}
    for name in ('__init__.py', 'train_past_covariate_attention.py',
                 'train_multisource_neural.py', 'train_temporal_mlp.py', 'train_temporal_linear.py'):
        files['scripts/'+name] = Path('scripts')/name
    for name in ('data.yaml', 'evaluation.yaml', 'past_covariate_attention.yaml'):
        files['configs/'+name] = Path('configs')/name
    files['LICENSE'] = Path('LICENSE')
    commit = subprocess.check_output(['git', '-c', 'safe.directory=D:/5G', 'rev-parse', 'HEAD'], text=True).strip()
    # Refuse to label modified implementation files as a committed source snapshot.
    for name, path in files.items():
        committed = subprocess.check_output(['git', '-c', 'safe.directory=D:/5G', 'show', f'{commit}:{name}'])
        if committed.replace(b'\r\n', b'\n') != path.read_bytes().replace(b'\r\n', b'\n'):
            raise ValueError(f'Uncommitted source: {name}')
    manifest = {'source_commit': commit, 'data_or_checkpoint_included': False,
                'files': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}}
    raw = json.dumps(manifest, indent=2).encode()
    digest = hashlib.sha256(raw).hexdigest()
    with zipfile.ZipFile(stage/'past_covariate_code.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr('past_covariate_source_manifest.json', raw)
    dataset = 'truongbaokien/5g-ni-past-covariate-code'
    (stage/'dataset-metadata.json').write_text(json.dumps({'id': dataset,
        'title': '5G NI Past Covariate Code', 'licenses': [{'name': 'MIT'}]}, indent=2))
    notebook = json.loads(Path('notebooks/44_target_patches_and_past_kpi_attention.ipynb').read_text(encoding='utf-8'))
    reference = json.loads(Path('notebooks/41_telecom_data_and_pretraining_budget_ablation.ipynb').read_text(encoding='utf-8'))
    setup = ''.join(reference['cells'][3]['source'])
    ingress = f"""if RUN_TRAINING:
    assert Path('/kaggle/working').is_dir(), 'Hosted Kaggle only.'
    base_root, base = verified_input('source_manifest.json',
        '3069c57c9f64f809464ead2fae973fbaa35a86c3a46884e15cc693354eecab46')
    code_root, code = verified_input('past_covariate_source_manifest.json', '{digest}')
    assert not code['data_or_checkpoint_included']
    PROJECT = Path('/kaggle/working/past_covariate_study')
    PROJECT.mkdir(exist_ok=True)
    for name in ['data/processed/panel.npz', 'data/processed/panel_meta.json', 'data/processed/split.json']:
        target = PROJECT/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base_root/name, target)
    for name in code['files']:
        target = PROJECT/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(code_root/name, target)
    os.chdir(PROJECT)
    sys.path.insert(0, str(PROJECT))
    os.environ['SOURCE_GIT_COMMIT'] = code['source_commit']
    print('Verified source:', code['source_commit'])
"""
    def cell(source, identity):
        ast.parse(source)
        return dict(cell_type='code', id=identity, metadata={}, source=source.splitlines(keepends=True),
                    execution_count=None, outputs=[])
    notebook['cells'][1] = cell('RUN_TRAINING = True\n', 'hosted-worker-only')
    notebook['cells'][3:3] = [cell(setup, 'verify-input-function'), cell(ingress, 'verified-input-ingress')]
    for item in notebook['cells']:
        if item['cell_type'] == 'code' and 'raise RuntimeError(\'Use a separately pinned' in ''.join(item['source']):
            item.update(cell("""if RUN_TRAINING:
    import subprocess
    (PROJECT/'reports').mkdir(exist_ok=True)
    with (PROJECT/'reports/past_covariate_training.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.train_past_covariate_attention'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        exit_code = process.wait()
    if exit_code:
        raise RuntimeError('Inspect durable logs before any repeat fit.')
""", 'hosted-training'))
    name = '44_target_patches_and_past_kpi_attention.ipynb'
    (worker/name).write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
    (worker/'kernel-metadata.json').write_text(json.dumps({
        'id': 'truongbaokien/5g-ni-target-patches-and-past-kpi-attention',
        'title': '5G NI Target Patches and Past KPI Attention', 'code_file': name,
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
        'enable_gpu': True, 'enable_internet': False,
        'dataset_sources': ['truongbaokien/5g-ni-cell-multisource-v2-inputs', dataset],
        'kernel_sources': [], 'competition_sources': []}, indent=2))
    transport = {'source_commit': commit, 'manifest_sha256': digest, 'files': len(files),
                 'data_or_checkpoint_included': False, 'training_performed': False}
    (stage/'transport.json').write_text(json.dumps(transport, indent=2))
    print(json.dumps(transport, indent=2))


if __name__ == '__main__':
    main()
