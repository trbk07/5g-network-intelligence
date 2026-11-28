"""Prepare the Colab failover from pinned public data; never launches training."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import urllib.request
import zipfile


def main():
    if os.name == 'nt' or 'COLAB_RELEASE_TAG' not in os.environ or not Path('/content').is_dir():
        raise RuntimeError('Hosted Colab only')
    drive = Path('/content/drive/MyDrive')
    if not drive.is_dir():
        raise RuntimeError('Mounted Drive required')
    code_path = Path('/content/past_covariate_code.zip')
    if hashlib.sha256(code_path.read_bytes()).hexdigest() != os.environ['PAST_KPI_CODE_SHA256']:
        raise ValueError('Code archive differs')
    with zipfile.ZipFile(code_path) as archive:
        manifest = json.loads(archive.read('past_covariate_source_manifest.json'))
        members = archive.infolist()
        names = [p.filename for p in members]
        if len(names) != len(set(names)) or set(names) != set(manifest['files']) | {'past_covariate_source_manifest.json'}:
            raise ValueError('Code member coverage differs')
        for member in members:
            name = PurePosixPath(member.filename)
            if (name.is_absolute() or '..' in name.parts or ':' in member.filename or '\\' in member.filename
                    or member.is_dir() or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError('Unsafe archive path')
        for name, digest in manifest['files'].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError('Code member checksum differs')
        project = drive/'5g-network-intelligence/checkpoints'/('past_kpi_'+manifest['source_commit'][:12])
        if (project/'artifacts/past_covariate_attention').exists():
            raise FileExistsError('Inspect persisted training before preparing again')
        project.mkdir(parents=True, exist_ok=True)
        archive.extractall(project)  # Exact paths and member hashes checked above.
    os.chdir(project)
    sys.path.insert(0, str(project))
    os.environ['SOURCE_GIT_COMMIT'] = manifest['source_commit']
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'numpy', 'pandas',
                    'pyyaml', 'lightgbm>=4.6,<5', 'scikit-learn'], check=True)
    import numpy as np
    import torch
    from src.data.io import sha256
    from src.data.panel import build_panel, save_panel
    if not torch.cuda.is_available():
        raise RuntimeError('Hosted CUDA required')
    public_path = Path('/content/public_sources.json')
    if sha256(public_path) != os.environ['PAST_KPI_PUBLIC_MANIFEST_SHA256']:
        raise ValueError('Public input manifest differs')
    public = json.loads(public_path.read_text())
    paths = {}
    for kpi, item in public['train_files'].items():
        target = project/'data/raw'/item['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or sha256(target) != item['sha256']:
            partial = target.with_suffix('.part')
            print(json.dumps({'downloading_public_train_file': item['path']}), flush=True)
            with urllib.request.urlopen(item['url'], timeout=120) as response, partial.open('wb') as output:
                shutil.copyfileobj(response, output, length=8*1024**2)
            if sha256(partial) != item['sha256']:
                raise ValueError('Publisher input checksum differs')
            partial.replace(target)
        paths[kpi] = target
    panel = build_panel(paths, {'dataset_revision': public['revision'], 'license': 'mit', 'segment': 'train'})
    prefix = hashlib.sha256(panel.values[:504].tobytes()).hexdigest()
    if list(panel.values.shape) != [840, 2880, 4] or prefix != public['prefix_values_sha256']:
        raise ValueError('Public panel differs from the canonical development prefix')
    panel.values[504:] = np.nan
    processed = project/'data/processed'
    processed.mkdir(parents=True, exist_ok=True)
    save_panel(panel, processed/'panel.npz')
    (processed/'split.json').write_text(json.dumps(public['frozen_split'], indent=2))
    state = {'status': 'prepared_not_training', 'source_commit': manifest['source_commit'],
             'project': str(project), 'gpu': torch.cuda.get_device_name(0), 'torch': str(torch.__version__),
             'prefix_matches_canonical': True, 'values_504_onward_redacted': True,
             'local_beam_payload_uploaded': False, 'training_started': False,
             'launch_condition': 'Kaggle duplicate authoritatively cancelled before any Colab fitting'}
    (project/'runtime_state.json').write_text(json.dumps(state, indent=2))
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
