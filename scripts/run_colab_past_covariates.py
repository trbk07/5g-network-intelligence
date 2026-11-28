"""Launch the prepared paired attention study; persist all outputs on Drive."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading


def main():
    if os.name == 'nt' or 'COLAB_RELEASE_TAG' not in os.environ:
        raise RuntimeError('Hosted Colab only; local fitting forbidden')
    if os.environ.get('KAGGLE_ATTENTION_STATUS') != 'CANCEL_ACKNOWLEDGED':
        raise RuntimeError('Orchestrator must confirm queued duplicate cancellation first')
    project = Path('/content/drive/MyDrive/5g-network-intelligence/checkpoints/past_kpi_0062eaf296c3')
    state_path = project/'runtime_state.json'
    state = json.loads(state_path.read_text())
    if state['status'] != 'prepared_not_training' or state['training_started']:
        raise RuntimeError('Inspect existing execution before attempting any repeat fit')
    manifest = json.loads((project/'past_covariate_source_manifest.json').read_text())
    if manifest['source_commit'] != state['source_commit']:
        raise ValueError('Prepared source identity differs')
    for name, digest in manifest['files'].items():
        if hashlib.sha256((project/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Prepared source checksum differs: '+name)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Hosted CUDA required')
    if (project/'artifacts/past_covariate_attention').exists():
        raise FileExistsError('Inspect checkpoints before any repeat training')
    (project/'reports').mkdir(exist_ok=True)
    os.environ['SOURCE_GIT_COMMIT'] = manifest['source_commit']
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    finished = threading.Event()

    def heartbeat():
        while not finished.wait(20):
            print('[Colab] Attention study active; logs/checkpoints on Drive.', flush=True)

    threading.Thread(target=heartbeat, daemon=True).start()
    # Match the strict attention math used in the earlier verified cell study.
    command = ('import torch,runpy; torch.backends.cuda.enable_flash_sdp(False); '
               'torch.backends.cuda.enable_mem_efficient_sdp(False); '
               'runpy.run_module("scripts.train_past_covariate_attention",run_name="__main__")')
    try:
        with (project/'reports/past_covariate_training.log').open('w') as log:
            process = subprocess.Popen([sys.executable, '-u', '-c', command], cwd=project,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            state.update(status='running', training_started=True, training_pid=process.pid,
                         kaggle_duplicate_status='CANCEL_ACKNOWLEDGED',
                         attention_math='flash_sdp=False; mem_efficient_sdp=False; MHA_fastpath=False')
            state_path.write_text(json.dumps(state, indent=2))
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            state.update(status='complete' if process.wait() == 0 else 'failed', exit_code=process.returncode)
            state_path.write_text(json.dumps(state, indent=2))
        if process.returncode:
            raise RuntimeError('Inspect persisted logs/checkpoints; do not blindly repeat fitting')
    finally:
        finished.set()


if __name__ == '__main__':
    main()
