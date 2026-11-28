"""Independent in-place attention verification on Colab; aggregate export only."""

import hashlib
import json
import os
from pathlib import Path
import sys


def verify_tree(root):
    files = json.loads((root/'manifest.json').read_text())['files']
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != set(files) | {'manifest.json'}:
        raise ValueError('Artifact coverage differs')
    for name, digest in files.items():
        path = root/name
        if (path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
            raise ValueError('Artifact path/checksum differs')


def verify_checkpoint_identity(checkpoint, cfg, fold, case):
    ignored = {'max_epochs', 'variants', 'seeds', 'artifact_dir', 'gpu_budget_seconds'}
    expected = {'seed': cfg['seed'], 'config': {k: v for k, v in cfg.items() if k not in ignored},
                'run_identity': {'architecture': 'TimeXer-inspired-independent-adaptation/'+case,
                                 'fold': fold, 'variant': 'B'}}
    if checkpoint['identity'] != expected:
        raise ValueError('Checkpoint training identity differs')


def main():
    if os.name == 'nt' or 'COLAB_RELEASE_TAG' not in os.environ:
        raise RuntimeError('Verify weights in hosted storage; no local checkpoint transfer')
    project = Path('/content/drive/MyDrive/5g-network-intelligence/checkpoints/past_kpi_0062eaf296c3')
    state = json.loads((project/'runtime_state.json').read_text())
    if state['status'] != 'complete' or state['exit_code'] != 0:
        raise RuntimeError('All declared fits must finish before independent verification')
    source_path = project/'past_covariate_source_manifest.json'
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != 'b73dfd010acc2bb2794519a5fece5db1cd0b92af4ecff65d82c278f414707707':
        raise ValueError('Pinned source manifest differs')
    source = json.loads(source_path.read_text())
    if source['source_commit'] != state['source_commit']:
        raise ValueError('Source commit differs')
    for name, digest in source['files'].items():
        if hashlib.sha256((project/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Training source checksum differs')
    os.chdir(project)
    sys.path.insert(0, str(project))
    import numpy as np
    import pandas as pd
    import torch
    from src.data.io import sha256
    from src.data.panel import load_panel
    from src.evaluation.origins import evaluation_origins, train_origins_joint
    from src.features.gru_inputs import GRUPreprocessor
    from src.models.cell_attention import configure_attention_math
    from src.models.past_covariate_attention import PastCovariateAttention
    from src.utils.config import load_config

    torch.set_num_threads(2)
    configure_attention_math()
    root = Path('artifacts/past_covariate_attention')
    verify_tree(root)
    cfg = load_config('configs/past_covariate_attention.yaml')
    if cfg['cases'] != ['no_covariates', 'past_covariates'] or cfg['variants'] != ['B']:
        raise ValueError('Declared paired study differs')
    panel = load_panel('data/processed/panel.npz')
    if panel.values.shape != (840, 2880, 4) or not np.isnan(panel.values[504:]).all():
        raise ValueError('Canonical redacted panel required')
    if hashlib.sha256(panel.values[:504].tobytes()).hexdigest() != 'c0f9cbe86244df0f9658ba4c3fb541183e62c613328f1aeb7c41ebb8d09df661':
        raise ValueError('Canonical development values differ')
    cold_controls, rows, replay, stopping, diagnostics = {}, [], [], [], []
    beam_indices = np.linspace(0, 2879, 24, dtype=int)
    for case in cfg['cases']:
        verify_tree(root/case)
        case_rows = []
        for fold in cfg['folds']:
            directory = root/case/fold['name']/'B'
            meta = json.loads((directory/'meta.json').read_text())
            adaptation = json.loads((directory/'adaptation.json').read_text())
            if (meta['protocol'] != cfg or meta['fold'] != fold or meta['variant'] != 'B'
                    or meta['source_commit'] != state['source_commit'] or meta['holdout_used']
                    or meta['validation_504_671_used'] or adaptation['future_covariates_used']
                    or adaptation['source_commit'] != state['source_commit']
                    or adaptation['target_scale_fit_end'] != fold['fit_end']
                    or adaptation['case'] != case or adaptation['gpu_attention_fastpath'] is not False):
                raise ValueError('Protocol or causal scope differs')
            pre = GRUPreprocessor.load(directory/'preprocessor.npz')
            expected = GRUPreprocessor.fit(panel, fold['fit_end'], 'B')
            if pre.fit_end != fold['fit_end'] or pre.beam_ids != panel.beam_ids or pre.variant != 'B':
                raise ValueError('Scaler cutoff/axes differ')
            for key in ('median', 'mean', 'scale'):
                np.testing.assert_allclose(getattr(pre, key), getattr(expected, key), atol=1e-12, rtol=1e-12)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(cfg['seed'])
                model = PastCovariateAttention(cfg, case == 'past_covariates')
                model.configure_target_scale(pre.mean[0], pre.scale[0]).eval()
            cold = torch.load(directory/'cold_initialization.pt', weights_only=True, map_location='cpu')
            if (cold['seed'] != cfg['seed'] or cold['torch_initial_seed'] != cfg['seed']
                    or cold['case'] != case or cold['fold'] != fold
                    or adaptation['cold_initialization_sha256'] != sha256(directory/'cold_initialization.pt')
                    or meta['parameters'] != sum(p.numel() for p in model.parameters())
                    or set(cold['model']) != set(model.state_dict())
                    or any(not torch.equal(v, cold['model'][k]) for k, v in model.state_dict().items())):
                raise ValueError('Seeded cold state differs bitwise')
            previous = cold_controls.setdefault(fold['name'], cold['model'])
            if any(not torch.equal(v, previous[k]) for k, v in cold['model'].items()):
                raise ValueError('Cases have unequal cold states')
            best = torch.load(directory/'best.pt', weights_only=True, map_location='cpu')
            last = torch.load(directory/'last.pt', weights_only=True, map_location='cpu')
            verify_checkpoint_identity(best, cfg, fold, case)
            verify_checkpoint_identity(last, cfg, fold, case)
            history = [json.loads(line) for line in (directory/'history.jsonl').read_text().splitlines()]
            if not history or [x['epoch'] for x in history] != list(range(len(history))):
                raise ValueError('Epoch coverage differs')
            chosen = min(history, key=lambda row: (row['val_l1_scaled'], row['epoch']))['epoch']
            summary = meta['training']
            if (best['epoch'] != chosen or last['epoch'] != len(history)-1
                    or summary['best_epoch'] != chosen or summary['epochs_completed'] != len(history)
                    or summary['budget_exhausted'] or not summary['early_stopping_completed']
                    or last['stale'] != last['epoch']-chosen or last['best_epoch'] != chosen
                    or best['best'] != history[chosen]['val_l1_scaled']):
                raise ValueError('Stopping choice or genuine completion differs')
            training_origins = train_origins_joint(fold['fit_end'], stride=cfg['training_stride'])
            stopping_origins = evaluation_origins(fold['fit_end'], fold['stop_end'], 24)
            train_count = sum(int(np.isfinite(panel.values[training_origins+h, :, 0]).sum()) for h in cfg['horizons'])
            stop_count = sum(int(np.isfinite(panel.values[stopping_origins+h, :, 0]).sum()) for h in cfg['horizons'])
            if any(row['train_valid_targets'] != train_count or row['validation_valid_targets'] != stop_count for row in history):
                raise ValueError('Supervision population differs')
            updates = len(history)*int(np.ceil(len(training_origins)*2880/cfg['batch_size']))
            steps = [int(v['step']) for v in last['optimizer']['state'].values() if 'step' in v]
            if not steps or set(steps) != {updates}:
                raise ValueError('Actual optimizer exposure differs')
            stopping.append({'case': case, 'fold': fold['name'], 'epochs': len(history),
                             'best_epoch': chosen, 'optimizer_updates': updates})
            model.load_state_dict(best['model'])
            for j, h in enumerate(cfg['horizons']):
                with np.load(directory/f'score_h{h}.npz', allow_pickle=False) as saved:
                    origins = evaluation_origins(fold['stop_end'], fold['score_end'], h)
                    actual = panel.values[origins+h, :, 0].reshape(-1)
                    np.testing.assert_array_equal(saved['origins'], origins)
                    np.testing.assert_array_equal(saved['beam_ids'], panel.beam_ids)
                    np.testing.assert_array_equal(saved['actual'], actual)
                    predicted = saved['predicted']
                    if predicted.shape != actual.shape or not np.isfinite(predicted).all() or (predicted < 0).any():
                        raise ValueError('Forecast coverage/constraint differs')
                    mask = np.isfinite(actual)
                    error = np.abs(actual[mask]-predicted[mask])
                    zero = mask & (actual == 0)
                    # Target-conditioned groups explain error only. They cannot
                    # become inference-time routing labels or achievable floors.
                    positive = mask & (actual > 0)
                    # Forecasts are float32; accumulate in float64 so the
                    # accounting check uses the same precision as target errors.
                    zero_error = float(np.abs(predicted[zero]).sum(dtype=np.float64))
                    positive_error = float(np.abs(actual[positive]-predicted[positive]).sum())
                    np.testing.assert_allclose(zero_error+positive_error, error.sum(), rtol=1e-12, atol=1e-12)
                    diagnostics.append({'case': case, 'fold': fold['name'], 'horizon': h,
                        'n_valid': int(mask.sum()), 'zero_target_fraction': float(zero.sum()/mask.sum()),
                        'zero_absolute_error': zero_error, 'positive_absolute_error': positive_error,
                        'zero_share_of_absolute_error': zero_error/float(error.sum()) if error.sum() else 0.,
                        'zero_conditional_mae': float(np.abs(predicted[zero]).mean(dtype=np.float64)) if zero.any() else None,
                        'positive_conditional_mae': float(np.abs(actual[positive]-predicted[positive]).mean()) if positive.any() else None,
                        'positive_signed_bias': float((predicted[positive]-actual[positive]).mean()) if positive.any() else None,
                        'wape_if_only_zero_errors_were_perfect': positive_error/float(np.abs(actual[mask]).sum())})
                    case_rows.append({'fold': fold['name'], 'variant': 'B', 'horizon': h,
                        'mae': float(error.mean()), 'wape': float(error.sum()/np.abs(actual[mask]).sum()),
                        'zero_fp_rate': float((predicted[zero] > 0).mean()), 'n_valid': int(mask.sum())})
                    for origin in (fold['stop_end'], fold['score_end']-24):
                        values = pre.transform(panel.values[origin-167:origin+1], 'B')
                        inputs = torch.as_tensor(values[:, beam_indices].transpose(1, 0, 2))
                        phase = torch.as_tensor(pre.static_at(np.full(24, origin), beam_indices))
                        with torch.no_grad():
                            cpu = model.predict_physical(inputs, phase).numpy()[:, j]
                        gpu = predicted.reshape(len(origins), 2880)[origin-origins[0], beam_indices]
                        np.testing.assert_allclose(cpu, gpu, atol=1e-4, rtol=0)
                        replay.append(float(np.abs(cpu-gpu).max()))
            pd_frame = pd.DataFrame(case_rows)
        pd.testing.assert_frame_equal(pd_frame, pd.read_csv(f'reports/past_covariate_attention_{case}_development.csv'),
                                      check_dtype=False, atol=1e-12, rtol=1e-12)
        rows.extend({'case': case, **row} for row in case_rows)
    proof = {'all_metrics_reconstructed': True, 'matched_seeded_cold_states_verified': True,
             'stopping_and_optimizer_exposure_verified': True, 'maximum_cpu_gpu_error': max(replay),
             'history_only_predictions_checked': len(replay)*24, 'termination': stopping,
             'source_commit': state['source_commit'], 'training_performed_by_audit': False,
             'holdout_used': False, 'production_promoted': False, 'metrics': rows,
             'error_decomposition': diagnostics,
             'diagnostic_scope': 'target-conditioned accounting only; no routing labels or irreducible-error claim',
             'scope': 'reused development; quality assessment separate; weights remain on Drive'}
    output = project/'reports/past_covariate_attention_verified_metrics.json'
    output.write_text(json.dumps(proof, indent=2))
    print(json.dumps({k: v for k, v in proof.items() if k not in ('metrics', 'error_decomposition')}, indent=2))


if __name__ == '__main__':
    main()
