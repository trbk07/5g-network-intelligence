"""Reconstruct the paired partial-label experiment in Kaggle; never fit models."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.configure_partial_supervision import protocols
from scripts.train_partial_supervision import supervision_counts
from src.data.io import sha256
from src.data.panel import load_panel
from src.evaluation.origins import evaluation_origins, train_origins, train_origins_joint
from src.features.cell_inputs import cell_groups
from src.features.gru_inputs import GRUPreprocessor
from src.models.cell_attention import CellPatchAttention, configure_attention_math
from src.models.initialization_replay import verify_cold_state
from src.models.training_status import stopping_status
from src.utils.config import load_config


def verify_manifest(root):
    root = Path(root)
    files = json.loads((root/'manifest.json').read_text())['files']
    for name, digest in files.items():
        path = root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or sha256(path) != digest:
            raise ValueError('Artifact path or checksum differs')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != set(files) | {'manifest.json'}:
        raise ValueError('Artifact manifest coverage differs')


def verify_history(history, best, last, summary, cfg, counts, stopping_count):
    """Check stopping, loss/count evidence and actual optimizer exposure."""
    if not history or [x['epoch'] for x in history] != list(range(len(history))):
        raise ValueError('Epoch sequence incomplete')
    chosen = min(history, key=lambda x: (x['val_l1_scaled'], x['epoch']))['epoch']
    status = stopping_status(summary, cfg)
    if (best['epoch'] != chosen or last['epoch'] != len(history)-1
            or summary['best_epoch'] != chosen or summary['epochs_completed'] != len(history)
            or summary['budget_exhausted'] or not status['early_stopping_completed']
            or last['best_epoch'] != chosen or last['stale'] != last['epoch']-chosen):
        raise ValueError('Stopping/termination evidence differs')
    total = sum(counts['horizon_valid_labels'].values())
    for row in history:
        if (row['train_valid_targets'] != total or row['validation_valid_targets'] != stopping_count
                or row['training_objective'] != cfg['training_objective']
                or not np.isfinite([row['train_objective'], row['val_l1_scaled']]).all()):
            raise ValueError('Loss or supervision counts differ')
    updates = len(history)*int(np.ceil(counts['training_cell_windows']/cfg['batch_size']))
    actual_steps = [int(v['step']) for v in last['optimizer']['state'].values() if 'step' in v]
    if not actual_steps or set(actual_steps) != {updates}:
        raise ValueError('Actual optimizer exposure differs')
    if best['best'] != history[chosen]['val_l1_scaled'] or last['best'] != best['best']:
        raise ValueError('Saved best stopping metric differs')
    return updates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('Verify in Kaggle; checkpoints must not be transferred locally')
    torch.set_num_threads(2)
    configure_attention_math()
    verify_manifest(args.artifacts)
    panel = load_panel('data/processed/panel.npz')
    if panel.values.shape != (840, 2880, 4) or not np.isnan(panel.values[504:]).all():
        raise ValueError('Canonical redacted panel required')
    declared = protocols(load_config('configs/cell_attention_v2.yaml'))
    groups = cell_groups(panel.beam_ids)[[0, 45, 89]]
    cold_controls, rows, parity, exposure = {}, [], [], []
    for arm in ('joint', 'available'):
        cfg = load_config(f'configs/partial_supervision_{arm}.yaml')
        if cfg != declared[arm]:
            raise ValueError('Declared paired protocol differs')
        verify_manifest(args.artifacts/arm)
        arm_rows = []
        for fold in cfg['folds']:
            directory = args.artifacts/arm/fold['name']/'cell'
            meta = json.loads((directory/'meta.json').read_text())
            if (meta['protocol'] != cfg or meta['fold'] != fold or meta['mode'] != 'cell'
                    or meta['source_commit'] != args.source_commit or not meta['training_complete']
                    or meta['holdout_used'] or meta['validation_504_671_used']
                    or meta['gpu_attention_fastpath'] is not False):
                raise ValueError('Training identity differs')
            pre = GRUPreprocessor.load(directory/'preprocessor.npz')
            expected = GRUPreprocessor.fit(panel, fold['fit_end'], 'B')
            if pre.fit_end != expected.fit_end or pre.beam_ids != expected.beam_ids or pre.variant != 'B':
                raise ValueError('Scaler cutoff/axes differ')
            for key in ('median', 'mean', 'scale'):
                np.testing.assert_allclose(getattr(pre, key), getattr(expected, key), atol=1e-12, rtol=1e-12)
            model = CellPatchAttention(cfg, pre.mean[0], pre.scale[0], 'cell').eval()
            cold = torch.load(directory/'cold_initialization.pt', weights_only=True, map_location='cpu')
            if (cold['seed'] != cfg['seed'] or cold['torch_initial_seed'] != cfg['seed']
                    or meta['torch_initial_seed'] != cfg['seed'] or cold['fold'] != fold or cold['mode'] != 'cell'
                    or meta['cold_initialization_sha256'] != sha256(directory/'cold_initialization.pt')
                    or meta['parameters'] != sum(p.numel() for p in model.parameters())):
                raise ValueError('Cold initialization identity differs')
            verify_cold_state(lambda: CellPatchAttention(cfg, pre.mean[0], pre.scale[0], 'cell'), cfg['seed'], cold['model'])
            previous = cold_controls.setdefault(fold['name'], cold['model'])
            if set(previous) != set(cold['model']) or any(not torch.equal(previous[k], v) for k, v in cold['model'].items()):
                raise ValueError('Arms have unequal cold states')
            counts = json.loads((directory/'supervision_counts.json').read_text())
            origins = train_origins_joint(fold['fit_end']) if arm == 'joint' else train_origins(fold['fit_end'], 1)
            n_cells = len(cell_groups(panel.beam_ids))
            if (counts['arm'] != arm or counts['fit_end'] != fold['fit_end']
                    or counts['horizon_valid_labels'] != supervision_counts(panel, origins, fold['fit_end'])
                    or counts['training_cell_windows'] != len(origins)*n_cells
                    or counts['training_objective'] != 'mean_of_observed_per_horizon_batch_MAEs'
                    or counts['stopping_policy'] != 'unchanged_complete_joint_windows'):
                raise ValueError('Available supervision differs')
            stopping_origins = evaluation_origins(fold['fit_end'], fold['stop_end'], 24)
            stopping_count = sum(int(np.isfinite(panel.values[stopping_origins+h, :, 0]).sum()) for h in (1, 6, 24))
            best = torch.load(directory/'best.pt', weights_only=True, map_location='cpu')
            last = torch.load(directory/'last.pt', weights_only=True, map_location='cpu')
            history = [json.loads(line) for line in (directory/'history.jsonl').read_text().splitlines()]
            updates = verify_history(history, best, last, meta['training'], cfg, counts, stopping_count)
            exposure.append({'arm': arm, 'fold': fold['name'], 'optimizer_updates': updates,
                             'epochs': len(history), **counts})
            model.load_state_dict(best['model'])
            for j, h in enumerate((1, 6, 24)):
                with np.load(directory/f'score_h{h}.npz', allow_pickle=False) as cache:
                    origins = evaluation_origins(fold['stop_end'], fold['score_end'], h)
                    actual = panel.values[origins+h, :, 0].reshape(-1)
                    np.testing.assert_array_equal(cache['origins'], origins)
                    np.testing.assert_array_equal(cache['beam_ids'], panel.beam_ids)
                    np.testing.assert_array_equal(cache['actual'], actual)
                    predicted = cache['predicted']
                    if predicted.shape != actual.shape or not np.isfinite(predicted).all() or (predicted < 0).any():
                        raise ValueError('Prediction coverage differs')
                    mask = np.isfinite(actual)
                    error = np.abs(actual[mask]-predicted[mask])
                    zero = mask & (actual == 0)
                    arm_rows.append({'fold': fold['name'], 'mode': 'cell', 'horizon': h,
                        'mae': float(error.mean()), 'wape': float(error.sum()/np.abs(actual[mask]).sum()),
                        'n_valid': int(mask.sum()), 'zero_fp_rate': float((predicted[zero] > 0).mean())})
                    for origin in (fold['stop_end'], fold['score_end']-24):
                        values = pre.transform(panel.values[origin-167:origin+1], 'B')
                        inputs = torch.as_tensor(values[:, groups].transpose(1, 0, 2, 3))
                        phase = torch.as_tensor(pre.static_at(np.full(3, origin), np.arange(3)))
                        with torch.no_grad():
                            cpu = model(inputs, phase).numpy()[:, :, j]
                        gpu = predicted.reshape(len(origins), 2880)[origin-origins[0], groups]
                        np.testing.assert_allclose(cpu, gpu, atol=1e-4, rtol=0)
                        parity.append(float(np.abs(cpu-gpu).max()))
        saved = args.artifacts.parent.parent/'reports'/f'partial_supervision_{arm}_development.csv'
        pd.testing.assert_frame_equal(pd.DataFrame(arm_rows), pd.read_csv(saved),
                                      check_dtype=False, atol=1e-12, rtol=1e-12)
        rows.extend({'arm': arm, **row} for row in arm_rows)
    result = {'source_commit': args.source_commit, 'training_performed_by_audit': False,
              'all_metrics_reconstructed': True, 'matched_cold_states_verified': True,
              'supervision_and_optimizer_exposure_verified': True, 'exposure': exposure,
              'maximum_cpu_gpu_error': max(parity), 'history_only_predictions_checked': len(parity)*groups.size,
              'holdout_used': False, 'production_promoted': False, 'metrics': rows,
              'scope': 'reused development; unequal natural optimizer exposure; not a pure data-effect estimate'}
    Path('/kaggle/working/partial_supervision_verified_metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ('metrics', 'exposure')}, indent=2))


if __name__ == '__main__':
    main()
