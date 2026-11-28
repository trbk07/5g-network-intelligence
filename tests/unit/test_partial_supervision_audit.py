"""Audit contracts using fabricated metadata only; no model fitting."""

import copy
import hashlib
import json

import pytest

from scripts.audit_partial_supervision_cloud import verify_history, verify_manifest


def evidence():
    cfg = {'batch_size': 4, 'max_epochs': 20, 'early_stopping_patience': 2,
           'training_objective': 'balanced'}
    counts = {'horizon_valid_labels': {'1': 12, '6': 10, '24': 8}, 'training_cell_windows': 9}
    history = [{'epoch': i, 'val_l1_scaled': value, 'train_valid_targets': 30,
                'validation_valid_targets': 24, 'training_objective': 'balanced', 'train_objective': .2}
               for i, value in enumerate([.4, .3, .31, .32])]
    best = {'epoch': 1, 'best': .3}
    last = {'epoch': 3, 'best_epoch': 1, 'stale': 2, 'best': .3,
            'optimizer': {'state': {0: {'step': 12}, 1: {'step': 12}}}}
    summary = {'best_epoch': 1, 'epochs_completed': 4, 'budget_exhausted': False}
    return history, best, last, summary, cfg, counts, 24


def test_actual_exposure_and_stopping_verified():
    assert verify_history(*evidence()) == 12


@pytest.mark.parametrize('fault', ['steps', 'counts', 'selection', 'budget', 'objective', 'epochs'])
def test_inconsistent_training_evidence_rejected(fault):
    args = copy.deepcopy(evidence())
    history, best, last, summary, _, _, _ = args
    if fault == 'steps':
        last['optimizer']['state'][0]['step'] = 11
    elif fault == 'counts':
        history[-1]['train_valid_targets'] = 29
    elif fault == 'selection':
        best['epoch'] = 2
    elif fault == 'budget':
        summary['budget_exhausted'] = True
    elif fault == 'objective':
        history[-1]['training_objective'] = 'different'
    else:
        history[-1]['epoch'] = 5
    with pytest.raises(ValueError):
        verify_history(*args)


def test_manifest_requires_exact_file_coverage(tmp_path):
    path = tmp_path/'report.json'
    path.write_text('{}')
    manifest = {'files': {'report.json': hashlib.sha256(path.read_bytes()).hexdigest()}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    verify_manifest(tmp_path)
    (tmp_path/'unexpected.json').write_text('{}')
    with pytest.raises(ValueError, match='coverage'):
        verify_manifest(tmp_path)


def test_manifest_rejects_changed_content(tmp_path):
    path = tmp_path/'report.json'
    path.write_text('{}')
    (tmp_path/'manifest.json').write_text(json.dumps({'files': {'report.json': '0'*64}}))
    with pytest.raises(ValueError, match='checksum'):
        verify_manifest(tmp_path)
