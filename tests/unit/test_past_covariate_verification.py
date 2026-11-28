"""Metadata-only verification tests; no checkpoints, model fitting or inference."""

from copy import deepcopy
import hashlib
import json

import pytest

from scripts.verify_colab_past_covariates import verify_checkpoint_identity, verify_tree


def identity():
    cfg = {'seed': 17, 'max_epochs': 80, 'variants': ['B'], 'learning_rate': .0003}
    fold = {'name': 'inner1', 'fit_end': 335, 'stop_end': 383, 'score_end': 443}
    checkpoint = {'identity': {'seed': 17, 'config': {'seed': 17, 'learning_rate': .0003},
                  'run_identity': {'architecture': 'TimeXer-inspired-independent-adaptation/no_covariates',
                                   'fold': fold.copy(), 'variant': 'B'}}}
    return checkpoint, cfg, fold, 'no_covariates'


def test_checkpoint_identity_matches_controller_contract():
    verify_checkpoint_identity(*identity())


@pytest.mark.parametrize('field', ['seed', 'architecture', 'variant', 'cutoff', 'learning_rate'])
def test_cross_case_or_cross_protocol_checkpoint_rejected(field):
    checkpoint, cfg, fold, case = deepcopy(identity())
    if field == 'seed':
        checkpoint['identity']['seed'] = 29
    elif field == 'learning_rate':
        checkpoint['identity']['config']['learning_rate'] = .1
    elif field == 'cutoff':
        checkpoint['identity']['run_identity']['fold']['fit_end'] = 395
    else:
        checkpoint['identity']['run_identity'][field] = 'different'
    with pytest.raises(ValueError, match='identity'):
        verify_checkpoint_identity(checkpoint, cfg, fold, case)


def test_manifest_detects_changed_content_and_unlisted_files(tmp_path):
    report = tmp_path/'report.json'
    report.write_text('{}')
    (tmp_path/'manifest.json').write_text(json.dumps({'files': {
        'report.json': hashlib.sha256(report.read_bytes()).hexdigest()}}))
    verify_tree(tmp_path)
    report.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='checksum'):
        verify_tree(tmp_path)
    report.write_text('{}')
    (tmp_path/'unlisted.json').write_text('{}')
    with pytest.raises(ValueError, match='coverage'):
        verify_tree(tmp_path)
