"""Acceptance gate edge cases from fabricated aggregate metrics, no training."""

from copy import deepcopy

import pandas as pd
import pytest

from scripts.assess_past_covariate_attention import assess


def fixture():
    strong = pd.DataFrame([{'fold': fold, 'horizon': h, 'mae': 1., 'wape': .5, 'n_valid': 100}
                           for fold in ('inner1', 'inner2') for h in (1, 6, 24)])
    rows = [{**row, 'case': case, 'mae': .94 if case == 'past_covariates' else 1.}
            for case in ('no_covariates', 'past_covariates') for row in strong.to_dict('records')]
    evidence = dict(all_metrics_reconstructed=True, matched_seeded_cold_states_verified=True,
                    stopping_and_optimizer_exposure_verified=True,
                    source_commit='0062eaf296c396da12af583ce7a99f7129bce372', holdout_used=False,
                    production_promoted=False, training_performed_by_audit=False,
                    maximum_cpu_gpu_error=1e-6, history_only_predictions_checked=576, metrics=rows)
    return evidence, strong


def test_all_cases_reported_and_material_gate_is_point_evidence_only():
    table, result = assess(*fixture())
    assert len(table) == 18
    assert result['cases_vs_strong_control']['past_covariates']['material_point_gain_gate_passed']
    assert not result['production_promoted'] and not result['fresh_generalization_established']


def test_one_regressing_horizon_blocks_gate_despite_positive_mean():
    evidence, strong = fixture()
    evidence['metrics'][-1]['mae'] = 1.01
    _, result = assess(evidence, strong)
    assert result['cases_vs_strong_control']['past_covariates']['mean_relative_mae_gain_pct'] > 2
    assert not result['cases_vs_strong_control']['past_covariates']['initial_research_gate_passed']


@pytest.mark.parametrize('fault', ['missing_case', 'population', 'parity', 'source', 'verification'])
def test_incomplete_or_unverified_results_are_rejected(fault):
    evidence, strong = deepcopy(fixture())
    if fault == 'missing_case':
        evidence['metrics'] = evidence['metrics'][:6]
    elif fault == 'population':
        evidence['metrics'][-1]['n_valid'] = 99
    elif fault == 'parity':
        evidence['maximum_cpu_gpu_error'] = .001
    elif fault == 'source':
        evidence['source_commit'] = 'unverified'
    else:
        evidence['all_metrics_reconstructed'] = False
    with pytest.raises(ValueError):
        assess(evidence, strong)
