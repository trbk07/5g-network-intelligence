"""Assess verified paired attention scores only; never fit, select or calibrate."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.assess_telecom_quality import KEYS, gate, paired


def assess(evidence, strong):
    for key in ('all_metrics_reconstructed', 'matched_seeded_cold_states_verified',
                'stopping_and_optimizer_exposure_verified'):
        if evidence.get(key) is not True:
            raise ValueError('Complete independent verification required')
    if (evidence['source_commit'] != '0062eaf296c396da12af583ce7a99f7129bce372'
            or evidence['holdout_used'] or evidence['production_promoted'] or evidence['training_performed_by_audit']
            or not np.isfinite(evidence['maximum_cpu_gpu_error']) or evidence['maximum_cpu_gpu_error'] > 1e-4
            or evidence['history_only_predictions_checked'] != 576):
        raise ValueError('Source, temporal or numerical verification scope differs')
    frame = pd.DataFrame(evidence['metrics'])
    cases = ('no_covariates', 'past_covariates')
    if len(frame) != 12 or set(frame['case']) != set(cases):
        raise ValueError('Both cases and all fold/horizon scores required')
    tables, gates = [], {}
    for case in cases:
        comparison = paired(frame[frame['case'] == case], strong)
        comparison['comparison'] = 'strong_comparator'
        tables.append(comparison)
        gates[case] = gate(comparison)
    comparison = paired(frame[frame['case'] == 'past_covariates'], frame[frame['case'] == 'no_covariates'])
    comparison['comparison'] = 'past_covariates_vs_matched_control'
    tables.append(comparison)
    output = {'cases_vs_strong_control': gates, 'paired_covariate_effect': gate(comparison),
              'interpretation': 'Point estimates on reused development; no best-case or best-horizon selection',
              'fresh_generalization_established': False, 'production_promoted': False}
    return pd.concat(tables, ignore_index=True), output


def main():
    evidence = json.loads(Path('reports/past_covariate_attention_verified_metrics.json').read_text())
    proof = json.loads(Path('reports/cell_multisource_v2_multi_source_pretrain_verification.json').read_text())
    if not proof['all_metrics_reconstructed'] or not proof['controlled_neural_initialization_verified']:
        raise ValueError('Verified historical comparator required')
    historical = pd.read_csv('reports/cell_multisource_v2_multi_source_pretrain_comparison.csv')
    strong = historical[KEYS+['wape', 'n_valid', 'current_mae']].rename(columns={'current_mae': 'mae'})
    table, result = assess(evidence, strong)
    table.to_csv('reports/past_covariate_attention_paired_comparisons.csv', index=False)
    Path('reports/past_covariate_attention_quality_assessment.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
