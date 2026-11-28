"""Compare verified full-population aggregate scores; never fit or calibrate."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.assess_telecom_quality import KEYS, gate, paired


def main():
    evidence = json.loads(Path('reports/partial_supervision_verified_metrics.json').read_text())
    for key in ('all_metrics_reconstructed', 'matched_cold_states_verified',
                'supervision_and_optimizer_exposure_verified'):
        if evidence.get(key) is not True:
            raise ValueError('Complete independent verification required')
    if (evidence['holdout_used'] or evidence['production_promoted'] or evidence['training_performed_by_audit']
            or not np.isfinite(evidence['maximum_cpu_gpu_error']) or evidence['maximum_cpu_gpu_error'] > 1e-4):
        raise ValueError('Temporal/parity scope differs')
    frame = pd.DataFrame(evidence['metrics'])
    if len(frame) != 12 or set(frame.arm) != {'joint', 'available'}:
        raise ValueError('Complete paired study required')
    proof = json.loads(Path('reports/cell_multisource_v2_multi_source_pretrain_verification.json').read_text())
    if not proof['all_metrics_reconstructed'] or not proof['controlled_neural_initialization_verified']:
        raise ValueError('Verified historical comparator required')
    historical = pd.read_csv('reports/cell_multisource_v2_multi_source_pretrain_comparison.csv')
    strong = historical[KEYS+['wape', 'n_valid', 'current_mae']].rename(columns={'current_mae': 'mae'})
    tables, gates = [], {}
    for arm in ('joint', 'available'):
        comparison = paired(frame[frame.arm == arm], strong)
        comparison['comparison'] = 'strong_comparator'
        tables.append(comparison)
        gates[arm] = gate(comparison)
    comparison = paired(frame[frame.arm == 'available'], frame[frame.arm == 'joint'])
    comparison['comparison'] = 'available_vs_new_joint_control'
    tables.append(comparison)
    output = {'arms_vs_strong_control': gates, 'paired_label_policy_effect': gate(comparison),
              'compute_caveat': 'Natural optimizer exposure differs; not a pure label-count effect',
              'fresh_generalization_established': False, 'production_promoted': False}
    pd.concat(tables, ignore_index=True).to_csv('reports/partial_supervision_paired_comparisons.csv', index=False)
    Path('reports/partial_supervision_quality_assessment.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
