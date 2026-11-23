# Research validation comparison

Scope: selected saved replay families used by the current research extension. Original baseline/GRU release benchmarks remain in experiment_report.md; this table is not a replacement for that release evaluation.

All entries below are reconstructed from saved cloud validation forecasts. Origins, targets and beam axes match original C1 exactly; every observed target has a finite prediction. No training, coefficient search, final holdout evaluation or production promotion occurs in this report.

Validation 504–671 has been reused repeatedly. These are exploratory point estimates, not fresh-test evidence or a no-overfitting guarantee. MAE uses original dataset usage units; WAPE is displayed as a percentage of the same observed absolute-target sum. No target-unit conversion is assumed.

| Family / model | 1h MAE | 6h MAE | 24h MAE | 1h WAPE | 6h WAPE | 24h WAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original / lightgbm_c1 | 0.230635 | 0.253473 | 0.251246 | 41.55% | 44.96% | 45.31% |
| original / lightgbm_c2 | 0.238861 | 0.264953 | 0.263352 | 43.03% | 47.00% | 47.50% |
| ensemble / improved_ensemble | 0.227866 | 0.251576 | 0.249527 | 41.05% | 44.62% | 45.00% |
| ensemble / lightgbm_c1 | 0.230635 | 0.253473 | 0.251246 | 41.55% | 44.96% | 45.31% |
| ensemble / week7_deeper_l1 | 0.229728 | 0.252613 | 0.250547 | 41.38% | 44.81% | 45.19% |
| single_source / transfer_0 | 0.229483 | 0.250349 | 0.249615 | 41.34% | 44.40% | 45.02% |
| single_source / transfer_0.1 | 0.229843 | 0.250832 | 0.250048 | 41.40% | 44.49% | 45.10% |
| single_source / transfer_0.5 | 0.234448 | 0.255143 | 0.254344 | 42.23% | 45.26% | 45.87% |
| single_source / transfer_1 | 0.236605 | 0.257297 | 0.256165 | 42.62% | 45.64% | 46.20% |
| multisource / target_only | 0.229483 | 0.250349 | 0.249615 | 41.34% | 44.40% | 45.02% |
| multisource / pm01_nr1800 | 0.234448 | 0.255143 | 0.254344 | 42.23% | 45.26% | 45.87% |
| multisource / pm03_nr2100 | 0.229830 | 0.250462 | 0.249183 | 41.40% | 44.42% | 44.94% |
| multisource / pm03_nr3500 | 0.229472 | 0.250222 | 0.249100 | 41.34% | 44.38% | 44.93% |
| multisource / balanced_pool | 0.230361 | 0.251101 | 0.250214 | 41.50% | 44.54% | 45.13% |
| hurdle / hurdle_direct_l1 | 0.228382 | 0.251661 | 0.249713 | 41.14% | 44.64% | 45.04% |
| hurdle / hurdle_median | 0.227494 | 0.251004 | 0.249159 | 40.98% | 44.52% | 44.94% |
| fixed_blend / selected_linear | 0.233254 | 0.262158 | 0.256821 | 42.02% | 46.50% | 46.32% |
| fixed_blend / hurdle_median | 0.227494 | 0.251004 | 0.249159 | 40.98% | 44.52% | 44.94% |
| fixed_blend / hurdle_linear_blend | 0.226834 | 0.250427 | 0.249162 | 40.86% | 44.42% | 44.94% |

Interpretation:

- Hurdle improves all three point MAEs relative to the earlier improved ensemble. It is not the smallest point MAE at every horizon across all transfer experiments.
- The NR3500 source case has lower 6h and 24h point MAE than hurdle, while hurdle has lower 1h MAE. The 24h difference is tiny and reused validation does not justify a new horizon-wise selector.
- The target-only relative-shape control also has lower 6h point MAE than hurdle. Compare source cases with their matched control; extra dataset volume alone is not evidence of benefit.
- Identical controls/shared source cases in separate jobs are replications, not independent datasets or new evidence of generalization.
- The fixed neural blend passed development confirmation and its audit replay is independently verified. It lowers hurdle point MAE at 1h/6h but slightly regresses at 24h; NR3500 remains lower at 6h/24h. Do not derive a new horizon selector from these reused-validation comparisons.
- WAPE below 10% remains unmet. Production remains the frozen release; these research cases are not deployment replacements.

Evidence: [transfer decision](transfer_ablation_decision.md), [hurdle audit](hurdle_candidate_analysis.md), [calibration rejection](hurdle_calibration_analysis.md), [linear rejection](temporal_linear_analysis.md), [fixed blend confirmation](development_blend_analysis.md), [fixed blend audit](blend_candidate_analysis.md).
