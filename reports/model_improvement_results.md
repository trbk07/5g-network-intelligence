# Model improvement results

All training, target-recipe fitting and ensemble calibration ran on hosted Kaggle. Production champion and previously observed final holdout are unchanged.

## Reused-validation audit

| Horizon | Original C1 MAE | Week 7 deeper MAE | New ensemble MAE | Reduction vs C1 | Reduction vs Week 7 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1h | 0.230635 | 0.229728 | 0.227866 | 1.200% | 0.811% |
| 6h | 0.253473 | 0.252613 | 0.251576 | 0.748% | 0.410% |
| 24h | 0.251246 | 0.250547 | 0.249527 | 0.685% | 0.407% |

Mean normalized MAE reduction: 0.878% against C1; 0.543% against Week 7 deeper.

## What was tested

- Nine LightGBM recipes screened on the first inner fold; promoted recipes confirmed on a second chronological fold.
- Two CatBoost configurations on both folds, trained on GPU. All models use the original causal40 numerical inputs.
- Separate training, early-stopping and scoring windows; all search scores use targets through hour 503.
- Convex blends, small bias calibration, and final round selection fitted on cloud only. Weights/recipes were written and hashed before outer audit arrays were constructed.
- Final candidate refit on development targets through 503 with fixed rounds; no stopping on audit 504-671.
- Target normalization/residuals, more frequent training origins, recency weighting and model diversity were tested rather than assuming a larger model would help.

## Paired uncertainty

```csv
horizon,reference,difference,lower,upper,status,n_clusters,repetitions,seed
1,lightgbm_c1,-0.0027687106759999996,-0.0034124207808785456,-0.002214331359058596,ok,30,1000,17
1,week7_deeper_l1,-0.0018620247729055133,-0.0023754457789081293,-0.001402761706960964,ok,30,1000,17
6,lightgbm_c1,-0.0018965805209911046,-0.002242367939340982,-0.0015975884244676987,ok,30,1000,17
6,week7_deeper_l1,-0.0010363501590470736,-0.0012238401503480286,-0.0008298330846092596,ok,30,1000,17
24,lightgbm_c1,-0.001719820851057511,-0.00197320239607156,-0.0014591694044871547,ok,30,1000,17
24,week7_deeper_l1,-0.0010200505960832355,-0.0011765254811201002,-0.000835208580332479,ok,30,1000,17
```

Intervals describe this reused development sample. They are not corrected for model search/multiple comparisons and are not evidence of unseen-week generalization.

## Saved model and limitations

`artifacts/improved_candidate` contains a loadable research ensemble, exact weights/bias, member models and checksum manifest. `scripts.predict_research` performs CPU inference only.
Independent CPU inference matched 20 beam/origin pairs across 3 horizons within 1e-8; see improvement_cpu_parity.csv.
Audit 504-671 is previously seen validation. No new final holdout score or global-optimality claim is made. Fresh unseen chronological data and a separately frozen promotion protocol remain necessary.
Forecast MAE and exceedance alerting are different objectives: inspect improvement_risk.csv for precision/recall/FPR. Thresholds are historical scenarios, not verified congestion or predictive quantiles.
No changes were made to artifacts/champion, configs/champion.yaml or Week 6 holdout reports.

## Exceedance trade-off

```csv
horizon,threshold,precision,recall,fpr,positive_events
1,historical_warning_p90,0.6508191007319624,0.1699710524878475,0.011678359014532085,54927
1,historical_critical_p95,0.6445497630331753,0.09812409812409813,0.003463563313937379,29106
6,historical_warning_p90,0.6203454463885151,0.10194452124228182,0.00815299203969315,54255
6,historical_critical_p95,0.617522373999058,0.04553190011461119,0.0018427448728801058,28793
24,historical_warning_p90,0.6109369120782642,0.1026273096938668,0.008380680666888206,47463
24,historical_critical_p95,0.6088311688311688,0.04644895370957514,0.0019191167475431227,25232
```
The new ensemble favors point MAE. Its P90 recall remains roughly 10-17%; recall at 6h/24h is lower than C1 while precision improves and FPR falls. This is not a complete solution to peak alerting.

## Reproduction

Protocol: configs/model_improvement.yaml; explanation: docs/model_improvement.md.
Cloud workers: improve-lgbm (CPU), improve-catboost (GPU), improve-audit (cloud refit/calibration). Checkpoints persist in private Kaggle Output.
Review notebook displays saved reports and never fits models or reopens holdout.
