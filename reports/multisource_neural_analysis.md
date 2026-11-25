# Multisource neural pretraining: verified development result

Decision: do not promote this pretraining recipe or refit it on validation. More source data did not establish a consistent gain over the compute-matched target self-pretraining control. Preserve the frozen production model and the previous research candidates.

## Protocol and provenance

Kaggle private worker `truongbaokien/5g-ni-multisource-representation-pretraining`, version 3, completed with source commit `38f569c147e0203e5a9b5d412c0f98bc68560b07`. Versions 1 and 2 stopped before model creation due to verified input-layout/storage-detection errors; no completed fit was repeated to repair presentation.

Fifteen hosted fits: three fixed external pretraining runs, two fold-specific target self-pretraining runs and ten target fine-tuning runs. Every pretraining run completed five epochs and exactly 320 optimizer steps. Fine-tuning used the same 64-unit univariate residual MLP, seed 17 and frozen fit/stop/score windows 335/383/443 and 395/443/503. All original beams and complete per-horizon scoring coverage were retained. Original values from 504 onward were redacted in the cloud bundle.

Radio domains NR1800/NR2100/NR3500 are one published collection. CESNET adds a distinct IP-traffic collection, not PRB measurements. Pool sampling assigns half the mass to CESNET and half to radio. Global per-source prefix preprocessing, source checksums, transferred initialization tensors and optimizer step counts were independently verified.

## Inner2 point estimates

MAE is in original DLPRB dataset units; WAPE is a percentage of the same observed absolute-target sum. These are reused development results, not final validation or holdout estimates.

| Case | 1h MAE | 6h MAE | 24h MAE | 1h WAPE | 6h WAPE | 24h WAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| target_only | 0.239965 | 0.254540 | 0.261046 | 40.71% | 43.90% | 42.73% |
| target_self_pretrain | 0.239073 | 0.253509 | 0.262219 | 40.56% | 43.72% | 42.92% |
| radio | 0.238949 | 0.253519 | 0.262367 | 40.54% | 43.72% | 42.95% |
| cesnet | 0.239158 | 0.254205 | 0.261324 | 40.57% | 43.84% | 42.78% |
| radio_and_cesnet | 0.239293 | 0.253806 | 0.261698 | 40.60% | 43.77% | 42.84% |

## Negative transfer and controls

| External case | Inner2 1h gain vs self | Inner2 6h gain vs self | Inner2 24h gain vs self |
| --- | ---: | ---: | ---: |
| radio | +0.0518% | -0.0041% | -0.0564% |
| cesnet | -0.0357% | -0.2747% | +0.3414% |
| radio_and_cesnet | -0.0919% | -0.1171% | +0.1987% |

Positive means lower MAE; negative means regression. Radio improves five of six fold/horizon point MAEs versus the no-pretraining control, but it improves only one of three inner2 cases versus self-pretraining. CESNET-only and pooled effects are mixed. Additional updates explain part of the apparent benefit; dataset count alone is not a useful acceptance criterion.

No external recipe is consistently better than the historical hurdle reference: each beats it only at inner2 +6h. The target-only case exactly reproduces the earlier univariate MLP scores at all six fold/horizon points. These observations do not establish an error floor or rule out a different transfer architecture.

## Verification and limitations

- History-only CPU/GPU parity: 180 beam forecasts, maximum absolute discrepancy 1.8e-07 original units (tolerance 1e-5).
- Completed fine-tuning patience: 10/10; capped runs: 0.
- Archive SHA256: `ee5d15f49406d8e42ce490214ef907bc514a6ebe599e6e9f012b48e7959b4632`; 115 artifact files verified. Frozen artifacts were not touched.
- One seed, univariate inputs and repeatedly reused development windows. No fresh-test superiority, no no-overfitting guarantee, no WAPE <10% claim and no production promotion.
- Preserve this negative result. A future architecture should address the original multivariate signals and zero/nonzero conditional distribution, with matched controls; do not simply add more pooled rows under this unchanged recipe.
