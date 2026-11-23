# Fixed MLP blend validation audit: small consistent MAE gains

Hosted Kaggle GPU v1 refitted MLP-B through hour 503 for exactly ten epochs,
derived from the two development best-epoch-plus-one counts (9 and 11). The
64-unit architecture, seed 17, global per-KPI preprocessing, phase4, masks and
20% MLP coefficient were fixed before the audit. No early stopping or coefficient
change occurred. The verified hurdle replay was reused without retraining.

| Horizon | Fixed MLP blend MAE | WAPE | Valid targets |
| --- | --- | --- | --- |
| 1h | 0.226480 | 40.7966% | 483,840 |
| 6h | 0.249664 | 44.2832% | 469,440 |
| 24h | 0.248324 | 44.7879% | 417,600 |

MAE uses original dataset units. All 2,880 beams and eligible per-horizon origins
are retained. These point MAEs are lower than the prior Linear blend and every
other case in the selected common-population research replay table. This table
does not replace the original release's baseline/GRU benchmark. The candidate
is retained as research evidence; no production model is promoted.

The mean C1-normalized MAE difference versus hurdle is -0.004335, approximately
0.4335 percentage points of C1-normalized error. Per-horizon paired spatial
intervals versus hurdle and the earlier ensemble are negative. The 6h interval
versus NR3500 crosses zero, despite a lower point MAE; the 1h/24h intervals do
not. These 95% intervals resample 30 base stations with 1,000 paired draws and
seed 17, conditional on this reused validation period. They neither describe
future-week uncertainty nor adjust for repeated research/model selection.

Trade-off: strict zero false-positive rates increase to 99.84%, 99.93%, 99.97%.
Their magnitude is not captured by this rate; retain MAE/WAPE and the earlier
zero/positive error decomposition rather than treating every small positive
forecast as equally costly. No historical risk-quality improvement is claimed.

Verification checked all six artifact files, fixed ten-epoch provenance, pinned
selection and hurdle replay hashes, full replay arithmetic and every reported
All-population metric. Eighty-one history-only component predictions match the
cloud output within declared tolerance (maximum difference 9.20e-8 original
units; neural tolerance 1e-5, hurdle tolerance 1e-8). The original Linear blend
verification remained unchanged after shared-loader/verifier refactoring.

Validation 504-671 has been reused repeatedly and is not a fresh confirmation
set. This audit cannot prove no overfitting, independent generalization or an
irreducible error bound. WAPE <10% remains unmet. Frozen production and original
holdout artifacts remain unchanged; no holdout replay occurred. Do not enlarge
the coefficient grid or tune epoch counts from this audit.
