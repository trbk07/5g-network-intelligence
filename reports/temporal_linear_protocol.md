# Temporal linear development study

Question: can a shared linear projection of complete 168-hour sequences improve
over existing development forecasts? This is an adaptation inspired by
[NLinear / LTSF-Linear](https://github.com/cure-lab/LTSF-Linear), not a reproduction
of the authors' benchmark or a prediction of improvement.

The model subtracts the last imputed/scaled value of each physical KPI from its
history, retains missing masks, flattens the sequence, appends phase4 and predicts
three target residuals. The last target value is restored. Fixed initial weights
reproduce the causal seven-day target-seasonal mean. No beam identity, per-beam
fitted preprocessing or numerical beam summary enters the model.

Four hosted GPU fits: variants A (target only) and B (four KPIs) on development
folds 335/383/443 and 395/443/503. Fit a global median/mean/std per selected KPI
using the training prefix only. Preserve missing labels in supervision. Train
joint horizons 1/6/24 with masked L1, all beams, stride 1, seed 17, batch 1024,
AdamW learning rate 0.001 and fixed default weight decay 0.01, clipping 1, maximum
30 epochs, patience 5. The shared trainer's optimizer default is checked at
runtime. Save best/last model, optimizer/RNG states and per-epoch history.

Joint stopping uses origins with all three targets inside the stopping interval;
this gives a shorter target range at short horizons. Scoring separately uses
every eligible origin for each horizon, so evaluation coverage is not truncated
to joint +24h availability. Epoch selection uses joint scaled L1; final reported
MAE/WAPE use original units after nonnegative clipping.

One seed and these reused development folds cannot establish generalization or
absence of overfitting. Report A/B, every fold/horizon, coverage and zero false-
positive rate, including negative results. Compare the same scoring arrays with
saved hurdle development predictions before considering any reused-validation
audit. No validation 504–671 or final holdout labels enter this study. Production
LightGBM/GRU artifacts and their feature/preprocessing contracts are unchanged.

Reader notebook `16_temporal_linear_forecasting.ipynb` defaults to no training.
Only the separately enabled hosted worker may fit. Local checks use initialized
forward passes without backward, optimizer steps or fitting. No claim is made
that the initialization or inference checks establish a trained model's quality.

Completed v1: `temporal_linear_analysis.md` records the negative result and
independent verification. To reproduce intake and verification without fitting:

```powershell
python -m scripts.import_hurdle_results .cache/temporal-linear-results/temporal_linear_results.tar.gz --kind temporal_linear
python -m scripts.summarize_temporal_linear
```
