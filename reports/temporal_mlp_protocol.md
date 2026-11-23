# Nonlinear temporal residual study

A fit-only descriptive diagnostic measured mean simultaneous residual correlation
0.0217 for same-cell beam pairs versus 0.0040 for shuffled 32-beam groups, after
subtracting the causal seven-day seasonal mean. All 44,640 pairs in each grouping
had finite correlations. This modest average association is not future
predictability, causation, significance or an error lower bound. It does not
justify adding graph complexity as the next priority. The diagnostic never fits
a forecasting model or supplies numerical beam summaries as inputs.

Instead test nonlinear history representation against the existing temporal
linear reference. A shared 64-unit GELU MLP, dropout 0.1 and a three-output head
predicts a residual added to the causal seven-day target-seasonal mean. The last
head is initialized to zero, so initialization has an exact analytical meaning.
Physical KPI channels are centered by their last imputed/scaled observation;
missing masks and phase4 are retained. No beam identifiers, fitted beam summaries,
neighbour values or external pretrained weights enter the model.

Four hosted GPU fits: target-only A / four-KPI B on 335/383/443 and 395/443/503
fit/stopping/scoring folds, all original beams, stride 1, seed 17. Global per-KPI
preprocessing fits only the training prefix. Shared masked-L1 trainer, AdamW
learning rate 0.001 and checked default weight decay 0.01, batch 1024, clipping 1,
30 maximum epochs and patience 5. This retains the preceding linear study's
budget; only the declared nonlinear residual architecture changes.

Joint training and stopping require all three targets; per-horizon scoring
retains every eligible origin rather than truncating short horizons to +24h
availability. Save checkpoint/optimizer/RNG state each epoch, best/last state,
source commit, preprocessing, parameter count, metrics and complete replay.
Report negative results for every fold, variant and horizon.

No local fitting or synthetic optimizer test is allowed. Local tests use only
initialized forward passes and safe review execution. No validation 504–671,
original final holdout, coefficient search or production promotion. These
reused development folds cannot prove future generalization or no overfitting.
Compare predictions with the matched linear and hurdle arrays before any
separate audit. The WAPE <10% goal remains unachieved until valid evidence proves it.

Reader notebook: `20_temporal_residual_mlp.ipynb`, default training off. Hosted
worker explicitly enables training and retains checkpoints in private Kaggle
Output. Original LightGBM causal40 and GRU phase4 contracts remain unchanged.
