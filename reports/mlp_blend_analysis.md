# MLP and hurdle blend: small positive reused-development result

Hosted Kaggle v1 chose variant B and MLP weight 0.2 for all three horizons on
inner1. Every choice was saved before the selector read inner2 forecast/target
arrays. The fixed grid was 0/0.05/0.1/0.2; ties favor the smaller weight, then A.
No base models were retrained and no coefficients were selected locally.

| Inner2 horizon | Fixed blend MAE | WAPE | MAE reduction vs hurdle |
| --- | --- | --- | --- |
| 1h | 0.237189 | 40.2405% | 0.5185% |
| 6h | 0.253525 | 43.7230% | 0.6023% |
| 24h | 0.259279 | 42.4432% | 0.4343% |

MAE uses original target units. All observed original beam targets and scoring
origins are retained. The fixed blend improves all horizons versus hurdle,
passing the predeclared development rule, and its point MAEs are also lower
than the earlier fixed Linear/hurdle blend on the same inner2 targets. This is
not a fresh-test comparison: both folds have supported multiple research
decisions. Selection is at the maximum allowed weight; do not expand the grid
after viewing confirmation or interpret that boundary as an optimal coefficient.

Archive integrity verified five artifact files. Pinned component manifests,
saved grid coverage, tie-rule consistency and choices matched the protocol.
All 437,760 fixed inner2 forecasts were reconstructed exactly from unchanged
component predictions, and MAE/WAPE/counts/decision matched the cloud report.
The verifier inspects the saved cloud grid rather than rerunning coefficient
selection locally. It does not claim that replay arithmetic proves every aspect
of scientific validity or independent generalization.

Decision: retain as a research candidate for a separately frozen fixed-refit
audit. No audit is automatically performed by this selection worker. If defined,
such an audit must keep weights, architecture, feature policy and epoch-count
rule fixed before reading its results; existing validation is reused and cannot
establish unseen-time performance. A gain here is not grounds to change frozen
production, replay final holdout, claim no overfitting or claim WAPE below 10%.

Notebook 22 defaults to review only. Original hosted worker outputs remain
immutable. The shared verifier also reproduced the prior Linear blend evidence
unchanged after its component-name refactor.
