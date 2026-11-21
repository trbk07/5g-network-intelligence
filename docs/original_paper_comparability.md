# Comparing the original paper with this project

The [original paper, v2](https://arxiv.org/html/2504.17493v2), section 5.1,
uses a 70/10/20 split and 96-input/24-output configuration for BLW-TrafficDS,
with channels cropped to 100. Its interval-oriented policy comparisons do not
establish full-population WAPE below 10%. Table 2 reports scaled MAE, so its
values cannot be substituted for this project's original-unit MAE.

The paper's [historical dataset URL](https://huggingface.co/datasets/netop/Beam-Level-Traffic-Timeseries-Dataset)
redirects to [netop/gotsf-ds](https://huggingface.co/datasets/netop/gotsf-ds).
Treat these URLs as aliases, not independent datasets. The current card describes
2,880 beams and a different released layout from the paper's experimental subset.

Project-specific interpretation: current experiments use 168 history hours,
three direct horizons, a frozen fallback split and all eligible original beams.
Architecture names or headline paper scores alone cannot establish an improvement
under this protocol. Any reproduction needs compatible population, units,
forecast horizons, chronological boundaries, metrics and source revision, while
keeping the already observed holdout unavailable for tuning.

This literature check does not modify the frozen split, train a model, evaluate
an unused target week or establish a forecasting-error lower bound.
