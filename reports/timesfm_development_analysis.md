# TimesFM 2.5 median screen: aggregate results, verification pending

The six-row aggregate and runtime metadata were recovered from Drive after the
Colab runtime ended. They identify source
`6595374dbe378344b0296ba834fe71c7990e44a2`, Tesla T4 and Torch 2.14.1+cu130.
Drive folder listings show all 120 per-origin forecast caches: 383–442 and
443–502. Pretrained weights remain on Drive. No training occurred.

| Fold | Horizon | Reported MAE | Reported WAPE |
|---|---:|---:|---:|
| inner1 | 1h | 0.231886 | 42.6614% |
| inner1 | 6h | 0.257737 | 45.4566% |
| inner1 | 24h | 0.245950 | 47.1609% |
| inner2 | 1h | 0.238893 | 40.5296% |
| inner2 | 6h | 0.253616 | 43.7387% |
| inner2 | 24h | 0.261163 | 42.7517% |

These reported MAEs are worse than the fixed historical MLP/hurdle blend in
all six comparisons. The screen therefore supplies no present justification
for model promotion or a material-gain claim. It does not meet WAPE below 10%.

**Verification status:** protocol and complete aggregate coverage are available,
but independent reconstruction from cached predictions and numerical replay
have not completed. Do not label these as independently verified scores.
Two proposed reference-transfer paths were rejected by automatic review:
persisting signed access URLs and delivering those URLs through a terminal.
Neither transfer occurred, and the unused helper was removed. No credential
URL manifest or checkpoint was written locally. Verify caches in a future
authenticated cloud runtime rather than repeat model fitting or change protocol.

The model/wheel hashes and median index were fixed before inference. No future
covariates or original observations from index 504 onward were supplied.
Pretraining corpus overlap remains uncertified; both folds are reused
development evidence. No fresh generalization or production claim is warranted.
