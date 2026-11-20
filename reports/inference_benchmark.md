# CPU inference benchmark

Local CPU shared Predictor.predict_risk, 3 horizons + 168-point histories; no training

CPU: 13th Gen Intel(R) Core(TM) i7-13650HX. Python: 3.12.13.
Model load: 4161.57 ms. RSS after load: 182505472 bytes.

| Beams | Warm-up | Requests | P50 (ms) | P95 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | 100 | 1.656 | 2.339 |
| 10 | 10 | 100 | 3.857 | 4.850 |

## Actual Docker footprint

- ['5g-network-intelligence-api:week7']: 142,256,948 bytes.
- ['5g-network-intelligence-dashboard:week7']: 196,415,065 bytes.

## Limits

- Single process, sequential warm requests; not a production SLA or load test.
- Cold model loading and warm inference are reported separately.
- Image sizes reflect the deployed LightGBM profile only; baseline/GRU image sizes are not measured.
