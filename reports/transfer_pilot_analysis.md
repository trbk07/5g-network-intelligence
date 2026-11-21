# transfer_pilot - original-network transfer audit

Cloud-only fitting. Original fit/stopping through 503; fixed refit and reused validation audit 504-671. No final holdout replay.
Source domains are sector data, not additional original-network beams; relative-shape transfer does not assert physical-unit equivalence.

```csv
model,horizon,mae,wape,n_valid
transfer_0,1,0.2294829050394352,0.41337579199529706,483840
transfer_0.1,1,0.2298434493151097,0.41402525333744233,483840
transfer_0.5,1,0.23444802417397792,0.42231963926897476,483840
transfer_1,1,0.23660532957127872,0.4262056708974819,483840
transfer_0,6,0.25034859829978345,0.4440468172067057,469440
transfer_0.1,6,0.25083188772597637,0.4449040344347951,469440
transfer_0.5,6,0.2551432441011774,0.45255114765713234,469440
transfer_1,6,0.2572968782620756,0.4563710787494939,469440
transfer_0,24,0.24961544513343606,0.45020788800543243,417600
transfer_0.1,24,0.2500475744677131,0.4509872790195824,417600
transfer_0.5,24,0.25434362940593225,0.4587356690258861,417600
transfer_1,24,0.2561649110866502,0.46202054339934756,417600
original_c1,1,0.23063451146783154,0.4154502219809354,483840
original_c1,6,0.25347289601808365,0.4495884278537767,469440
original_c1,24,0.2512464622398822,0.4531496000713717,417600
```

The external ratio-zero / target-only case is the transfer ablation control; compare source cases with it, not just with a differently featured original model.
Target-only relative inputs drop incompatible exogenous KPIs for both domains. This isolates transfer under that representation; it does not prove the representation is better.
All horizon/beam coverage must match original replay. Additional dataset volume is not evidence of higher quality.
WAPE <10% is achieved only if every declared horizon meets it on a separately frozen unseen evaluation. Reused-validation results cannot establish that final claim.
CPU history-only parity checks: 36; maximum absolute difference 0.
