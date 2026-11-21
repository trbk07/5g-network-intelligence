# multisource_pilot - original-network transfer audit

Cloud-only fitting. Original fit/stopping through 503; fixed refit and reused validation audit 504-671. No final holdout replay.
Source domains are sector data, not additional original-network beams; relative-shape transfer does not assert physical-unit equivalence.

```csv
model,horizon,mae,wape,n_valid
target_only,1,0.2294829050394352,0.41337579199529706,483840
pm01_nr1800,1,0.23444802417397792,0.42231963926897476,483840
pm03_nr2100,1,0.2298298732718166,0.4140007983234731,483840
pm03_nr3500,1,0.22947189821205236,0.41335596500214494,483840
balanced_pool,1,0.2303608590882061,0.4149572821293172,483840
target_only,6,0.25034859829978345,0.4440468172067057,469440
pm01_nr1800,6,0.2551432441011774,0.45255114765713234,469440
pm03_nr2100,6,0.25046155818956733,0.444247175746243,469440
pm03_nr3500,6,0.25022174364849453,0.44382181333396475,469440
balanced_pool,6,0.2511014692077,0.4453821948868616,469440
target_only,24,0.24961544513343606,0.45020788800543243,417600
pm01_nr1800,24,0.25434362940593225,0.4587356690258861,417600
pm03_nr2100,24,0.24918265577049362,0.44942730655969143,417600
pm03_nr3500,24,0.24909999981342287,0.44927822778836785,417600
balanced_pool,24,0.2502142251734668,0.4512878512146719,417600
```

The external ratio-zero / target-only case is the transfer ablation control; compare source cases with it, not just with a differently featured original model.
Target-only relative inputs drop incompatible exogenous KPIs for both domains. This isolates transfer under that representation; it does not prove the representation is better.
All horizon/beam coverage must match original replay. Additional dataset volume is not evidence of higher quality.
WAPE <10% is achieved only if every declared horizon meets it on a separately frozen unseen evaluation. Reused-validation results cannot establish that final claim.
CPU history-only parity checks: 45; maximum absolute difference 0.
