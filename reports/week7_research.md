# Week 7 validation research

Four predeclared parameter sets use the original 40 causal inputs. All training runs on hosted Kaggle CPU.
The squared-error candidate tests whether stronger peak sensitivity helps historical exceedance recall at a cost to MAE.
No threshold tuning, holdout reopening, production replacement or new final generalization claim.

model,1,6,24
deeper_l1,0.22972782556473698,0.2526126656561397,0.25054669198490803
lightgbm_c1,0.23063451146783154,0.25347289601808365,0.2512464622398822
regularized_l1,0.23028705045484316,0.2530179051979234,0.25100153945330406
shallow_l1,0.2309918133247733,0.2538728239548511,0.25189962286587736
squared_error,0.23820928993581159,0.2640707741764118,0.2625631596692876


{
  "research_only": true,
  "production_champion_unchanged": true,
  "holdout_reopened": false,
  "validation_best": "deeper_l1",
  "mean_normalized_mae": {
    "deeper_l1": 0.9966299205889261,
    "lightgbm_c1": 1.0,
    "regularized_l1": 0.9985745326074413,
    "shallow_l1": 1.0019088956775641,
    "squared_error": 1.0398987084090645
  },
  "limitation": "Repeated validation experiments are exploratory. Fresh unseen chronological data required for a new final claim or promotion."
}