# Temporal split decision

Mode: **fallback**.

Use internal holdout: continuity has not passed evidence and structure checks.

Evidence: Not approved.

Source: dataset_card.md at pinned revision. Source indices reset for week 6; no calendar timestamps are supplied. No profile correlation is used as evidence.

Inclusive bounds:
```json
{
  "mode": "fallback",
  "bounds": "inclusive",
  "lookback": 168,
  "horizons": [
    1,
    6,
    24
  ],
  "fit": [
    0,
    503
  ],
  "validation": [
    504,
    671
  ],
  "holdout": [
    672,
    839
  ],
  "final_refit": [
    0,
    671
  ],
  "week11": "excluded"
}
```
Week 11 is separate. Model selection uses validation only.
