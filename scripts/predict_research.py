"""CPU inference only, for the saved research ensemble; never exposes final holdout actuals."""

import argparse
import json
from pathlib import Path

from src.data.panel import load_panel
from src.models.research_ensemble import ResearchEnsemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts/improved_candidate")
    parser.add_argument("--origin", type=int, default=503)
    parser.add_argument("--beam-ids", nargs="+", default=["0_0_0"])
    args = parser.parse_args()
    if not 503 <= args.origin <= 647:
        raise ValueError(
            "Research replay origin must be 503..647; all displayed targets stay <=671"
        )
    panel = load_panel(Path("data/processed/panel.npz"))
    indices = [panel.beam_ids.index(b) for b in args.beam_ids]
    model = ResearchEnsemble(args.artifacts)
    history = panel.values[args.origin - 167 : args.origin + 1][:, indices, :].copy()
    result = {
        "research_only": True,
        "deployment_allowed": False,
        "origin": args.origin,
        "beam_ids": args.beam_ids,
        "predictions": {},
    }
    for h in (1, 6, 24):
        result["predictions"][h] = model.predict_at(history, args.origin, h).tolist()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
