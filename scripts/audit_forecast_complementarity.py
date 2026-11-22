"""Descriptive development-error associations only; no blend-weight fitting."""

from pathlib import Path

import numpy as np
import pandas as pd


def main():
    root = Path("artifacts")
    rows = []
    for fold in ("inner1", "inner2"):
        for variant in ("A", "B"):
            for h in (1, 6, 24):
                with np.load(root / "temporal_linear" / fold / variant / f"score_h{h}.npz", allow_pickle=False) as linear, np.load(root / "hurdle_research" / fold / f"h{h}" / "score.npz", allow_pickle=False) as hurdle:
                    for key in ("actual", "origins", "beam_ids"):
                        np.testing.assert_array_equal(linear[key], hurdle[key])
                    target = linear["actual"]
                    mask = np.isfinite(target)
                    a = linear["predicted"][mask] - target[mask]
                    b = hurdle["hurdle_median"][mask] - target[mask]
                    if not np.isfinite(a).all() or not np.isfinite(b).all():
                        raise ValueError("Missing paired forecasts cannot shrink coverage")
                    rows.append({"fold": fold, "variant": variant, "horizon": h,
                                 "signed_error_pearson": float(np.corrcoef(a, b)[0, 1]),
                                 "opposite_error_fraction": float(np.mean(a*b < 0)),
                                 "n_valid": int(mask.sum())})
    frame = pd.DataFrame(rows)
    frame.to_csv("reports/forecast_complementarity.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
