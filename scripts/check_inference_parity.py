"""20 real beam/origin pairs: offline artifacts, shared predictor, HTTP API and dashboard transport."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from scripts.run_risk import risk_records
from src.api.main import create_app
from src.data.io import sha256
from src.inference.predictor import Predictor


def check(url=None):
    """Only CPU inference on saved artifacts; no fitting or new benchmark aggregate."""
    completion = json.loads(Path("reports/holdout_completion.json").read_text())
    path = Path("reports/forecasts/champion_holdout.npz")
    if sha256(path) != completion["files"][path.as_posix()]:
        raise ValueError("Offline forecast checksum mismatch")
    records = risk_records(path)
    predictor = Predictor()
    from app.backends import ApiBackend

    backend = ApiBackend(url) if url else None
    rows = []
    with TestClient(create_app(lambda: predictor)) as client:
        for beam in ("0_0_0", "5_1_8", "15_2_0", "29_0_8"):
            for origin in (671, 695, 719, 743, 767):
                shared = predictor.predict([beam], origin)
                payload = {"origin": origin, "beam_ids": [beam]}
                response = client.post("/forecast", json=payload)
                response.raise_for_status()
                api = response.json()
                dashboard = backend.forecast([beam], origin) if backend else api
                for j, h in enumerate((1, 6, 24)):
                    record = records[h]["lightgbm_c1"]
                    i = int(np.flatnonzero(record["origins"] == origin)[0])
                    b = record["beam_ids"].index(beam)
                    actual = float(record["predicted"][i, b])
                    values = [
                        shared["beams"][0]["predictions"][j]["point"],
                        api["beams"][0]["predictions"][j]["point"],
                        dashboard["beams"][0]["predictions"][j]["point"],
                    ]
                    errors = [abs(x - actual) for x in values]
                    if max(errors) > 1e-9:
                        raise AssertionError(f"Inference parity failed: {beam}/{origin}/+{h}h")
                    rows.append(
                        {
                            "beam_id": beam,
                            "origin": origin,
                            "horizon": h,
                            "offline": actual,
                            "shared_abs_error": errors[0],
                            "api_abs_error": errors[1],
                            "dashboard_transport_abs_error": errors[2],
                            "atol": 1e-9,
                            "passed": True,
                            "transport": "real_http" if backend else "in_process",
                        }
                    )
    pd.DataFrame(rows).to_csv("reports/inference_parity.csv", index=False)
    print(
        "Inference parity:",
        20,
        "pairs x 3 horizons; max error:",
        max(
            max(
                r[k] for k in ("shared_abs_error", "api_abs_error", "dashboard_transport_abs_error")
            )
            for r in rows
        ),
    )
    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Also verify real HTTP through the dashboard backend")
    check(parser.parse_args().url)
