"""Asset -> shared inference -> HTTP JSON contract; no model fitting."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.inference.predictor import Predictor
from tests.utils.demo import baseline_assets


def test_baseline_asset_to_api_and_unknown_target(tmp_path):
    assets = baseline_assets(tmp_path / "demo")
    with TestClient(create_app(lambda: Predictor(assets))) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/risk", json={"origin": 239, "beam_ids": ["0_0_0", "0_0_1"]})
        assert response.status_code == 200
        data = response.json()
        assert all(p["actual"] is None for b in data["beams"] for p in b["predictions"])
        assert all(p["status"] == "Excluded" for p in data["beams"][1]["predictions"])


@pytest.mark.skipif(
    not Path("demo_assets/manifest.json").exists(), reason="Verified real demo assets not present"
)
def test_frozen_lightgbm_loaded_once_and_ordered_batch():
    calls = []

    def load():
        calls.append(1)
        return Predictor()

    with TestClient(create_app(load)) as client:
        assert client.get("/model-info").json()["champion"] == "lightgbm_c1"
        for origin in (671, 695):
            forward = client.post(
                "/forecast", json={"origin": origin, "beam_ids": ["0_0_0", "5_1_8"]}
            ).json()
            reverse = client.post(
                "/forecast", json={"origin": origin, "beam_ids": ["5_1_8", "0_0_0"]}
            ).json()
            assert forward["beams"] == list(reversed(reverse["beams"]))
    assert calls == [1]
