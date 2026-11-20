"""Positive and negative HTTP contracts; baseline fixtures never train."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.inference.predictor import Predictor
from tests.utils.demo import baseline_assets


@pytest.fixture
def client(tmp_path):
    root = baseline_assets(tmp_path / "demo")
    with TestClient(create_app(lambda: Predictor(root))) as value:
        yield value


def test_five_endpoints_and_shared_risk_forecast(client):
    assert client.get("/health").json()["beams_loaded"] == 2
    assert client.get("/model-info").json()["forecast_quantiles_available"] is False
    assert client.get("/replay/meta").json()["origin_min"] == 199
    payload = {"origin": 199, "beam_ids": ["0_0_0"]}
    f = client.post("/forecast", json=payload).json()
    r = client.post("/risk", json=payload).json()
    assert len(f["beams"][0]["history"]) == 168
    assert len(f["beams"][0]["predictions"]) == 3
    assert [p["point"] for p in f["beams"][0]["predictions"]] == [
        p["point"] for p in r["beams"][0]["predictions"]
    ]
    assert "historical_warning_p90" in r["beams"][0]
    assert client.get("/docs").status_code == 200


@pytest.mark.parametrize(
    "payload,status",
    [
        ({"origin": 199, "beam_ids": ["unknown"]}, 404),
        ({"origin": 198, "beam_ids": ["0_0_0"]}, 422),
        ({"origin": 240, "beam_ids": ["0_0_0"]}, 422),
        ({"origin": True, "beam_ids": ["0_0_0"]}, 422),
        ({"origin": 199.5, "beam_ids": ["0_0_0"]}, 422),
        ({"origin": 199, "beam_ids": ["0_0_0"] * 33}, 422),
        ({"origin": 199, "beam_ids": ["0_0_0", "0_0_0"]}, 422),
        ({"origin": 199, "beam_ids": []}, 422),
        ({"origin": 199, "beam_ids": ["0_0_0"], "horizon": 7}, 422),
    ],
)
def test_negative_requests(client, payload, status):
    assert client.post("/forecast", json=payload).status_code == status
    assert client.post("/risk", json=payload).status_code == status


def test_missing_artifacts_503_and_no_internal_trace():
    def unavailable():
        raise FileNotFoundError("private/internal/path")

    with TestClient(create_app(unavailable)) as client:
        for path in ("/health", "/model-info", "/replay/meta"):
            response = client.get(path)
            assert response.status_code == 503
            assert "private/internal" not in response.text
        assert (
            client.post("/forecast", json={"origin": 199, "beam_ids": ["0_0_0"]}).status_code == 503
        )
