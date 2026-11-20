"""Real HTTP product smoke, enabled explicitly after Compose startup; no training."""

import os
from pathlib import Path

import httpx
import pytest


@pytest.mark.skipif(
    not os.getenv("FIVEG_API_URL"), reason="Start Compose and set FIVEG_API_URL for product smoke"
)
def test_real_http_and_dashboard_interactions():
    from streamlit.testing.v1 import AppTest

    from app.backends import ApiBackend
    from scripts.check_inference_parity import check

    url = os.environ["FIVEG_API_URL"]
    client = ApiBackend(url)
    meta = client.meta()
    assert httpx.get(url + "/health").status_code == 200
    assert len(check(url)) == 60
    os.environ["API_URL"] = url
    app = AppTest.from_file(Path("app/dashboard.py").resolve(), default_timeout=30).run()
    assert not app.exception
    assert app.metric[0].value == str(meta["beam_subset_count"])
    # Beam A -> beam B -> new origin, using the real API every time.
    app.selectbox(key="beam").set_value("0_0_8").run()
    assert not app.exception
    app.slider(key="origin").set_value(meta["origin_min"] + 24).run()
    assert not app.exception
    expected = client.risk(["0_0_8"], meta["origin_min"] + 24)["beams"][0]["predictions"][0][
        "status"
    ]
    assert app.metric[3].value == expected
