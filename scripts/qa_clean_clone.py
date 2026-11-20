"""Real clean-clone Docker QA from committed remote source and explicit verified asset archive."""

import argparse
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

import httpx
import numpy as np

from src.data.io import sha256


def run(keep=False):
    root = Path.cwd()
    target = root / ".cache" / f"clean-clone-{int(time.time())}"
    package = root / ".cache/release/5g-demo-assets.tar.gz"
    expected = json.loads((root / "reports/demo_asset_package.json").read_text())
    if sha256(package) != expected["sha256"]:
        raise ValueError("Asset archive checksum mismatch")
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/trbk07/5g-network-intelligence.git",
            str(target),
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-c", f"safe.directory={target.as_posix()}", "rev-parse", "HEAD"],
        cwd=target,
        text=True,
    ).strip()
    with tarfile.open(package) as archive:
        members = archive.getmembers()
        if any(
            not (target / m.name).resolve().is_relative_to(target) or m.issym() or m.islnk()
            for m in members
        ):
            raise ValueError("Unsafe demo archive")
        archive.extractall(target, filter="data")
    env = {**os.environ, "API_PORT": "18000", "DASHBOARD_PORT": "18501"}
    command = ["docker", "compose", "-p", "fiveg-cleanqa"]
    log = root / "reports/clean_clone_build.log"
    results = {
        "commit": commit,
        "source": "fresh remote GitHub clone",
        "assets_sha256": expected["sha256"],
        "local_training": False,
        "checks": {},
    }
    try:
        with log.open("w", encoding="utf-8") as handle:
            subprocess.run(
                command + ["up", "--build", "-d", "--wait", "--wait-timeout", "120"],
                cwd=target,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=600,
            )
        with httpx.Client(base_url="http://127.0.0.1:18000", timeout=20) as client:
            health = client.get("/health")
            health.raise_for_status()
            assert health.json()["beams_loaded"] == 24
            assert client.get("/docs").status_code == 200
            assert client.get("/model-info").json()["is_replay"] is True
            meta = client.get("/replay/meta").json()
            assert meta["beam_subset_count"] == 24
            payload = {"origin": 671, "beam_ids": ["0_0_0", "5_1_8"]}
            forecast = client.post("/forecast", json=payload)
            forecast.raise_for_status()
            risk = client.post("/risk", json=payload)
            risk.raise_for_status()
            for a, b in zip(forecast.json()["beams"], risk.json()["beams"]):
                assert len(a["history"]) == 168
                np.testing.assert_array_equal(
                    [p["point"] for p in a["predictions"]], [p["point"] for p in b["predictions"]]
                )
            for payload, status in [
                ({"origin": 670, "beam_ids": ["0_0_0"]}, 422),
                ({"origin": 671, "beam_ids": ["unknown"]}, 404),
            ]:
                assert client.post("/forecast", json=payload).status_code == status
            results["checks"].update(
                health=True,
                model_info=True,
                replay_meta=True,
                forecast=True,
                risk=True,
                docs=True,
                negative_404_422=True,
                forecast_risk_parity=True,
            )
        response = httpx.get("http://127.0.0.1:18501/_stcore/health", timeout=15)
        response.raise_for_status()
        results["checks"]["dashboard_health"] = True
        output = subprocess.check_output(
            command + ["ps", "--format", "json"], cwd=target, env=env, text=True
        )
        results["compose_ps"] = [json.loads(line) for line in output.splitlines() if line.strip()]
        assert all(s["Health"] == "healthy" for s in results["compose_ps"])
        results["status"] = "PASS"
    except Exception as exc:
        results.update(status="FAIL", error=str(exc))
        raise
    finally:
        (root / "reports/clean_clone_qa.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8", newline="\n"
        )
        if not keep:
            subprocess.run(command + ["down"], cwd=target, env=env, check=False, timeout=60)
    print("Clean remote clone + Docker + explicit assets:", results["status"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true")
    run(parser.parse_args().keep)
