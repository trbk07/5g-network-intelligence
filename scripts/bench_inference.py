"""Measure CPU inference only: load, memory, warm latency and actual container sizes."""

import argparse
import json
import platform
import subprocess
import time
import tracemalloc
from pathlib import Path

import numpy as np

from src.inference.predictor import Predictor


def benchmark(repetitions=100, warmup=10):
    if repetitions < 20 or warmup < 1:
        raise ValueError("At least 20 measured requests and one warm-up required")
    tracemalloc.start()
    start = time.perf_counter()
    predictor = Predictor()
    load_ms = 1000 * (time.perf_counter() - start)
    memory = {
        "python_traced_current_bytes": tracemalloc.get_traced_memory()[0],
        "python_traced_peak_bytes": tracemalloc.get_traced_memory()[1],
        "limitation": "Python allocation tracing excludes native LightGBM/NumPy memory.",
    }
    tracemalloc.stop()
    try:
        import psutil

        memory["process_rss_bytes_after_load"] = psutil.Process().memory_info().rss
    except ImportError:
        memory["process_rss_bytes_after_load"] = None
    results = []
    ids = list(predictor.store.index)
    for count in (1, 10):
        beams = ids[:count]
        for _ in range(warmup):
            predictor.predict_risk(beams, 671)
        times = []
        for _ in range(repetitions):
            before = time.perf_counter()
            result = predictor.predict_risk(beams, 671)
            times.append(1000 * (time.perf_counter() - before))
            assert len(result["beams"]) == count
        results.append(
            {
                "beams": count,
                "warmup": warmup,
                "repetitions": repetitions,
                "p50_ms": float(np.percentile(times, 50)),
                "p95_ms": float(np.percentile(times, 95)),
                "min_ms": min(times),
                "max_ms": max(times),
            }
        )
    images = []
    try:
        raw = subprocess.check_output(
            [
                "docker",
                "image",
                "inspect",
                "5g-network-intelligence-api:week7",
                "5g-network-intelligence-dashboard:week7",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        images = [
            {"tags": i["RepoTags"], "size_bytes": i["Size"], "id": i["Id"]} for i in json.loads(raw)
        ]
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    cpu = platform.processor()
    if platform.system() == "Windows":
        try:
            cpu = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            pass
    evidence = {
        "scope": "Local CPU shared Predictor.predict_risk, 3 horizons + 168-point histories; no training",
        "model": predictor.info(),
        "cpu": cpu,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "model_load_ms": load_ms,
        "memory": memory,
        "latency": results,
        "docker_images": images,
        "limitations": [
            "Single process, sequential warm requests; not a production SLA or load test.",
            "Cold model loading and warm inference are reported separately.",
            "Image sizes reflect the deployed LightGBM profile only; baseline/GRU image sizes are not measured.",
        ],
    }
    reports = Path("reports")
    (reports / "inference_benchmark.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8", newline="\n"
    )
    lines = [
        "# CPU inference benchmark",
        "",
        evidence["scope"],
        "",
        f"CPU: {cpu}. Python: {platform.python_version()}.",
        f"Model load: {load_ms:.2f} ms. RSS after load: {memory['process_rss_bytes_after_load']} bytes.",
        "",
        "| Beams | Warm-up | Requests | P50 (ms) | P95 (ms) |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {r['beams']} | {warmup} | {repetitions} | {r['p50_ms']:.3f} | {r['p95_ms']:.3f} |"
        for r in results
    ]
    lines += ["", "## Actual Docker footprint", ""]
    lines += [f"- {i['tags']}: {i['size_bytes']:,} bytes." for i in images]
    lines += ["", "## Limits", ""] + [f"- {s}" for s in evidence["limitations"]]
    (reports / "inference_benchmark.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"load_ms": load_ms, "latency": results}, indent=2))
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    benchmark(args.repetitions, args.warmup)
