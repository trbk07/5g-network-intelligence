"""Build an explicit hosted training notebook; no source bundles or local fitting."""

import json
from pathlib import Path

from scripts.build_week6_job import build


def main():
    stage = build("preflight")
    notebook = json.loads((stage / "week6.ipynb").read_text())
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        source = source.replace("scripts.run_risk", "scripts.research_week7").replace(
            "['scripts.research_week7', '--segment', 'val']", "['scripts.research_week7']"
        )
        source = source.replace(
            "'reports','configs','artifacts/champion'", "'reports','configs','artifacts/lgbm_w7'"
        )
        # Packaging template stores the archive tuple without spaces.
        source = source.replace("week6_results.tar.gz", "week7_research_results.tar.gz")
        if "'m','pip','install'" in source:
            source = source.replace(".[dev,ml,neural]", ".[dev,ml,neural,app]")
        # This worker runs the research script, not the product test suite.
        if "week6_cloud_tests.log" in source:
            source = (
                "subprocess.run([sys.executable,'-m','pip','install','-e','.[ml]'],check=True)\n"
            )
        cell["source"] = source.splitlines(keepends=True)
    notebook["cells"][0]["source"] = [
        "# Week 7 - Validation model research\n\nHosted CPU training. Four explicit parameter sets, original fit/validation only. Production champion and final holdout remain unchanged.\n"
    ]
    output = Path(".cache/5g-network-intelligence-week7-research")
    output.mkdir(parents=True, exist_ok=True)
    (output / "research.ipynb").write_text(
        json.dumps(notebook, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    metadata = json.loads((stage / "kernel-metadata.json").read_text())
    metadata.update(
        id="truongbaokien/5g-network-intelligence-week7-research",
        title="5G Network Intelligence Week7 Research",
        code_file="research.ipynb",
    )
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    print(output)


if __name__ == "__main__":
    main()
