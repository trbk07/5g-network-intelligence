"""Package a self-contained Colab/Kaggle setup notebook without credentials or raw data."""
import base64
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path


def build() -> None:
    """Embed an allowlisted source bundle and verify its hash before cloud extraction."""
    root = Path(__file__).resolve().parents[1]
    paths = [root / "pyproject.toml", root / "dataset_card.md"]
    for folder in ("src", "scripts", "configs", "tests"):
        paths.extend(p for p in (root / folder).rglob("*")
                     if p.suffix in (".py", ".yaml") and "__pycache__" not in p.parts)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.writestr(path.relative_to(root).as_posix(), path.read_bytes())
    source = payload.getvalue()
    encoded = base64.b64encode(source).decode()
    digest = hashlib.sha256(source).hexdigest()
    revision = subprocess.check_output(
        ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
        cwd=root, text=True).strip()
    notebook = json.loads((root / "notebooks/00_runtime_setup.ipynb").read_text())
    notebook["cells"][0]["source"] = [
        "# 5G Network Intelligence - standalone cloud setup\n",
        "Source is embedded: no GitHub token, source ZIP or dataset upload is required.\n",
        "Colab: select a GPU runtime. Kaggle: enable Internet and select a GPU accelerator.\n",
        "Run all cells. Drive authorization must be completed by the account owner.\n",
        f"Source base commit: `{revision}`; bundle SHA256: `{digest}`.\n",
        "This checks infrastructure only; it does not train a model.\n",
    ]
    setup = f'''from pathlib import Path
import base64
import hashlib
import io
import os
import subprocess
import sys
import zipfile

IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or Path('/kaggle/working').is_dir()
IS_COLAB = not IS_KAGGLE and 'COLAB_RELEASE_TAG' in os.environ
if not (IS_COLAB or IS_KAGGLE):
    raise RuntimeError('Open this notebook in Colab or Kaggle')
WORK = Path('/kaggle/working' if IS_KAGGLE else '/content')
PROJECT = WORK / '5g-network-intelligence'
source = base64.b64decode({encoded!r})
assert hashlib.sha256(source).hexdigest() == {digest!r}, 'Source bundle hash mismatch'
PROJECT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(source)) as archive:
    for member in archive.infolist():
        if not (PROJECT / member.filename).resolve().is_relative_to(PROJECT.resolve()):
            raise ValueError('Unsafe archive member')
    archive.extractall(PROJECT)
os.chdir(PROJECT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.[dev]'], check=True)
print('Project ready:', PROJECT)
'''
    notebook["cells"][1]["source"] = setup.splitlines(keepends=True)
    for cell in notebook["cells"]:
        cell["id"] = hashlib.sha256("".join(cell["source"]).encode()).hexdigest()[:12]
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "<cloud-notebook>", "exec")
    output = root / "notebooks/00_cloud_standalone.ipynb"
    output.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Built {output.name}: {len(source)} source bytes; Python cells compile")


if __name__ == "__main__":
    build()
