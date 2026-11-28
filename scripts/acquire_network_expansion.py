"""Acquire licensed additional network corpora with publisher MD5/SHA256 checks."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


def fetch(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-large", action="store_true", help="Also acquire the 5.1 GB hourly Telecom Italia file")
    args = parser.parse_args()
    record_id = "22721975"
    record = json.loads(fetch(f"https://zenodo.org/api/records/{record_id}"))
    by_name = {item["key"]: item for item in record["files"]}
    sums = fetch(by_name["SHA256SUMS.txt"]["links"]["self"]).decode()
    sha_by_name = {}
    for line in sums.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            sha_by_name[parts[1].lstrip(" *")] = parts[0]
    selected = [name for name in by_name if
                (name.startswith("Vigo-NetworkOperatorKPIs-") and name.endswith("-1_hour.h5"))
                or name == "CESNET-AGG23-CESNET2-minute.h5"]
    if args.include_large:
        selected.append("TelecomItalia-telecommunications-service_provider-1_hour.h5")
    if len(selected) != (6 if args.include_large else 5):
        raise ValueError("Expected four Vigo domains, AGG23 and optional Telecom Italia")
    root = Path("data/external/network_expansion")
    root.mkdir(parents=True, exist_ok=True)
    (root/"publisher_metadata.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (root/"SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    evidence = {"record": f"https://zenodo.org/records/{record_id}", "files": [],
                "training_performed": False, "row_level_audit_complete": False,
                "note": "Raw acquisition only; prefix, units, gaps and overlap still require audit"}
    for name in selected:
        if Path(name).name != name or name not in sha_by_name:
            raise ValueError("Unsafe or unchecksummed publisher file")
        item = by_name[name]
        destination, partial = root/name, root/(name+".part")
        if not destination.exists():
            offset = partial.stat().st_size if partial.exists() else 0
            if shutil.disk_usage(root).free < item["size"]-offset+512*1024**2:
                raise OSError("Insufficient disk space for selected file")
            request = urllib.request.Request(item["links"]["self"],
                headers={"Range": f"bytes={offset}-"} if offset else {})
            with urllib.request.urlopen(request, timeout=120) as response:
                if offset and response.status != 206:
                    raise RuntimeError("Server did not honor resume; preserve partial file")
                with partial.open("ab" if offset else "wb") as output:
                    progress = offset
                    print(json.dumps({"file": name, "download_started_bytes": offset, "expected_bytes": item["size"]}), flush=True)
                    while chunk := response.read(8*1024**2):
                        output.write(chunk)
                        offset += len(chunk)
                        if offset-progress >= 128*1024**2:
                            print(json.dumps({"file": name, "downloaded_bytes": offset}), flush=True)
                            progress = offset
            candidate = partial
        else:
            candidate = destination
        md5, sha = hashlib.md5(), hashlib.sha256()
        with candidate.open("rb") as stream:
            while chunk := stream.read(8*1024**2):
                md5.update(chunk)
                sha.update(chunk)
        if (candidate.stat().st_size != item["size"] or "md5:"+md5.hexdigest() != item["checksum"]
                or sha.hexdigest() != sha_by_name[name]):
            raise ValueError("Publisher checksum/size mismatch: " + name)
        if candidate != destination:
            candidate.replace(destination)
        telecom = name.startswith("TelecomItalia-")
        evidence["files"].append({"path": destination.as_posix(), "bytes": item["size"],
            "sha256": sha.hexdigest(), "md5": md5.hexdigest(),
            "license": "ODbL-1.0" if telecom else "CC-BY-4.0",
            "attribution": "from BigDataChallenge contest — http://www.telecomitalia.com/tit/en/bigdatachallenge.html" if telecom else
                "Original dataset authors and CESNET TS-Zoo conversion; see publisher metadata",
            "training_ready": False})
        Path("reports/network_expansion_acquisition.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print(json.dumps(evidence["files"][-1]), flush=True)


if __name__ == "__main__":
    main()
