"""Acquire publisher-licensed hourly CESNET data for a cross-domain audit, no training."""

import hashlib
import json
from pathlib import Path
import urllib.request


def get_json(url):
    request=urllib.request.Request(url,headers={"User-Agent":"5g-network-intelligence-research/1.0"})
    with urllib.request.urlopen(request,timeout=60) as response:
        return json.load(response)


def main():
    original=get_json("https://zenodo.org/api/records/13382427")
    distribution=get_json("https://zenodo.org/api/records/23296272")
    for record in (original,distribution):
        license_id=record["metadata"].get("license",{}).get("id")
        if license_id not in ("cc-by-4.0","CC-BY-4.0"):
            raise ValueError("Publisher CC-BY-4.0 license must be verified before acquisition")
    root=Path("data/external/cesnet/23296272")
    root.mkdir(parents=True,exist_ok=True)
    selected=("CESNET-TimeSeries24-institutions-hour.h5","SHA256SUMS.txt")
    files=[]
    for name in selected:
        entry=next(f for f in distribution["files"] if f["key"]==name)
        expected=entry["checksum"]
        if not expected.startswith("md5:"):
            raise ValueError("Expected publisher MD5 checksum")
        target=root/name
        if not target.exists():
            temporary=target.with_suffix(target.suffix+".part")
            request=urllib.request.Request(entry["links"]["self"],headers={"User-Agent":"5g-network-intelligence-research/1.0"})
            downloaded=0
            with urllib.request.urlopen(request,timeout=60) as response,temporary.open("wb") as output:
                while chunk:=response.read(1024*1024):
                    output.write(chunk)
                    downloaded+=len(chunk)
                    if downloaded%(16*1024*1024)==0:
                        print(name,downloaded,"bytes downloaded",flush=True)
            if hashlib.md5(temporary.read_bytes()).hexdigest()!=expected.split(":",1)[1]:
                raise ValueError("Publisher download checksum mismatch")
            temporary.replace(target)
        content=target.read_bytes()
        if len(content)!=entry["size"] or hashlib.md5(content).hexdigest()!=expected.split(":",1)[1]:
            raise ValueError("Existing CESNET download differs from publisher metadata")
        files.append({"path":str(target),"file":name,"bytes":len(content),"publisher_checksum":expected,
            "sha256":hashlib.sha256(content).hexdigest(),"source_url":entry["links"]["self"]})
    sums=(root/"SHA256SUMS.txt").read_text()
    primary=files[0]
    matching=[line for line in sums.splitlines() if line.rstrip().endswith(primary["file"])]
    if len(matching)!=1 or matching[0].split()[0].lower()!=primary["sha256"]:
        raise ValueError("Hourly file does not match publisher SHA256SUMS")
    evidence={"original_record":13382427,"distribution_record":23296272,"license":"CC-BY-4.0",
        "original_doi":original["metadata"]["doi"],"distribution_doi":distribution["metadata"]["doi"],
        "title":original["metadata"]["title"],"creators":original["metadata"]["creators"],
        "distribution_metadata_modified":distribution["modified"],"files":files,
        "publisher_MD5_and_SHA256_verified":True,"training_performed":False,
        "domain":"IP traffic, not PRB; candidate for separately evaluated temporal representation pretraining only",
        "aggregation_source":"Publisher-provided hourly HDF5; no local resampling or fit",
        "remaining_gate":"Schema/time/duplicates/coverage and source overlap audit; no accuracy improvement claimed"}
    Path("reports/cesnet_acquisition.json").write_text(json.dumps(evidence,indent=2,ensure_ascii=True))
    (root/"publisher_metadata.json").write_text(json.dumps({"original":original,"distribution":distribution},ensure_ascii=True))
    print(json.dumps({"files":len(files),"hourly_bytes":primary["bytes"],"sha256":primary["sha256"],"license":evidence["license"]}),flush=True)


if __name__=="__main__":
    main()
