"""Public-source metadata/license audit only; never download or execute model weights locally."""

import json
import zipfile
from pathlib import Path
from urllib.request import urlopen

from huggingface_hub import HfApi, hf_hub_download
from packaging.version import Version

from src.data.io import sha256


def main():
    model_id="amazon/chronos-bolt-base"
    api=HfApi()
    info=api.model_info(model_id,files_metadata=True)
    if info.gated or info.private:
        raise ValueError("Only a public ungated checkpoint is eligible")
    license_name=getattr(info.card_data,"license",None)
    if license_name!="apache-2.0":
        raise ValueError("Unexpected checkpoint license; review before use")
    root=Path(".cache/pretrained-source-audit")
    root.mkdir(parents=True,exist_ok=True)
    names={entry.rfilename for entry in info.siblings}
    metadata={}
    for name in ("README.md","config.json"):
        if name not in names:
            raise ValueError(f"Required public provenance file missing: {name}")
        path=Path(hf_hub_download(model_id,name,revision=info.sha,local_dir=root))
        metadata[name]={"sha256":sha256(path)}
    if "LICENSE" in names:
        path=Path(hf_hub_download(model_id,"LICENSE",revision=info.sha,local_dir=root))
        metadata["LICENSE"]={"sha256":sha256(path)}
    config=json.loads((root/"config.json").read_text())
    weights=[]
    for entry in info.siblings:
        if entry.rfilename.endswith(".safetensors"):
            weights.append({"name":entry.rfilename,"size_bytes":entry.size,
                            "sha256":entry.lfs.sha256 if entry.lfs else None})
    if not weights or any(not row["sha256"] for row in weights):
        raise ValueError("Pinned safetensors SHA256 metadata required")
    with urlopen("https://pypi.org/pypi/chronos-forecasting/json",timeout=30) as response:
        package=json.load(response)
    wheel=next(item for item in package["urls"] if item["packagetype"]=="bdist_wheel")
    wheel_path=root/wheel["filename"]
    with urlopen(wheel["url"],timeout=30) as response:
        wheel_path.write_bytes(response.read())
    if sha256(wheel_path)!=wheel["digests"]["sha256"]:
        raise ValueError("Library distribution checksum mismatch")
    with zipfile.ZipFile(wheel_path) as archive:
        for member in ("chronos/chronos_bolt.py","chronos/base.py"):
            (root/Path(member).name).write_bytes(archive.read(member))
        library_license=next(name for name in archive.namelist() if name.endswith("/LICENSE"))
        (root/"library_LICENSE").write_bytes(archive.read(library_license))
    with urlopen("https://pypi.org/pypi/transformers/json",timeout=30) as response:
        transformers=json.load(response)
    versions=[Version(version) for version in transformers["releases"]]
    transformer_pin=str(max(version for version in versions if version.major==4 and not version.is_prerelease and not version.is_devrelease))
    model_history=api.list_repo_commits(model_id)
    data_history=api.list_repo_commits("netop/gotsf-ds",repo_type="dataset")
    oldest_model=min(model_history,key=lambda commit:commit.created_at)
    oldest_data=min(data_history,key=lambda commit:commit.created_at)
    matching_weights_commit=None
    for commit in sorted(model_history,key=lambda item:item.created_at):
        historical=api.model_info(model_id,revision=commit.commit_id,files_metadata=True)
        digests={entry.rfilename:entry.lfs.sha256 for entry in historical.siblings if entry.lfs}
        if all(digests.get(row["name"])==row["sha256"] for row in weights):
            matching_weights_commit={"revision":commit.commit_id,"created_at":commit.created_at.isoformat()}
            break
    result={"model_id":model_id,"revision":info.sha,"license":license_name,"gated":False,
        "metadata_files":metadata,"weights":weights,"weights_downloaded_locally":False,
        "model_license_evidence":"author-declared apache-2.0 card metadata at pinned revision",
        "separate_model_license_file_present":"LICENSE" in names,
        "model_config":config,"library":"chronos-forecasting","library_version":package["info"]["version"],
        "library_requires_python":package["info"]["requires_python"],
        "library_requires_dist":package["info"].get("requires_dist"),
        "library_distribution":{"filename":wheel["filename"],"sha256":wheel["digests"]["sha256"],"url":wheel["url"]},
        "library_license_sha256":sha256(root/"library_LICENSE"),
        "transformers_version_pin":transformer_pin,
        "oldest_visible_model_commit":{"revision":oldest_model.commit_id,"created_at":oldest_model.created_at.isoformat()},
        "oldest_visible_target_dataset_commit":{"revision":oldest_data.commit_id,"created_at":oldest_data.created_at.isoformat()},
        "earliest_visible_matching_weights_commit":matching_weights_commit,
        "pretraining_overlap_status":"not_proven_absent; visible publication dates and model card are not a complete corpus audit",
        "eligible_scope":"exploratory development-only zero-shot inference; no unseen-test or no-leakage certification",
        "fine_tuning_allowed_by_this_audit":False,"holdout_use_allowed":False,"production_promotion_allowed":False}
    Path("reports/pretrained_forecaster_source_audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({key:result[key] for key in ("model_id","revision","license","weights","library_version",
        "oldest_visible_model_commit","oldest_visible_target_dataset_commit","pretraining_overlap_status")},indent=2))


if __name__=="__main__":
    main()
