"""Tiny publisher-response fixtures; no real downloads or model training."""

import hashlib
import io
import json

import pytest

import scripts.acquire_cesnet_timeseries as module


def fixture_responses(monkeypatch,bad_hash=False,license_id="cc-by-4.0"):
    name="CESNET-TimeSeries24-institutions-hour.h5"
    content=b"small transport fixture, not real HDF5"
    sums=(hashlib.sha256(content).hexdigest()+"  "+name+"\n").encode()
    payload={"hourly":content,"sums":sums}
    metadata={"license":{"id":license_id},"doi":"fixture-doi","title":"Fixture","creators":[{"name":"Fixture"}]}
    entries=[{"key":n,"size":len(b),"checksum":"md5:"+("bad" if bad_hash else hashlib.md5(b).hexdigest()),"links":{"self":url}}
             for n,b,url in ((name,content,"hourly"),("SHA256SUMS.txt",sums,"sums"))]
    monkeypatch.setattr(module,"get_json",lambda url:{"metadata":metadata,"files":entries,"modified":"fixture"})
    monkeypatch.setattr(module.urllib.request,"Request",lambda url,headers:url)
    monkeypatch.setattr(module.urllib.request,"urlopen",lambda request,timeout:io.BytesIO(payload[request]))


def test_both_publisher_checksums_and_license_are_required(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path/"reports").mkdir()
    fixture_responses(monkeypatch)
    module.main()
    evidence=json.loads((tmp_path/"reports/cesnet_acquisition.json").read_text())
    assert evidence["publisher_MD5_and_SHA256_verified"]
    assert evidence["training_performed"] is False


def test_wrong_checksum_is_not_installed_as_a_verified_file(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    fixture_responses(monkeypatch,bad_hash=True)
    with pytest.raises(ValueError,match="checksum"):
        module.main()
    assert not list(tmp_path.rglob("*.h5"))


def test_unverified_license_stops_before_acquisition(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    fixture_responses(monkeypatch,license_id="unspecified")
    with pytest.raises(ValueError,match="license"):
        module.main()
    assert not (tmp_path/"data").exists()
