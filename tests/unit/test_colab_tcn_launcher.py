"""Transport validation and local refusal only; never start a training process."""

import hashlib
import json
import zipfile

import pytest

from scripts.run_colab_tcn import main, verify_transport


def make_archive(path, extra=None, bad_hash=False):
    content=b"not training data or a model"
    with zipfile.ZipFile(path,"w") as archive:
        archive.writestr("src/example.py",content)
        archive.writestr("colab_source_manifest.json",json.dumps({"source_commit":"fixture","files":{
            "src/example.py":"bad" if bad_hash else hashlib.sha256(content).hexdigest()}}))
        if extra:
            archive.writestr(extra,b"unexpected")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_transport_checks_every_file_and_exact_manifest_coverage(tmp_path):
    path=tmp_path/"source.zip"
    digest=make_archive(path)
    assert verify_transport(path,digest)["source_commit"]=="fixture"
    with pytest.raises(ValueError,match="Transport checksum"):
        verify_transport(path,"bad")


@pytest.mark.parametrize("extra",["../escape.py","C:/escape.py","other.py"])
def test_unapproved_archive_entries_fail_before_extraction(tmp_path,extra):
    path=tmp_path/"source.zip"
    digest=make_archive(path,extra=extra)
    with pytest.raises(ValueError):
        verify_transport(path,digest)
    assert not (tmp_path/"src").exists()


def test_local_launcher_refuses_before_reading_files(monkeypatch):
    monkeypatch.delenv("COLAB_RELEASE_TAG",raising=False)
    with pytest.raises(RuntimeError,match="local training is forbidden"):
        main()
