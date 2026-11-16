import hashlib
import importlib.util
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

SPEC = importlib.util.spec_from_file_location("download_data", Path(__file__).resolve().parents[1] / "scripts/download_data.py")
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


def archive_manifest(root, payload):
    relative = "skillsets/archives/skills_200.tar.gz"
    record = {"path": relative, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    (root / "manifest.json").write_text(json.dumps({"files": [record]}))
    target = root / relative
    target.parent.mkdir(parents=True)
    return target


def test_resolves_lfs_pointer_and_skips_verified_archive(tmp_path):
    payload = b"archive payload"
    target = archive_manifest(tmp_path, payload)
    target.write_bytes(downloader.LFS_HEADER + b"oid sha256:placeholder\n")
    with patch.object(downloader, "urlopen", return_value=io.BytesIO(payload)) as fetch:
        downloader.download(tmp_path)
        downloader.download(tmp_path)
    assert fetch.call_count == 1
    assert fetch.call_args.args[0].startswith(downloader.BASE_URL)
    assert target.read_bytes() == payload


def test_failed_download_preserves_pointer_and_removes_partial(tmp_path):
    target = archive_manifest(tmp_path, b"expected payload")
    pointer = downloader.LFS_HEADER + b"oid sha256:placeholder\n"
    target.write_bytes(pointer)
    with patch.object(downloader, "urlopen", return_value=io.BytesIO(b"wrong content")):
        with pytest.raises(ValueError, match="failed checksum"):
            downloader.download(tmp_path)
    assert target.read_bytes() == pointer
    assert not list(target.parent.glob(".download-*"))


def test_preserves_existing_archive_with_different_checksum(tmp_path):
    target = archive_manifest(tmp_path, b"expected payload")
    target.write_bytes(b"local modified archive")
    with patch.object(downloader, "urlopen") as fetch:
        with pytest.raises(ValueError, match="Existing archive"):
            downloader.download(tmp_path)
    fetch.assert_not_called()
    assert target.read_bytes() == b"local modified archive"
