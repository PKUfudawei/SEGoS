"""Fetch missing review archives through the anonymous URL and verify SHA-256."""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://anonymous.4open.science/api/repo/SEGoS/file/paper/data/"
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1\n"


def matches(path, record):
    if path.stat().st_size != record["bytes"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == record["sha256"]


def download(root, base_url=BASE_URL):
    records = json.loads((root / "manifest.json").read_text())["files"]
    for record in records:
        relative = PurePosixPath(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid manifest path: {relative}")
        if not str(relative).endswith(".tar.gz"):
            continue
        target = root / relative
        if target.exists():
            if matches(target, record):
                print(f"Verified {relative}", flush=True)
                continue
            with target.open("rb") as stream:
                pointer = stream.read(len(LFS_HEADER)) == LFS_HEADER
            if not pointer:
                raise ValueError(f"Existing archive has a different checksum: {target}; preserve or remove it before retrying")
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = None
        print(f"Downloading {relative} ({record['bytes']:,} bytes)", flush=True)
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".download-", delete=False) as output:
                temp = Path(output.name)
                with urlopen(base_url.rstrip("/") + "/" + quote(str(relative), safe="/"), timeout=120) as response:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            if not matches(temp, record):
                raise ValueError(f"Downloaded archive failed checksum verification: {relative}")
            os.replace(temp, target)
            print(f"Verified {relative}", flush=True)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "paper" / "data")
    args = parser.parse_args()
    download(args.data_root.resolve())
