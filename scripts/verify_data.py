"""Verify a data or paper manifest without downloading assets."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for record in manifest["files"]:
        path = root / record["path"]
        if not path.is_file():
            raise ValueError(f"Missing file: {path}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != record["sha256"]:
            raise ValueError(f"Checksum mismatch (or unhydrated LFS pointer): {path}")
    print(f"Verified {len(manifest['files'])} files")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    verify(p.parse_args().data_root)
