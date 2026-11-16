#!/usr/bin/env python3
"""Verify and unpack the skill archives included in the review materials."""
import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'paper' / 'data' / 'skillsets'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(size):
    name = f'skills_{size}'
    archive = BASE / 'archives' / f'{name}.tar.gz'
    records = json.loads((BASE / 'archive_manifest.json').read_text())['archives']
    expected = next(row for row in records if row['name'] == archive.name)
    if not archive.is_file() or digest(archive) != expected['sha256']:
        raise ValueError(f'Missing archive or SHA-256 mismatch: {archive}; run scripts/download_data.sh first')
    dest = BASE / name
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        if any(Path(m.name).parts[0] != name for m in members):
            raise ValueError(f'Unexpected archive root: {archive}')
        expected_files = [m for m in members if m.isfile()]
        if dest.exists():
            # Do not silently reuse a partial or modified skill library.
            for member in expected_files:
                relative = Path(member.name).relative_to(name)
                file = dest / relative
                if not file.is_file() or file.is_symlink() or file.stat().st_size != member.size:
                    raise ValueError(f'Existing skillset is incomplete or modified: {file}')
                with tar.extractfile(member) as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != digest(file):
                        raise ValueError(f'Existing skill file differs: {file}')
            actual = {p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()}
            wanted = {Path(m.name).relative_to(name).as_posix() for m in expected_files}
            if actual != wanted:
                raise ValueError(f'Existing skillset has extra or missing files: {dest}')
            print(f'Verified existing {dest}')
            return
        with tempfile.TemporaryDirectory(prefix='.unpack-', dir=BASE) as tmp:
            tar.extractall(tmp, filter='data')
            unpacked = Path(tmp) / name
            found = len(list(unpacked.glob('*/SKILL.md')))
            if found != size:
                raise ValueError(f'{name}: expected {size} skills, got {found}')
            unpacked.rename(dest)
    print(f'Prepared {size} skills: {dest}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'paper' / 'data')
    parser.add_argument('--sizes', type=int, nargs='+', choices=[200, 500, 1000, 2000], default=[200, 500, 1000, 2000])
    args = parser.parse_args()
    BASE = args.data_root.resolve() / 'skillsets'
    for size in dict.fromkeys(args.sizes):
        prepare(size)
