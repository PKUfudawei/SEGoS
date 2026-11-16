"""Export paper evidence and skill archives, preserving the reviewed task archive."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def public_metadata(value):
    """Keep runtime graph fields and scores; omit free-form relation review notes."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "reviewed_relation" and isinstance(item, dict):
                item = {k: v for k, v in item.items() if k not in {"reason", "condition"}}
            result[key] = public_metadata(item)
        return result
    if isinstance(value, list):
        return [public_metadata(item) for item in value]
    if isinstance(value, str) and value.startswith("round(p/n,3), 0<=p<=n<=48, n>=1;"):
        return "round(p/n,3), 0<=p<=n<=48, n>=1; 0 or 1 when no valid counts are available"
    return value


def export(data_root, output):
    for relative in ("benchmarks/skillsbench/tasks.tar.gz", "benchmarks/skillsbench/archive_manifest.json"):
        if not (output / "data" / relative).is_file():
            raise ValueError(f"Provide the reviewed task package before export: {relative}")
    manifest = json.loads((data_root / "manifest.json").read_text())
    selected = []
    for record in manifest["files"]:
        relative = Path(record["path"])
        if relative.parts[0] not in {"experiments", "benchmarks", "skillsets"}:
            continue
        if "traces" in relative.parts or relative.name in {"tasks.tar.gz", "rename_report.json", "path_migration.json"}:
            continue
        if relative.suffix not in {".json", ".jsonl"} and "evolution" not in relative.parts and relative.parts[0] != "skillsets":
            continue
        source = data_root / relative
        if digest(source) != record["sha256"]:
            raise ValueError(f"Source checksum mismatch: {source}")
        target = output / "data" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative.suffix == ".json":
            payload = public_metadata(json.loads(source.read_text()))
            target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        elif relative.suffix == ".jsonl":
            rows = [public_metadata(json.loads(line)) for line in source.read_text().splitlines() if line.strip()]
            target.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows))
        else:
            shutil.copy2(source, target)
        selected.append(str(relative))
    # The task archive has its own reviewed packaging (local Git reflogs removed).
    # Never overwrite it with an unsanitized archive from the source data tree.
    selected.extend(["benchmarks/skillsbench/tasks.tar.gz", "benchmarks/skillsbench/archive_manifest.json", "benchmarks/alfworld/alfworld_textworld.tar.gz", "benchmarks/alfworld/archive_manifest.json", "manifest.json"])
    existing = output / "data"
    def unpacked_input(path):
        parts = path.relative_to(output).parts
        return len(parts) >= 3 and parts[:2] == ("data", "skillsets") and parts[2].startswith("skills_")
    if existing.exists():
        for path in existing.rglob("*"):
            if path.is_file() and not unpacked_input(path) and str(path.relative_to(existing)) not in selected:
                raise ValueError(f"Unexpected old export file; use a clean output directory: {path}")
    data_records = [{"path": str(p.relative_to(existing)), "bytes": p.stat().st_size, "sha256": digest(p)}
                    for p in sorted(existing.rglob("*")) if p.is_file() and not unpacked_input(p) and p != existing / "manifest.json"]
    (existing / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": data_records}, indent=2)+"\n")
    records = [{"path": str(p.relative_to(output)), "bytes": p.stat().st_size, "sha256": digest(p)}
               for p in sorted(output.rglob("*")) if p.is_file() and not unpacked_input(p) and p != output / "manifest.json"]
    (output / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": records}, indent=2)+"\n")
    print(f"Exported {len(selected)} data files; manifest covers {len(records)} paper files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "paper")
    args = parser.parse_args()
    export(args.data_root, args.output)
