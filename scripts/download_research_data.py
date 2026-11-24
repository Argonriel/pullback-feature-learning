"""Download and verify the published research arrays and model checkpoints."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", action="append", help="Study name; repeat to select several. Omit to select all.")
    parser.add_argument("--list", action="store_true", help="List available archives without downloading.")
    parser.add_argument("--verify-only", action="store_true", help="Verify extracted data without downloading or writing.")
    parser.add_argument("--cache", type=Path, default=ROOT / ".research-downloads")
    parser.add_argument("--output", type=Path, default=ROOT / "research/archive")
    args = parser.parse_args()
    assets = json.loads((ROOT / "research/DATA_ASSETS.json").read_text())["assets"]
    selected = [a for a in assets if not args.study or a["study"] in args.study]
    if args.study and set(args.study) - {a["study"] for a in assets}:
        parser.error("Unknown study; use --list to see available names.")
    if args.list:
        for asset in selected:
            print(f"{asset['study']}: {asset['name']} ({asset['bytes'] / 1024**2:.1f} MiB)")
        return
    names = {a["name"] for a in selected}
    manifest = json.loads((ROOT / "research/PUBLICATION_MANIFEST.json").read_text())
    records = {r["path"]: r for r in manifest["files"] if r.get("asset") in names}
    output = args.output.resolve()
    if args.verify_only:
        failures = [name for name, record in records.items()
                    if not (output / name).is_file() or digest(output / name) != record["sha256"]]
        print(json.dumps({"files": len(records), "mismatches_or_missing": failures}, indent=2))
        raise SystemExit(bool(failures))
    args.cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    for asset in selected:
        target = args.cache / asset["name"]
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".partial")
            print(f"Downloading {asset['name']}", flush=True)
            request = urllib.request.Request(asset["url"], headers={"User-Agent": "pullback-research-data"})
            with urllib.request.urlopen(request, timeout=120) as source, temporary.open("wb") as dest:
                shutil.copyfileobj(source, dest, length=4 * 1024 * 1024)
            if temporary.stat().st_size != asset["bytes"] or digest(temporary) != asset["sha256"]:
                raise RuntimeError(f"Downloaded archive failed verification: {asset['name']}")
            temporary.replace(target)
        if target.stat().st_size != asset["bytes"] or digest(target) != asset["sha256"]:
            raise RuntimeError(f"Cached archive failed verification: {target}")
        with zipfile.ZipFile(target) as archive:
            expected = {name for name, r in records.items() if r["asset"] == asset["name"]}
            if set(archive.namelist()) != expected:
                raise RuntimeError(f"Archive members do not match the manifest: {asset['name']}")
            for member in archive.infolist():
                dest = (output / member.filename).resolve()
                if not dest.is_relative_to(output):
                    raise RuntimeError("Archive entry escapes the output directory")
                if dest.exists():
                    if digest(dest) == records[member.filename]["sha256"]:
                        continue
                    raise RuntimeError(f"Refusing to overwrite a different existing file: {dest}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, dest.open("xb") as stream:
                    shutil.copyfileobj(source, stream, length=4 * 1024 * 1024)
                if digest(dest) != records[member.filename]["sha256"]:
                    raise RuntimeError(f"Extracted file failed verification: {dest}")
        print(f"Verified {asset['name']}", flush=True)

if __name__ == "__main__":
    main()
