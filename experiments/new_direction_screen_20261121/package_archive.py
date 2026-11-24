"""Package a bounded research screen without modifying historical archives."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_new(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(source) != sha(target):
            raise FileExistsError(target)
        return
    shutil.copy2(source, target)


def package(repo: Path, staging: Path, archive: Path, previous: list[Path]) -> None:
    if archive.name != "new-direction-screen-20261121" or not archive.is_dir():
        raise ValueError("Use the explicitly named existing new archive.")
    required = ["REPORT.md", "DATA_DICTIONARY.md", "VALIDATION_CHECKLIST.md",
                "SCREEN_REGISTER.md", "INVENTORY.md", "DEPTH_AUDIT.md",
                "ENDPOINT_PRIOR_AUDIT.md", "ENDPOINT_MATH_REVIEW.md",
                "STOCHASTIC_AUDIT.md", "POISONING_AUDIT.md", "TOPOLOGY_AUDIT.md"]
    for name in required:
        if not (staging / name).is_file():
            raise FileNotFoundError(name)
    previous_checks = []
    for old_archive in previous:
        old = json.loads((old_archive / "manifest.json").read_text(encoding="utf-8"))
        for name, row in old["files"].items():
            if sha(old_archive / name) != row["sha256"]:
                raise ValueError(f"Historical archive changed: {old_archive / name}")
        previous_checks.append({"archive": str(old_archive), "unchanged_files": len(old["files"]),
                                "manifest_sha256": sha(old_archive / "manifest.json")})
    for source in staging.rglob("*"):
        if not source.is_file() or "__pycache__" in source.parts:
            continue
        relative = source.relative_to(staging)
        destination = archive / relative
        copy_new(source, destination)
    sources = [p for p in (repo / "experiments/new_direction_screen_20261121").iterdir()
               if p.is_file() and p.suffix in (".py", ".md")]
    sources += [repo / "src/pullback_feature_learning" / n for n in ("core.py", "data.py", "__init__.py")]
    sources += [repo / n for n in ("AGENTS.md", "pyproject.toml", "requirements-lock.txt")]
    for source in sources:
        copy_new(source, archive / "source_snapshot" / source.relative_to(repo))
    source_lookup = {sha(p): str(p.relative_to(archive)) for p in archive.rglob("*.py")}
    provenance = []
    for path in archive.rglob("*.json"):
        if path.name in ("manifest.json", "ARCHIVE_CHECK.json"):
            continue
        record = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(record, dict):
            continue
        for field, value in record.items():
            if (field.endswith("source_sha256") or field in ("core_sha256", "endpoint_probe_sha256")) and isinstance(value, str):
                if value not in source_lookup:
                    raise ValueError(f"Unresolved source hash {path}: {field}")
                provenance.append({"record": str(path.relative_to(archive)), "field": field,
                                   "resolved": source_lookup[value]})
    broken = []
    for report in archive.glob("*.md"):
        for link in re.findall(r"\]\(([^)]+)\)", report.read_text(encoding="utf-8")):
            if link in ("ARCHIVE_CHECK.json", "manifest.json"):
                continue  # Both are produced immediately below.
            if "://" not in link and not link.startswith("#") and not (report.parent / link.split("#")[0]).exists():
                broken.append({"report": report.name, "target": link})
    if broken:
        raise ValueError(broken)
    files = {p.relative_to(archive).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
             for p in sorted(archive.rglob("*")) if p.is_file()
             and p.name not in ("manifest.json", "ARCHIVE_CHECK.json")}
    manifest = archive / "manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    manifest.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(), "files": files}, indent=2) + "\n", encoding="utf-8")
    for name, row in files.items():
        assert sha(archive / name) == row["sha256"]
    checks = {"status": "passed", "file_count": len(files), "total_bytes": sum(row["bytes"] for row in files.values()),
              "manifest_sha256": sha(manifest), "source_provenance": provenance,
              "previous_archives": previous_checks, "broken_report_links": broken,
              "checked_utc": datetime.now(timezone.utc).isoformat()}
    (archive / "ARCHIVE_CHECK.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in checks.items() if key != "source_provenance"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("repo", "staging", "archive"):
        parser.add_argument("--" + field, required=True, type=Path)
    parser.add_argument("--previous", required=True, type=Path, nargs="+")
    args = parser.parse_args()
    package(args.repo.resolve(), args.staging.resolve(), args.archive.resolve(), [p.resolve() for p in args.previous])
