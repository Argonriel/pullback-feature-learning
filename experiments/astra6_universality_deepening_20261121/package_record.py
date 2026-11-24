"""Snapshot this new research package and verify its read-only dependencies."""
from __future__ import annotations
import hashlib
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\Users\liwei\Documents\ChatGPT\277")
STAGE = ROOT / "tmp/astra6-universality-deepening-20261121"
CODE = ROOT / "pullback-feature-learning/experiments/astra6_universality_deepening_20261121"
OLD = Path(r"C:\Users\liwei\Documents\Research\pullback-feature-learning\explorations")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    files_to_copy = {
        "three-candidate-exploration-20261121": [
            "REPORT.md", "DATA_DICTIONARY.md", "universality/THEOREM.md",
            "universality/REVIEW.md", "universality/SEARCH_LEDGER.md",
            "universality/validation/results.json"],
        "endpoint-classification-audit-20261121": [
            "ALGEBRA_REDUCTION.md", "FINITEDAMPING_AUDIT.md", "REFEREE_AUDIT.md"]}
    original_records = []
    for package, names in files_to_copy.items():
        for name in names:
            src = OLD/package/name
            dest = STAGE/"reference_inputs"/package/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            original_records.append({"original": str(src), "snapshot": dest.relative_to(STAGE).as_posix(),
                                     "sha256": digest(src), "identical": digest(src) == digest(dest)})
    source_snapshot = STAGE/"source_snapshot"/CODE.name
    source_snapshot.mkdir(parents=True, exist_ok=True)
    for src in CODE.iterdir():
        if src.is_file():
            shutil.copyfile(src, source_snapshot/src.name)
    archive_checks = []
    for package in files_to_copy:
        base = OLD/package
        manifest = json.loads((base/"manifest.json").read_text(encoding="utf-8"))
        failures = []
        for rel, record in manifest["files"].items():
            path = base/rel
            expected = record["sha256"] if isinstance(record, dict) else record
            if not path.is_file() or digest(path) != expected:
                failures.append(rel)
        archive_checks.append({"archive": str(base), "manifest_sha256": digest(base/"manifest.json"),
                               "files_verified": len(manifest["files"]), "failures": failures})
    result = json.loads((STAGE/"validation-v1/results.json").read_text(encoding="utf-8"))
    code_matches = result["source_sha256"] == digest(CODE/"validate_nonanalytic.py")
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(),
                  "user_facing_date": "2026-11-21", "user_timezone": "America/Los_Angeles",
                  "python": platform.python_version(), "inputs": original_records,
                  "validation_source_hash_resolved": code_matches,
                  "old_archive_preservation": archive_checks,
                  "notes": "No old output or shared code was written. Source PDFs have separate byte hashes. No training or reserved cohort was used."}
    (STAGE/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n", encoding="utf-8")
    assert code_matches and all(not x["failures"] for x in archive_checks)
    manifest_files = {}
    for path in sorted(STAGE.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.name not in {"manifest.json", "ARCHIVE_CHECK.json"}:
            manifest_files[path.relative_to(STAGE).as_posix()] = {"sha256": digest(path), "bytes": path.stat().st_size}
    (STAGE/"manifest.json").write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(),
                                                  "files": manifest_files}, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"files": len(manifest_files), "old_packages_verified": archive_checks,
                      "validation_source_hash_resolved": code_matches}, indent=2))


if __name__ == "__main__":
    main()
