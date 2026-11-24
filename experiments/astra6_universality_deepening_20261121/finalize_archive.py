"""Verify the completed package, then optionally copy to the named new archive."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

STAGE = Path(r"C:\Users\liwei\Documents\ChatGPT\277\tmp\astra6-universality-deepening-20261121")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    manifest = json.loads((STAGE/"manifest.json").read_text(encoding="utf-8"))
    failures = []
    for rel, rec in manifest["files"].items():
        p = STAGE/rel
        if not p.is_file() or sha(p) != rec["sha256"]:
            failures.append("content hash: "+rel)
    broken_links = []
    for p in STAGE.rglob("*.md"):
        if "reference_inputs" in p.parts:
            continue
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", p.read_text(encoding="utf-8")):
            target = match.group(1).strip("<>")
            if re.match(r"https?://|#|mailto:", target):
                continue
            target = unquote(target.split("#")[0])
            if target and not (p.parent/target).exists():
                broken_links.append({"file": str(p.relative_to(STAGE)), "target": target})
    # Resolve each generated result's embedded source hash to a preserved script.
    available_hashes = {sha(p) for p in STAGE.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    embedded_checks = []
    for p in STAGE.rglob("*.json"):
        if "reference_inputs" in p.parts:
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ["source_sha256", "theorem_sha256"]:
                if key in data:
                    ok = data[key] in available_hashes
                    embedded_checks.append({"file": p.relative_to(STAGE).as_posix(), "field": key, "resolved": ok})
                    if not ok:
                        failures.append("unresolved embedded hash: "+str(p)+" "+key)
    source_checks = []
    for name in ["SOURCE_HASHES.json", "EXTRA_SOURCE_HASHES.json"]:
        p = STAGE/"independent-literature"/name
        data = json.loads(p.read_text(encoding="utf-8"))
        records = data["records"] if "records" in data else [data]
        for rec in records:
            if rec.get("status") == "saved":
                candidate = p.parent/rec["file"]
                ok = candidate.exists() and sha(candidate) == rec["sha256"]
                source_checks.append({"file": rec["file"], "verified": ok})
                if not ok:
                    failures.append("source byte hash: "+rec["file"])
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "passed" if not failures and not broken_links else "failed",
              "manifest_content_files": len(manifest["files"]),
              "content_failures": failures, "broken_new_document_links": broken_links,
              "embedded_hash_resolution": embedded_checks, "primary_source_hashes": source_checks,
              "prior_archive_check": "provenance.json verifies 39+13 old manifest-listed content files",
              "bookkeeping_exclusion": "manifest.json and ARCHIVE_CHECK.json do not hash themselves; bytecode caches excluded",
              "archived_to": None}
    if failures or broken_links:
        print(json.dumps(result, indent=2))
        raise SystemExit(1)
    if args.archive:
        archive = args.archive.resolve()
        expected = Path(r"C:\Users\liwei\Documents\Research\pullback-feature-learning\explorations\astra6-universality-deepening-20261121").resolve()
        if archive != expected:
            raise ValueError("Archive must be the exact user-designated new directory")
        if archive.exists():
            raise FileExistsError("Refusing to overwrite an existing archive")
        archive.mkdir(parents=True)
        for rel in list(manifest["files"])+["manifest.json"]:
            dest = archive/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(STAGE/rel, dest)
            if sha(dest) != sha(STAGE/rel):
                raise RuntimeError("Archive byte mismatch: "+rel)
        result["archived_to"] = str(archive)
        result["archive_content_equality"] = True
        (archive/"ARCHIVE_CHECK.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    (STAGE/"ARCHIVE_CHECK.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "manifest_content_files": len(manifest["files"]),
                      "embedded_hashes": len(embedded_checks), "source_hashes": len(source_checks),
                      "archived_to": result["archived_to"]}, indent=2))


if __name__ == "__main__":
    main()
