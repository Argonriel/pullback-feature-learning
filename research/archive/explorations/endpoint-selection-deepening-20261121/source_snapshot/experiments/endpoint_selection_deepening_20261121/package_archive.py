"""Preserve this completed mathematical investigation and verify its provenance."""
import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_PREVIOUS = {
    "learned-geometry-gate-20261121": "7164aede3c183ea07325b4cda8aa2a1fa8b820aee34d3782043fa4e4a5c9baf6",
    "geometry-condition-screen-20261121": "b03f4388c8e890343dd1b8d4a79de50a34c1739978da374fa0de5edcb6a9abaf",
    "new-direction-screen-20261121": "1119aeed6ad606bed615dbec134b83bba97c6d47fe038137fee5b07d7da07e1a",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_new(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(source) != sha(target):
            raise FileExistsError(target)
        return
    shutil.copy2(source, target)


def package(repo, staging, archive):
    if archive.name != "endpoint-selection-deepening-20261121":
        raise ValueError("Use the explicitly named new research archive.")
    required = ["REPORT.md", "SCREEN_REGISTER.md", "GENERAL_GEOMETRY.md",
                "STATIC_WEIGHT_THEORY.md", "LITERATURE_AUDIT.md", "METRIC_CONTROL.md",
                "DATA_DICTIONARY.md", "VALIDATION_CHECKLIST.md"]
    for name in required:
        if not (staging/name).is_file():
            raise FileNotFoundError(name)
    previous_checks=[]
    for name, expected in EXPECTED_PREVIOUS.items():
        previous=archive.parent/name
        manifest=previous/"manifest.json"
        if sha(manifest) != expected:
            raise ValueError(f"Prior manifest changed: {manifest}")
        old=json.loads(manifest.read_text(encoding="utf-8"))
        for relative, row in old["files"].items():
            if sha(previous/relative) != row["sha256"]:
                raise ValueError(f"Prior file changed: {previous/relative}")
        previous_checks.append({"archive":str(previous),"unchanged_files":len(old["files"]),"manifest_sha256":expected})
    archive.mkdir(parents=True,exist_ok=True)
    if (archive/"manifest.json").exists():
        raise FileExistsError(archive/"manifest.json")
    for source in staging.rglob("*"):
        if source.is_file() and "__pycache__" not in source.parts:
            copy_new(source,archive/source.relative_to(staging))
    sources=[p for p in (repo/"experiments/endpoint_selection_deepening_20261121").iterdir()
             if p.is_file() and p.suffix in (".py",".md")]
    sources += [repo/name for name in ("AGENTS.md","pyproject.toml","requirements-lock.txt")]
    for source in sources:
        copy_new(source,archive/"source_snapshot"/source.relative_to(repo))
    lookup={sha(p):p.relative_to(archive).as_posix() for p in archive.rglob("*") if p.is_file()}
    provenance=[]
    for data in archive.rglob("*.json"):
        record=json.loads(data.read_text(encoding="utf-8-sig"))
        if not isinstance(record,dict):
            continue
        for key,value in record.items():
            if key.endswith("_sha256") and isinstance(value,str):
                if value not in lookup:
                    raise ValueError(f"Unresolved hash in {data.name}: {key}")
                provenance.append({"record":data.relative_to(archive).as_posix(),"field":key,"resolved":lookup[value]})
    broken=[]
    for report in archive.glob("*.md"):
        for link in re.findall(r"\]\(([^)]+)\)",report.read_text(encoding="utf-8")):
            if link in ("ARCHIVE_CHECK.json","manifest.json"):
                continue
            if "://" not in link and not link.startswith("#") and not (report.parent/link.split("#")[0]).exists():
                broken.append({"report":report.name,"target":link})
    if broken:
        raise ValueError(broken)
    files={p.relative_to(archive).as_posix():{"sha256":sha(p),"bytes":p.stat().st_size}
           for p in sorted(archive.rglob("*")) if p.is_file()
           and p.name not in ("manifest.json","ARCHIVE_CHECK.json")}
    manifest=archive/"manifest.json"
    manifest.write_text(json.dumps({"created_utc":datetime.now(timezone.utc).isoformat(),"files":files},indent=2)+"\n",encoding="utf-8")
    for name,row in files.items():
        assert sha(archive/name)==row["sha256"]
    checks={"status":"passed","file_count":len(files),"total_bytes":sum(v["bytes"] for v in files.values()),
            "manifest_sha256":sha(manifest),"source_and_data_provenance":provenance,
            "previous_archives":previous_checks,"broken_report_links":broken,
            "checked_utc":datetime.now(timezone.utc).isoformat()}
    (archive/"ARCHIVE_CHECK.json").write_text(json.dumps(checks,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in checks.items() if k!="source_and_data_provenance"},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ("repo","staging","archive"):
        parser.add_argument("--"+field,type=Path,required=True)
    args=parser.parse_args()
    package(args.repo.resolve(),args.staging.resolve(),args.archive.resolve())
