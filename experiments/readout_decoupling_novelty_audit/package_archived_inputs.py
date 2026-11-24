"""Copy the exact read-only inputs named in an audit record into a portable tree."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.audit.read_text(encoding="utf-8"))
    archive = Path(record["source_archive"])
    repo = Path(record["repository"])
    args.out.mkdir(parents=True, exist_ok=False)
    copied = {}
    for name, expected in record["historical_files_sha256"].items():
        path = Path(name)
        data = path.read_bytes()
        assert sha(data) == expected
        if path.is_relative_to(archive):
            relative = Path("archive") / path.relative_to(archive)
        else:
            relative = Path("repository") / path.relative_to(repo)
        target = args.out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        assert sha(target.read_bytes()) == expected
        copied[relative.as_posix()] = expected
    (args.out / "INPUT_HASHES.json").write_text(json.dumps(copied, indent=2), encoding="utf-8")
    print(json.dumps(dict(copied_files=len(copied), out=str(args.out))))


if __name__ == "__main__":
    main()
