"""Refine only the pure-Bures control from coherent_blend.py."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from coherent_blend import simulate

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
old = json.loads(args.input.read_text(encoding="utf-8"))
args.output.mkdir(parents=True, exist_ok=False)
x0 = np.array(old["initial_state"])
target = np.diag(x0) + 0.1 * np.array(old["target_direction"])
runs = [simulate(x0, target, np.array(w), None, 0.005) for w in old["weights"]]
gap = runs[1]["offdiagonal"] - runs[0]["offdiagonal"]
expected = old["limit_controls"]["pure_bures"]["predicted_offdiagonal"]
result = {
    "runs": runs,
    "weight_gap": gap,
    "predicted_offdiagonal": expected,
    "coarse_to_fine_gap_ratio": old["limit_controls"]["pure_bures"]["weight_gap"] / gap,
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
assert abs(gap) < 2e-11
assert max(abs(r["offdiagonal"] - expected) for r in runs) < 2e-11
(args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
