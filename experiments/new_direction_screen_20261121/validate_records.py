"""Cross-check archived discovery/replay records and source provenance."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from poisoning import audit as poisoning_audit


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.staging
    read = lambda name: json.loads((root / name).read_text(encoding="utf-8"))
    reduced = read("depth_endpoint_refined/depth_endpoint_refined_results.json")
    parameter = read("endpoint_parameter_refined/endpoint_parameter_results.json")
    contrasts = []
    for damping in [0.0, 0.003, 0.3]:
        endpoints = []
        for result in [reduced, parameter]:
            values = []
            for weight in [1.0, 9.0]:
                rows = [row for row in result["records"] if row["damping"] == damping
                        and row["second_loss_weight"] == weight]
                row = min(rows, key=lambda row: row["rk4_step"])
                values.append(row["unobserved_x12"])
            endpoints.append(values)
        gap_difference = abs((endpoints[0][1] - endpoints[0][0]) - (endpoints[1][1] - endpoints[1][0]))
        assert gap_difference < 3e-10
        contrasts.append({"damping": damping, "independent_gap_difference": gap_difference,
                          "max_independent_endpoint_difference": float(np.max(np.abs(np.array(endpoints[0]) - endpoints[1])))})
    original_poisoning = read("poisoning_checks.json")
    assert poisoning_audit() == original_poisoning
    result = {"status": "passed", "endpoint_cross_implementation": contrasts,
              "poisoning_replay_exactly_matches": True,
              "poisoning_source_sha256": sha(Path(__file__).parent / "poisoning.py"),
              "source_sha256": sha(Path(__file__)),
              "scope": "Numerical/provenance validation, not originality or generalization validation."}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
