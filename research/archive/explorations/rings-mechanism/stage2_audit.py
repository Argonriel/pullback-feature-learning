"""Cross-check population integrals on an independent grid and common endpoints against stage 1."""
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from audit_polar import independent

ROOT = Path(__file__).resolve().parent


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    output = ROOT / "stage2"
    measured = rows(output / "rows.csv")
    original = {(int(r["pair"]), r["schedule"], int(r["step"])): r
                for r in rows(ROOT / "stage1/population/states.csv")}
    fields = ("population_loss", "radial_excess", "angular_excess")
    discrepancies, spot_checks = [], []
    with np.load(output / "states.npz") as bank, threadpool_limits(1):
        selected = {}
        for r in measured:
            pair, weight, step = int(r["pair"]), float(r["weight"]), int(r["step"])
            if pair <= 8 and weight == 1 and r["size_rule"] == "native":
                s = original[pair, "pullback" if r["schedule"] == "full" else "warmup", step]
                assert r["state_sha256"] == s["state_sha256"]
                discrepancies.append(max(abs(float(s[f])-float(r[f])) for f in fields))
            if pair == 9 or int(r["angular_nodes"]) > 768:
                selected[r["state_sha256"]] = r
        for key,r in selected.items():
            ind = independent(bank[key])
            spot_checks.append(dict(state=key, pair=int(r["pair"]), differences={
                f: abs(ind[f]-float(r[f])) for f in fields}))
    worst = max(max(s["differences"].values()) for s in spot_checks)
    assert len(discrepancies)==32 and max(discrepancies)<1e-12
    assert len(spot_checks)>=18 and worst<2e-6, worst
    audit = dict(original_endpoints=len(discrepancies),max_old_error=max(discrepancies),
                 distinct_spot_checks=len(spot_checks),max_independent_error=worst,
                 max_logit_matching_residual=max(float(r["match_max_error"]) for r in measured),
                 denominator="RMS predicted z1 displacement on the arm's own training batch",
                 independent=spot_checks)
    (output / "numerical_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(f"{len(discrepancies)} original endpoints matched, {len(spot_checks)} grid checks, max error {worst:.3g}")


if __name__ == "__main__":
    main()
