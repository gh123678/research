"""FP-XRULE-002 evaluator: the same four cells and families at `K = 12`.

Registered in `docs/research_tasks/FP-XRULE-002.md` before this run. The only
change from FP-XRULE-001 is the horizon, because `K = 4` saturated: every
per-state cell reached it in both families, so it could not distinguish "the loop
can go further" from "the loop stops there".

`evaluate_fp_xrule_001.run_family` reads its horizon from the module global, so
this script parameterises it by assignment rather than by editing that file --
keeping FP-XRULE-001's recorded file hash valid.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import evaluate_fp_xrule_001 as x1  # noqa: E402

TASK_ID = "FP-XRULE-002"
HORIZON = 12


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="f1f2_K12")
    parser.add_argument("--families", default="f1,f2")
    args = parser.parse_args()

    x1.HORIZON = HORIZON  # the only parameterisation; x1's file is untouched
    started = time.time()
    records: list[dict[str, Any]] = []
    for fam_name in args.families.split(","):
        records += x1.run_family(fam_name, x1.FAMILIES[fam_name])

    cells = list(x1.CELLS)
    bundle = {
        "task_id": TASK_ID, "cells": cells, "rules": list(x1.RULES),
        "certificates": list(x1.CERTS), "routes": list(x1.ROUTES),
        "families": args.families.split(","), "horizon": HORIZON,
        "chains": x1.CHAINS, "chain_length": x1.CHAIN_LENGTH,
        "items_per_step": x1.CHAINS * x1.CHAIN_LENGTH,
        "min_visits": x1.MIN_VISITS, "delta_total": x1.DELTA_TOTAL,
        "delta_step": x1.DELTA_TOTAL / HORIZON,
        "split_fraction": x1.SPLIT_FRACTION,
        "delta_prop_fraction": x1.DELTA_PROP_FRACTION,
        "task_salt": x1.TASK_SALT, "record_count": len(records),
        "wall_seconds_total": time.time() - started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "task_results.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "config.json").write_text(
        json.dumps({
            "task_id": TASK_ID, "label": args.label, "families": args.families.split(","),
            "horizon": HORIZON, "chains": x1.CHAINS, "chain_length": x1.CHAIN_LENGTH,
            "cells": cells, "min_visits": x1.MIN_VISITS, "delta_total": x1.DELTA_TOTAL,
            "delta_step": bundle["delta_step"], "split_fraction": x1.SPLIT_FRACTION,
            "delta_prop_fraction": x1.DELTA_PROP_FRACTION,
            "reuses": "evaluate_fp_xrule_001.run_family with module-level HORIZON set; "
                      "that file is byte-identical to its FP-XRULE-001 record",
            "sealed_file_hashes": {n: x1.sha256(PROJECT / n) for n in x1.SEALED_FILES},
            "new_file_hashes": {n: x1.sha256(PROJECT / n)
                                for n in ("fixed_policy_tight_certificate.py",
                                          "evaluate_fp_xrule_002.py")},
        }, indent=2, sort_keys=True), encoding="utf-8",
    )
    (args.output_dir / "environment.json").write_text(
        json.dumps({"python": sys.version, "platform": platform.platform(),
                    "numpy": np.__version__}, indent=2, sort_keys=True), encoding="utf-8",
    )

    def agg(fam: str, cell: str, fn):
        return fn([rt[cell] for r in records if r["family"] == fam for rt in r["routes"].values()])

    summary = {
        "task_id": TASK_ID, "records": len(records), "horizon": HORIZON,
        "wall_seconds_total": bundle["wall_seconds_total"],
        "by_family": {
            fam: {
                "total_emitted": {c: agg(fam, c, lambda rr: sum(r["emitted_steps"] for r in rr))
                                  for c in cells},
                "mean_simulated_steps": {
                    c: float(np.mean([r["simulated_steps"] for r in
                                      [rt[c] for rec in records if rec["family"] == fam
                                       for rt in rec["routes"].values()]]))
                    for c in cells
                },
                "reached_horizon": {
                    c: sum(1 for rec in records if rec["family"] == fam
                           for rt in rec["routes"].values() if rt[c]["stopped_at"] is None)
                    for c in cells
                },
                "mean_value_gain": {
                    c: agg(fam, c, lambda rr: float(np.mean([r["total_value_gain"] for r in rr])))
                    for c in cells
                },
                "items_if_run_alone": {
                    c: agg(fam, c, lambda rr: sum(r["items_if_run_alone"] for r in rr))
                    for c in cells
                },
                "coverage_violations": {
                    c: agg(fam, c, lambda rr: sum(1 for r in rr for e in r["steps"]
                                                  if not e["covers_realized"]))
                    for c in cells
                },
                "degradations": {
                    c: agg(fam, c, lambda rr: sum(1 for r in rr for e in r["steps"]
                                                  if e["emitted"]
                                                  and not e["componentwise_nondegrading"]))
                    for c in cells
                },
                "partial_update_share": {
                    c: agg(fam, c, lambda rr: float(np.mean(
                        [1.0 if e["states_updated"] < x1.FAMILIES[fam]["n_states"] else 0.0
                         for r in rr for e in r["steps"] if e["emitted"]] or [0.0])))
                    for c in cells
                },
            }
            for fam in args.families.split(",")
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
