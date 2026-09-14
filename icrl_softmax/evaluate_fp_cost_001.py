"""FP-COST-001 evaluator: the conjunctive rule's data-cost curve on both new families.

Registered in `docs/research_tasks/FP-COST-001.md` before this run. The question is
the one the audit's item 6 and the user's follow-up both point at: at EQUAL TOTAL
CERTIFICATION DATA, does the per-state rule still beat the conjunctive one?

The conjunctive arm stops on its own (mean 1.32 steps on F1, 4.91 on F2), so its
only way to spend more is fatter batches. This runs it at a chain ladder whose top
rung exceeds the sealed per-state arm's total spend, so the matched-cost point is
bracketed by measurements rather than extrapolated.

`evaluate_fp_xrule_001.run_family` reads `CHAINS`, `CELLS` and `HORIZON` from its
module globals, so this parameterises it by assignment instead of by editing that
file, keeping its recorded hash valid.

    python evaluate_fp_cost_001.py --output-dir results/FP-COST-001/claude/all
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

TASK_ID = "FP-COST-001"
RULES = ("conj",)
CERT = "frozen"
CELLS = (f"{RULES[0]}|{CERT}",)
#: top rung chosen to exceed the sealed per-state arm's total spend per family
LADDERS = {"f1": (16384, 65536, 131072), "f2": (16384, 65536)}
PERSTATE_REF = {
    "f1": {"items": 1_015_021_568, "gain": 8.6893, "cell": "perstate|frozen"},
    "f2": {"items": 1_169_162_240, "gain": 8.1739, "cell": "perstate|frozen"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="all")
    args = parser.parse_args()

    x1.CELLS = CELLS              # only the conjunctive cell
    x1.RULES = RULES
    x1.HORIZON = 12               # same as FP-XRULE-002

    started = time.time()
    bundle: dict[str, Any] = {
        "task_id": TASK_ID, "cells": list(CELLS), "certificate": CERT,
        "horizon": 12, "chain_length": x1.CHAIN_LENGTH,
        "min_visits": x1.MIN_VISITS, "delta_total": x1.DELTA_TOTAL,
        "delta_step": x1.DELTA_TOTAL / 12,
        "task_salt": x1.TASK_SALT,
        "perstate_reference": PERSTATE_REF,
        "ladders": {k: list(v) for k, v in LADDERS.items()},
        "runs": [],
    }
    for fam_name, ladder in LADDERS.items():
        for chains in ladder:
            x1.CHAINS = chains
            print(f"  {fam_name} conj|{CERT} @ {chains} chains", flush=True)
            t0 = time.time()
            records = x1.run_family(fam_name, x1.FAMILIES[fam_name])
            bundle["runs"].append({
                "family": fam_name, "chains": chains, "cell": CELLS[0],
                "items_per_step": chains * x1.CHAIN_LENGTH,
                "wall_seconds": time.time() - t0,
                "records": records,
            })
    bundle["wall_seconds_total"] = time.time() - started
    bundle["finished_utc"] = datetime.now(timezone.utc).isoformat()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "task_results.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "config.json").write_text(
        json.dumps({
            "task_id": TASK_ID, "label": args.label, "cells": list(CELLS),
            "certificate": CERT, "horizon": 12, "ladders": {k: list(v) for k, v in LADDERS.items()},
            "perstate_reference": PERSTATE_REF,
            "reuses": "evaluate_fp_xrule_001.run_family with module-level CHAINS/CELLS/HORIZON set; "
                      "that file is byte-identical to its FP-XRULE-001 record",
        }, indent=2, sort_keys=True), encoding="utf-8",
    )
    (args.output_dir / "environment.json").write_text(
        json.dumps({"python": sys.version, "platform": platform.platform(),
                    "numpy": np.__version__}, indent=2, sort_keys=True), encoding="utf-8",
    )
    print(f"done: {len(bundle['runs'])} ladder runs, "
          f"{bundle['wall_seconds_total'] / 60:.1f} min")


if __name__ == "__main__":
    main()
