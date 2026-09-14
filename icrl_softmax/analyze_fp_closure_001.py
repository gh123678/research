"""Two cheap closures opened by the 2026-09-14 rulings.

(甲) BATCH CONSISTENCY between FP-RERUN-001 and the WITHDRAWN FP-ITER-BOUND-001.
     The H6 ruling revoked that clause's acceptance force and recorded that the
     batch consistency of the two runs is still unchecked and is NOT superseded by
     anything else. This checks it directly: compare the constants, then draw the
     same (record, step) batches under each task's documented schedule and compare
     digests. No experiment is re-run; the batches are deterministic functions of
     (seed schedule, chains, chain length, environment).

(乙) IS THE LONG LOOP CONVERGING OR SPINNING? `FP-XRULE-002` showed the per-state
     loop does not terminate within K=12, but reported only per-step value gains.
     The sealed bundle records each step's total value gain, so the cumulative gain
     is the current value sum minus the initial one -- and the denominator
     (v* - v_0) is computable offline from the MDP. That gives the suboptimality
     closure curve with no replay at all.

    python analyze_fp_closure_001.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT))

import evaluate_fp_iter_bound_001 as it1  # noqa: E402
import evaluate_fp_rerun_001 as rr1  # noqa: E402
import evaluate_fp_xrule_002 as x2mod  # noqa: E402
import fixed_policy_expected_sarsa_scaled as fs  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402
from evaluate_fp_xfam_001 import (  # noqa: E402
    FAMILIES,
    build_family_task,
    optimal_values_generic,
    training_batch,
    vectorised_batch_generic,
)
from fp_certfix_first_n import step_seed_parts  # noqa: E402
from fp_sample_vectorised_batch import vectorised_batch  # noqa: E402

XRULE = PROJECT / "results" / "FP-XRULE-002" / "claude" / "f1f2_K12"


def digest(batch: dict) -> str:
    h = hashlib.sha256()
    for key in ("states", "actions", "rewards", "next_states"):
        h.update(np.ascontiguousarray(batch[key]).tobytes())
    return h.hexdigest()[:16]


def part_batch_consistency() -> dict:
    """FP-RERUN-001 vs the withdrawn FP-ITER-BOUND-001: same batches?"""
    print("=" * 92)
    print("(甲) batch consistency: FP-RERUN-001 vs the withdrawn FP-ITER-BOUND-001")
    print("=" * 92)
    a = {"task_salt": rr1.TASK_SALT, "chains": rr1.CHAINS, "length": rr1.CHAIN_LENGTH,
         "tasks": list(rr1.FRESH_TASKS), "sampler": "vectorised_batch",
         "horizon": rr1.HORIZON}
    b = {"task_salt": it1.TASK_SALT, "chains": it1.CHAINS, "length": it1.CHAIN_LENGTH,
         "tasks": list(it1.FRESH_TASKS), "sampler": "vectorised_batch",
         "horizon": it1.HORIZON}
    same_params = all(a[k] == b[k] for k in ("task_salt", "chains", "length", "tasks"))
    print(f"  FP-RERUN-001     : salt={a['task_salt']} chains={a['chains']} L={a['length']} "
          f"K={a['horizon']} tasks={a['tasks'][0]}..{a['tasks'][-1]}")
    print(f"  FP-ITER-BOUND-001: salt={b['task_salt']} chains={b['chains']} L={b['length']} "
          f"K={b['horizon']} tasks={b['tasks'][0]}..{b['tasks'][-1]}")
    print(f"  parameters identical: {same_params}")

    rows = []
    for mixing, task_index, step in ((0.08, 12, 1), (0.08, 17, 3), (0.5, 20, 7), (0.5, 23, 12)):
        mdp, behaviour, rng = fs.build_task(task_index=task_index, mixing=mixing)
        mu = np.asarray(policy_quantities(mdp, behaviour)["mu_state"], dtype=np.float64)
        # each task's documented schedule, same sampler
        da = digest(vectorised_batch(mdp, behaviour, mu,
                                     step_seed_parts(fs.SEED, a["task_salt"], mixing, task_index, step),
                                     a["chains"], a["length"]))
        db = digest(vectorised_batch(mdp, behaviour, mu,
                                     step_seed_parts(fs.SEED, b["task_salt"], mixing, task_index, step),
                                     b["chains"], b["length"]))
        rows.append({"mixing": mixing, "task_index": task_index, "step": step,
                     "digest_rerun": da, "digest_iter": db, "equal": da == db})
        print(f"  {mixing}/{task_index} step {step:>2}: {da} vs {db}  "
              f"{'EQUAL' if da == db else 'DIFFERENT'}")
    all_equal = all(r["equal"] for r in rows)
    print(f"\n  VERDICT: the two runs drew {'the SAME' if all_equal else 'DIFFERENT'} batches "
          f"for every probed (record, step)")
    print("  (scope: four probed (record, step) pairs on the shared population; the schedules "
          "are pure functions of the probed keys, so this settles the schedule, not a full "
          "per-step digest of all 48x12 draws)")
    return {"params": {"rerun": a, "iter": b}, "params_identical": same_params,
            "probes": rows, "all_equal": all_equal}


def closure_curve() -> dict:
    """Suboptimality closed vs step, from the sealed FP-XRULE-002 bundle."""
    print()
    print("=" * 92)
    print("(乙) is the K=12 loop converging or spinning? (zero replay)")
    print("=" * 92)
    bundle = json.loads((XRULE / "task_results.json").read_text(encoding="utf-8"))
    cells = list(bundle["cells"])
    K = int(bundle["horizon"])
    out: dict = {"horizon": K, "cells": cells, "by_family": {}}
    for fam_name in bundle["families"]:
        fam = FAMILIES[fam_name]
        recs = [r for r in bundle["records"] if r["family"] == fam_name]
        # denominator per record: v*_sum - v0_sum
        denom: dict = {}
        for rec in recs:
            key = (float(rec["mixing"]), int(rec["task_index"]))
            mdp, behaviour, _rng = build_family_task(fam, key[0], key[1])
            v0 = np.asarray(policy_quantities(mdp, behaviour)["v_pi"], dtype=np.float64)
            vstar = np.asarray(optimal_values_generic(mdp, fam["gamma"]), dtype=np.float64)
            denom[key] = float(np.sum(vstar) - np.sum(v0))
        curves: dict = {}
        for cell in cells:
            cum = np.zeros(K)
            cnt = np.zeros(K)
            for rec in recs:
                key = (float(rec["mixing"]), int(rec["task_index"]))
                d = denom[key]
                if d <= 1e-12:
                    continue
                run = 0.0
                for route in rec["routes"].values():
                    steps = route[cell]["steps"]
                    run = 0.0
                    for i, e in enumerate(steps):
                        run += float(e.get("total_value_gain", 0.0))
                        if i < K:
                            cum[i] += run / d
                            cnt[i] += 1
            curves[cell] = [float(cum[i] / cnt[i]) if cnt[i] else float("nan")
                            for i in range(K)]
        out["by_family"][fam_name] = {
            "n_records": len(recs),
            "mean_gap_v_star_minus_v0": float(np.mean(list(denom.values()))),
            "closure_by_step": curves,
        }
        print(f"\n  family {fam_name} ({len(recs)} records; mean (v*-v0) sum = "
              f"{np.mean(list(denom.values())):.4f})")
        print(f"    {'cell':<16}" + "".join(f"{f'k={i+1}':>9}" for i in range(K)))
        for cell in cells:
            row = "".join(f"{100 * v:>8.1f}%" for v in curves[cell])
            print(f"    {cell:<16}{row}")
        # marginal closure over the last third
        print(f"    {'':<16}" + "".join(f"{'--' if i == 0 else f'{100 * (curves[cell][i] - curves[cell][i - 1]):>+7.1f}%':>9}"
                                          for i in range(K)))
        for cell in ("perstate|frozen", "perstate|L12S"):
            if cell not in curves:
                continue
            tail = curves[cell][-1] - curves[cell][7] if K > 8 else float("nan")
            print(f"    {cell}: closure at K={K} = {100 * curves[cell][-1]:.2f}%; "
                  f"steps 8..{K} contribute {100 * tail:.2f} points of it")
    return out


def main() -> int:
    res = {"batch_consistency": part_batch_consistency(), "closure": closure_curve()}
    (ROOT / "tmp").mkdir(exist_ok=True)
    (ROOT / "tmp" / "closure_and_batch_check.json").write_text(
        json.dumps(res, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
