"""FP-BOUND-004: does the SOUND (split) propagation still beat the alternatives?

`FP-BOUND-002`/`003` reported the data-driven propagation `L12M` at `-39.6%`
(`c64k`) and `-40.4%` (`c16k`). The 2026-09-13 audit found that `L12M`'s
concentration step is unlicensed: it applies Hoeffding to `W_k`, which is itself a
function of the same successor draws. This script measures the repaired
construction `L12S`, in which the per-pair sample is split, the intervals come
from half A and every propagation estimate from half B.

One pass, one batch per record-route, all levers computed on it, so the
comparison is paired. Same population and seeds as `FP-BOUND-001`/`003`
(`task_index 12..23`, both mixings, both routes = 48 record-routes, step 1).

    python measure_fp_bound_004.py --chains 65536
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT))

import fixed_policy_expected_sarsa_scaled as fs  # noqa: E402
import fixed_policy_tight_certificate as tc  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402
from fp_certfix_first_n import first_visit_batch, step_seed_parts  # noqa: E402
from fp_sample_vectorised_batch import vectorised_batch  # noqa: E402

LEVERS = ("frozen", "L1", "L12", "L123", "L12M", "L12S")
TASK_SALT = 90417
CHAINS_LEN = 64
MIN_VISITS = 2000
DELTA = 0.05
FRESH = tuple(range(12, 24))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chains", type=int, default=65536)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    rows = []
    for mixing in fs.MIXING:
        for task_index in FRESH:
            mdp, behaviour, rng = fs.build_task(task_index=task_index, mixing=mixing)
            exact0 = policy_quantities(mdp, behaviour)
            mu = np.asarray(exact0["mu_state"], dtype=np.float64)
            train = fs.training_batch(mdp, behaviour, mu, rng)
            q_pi = np.asarray(exact0["q_pi"], dtype=np.float64)
            P = np.asarray(mdp["P"], dtype=np.float64)
            R = np.asarray(mdp["R"], dtype=np.float64)
            raw = vectorised_batch(
                mdp, behaviour, mu,
                step_seed_parts(fs.SEED, TASK_SALT, mixing, task_index, 1),
                int(args.chains), CHAINS_LEN,
            )
            reduced, _ = first_visit_batch(raw, CHAINS_LEN)
            for route in ("expected_exact", "expected_finite"):
                q_hat = np.asarray(
                    fs.run_route(route, behaviour, train)["q_hat"], dtype=np.float64
                ).reshape(fs.N_STATES, fs.N_ACTIONS)
                realized = float(np.max(np.abs(q_hat - q_pi)))
                for lever in LEVERS:
                    cert = tc.certificate(
                        q_hat, behaviour, reduced, min_visits=MIN_VISITS,
                        delta_step=DELTA, lever=lever, transition=P, reward=R,
                    )
                    dec = fs.improvement_for(behaviour, q_hat, cert)
                    e_q = cert["e_q"]
                    rows.append({
                        "mixing": mixing, "task_index": task_index, "route": route,
                        "lever": lever, "e_q": e_q,
                        "max_eps": float(np.max(cert["epsilon_res_by_pair"]))
                        if cert["status"] == "certificate_emitted" else None,
                        "emitted": dec["status"] == "safe_update_emitted",
                        "covers": bool(e_q is not None and float(e_q) >= realized),
                    })

    print(f"SOUND propagation check at {args.chains} chains "
          f"({len(rows) // len(LEVERS)} record-routes)")
    print("=" * 88)
    base = float(np.mean([r["e_q"] for r in rows if r["lever"] == "frozen" and r["e_q"]]))
    print(f"  {'lever':<8}{'mean E_Q':>12}{'vs frozen':>12}{'emitted':>9}"
          f"{'max eps mean':>15}{'coverage viol':>15}")
    summary = {}
    for lever in LEVERS:
        vals = [float(r["e_q"]) for r in rows if r["lever"] == lever and r["e_q"]]
        eps = [r["max_eps"] for r in rows if r["lever"] == lever and r["max_eps"]]
        em = sum(1 for r in rows if r["lever"] == lever and r["emitted"])
        viol = sum(1 for r in rows if r["lever"] == lever and r["e_q"] and not r["covers"])
        summary[lever] = {
            "mean_e_q": float(np.mean(vals)), "rel": float(np.mean(vals) / base - 1.0),
            "emitted": em, "mean_max_eps": float(np.mean(eps)), "coverage_violations": viol,
        }
        print(f"  {lever:<8}{np.mean(vals):>12.5f}{np.mean(vals) / base - 1:>+12.2%}{em:>9}"
              f"{np.mean(eps):>15.5f}{viol:>15}")
    print()
    print(f"  frozen mean E_Q = {base:.5f}")
    d = np.mean([r["e_q"] for r in rows if r["lever"] == "L12S" and r["e_q"]]) - \
        np.mean([r["e_q"] for r in rows if r["lever"] == "L12" and r["e_q"]])
    print(f"  L12S - L12 = {d:+.5f}  "
          f"({'SOUND propagation still helps' if d < 0 else 'the split costs more than it buys'})")
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (ROOT / "tmp").mkdir(exist_ok=True)
    (ROOT / "tmp" / f"bound_004_c{args.chains}.json").write_text(
        json.dumps({"rows": rows, "summary": summary}, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
