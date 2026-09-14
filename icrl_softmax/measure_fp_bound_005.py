"""Grid over the SOUND propagation's two design knobs.

`measure_fp_bound_004.py` showed that `L12S` at (split 0.5, delta_prop 0.5) is
slightly WORSE than the plain `L12` (+1.3%): splitting costs x1.34 on the interval
while the propagation only buys x1.32. Two knobs were left at their defaults and
both are known to matter:

* `split_fraction` -- how much of each pair's sample goes to the interval half;
  the interval is the thing being penalised, so a larger share should help;
* `delta_prop_fraction` -- FP-BOUND-003 already showed the propagation estimates
  need very little risk (`0.05` beat `0.5` for the withdrawn arm), because their
  confidence terms are O(1e-3) against a gain of O(3e-2).

This script measures the grid on one batch per record-route, so every cell is
paired with every other and with `L12`/`frozen`. It reports the whole grid rather
than a winner: choosing the best cell after the fact and quoting it as "the"
result would be exactly the kind of tuning the audit warns about.

    python measure_fp_bound_005.py --chains 65536
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

FRESH = tuple(range(12, 24))
TASK_SALT = 90417
DELTA = 0.05
MIN_VISITS = 2000
GRID = [(f, p) for f in (0.5, 0.7, 0.9) for p in (0.5, 0.05)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chains", type=int, default=65536)
    args = parser.parse_args()

    acc = {
        "frozen": [[], 0, 0], "L12": [[], 0, 0],
        **{f"L12S(f={f},p={p})": [[], 0, 0] for f, p in GRID},
    }
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
                int(args.chains), 64,
            )
            reduced, _ = first_visit_batch(raw, 64)
            for route in ("expected_exact", "expected_finite"):
                q_hat = np.asarray(
                    fs.run_route(route, behaviour, train)["q_hat"], dtype=np.float64
                ).reshape(fs.N_STATES, fs.N_ACTIONS)
                realized = float(np.max(np.abs(q_hat - q_pi)))
                cells = [("frozen", "frozen", 0.5), ("L12", "L12", 0.5)]
                cells += [(f"L12S(f={f},p={p})", "L12S", p) for f, p in GRID]
                for name, lever, pfrac in cells:
                    f = 0.5
                    if name.startswith("L12S"):
                        f = float(name.split("f=")[1].split(",")[0])
                    cert = tc.certificate(
                        q_hat, behaviour, reduced, min_visits=MIN_VISITS,
                        delta_step=DELTA, lever=lever, transition=P, reward=R,
                        delta_prop_fraction=pfrac, split_fraction=f,
                    )
                    dec = fs.improvement_for(behaviour, q_hat, cert)
                    e_q = cert["e_q"]
                    if e_q is None:
                        continue
                    acc[name][0].append(float(e_q))
                    acc[name][1] += int(dec["status"] == "safe_update_emitted")
                    acc[name][2] += int(float(e_q) < realized)

    base = float(np.mean(acc["frozen"][0]))
    print(f"SOUND propagation grid at {args.chains} chains "
          f"({len(acc['frozen'][0])} record-routes)")
    print("=" * 88)
    print(f"  {'configuration':<22}{'mean E_Q':>11}{'vs frozen':>11}{'vs L12':>10}"
          f"{'emitted':>9}{'cov viol':>10}")
    l12 = float(np.mean(acc["L12"][0]))
    out = {}
    for name, (vals, emitted, viol) in acc.items():
        m = float(np.mean(vals))
        vs12 = f"{m / l12 - 1:+.1%}" if name not in ("frozen",) else "--"
        print(f"  {name:<22}{m:>11.5f}{m / base - 1:>+11.2%}{vs12:>10}"
              f"{emitted:>9}{viol:>10}")
        out[name] = {"mean_e_q": m, "rel_frozen": m / base - 1.0, "rel_l12": m / l12 - 1.0,
                     "emitted": emitted, "coverage_violations": viol, "n": len(vals)}
    best = min((v["mean_e_q"], k) for k, v in out.items() if k.startswith("L12S"))
    print()
    print(f"  best SOUND cell: {best[1]} at {best[0]:.5f} "
          f"({best[0] / l12 - 1:+.1%} vs L12)")
    (ROOT / "tmp").mkdir(exist_ok=True)
    (ROOT / "tmp" / f"bound_005_grid_c{args.chains}.json").write_text(
        json.dumps(out, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
