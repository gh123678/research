"""FP-BOUND-002 verifier: recompute the model-free arm from the sealed bundle alone.

C1  the L12M propagation, recomputed from the sealed successor histogram, pair
    sizes and per-pair intervals -- the whole model-free bound, no kernel, no
    code shared with the certificate module's arithmetic.
C2  per-pair interval coverage against the EXACT residual `rho_x` for every lever.
C3  `E_Q` reconstruction for the non-propagation levers.
C4  the four shared arms against FP-BOUND-001's sealed bundles, bit-for-bit.
C5  the decision chain `emitted <=> E_Q < h`.
C6  risk accounting: `frozen/L1` spend `delta_step/2`, `L12/L123` spend
    `delta_step`, `L12M` spends `delta_step/2 + delta_step/2 = delta_step`.

Importing helpers from `verify_fp_bound_001` keeps that file byte-identical.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import fixed_policy_expected_sarsa_scaled as fs  # noqa: E402
from verify_fp_bound_001 import exact_rho, gate_margin  # noqa: E402

GAMMA = fs.GAMMA
D = fs.N_STATES * fs.N_ACTIONS
LEVERS = ("frozen", "L1", "L12", "L123", "L12M")


def recompute_l12m(prop: dict, eps: np.ndarray, sizes: np.ndarray, policy, gamma=GAMMA):
    """Rebuild the model-free bound from the sealed successor histogram."""
    pi = np.asarray(policy, dtype=np.float64)
    counts = np.asarray(prop["successor_counts"], dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    eps_grid = np.asarray(eps, dtype=np.float64).reshape(fs.N_STATES, fs.N_ACTIONS)
    epsbar = (pi * eps_grid).sum(axis=1)
    n_iter = int(prop["n_iter"])
    dk = float(prop["delta_per_iteration"])
    df = float(prop["delta_final"])
    log_k = math.log(1.0 / dk)
    log_f = math.log(1.0 / df)

    def estimate(f):
        t = (counts * f[None, :]).sum(axis=1) / sizes
        return t

    w = np.full(fs.N_STATES, float(np.max(eps)) / (1.0 - gamma))
    for _ in range(n_iter):
        t = estimate(w)
        c = float(np.max(w)) * np.sqrt(log_k / (2.0 * sizes))
        nxt = epsbar + gamma * (pi * (t + c).reshape(fs.N_STATES, fs.N_ACTIONS)).sum(axis=1)
        if float(np.max(np.abs(nxt - w))) <= 1e-14:
            w = nxt
            break
        w = nxt
    t = estimate(w)
    c = float(np.max(w)) * np.sqrt(log_f / (2.0 * sizes))
    bound = eps_grid + gamma * (t + c).reshape(fs.N_STATES, fs.N_ACTIONS)
    return float(np.max(bound)), w


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT / "results" / "FP-BOUND-002" / "claude" / "c64k",
    )
    parser.add_argument(
        "--bound-001-dir",
        type=Path,
        default=PROJECT / "results" / "FP-BOUND-001" / "claude",
    )
    args = parser.parse_args()
    bundle = json.loads((args.results / "task_results.json").read_text(encoding="utf-8"))
    rung = f"c{int(bundle['chains']) // 1024}k"
    delta_step = float(bundle["delta_step"])

    failures: list[str] = []
    pair_checks = pair_pass = 0
    max_l12m_delta = 0.0
    max_e_q_delta = 0.0
    max_h_delta = 0.0
    emit_mismatch = 0
    risk_bad = 0

    for rec in bundle["records"]:
        mixing, task_index = float(rec["mixing"]), int(rec["task_index"])
        mdp, behaviour, rng = fs.build_task(task_index=task_index, mixing=mixing)
        pi = np.asarray(behaviour, dtype=np.float64)
        for route_name, route in rec["routes"].items():
            q_hat = np.asarray(route["q_hat"], dtype=np.float64).reshape(
                fs.N_STATES, fs.N_ACTIONS
            )
            rho = exact_rho(mdp, behaviour, q_hat)
            h = gate_margin(behaviour, q_hat)
            max_h_delta = max(max_h_delta, abs(h - float(route["gate_margin_h"])))
            for lever in LEVERS:
                arm = route["arms"][lever]
                if arm["e_q"] is None:
                    continue
                means = np.asarray(arm["residual_means"], dtype=np.float64)
                radii = np.asarray(arm["radii"], dtype=np.float64)
                eps = np.abs(means) + radii
                sizes = np.asarray(arm["pair_sizes"], dtype=np.int64)
                pair_checks += 1
                if np.all(np.abs(means - rho) <= radii + 1e-15):
                    pair_pass += 1
                else:
                    failures.append(f"pair coverage {lever} {mixing}/{task_index}/{route_name}")

                if lever == "L12M":
                    e_q_re, _ = recompute_l12m(
                        arm["propagation"], eps, sizes, behaviour
                    )
                    max_l12m_delta = max(max_l12m_delta, abs(e_q_re - float(arm["e_q"])))
                elif lever == "L123":
                    # the kernel propagation, rebuilt from the sealed matrices
                    prop = arm["propagation"]
                    inv = np.asarray(prop["inverse"], dtype=np.float64)
                    P = np.asarray(mdp["P"], dtype=np.float64)
                    epsbar = (pi * eps.reshape(fs.N_STATES, fs.N_ACTIONS)).sum(axis=1)
                    w = inv @ epsbar
                    bound = eps.reshape(fs.N_STATES, fs.N_ACTIONS) + GAMMA * (
                        P * w[None, None, :]
                    ).sum(axis=2)
                    e_q_re = float(np.max(bound))
                    max_e_q_delta = max(max_e_q_delta, abs(e_q_re - float(arm["e_q"])))
                else:
                    e_q_re = float(np.max(eps)) / (1.0 - GAMMA)
                    max_e_q_delta = max(max_e_q_delta, abs(e_q_re - float(arm["e_q"])))

                if bool(arm["emitted"]) != bool(float(arm["e_q"]) < h):
                    emit_mismatch += 1

                de = float(arm["delta_each"])
                if lever in ("L12", "L123"):
                    ok = abs(D * de - delta_step) <= 1e-15
                elif lever == "L12M":
                    # the interval budget is delta_eps, whatever split is used
                    prop = arm["propagation"]
                    ok = abs(D * de - float(prop["delta_eps"])) <= 1e-15
                else:
                    ok = abs(2.0 * D * de - delta_step) <= 1e-15
                if not ok:
                    risk_bad += 1
                if lever == "L12M":
                    prop = arm["propagation"]
                    spent = float(prop["delta_eps"]) + (
                        float(prop["delta_per_iteration"]) * int(prop["n_iter"]) * D
                        + float(prop["delta_final"]) * D
                    )
                    if abs(spent - delta_step) > 1e-12:
                        risk_bad += 1

    # C4: the shared arms must reproduce FP-BOUND-001 exactly
    ref_path = args.bound_001_dir / rung / "task_results.json"
    shared_worst = None
    shared_mismatch = None
    if ref_path.exists():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        ref_map = {
            (float(r["mixing"]), int(r["task_index"]), name): rt
            for r in ref["records"]
            for name, rt in r["routes"].items()
        }
        shared_worst = 0.0
        shared_mismatch = 0
        for rec in bundle["records"]:
            for name, route in rec["routes"].items():
                key = (float(rec["mixing"]), int(rec["task_index"]), name)
                for lever in ("frozen", "L1", "L12", "L123"):
                    a = ref_map[key]["arms"][lever]
                    b = route["arms"][lever]
                    if a["e_q"] is None or b["e_q"] is None:
                        if a["e_q"] != b["e_q"]:
                            shared_mismatch += 1
                        continue
                    shared_worst = max(shared_worst, abs(float(a["e_q"]) - float(b["e_q"])))
                    if a["emitted"] != b["emitted"]:
                        shared_mismatch += 1
    else:
        failures.append(f"FP-BOUND-001 bundle missing at {ref_path}")

    report = {
        "results_dir": str(args.results),
        "C1_l12m_recomputed": {"max_abs_delta": max_l12m_delta},
        "C2_pair_level_coverage": {"cells": pair_checks, "passing": pair_pass,
                                   "pairs_per_cell": D},
        "C3_e_q_reconstruction": {"max_abs_delta": max_e_q_delta},
        "C4_shared_arms_vs_bound_001": {
            "max_abs_delta_e_q": shared_worst,
            "decision_mismatches": shared_mismatch,
        },
        "C5_decision_chain": {"max_abs_delta_h": max_h_delta,
                              "emission_mismatches": emit_mismatch},
        "C6_risk_accounting_bad": risk_bad,
        "failure_count": len(failures)
        + emit_mismatch
        + risk_bad
        + (shared_mismatch or 0)
        + (1 if shared_worst not in (0.0, None) else 0)
        + (1 if max_e_q_delta > 1e-12 else 0)
        + (1 if max_l12m_delta > 1e-12 else 0)
        + (1 if pair_pass != pair_checks else 0)
        + (1 if max_h_delta > 1e-12 else 0),
        "failures": failures[:20],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["failure_count"] else 0


if __name__ == "__main__":
    sys.exit(main())
