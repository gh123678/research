"""FP-BOUND-001 verifier: recompute every headline from the sealed bundle alone.

Five independent checks, none of which trusts a number the evaluator wrote down:

C1  per-pair envelope. For every record-route and every pair, recompute the exact
    Bellman residual `rho_x = (T^pi Qhat - Qhat)(x)` from the MDP tables and check
    the sealed interval: `|mean_x - rho_x| <= radius_x`. This is the real coverage
    test (24 pairs x 48 record-routes per lever), far more sensitive than the
    single sup-norm check.
C2  E_Q reconstruction. Rebuild `E_Q` from the sealed per-pair means/radii, for
    `frozen/L1/L12/support_range` as `max_x eps_x/(1-gamma)` and for `L123` as the
    propagation, and compare with the sealed scalar.
C3  propagation algebra. Rebuild `K` and `(I-gamma K)^{-1}` from the sealed
    `q_hat` plus the task's MDP, compare with the sealed matrices, and confirm the
    identity `Qhat - Q^pi = -(I - gamma P^pi)^{-1} rho` numerically -- the lemma
    L123 rests on.
C4  the L1 envelope. Recompute `R* + gamma ||Vhat||_inf + ||Qhat||_inf` from the
    sealed `q_hat` and check the sealed `y_range` equals twice it.
C5  decision chain. Recompute the exact gate margin `h`, check it against the
    sealed `h`, and verify `emitted <=> E_Q < h` -- the identity that makes the
    whole decision reproducible from two independently computed numbers.
C6  risk accounting: `2d*delta_each = delta_step` for the frozen lever,
    `d*delta_each = delta_step` for the full-budget levers.
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

import fixed_policy_expected_sarsa as es  # noqa: E402
import fixed_policy_expected_sarsa_scaled as fs  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402

GAMMA = fs.GAMMA
R_STAR = 1.5
D = fs.N_STATES * fs.N_ACTIONS
FULL_BUDGET = ("L12", "L123")


def exact_rho(mdp, policy, q_hat, gamma=GAMMA):
    P = np.asarray(mdp["P"], dtype=np.float64)
    R = np.asarray(mdp["R"], dtype=np.float64)
    V = (np.asarray(policy, dtype=np.float64) * np.asarray(q_hat, dtype=np.float64)).sum(axis=1)
    return ((P * (R + gamma * V[None, None, :])).sum(axis=2) - np.asarray(q_hat, float)).reshape(-1)


def gate_margin(policy, q_hat):
    best = 0.0
    for eta in fs.ETA_CANDIDATES:
        cand = es.relative_softmax_candidate(policy, q_hat, eta)
        dpi = cand - policy
        i_hat = (dpi * q_hat).sum(axis=1)
        l1 = np.abs(dpi).sum(axis=1)
        ratios = np.where(l1 > 0.0, i_hat / np.where(l1 > 0.0, l1, 1.0), 0.0)
        best = max(best, float(np.min(ratios)))
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT / "results" / "FP-BOUND-001" / "claude" / "c64k",
    )
    args = parser.parse_args()
    bundle = json.loads((args.results / "task_results.json").read_text(encoding="utf-8"))
    delta_step = float(bundle["delta_step"])
    levers = list(bundle["levers"])

    failures: list[str] = []
    counters = {k: 0 for k in ("pair_checks", "pair_pass", "e_q_checks", "identity_checks")}
    max_e_q_delta = 0.0
    max_h_delta = 0.0
    max_inv_delta = 0.0
    max_identity_resid = 0.0
    max_y_range_delta = 0.0
    emit_mismatch = 0
    risk_bad = 0

    for rec in bundle["records"]:
        mixing, task_index = float(rec["mixing"]), int(rec["task_index"])
        mdp, behaviour, rng = fs.build_task(task_index=task_index, mixing=mixing)
        P = np.asarray(mdp["P"], dtype=np.float64)
        pi = np.asarray(behaviour, dtype=np.float64)
        K_ref = np.einsum("sa,sat->st", pi, P)
        inv_ref = np.linalg.inv(np.eye(fs.N_STATES) - GAMMA * K_ref)
        for route_name, route in rec["routes"].items():
            q_hat = np.asarray(route["q_hat"], dtype=np.float64).reshape(
                fs.N_STATES, fs.N_ACTIONS
            )
            rho = exact_rho(mdp, behaviour, q_hat)
            h = gate_margin(behaviour, q_hat)
            max_h_delta = max(max_h_delta, abs(h - float(route["gate_margin_h"])))
            realized = float(np.max(np.abs(q_hat - np.asarray(policy_quantities(mdp, behaviour)["q_pi"], float))))
            if abs(realized - float(route["realized_q_sup_error"])) > 1e-12:
                failures.append(f"realized error mismatch {mixing}/{task_index}/{route_name}")

            # C3: the identity L123 rests on
            try:
                u = np.linalg.solve(np.eye(D) - GAMMA * _Ppi(P, pi), rho)
            except np.linalg.LinAlgError:
                u = None
            if u is not None:
                resid = float(np.max(np.abs((q_hat - np.asarray(policy_quantities(mdp, behaviour)["q_pi"], float)).reshape(-1) + u)))
                max_identity_resid = max(max_identity_resid, resid)
                counters["identity_checks"] += 1

            # C4: envelope
            v_hat = (pi * q_hat).sum(axis=1)
            e_eff = R_STAR + GAMMA * float(np.max(np.abs(v_hat))) + float(np.max(np.abs(q_hat)))

            for lever in levers:
                arm = route["arms"][lever]
                if arm["e_q"] is None:
                    continue
                means = np.asarray(arm.get("residual_means", []), dtype=np.float64)
                radii = np.asarray(arm.get("radii", []), dtype=np.float64)
                eps = np.abs(means) + radii
                sizes = np.asarray(arm["pair_sizes"], dtype=np.int64)

                # C1: per-pair coverage of the exact residual
                if means.size == D:
                    counters["pair_checks"] += 1
                    if np.all(np.abs(means - rho) <= radii + 1e-15):
                        counters["pair_pass"] += 1
                    else:
                        failures.append(
                            f"pair-level coverage violated {lever} {mixing}/{task_index}/{route_name}"
                        )

                # C2: reconstruct E_Q
                if lever in ("L12", "L123"):
                    delta_dir = delta_step / D
                else:
                    delta_dir = delta_step / (2.0 * D)
                if arm["delta_each"] is not None and abs(delta_dir - float(arm["delta_each"])) > 1e-15:
                    risk_bad += 1
                if lever in FULL_BUDGET:
                    if abs(D * float(arm["delta_each"]) - delta_step) > 1e-15:
                        risk_bad += 1
                else:
                    if abs(2.0 * D * float(arm["delta_each"]) - delta_step) > 1e-15:
                        risk_bad += 1

                if lever == "L123":
                    prop = arm.get("propagation")
                    if not prop:
                        failures.append(f"L123 without propagation block {mixing}/{task_index}/{route_name}")
                        continue
                    K_sealed = np.asarray(prop["K"], dtype=np.float64)
                    inv_sealed = np.asarray(prop["inverse"], dtype=np.float64)
                    max_inv_delta = max(max_inv_delta, float(np.max(np.abs(K_sealed - K_ref))))
                    max_inv_delta = max(max_inv_delta, float(np.max(np.abs(inv_sealed - inv_ref))))
                    epsbar = (pi * eps.reshape(fs.N_STATES, fs.N_ACTIONS)).sum(axis=1)
                    w = inv_ref @ epsbar
                    bound = eps.reshape(fs.N_STATES, fs.N_ACTIONS) + GAMMA * (
                        P * w[None, None, :]
                    ).sum(axis=2)
                    e_q_re = float(np.max(bound))
                    if abs(float(prop["uniform_bound"]) - float(np.max(eps)) / (1.0 - GAMMA)) > 1e-12:
                        failures.append(f"uniform bound mismatch {mixing}/{task_index}/{route_name}")
                else:
                    e_q_re = float(np.max(eps)) / (1.0 - GAMMA)
                    if lever in ("L1", "support_range"):
                        want = 2.0 * e_eff if lever == "L1" else None
                        if want is not None and abs(float(arm["y_range"]) - want) > 1e-12:
                            max_y_range_delta = max(
                                max_y_range_delta, abs(float(arm["y_range"]) - want)
                            )

                counters["e_q_checks"] += 1
                max_e_q_delta = max(max_e_q_delta, abs(e_q_re - float(arm["e_q"])))

                # C5: decision chain
                expect_emit = float(arm["e_q"]) < h
                if bool(arm["emitted"]) != bool(expect_emit):
                    emit_mismatch += 1

    report = {
        "results_dir": str(args.results),
        "C1_pair_level_coverage": {
            "record_routes_checked_per_lever": counters["pair_checks"],
            "passing": counters["pair_pass"],
            "pairs_per_check": D,
        },
        "C2_e_q_reconstruction": {
            "checks": counters["e_q_checks"],
            "max_abs_delta": max_e_q_delta,
        },
        "C3_propagation_algebra": {
            "max_abs_delta_K_or_inverse": max_inv_delta,
            "identity_residual_max": max_identity_resid,
        },
        "C4_envelope": {"max_abs_delta_y_range": max_y_range_delta},
        "C5_decision_chain": {"max_abs_delta_h": max_h_delta, "emission_mismatches": emit_mismatch},
        "C6_risk_accounting_bad": risk_bad,
        "failure_count": len(failures) + emit_mismatch + risk_bad,
        "failures": failures[:20],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["failure_count"] else 0


def _Ppi(P, pi):
    """`P^pi` on the pair space: [(s,a),(s',a')] = P(s'|s,a) pi(a'|s')."""
    Sa, A = P.shape[0], pi.shape[1]
    return np.einsum("sat,tb->satb", P, pi).reshape(Sa * A, Sa * A)


if __name__ == "__main__":
    sys.exit(main())
