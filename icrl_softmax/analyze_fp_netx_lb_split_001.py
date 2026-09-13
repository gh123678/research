"""FP-NETX-001 targeted check: at the deepest emitting steps, how can the two
producers agree when the certified margin is far below their Qhat gap?

At record ``0.08/11`` steps 19-21 the sealed numpy margin reaches ``3.4e-16``
while the network-vs-numpy sup gap on the same record is ``~1.6e-6`` (exact
route) / ``~3e-6`` (finite route). A naive reading says the decision should have
flipped. This script replays those exact decision points with BOTH producers and
prints, per candidate eta and per state:

* ``LB_s`` from the numpy ``Qhat`` and from the network ``Qhat``,
* their difference,
* whether the first-passage eta and the emitted verdict are the same.

If the gap between the two LB vectors is orders smaller than the Qhat gap, the
explanation is structural (the tilt at the selected eta is nearly orthogonal to
the producers' difference); if it is the same order, the agreement in the sealed
run was luck and must be reported as such.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import fixed_policy_expected_sarsa as es  # noqa: E402
import fixed_policy_expected_sarsa_scaled as fs  # noqa: E402
import fixed_policy_mp_certificate as mc  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402
from evaluate_fp_attn_iter_001 import network_qhat  # noqa: E402
from fp_certfix_first_n import first_visit_batch, step_seed_parts  # noqa: E402
from fp_sample_vectorised_batch import vectorised_batch  # noqa: E402
from model import (  # noqa: E402
    EndToEndFiniteSoftmaxExpectedSARSA,
    EndToEndMaskedSoftmaxExpectedSARSA,
)

ETA_GRID = fs.ETA_CANDIDATES


def lbs_for(policy: np.ndarray, q_hat: np.ndarray, e_q: float):
    out = []
    for eta in ETA_GRID:
        cand = es.relative_softmax_candidate(policy, q_hat, eta)
        delta_pi = cand - policy
        i_hat = (delta_pi * q_hat).sum(axis=1)
        lb = i_hat - e_q * np.abs(delta_pi).sum(axis=1)
        out.append((float(eta), lb, cand))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", default="0.08/11")
    parser.add_argument("--steps", type=str, default="18,19,20,21")
    args = parser.parse_args()

    data = json.loads((args.results / "task_results.json").read_text(encoding="utf-8"))
    chains = int(data["chains_per_step"])
    delta_step = float(data["delta_step"])
    salt = int(data["task_salt"])
    min_visits = int(data.get("min_visits", 2000))
    want_steps = {int(v) for v in args.steps.split(",")}
    mixing_s, task_s = args.record.split("/")
    mixing, task_index = float(mixing_s), int(task_s)

    networks = {
        "expected_exact": EndToEndMaskedSoftmaxExpectedSARSA(
            gamma=fs.GAMMA, alpha=fs.ALPHA
        ),
        "expected_finite": EndToEndFiniteSoftmaxExpectedSARSA(
            gamma=fs.GAMMA, alpha=fs.ALPHA, zeta=fs.ZETA, xi=fs.XI, tau=fs.TAU
        ),
    }

    mdp, behaviour, rng = fs.build_task(task_index=task_index, mixing=mixing)
    exact0 = policy_quantities(mdp, behaviour)
    mu_state = np.asarray(exact0["mu_state"], dtype=np.float64)
    train = fs.training_batch(mdp, behaviour, mu_state, rng)

    rec = next(
        r
        for r in data["records"]
        if float(r["mixing"]) == mixing and int(r["task_index"]) == task_index
    )

    rows: list[dict[str, Any]] = []
    for route in data["routes"]:
        sealed_steps = rec["routes"][route]["mp"]["steps"]
        current = behaviour.copy()
        for s in sealed_steps:
            step_index = int(s["step"])
            if step_index > max(want_steps):
                break
            q_np = np.asarray(
                fs.run_route(route, current, train)["q_hat"], dtype=np.float64
            ).reshape(fs.N_STATES, fs.N_ACTIONS)
            q_net, _ = network_qhat(networks[route], current, train)
            q_net = np.asarray(q_net, dtype=np.float64).reshape(
                fs.N_STATES, fs.N_ACTIONS
            )
            if step_index in want_steps:
                raw = vectorised_batch(
                    mdp,
                    behaviour,
                    mu_state,
                    step_seed_parts(fs.SEED, salt, mixing, task_index, step_index),
                    chains,
                    64,
                )
                reduced, _ = first_visit_batch(raw, 64)
                cert_np = mc.mp_certificate_firstvisit(
                    q_np, current, reduced, min_visits=min_visits, delta_step=delta_step
                )
                cert_net = mc.mp_certificate_firstvisit(
                    q_net, current, reduced, min_visits=min_visits, delta_step=delta_step
                )
                lb_np = lbs_for(current, q_np, float(cert_np["e_q"]))
                lb_net = lbs_for(current, q_net, float(cert_net["e_q"]))
                first_np = next((eta for eta, lb, _ in lb_np if float(lb.min()) > 0), None)
                first_net = next(
                    (eta for eta, lb, _ in lb_net if float(lb.min()) > 0), None
                )
                detail = []
                for (eta, lb_a, _), (_, lb_b, _) in zip(lb_np, lb_net):
                    detail.append(
                        {
                            "eta": eta,
                            "min_lb_numpy": float(lb_a.min()),
                            "min_lb_network": float(lb_b.min()),
                            "min_lb_abs_diff": float(np.max(np.abs(lb_a - lb_b))),
                        }
                    )
                rows.append(
                    {
                        "record": f"{args.record}/{route}",
                        "step": step_index,
                        "qhat_sup_gap": float(np.max(np.abs(q_np - q_net))),
                        "e_q_numpy": float(cert_np["e_q"]),
                        "e_q_network": float(cert_net["e_q"]),
                        "e_q_abs_diff": abs(
                            float(cert_np["e_q"]) - float(cert_net["e_q"])
                        ),
                        "first_eta_numpy": first_np,
                        "first_eta_network": first_net,
                        "sealed_emitted": bool(s["update_emitted"]),
                        "sealed_eta": s["eta_selected"],
                        "per_eta": detail,
                    }
                )
            if not bool(s["update_emitted"]):
                break
            cand = es.relative_softmax_candidate(current, q_np, float(s["eta_selected"]))
            current = np.asarray(cand, dtype=np.float64)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "lb_split.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    for r in rows:
        print(
            f"{r['record']} step {r['step']}: qhat gap {r['qhat_sup_gap']:.3e}, "
            f"E_Q diff {r['e_q_abs_diff']:.3e}, eta numpy {r['first_eta_numpy']} "
            f"vs network {r['first_eta_network']}, sealed {r['sealed_eta']}"
        )
        for d in r["per_eta"]:
            print(
                f"    eta {d['eta']:<5} min LB numpy {d['min_lb_numpy']:+.3e} "
                f"network {d['min_lb_network']:+.3e} |max diff| {d['min_lb_abs_diff']:.3e}"
            )


if __name__ == "__main__":
    main()
