"""Read-only audit of model.py's two end-to-end Expected-SARSA classes.

The audit compares the literal torch implementations to the independent pure NumPy route
formulas, tests attention invariants and unvisited-query behavior, checks constructor/input
boundaries, and probes numerical sharpness limits. It does not edit model.py or use any
experiment output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))

import fixed_policy_expected_sarsa as ref  # noqa: E402
from model import (  # noqa: E402
    EndToEndFiniteSoftmaxExpectedSARSA,
    EndToEndMaskedSoftmaxExpectedSARSA,
)

RNG = np.random.default_rng(20260915)


def check(ok, name, detail, report):
    report["checks_total"] += 1
    if ok:
        report["checks_passed"] += 1
    else:
        report["failures"].append({"check": name, "detail": detail})
    return ok


def fixture(s, a, m, dtype=np.float64):
    q = RNG.normal(0.0, 1.0, size=(s, a)).astype(dtype)
    policy = RNG.dirichlet(np.ones(a), size=s).astype(dtype)
    states = RNG.integers(0, s, size=m, dtype=np.int64)
    actions = RNG.integers(0, a, size=m, dtype=np.int64)
    next_states = RNG.integers(0, s, size=m, dtype=np.int64)
    rewards = RNG.normal(0.0, 1.0, size=m).astype(dtype)
    return q, policy, states, actions, rewards, next_states


def torch_forward(net, q, policy, states, actions, rewards, next_states):
    with torch.no_grad():
        return net(
            torch.as_tensor(q), torch.as_tensor(states), torch.as_tensor(actions),
            torch.as_tensor(rewards), torch.as_tensor(next_states), torch.as_tensor(policy),
        )


def main():
    out = {
        "scope": "literal model.py implementation versus pure NumPy formulas plus API boundaries",
        "checks_total": 0, "checks_passed": 0, "failures": [],
        "random_cases": [], "stress": [], "input_contract": [],
        "environment": {"torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available())},
    }

    # 1. Random dimensions, duplicate pairs, and deliberately unvisited pairs.
    def run_random_case(dtype, tol, s, a, force_unvisited):
        d = s * a
        m = max(1, min(40, d + 3))
        q, pi, st, ac, rw, ns = fixture(s, a, m, dtype)
        if force_unvisited and d > 1:
            st[:] = 0
            ac[:] = 0
        q64 = np.asarray(q, np.float64)
        pi64 = np.asarray(pi, np.float64)
        exact = ref.run_expected_exact(
            q64, pi64, st, ac, np.asarray(rw, np.float64), ns,
            gamma=0.73, alpha=0.61, layers=1, value_bound=1e9,
        )
        finite = ref.run_expected_finite(
            q64, pi64, st, ac, np.asarray(rw, np.float64), ns,
            gamma=0.73, alpha=0.61, layers=1, value_bound=1e9,
            zeta=8.0, xi=8.0, tau=8.0,
        )
        masked = EndToEndMaskedSoftmaxExpectedSARSA(gamma=0.73, alpha=0.61)
        finite_net = EndToEndFiniteSoftmaxExpectedSARSA(
            gamma=0.73, alpha=0.61, zeta=8.0, xi=8.0, tau=8.0)
        q_m, dm = torch_forward(masked, q, pi, st, ac, rw, ns)
        q_f, df = torch_forward(finite_net, q, pi, st, ac, rw, ns)
        qm = q_m.detach().cpu().numpy().astype(np.float64)
        qf = q_f.detach().cpu().numpy().astype(np.float64)
        exact_ok = np.allclose(qm, exact["layer_snapshots"][0], rtol=0.0, atol=tol)
        finite_ok = np.allclose(qf, finite["layer_snapshots"][0], rtol=0.0, atol=tol)
        check(exact_ok, "masked_random_reference", {"shape": [s, a], "dtype": str(dtype),
              "force_unvisited": force_unvisited,
              "max_abs": float(np.max(np.abs(qm - exact["layer_snapshots"][0])))}, out)
        check(finite_ok, "finite_random_reference", {"shape": [s, a], "dtype": str(dtype),
              "force_unvisited": force_unvisited,
              "max_abs": float(np.max(np.abs(qf - finite["layer_snapshots"][0])))}, out)
        wa = dm["write_attention"].detach().cpu().numpy()
        ca = dm["current_attention"].detach().cpu().numpy()
        aa = dm["action_attention"].detach().cpu().numpy()
        check(np.allclose(wa.sum(axis=0), 1.0, atol=tol), "masked_write_normalized", {}, out)
        check(np.allclose(ca.sum(axis=1), 1.0, atol=tol), "masked_read_normalized", {}, out)
        check(np.allclose(aa.sum(axis=1), 1.0, atol=tol), "exact_action_normalized", {}, out)
        wa_f = df["write_attention"].detach().cpu().numpy()
        ra_f = df["read_attention"].detach().cpu().numpy()
        sa_f = df["action_attention"].detach().cpu().numpy()
        check(np.all(np.isfinite(wa_f)) and np.all(wa_f > 0), "finite_write_positive", {}, out)
        check(np.all(np.isfinite(ra_f)) and np.all(ra_f > 0), "finite_read_positive", {}, out)
        check(np.all(np.isfinite(sa_f)) and np.all(sa_f > 0), "finite_action_positive", {}, out)
        check(np.allclose(wa_f.sum(axis=0), 1.0, atol=tol), "finite_write_normalized", {}, out)
        check(np.allclose(ra_f.sum(axis=1), 1.0, atol=tol), "finite_read_normalized", {}, out)
        check(np.allclose(sa_f.sum(axis=1), 1.0, atol=tol), "finite_action_normalized", {}, out)
        if force_unvisited and d > 1:
            leakage = float(np.abs(df["update"].detach().cpu().numpy().reshape(-1)[1]))
            check(leakage > 0.0, "finite_unvisited_leakage", {"leakage": leakage}, out)
            exact_update = float(np.abs(dm["update"].detach().cpu().numpy().reshape(-1)[1]))
            check(exact_update <= tol, "masked_unvisited_zero", {"update": exact_update}, out)
        out["random_cases"].append({"shape": [s, a], "dtype": str(dtype),
                                    "force_unvisited": force_unvisited,
                                    "exact_max_abs": float(np.max(np.abs(qm - exact["layer_snapshots"][0]))),
                                    "finite_max_abs": float(np.max(np.abs(qf - finite["layer_snapshots"][0])))})

    for dtype, tol in ((np.float64, 2e-12), (np.float32, 4e-5)):
        for s in (1, 2, 3, 5, 6):
            for a in (1, 2, 4):
                for force_unvisited in (False, True):
                    run_random_case(dtype, tol, s, a, force_unvisited)

    # 2. No trainable state / no parameter mutation.
    for name, net in (("masked", EndToEndMaskedSoftmaxExpectedSARSA()),
                      ("finite", EndToEndFiniteSoftmaxExpectedSARSA())):
        check(sum(p.numel() for p in net.parameters()) == 0, f"{name}_no_parameters", {}, out)
        check(len(net.state_dict()) == 0, f"{name}_empty_state_dict", {}, out)

    # 3. Boundary and numerical sharpness checks.
    q, pi, st, ac, rw, ns = fixture(2, 2, 4, np.float64)
    for sharp in (8.0, 50.0, 100.0, 500.0, 1000.0):
        try:
            net = EndToEndFiniteSoftmaxExpectedSARSA(
                gamma=0.7, alpha=0.65, zeta=sharp, xi=sharp, tau=sharp)
            with torch.no_grad():
                y, diag = torch_forward(net, q, pi, st, ac, rw, ns)
            finite = bool(torch.isfinite(y).all() and all(torch.isfinite(v).all()
                         for v in diag.values() if torch.is_tensor(v)))
            out["stress"].append({"sharpness": sharp, "status": "finite" if finite else "nonfinite"})
        except Exception as exc:
            out["stress"].append({"sharpness": sharp, "status": "raises",
                                  "type": type(exc).__name__, "message": str(exc)})
    check(out["stress"][0]["status"] == "finite", "task_sharpness_finite", out["stress"][0], out)

    # 4. Policy contract: the module documents a policy distribution but does not validate
    # positivity or row normalization. Record this as a boundary, not as a valid-input FAIL.
    q = torch.zeros((2, 2), dtype=torch.float64)
    states = torch.tensor([0, 1]); actions = torch.tensor([0, 1]); ns_t = torch.tensor([1, 0])
    rewards = torch.ones(2, dtype=torch.float64)
    for label, bad_pi in (
        ("unnormalized_positive", torch.tensor([[2., 2.], [1., 1.]], dtype=torch.float64)),
        ("zero_policy", torch.tensor([[1., 0.], [0.5, 0.5]], dtype=torch.float64)),
        ("negative_policy", torch.tensor([[1., -0.1], [0.5, 0.5]], dtype=torch.float64)),
    ):
        try:
            with torch.no_grad():
                y, _ = EndToEndMaskedSoftmaxExpectedSARSA()(q, states, actions, rewards, ns_t, bad_pi)
            accepted = True
            finite = bool(torch.isfinite(y).all())
            exc = None
        except Exception as error:
            accepted = False
            finite = False
            exc = type(error).__name__
        out["input_contract"].append({"case": label, "accepted": accepted,
                                      "finite_output": finite, "exception": exc})

    out["verdict"] = "PASS_FOR_VALID_INPUTS_WITH_API_LIMITS" if not out["failures"] else "FAIL"
    out["limits"] = [
        ("The random equivalence checks establish equality to the existing pure NumPy route, "
         "not conformance to an independently specified paper implementation beyond that route."),
        ("The classes do not validate that policy is strictly positive and row-normalized; "
         "out-of-domain policies can be accepted and produce mathematically meaningless "
         "Expected-SARSA values. This is an API-contract weakness, not a valid-input mismatch."),
        ("The classes do not explicitly validate all tensors share a device; states/actions are "
         "cast to long without moving them to q_values.device. GPU mixed-device behavior was "
         "not tested because CUDA availability is reported in the output."),
        ("The finite sharpness stress test is a numerical boundary. Task sharpness (8,8,8) is "
         "finite; very large positive sharpness may overflow math.exp(tau), which is an API "
         "robustness boundary rather than a task-protocol failure."),
        ("This audit does not independently reimplement the MDP/training generators or the "
         "network architecture. It audits model.py against the local pure reference and "
         "existing invariants; it cannot exclude a shared specification error."),
    ]
    out_path = PROJECT / "results/FP-MODEL-REVIEW-001"
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "model_spec_audit.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"verdict": out["verdict"], "checks": [out["checks_passed"], out["checks_total"]],
                      "failures": len(out["failures"]), "stress": out["stress"],
                      "input_contract": out["input_contract"],
                      "cuda_available": out["environment"]["cuda_available"]}, indent=2))


if __name__ == "__main__":
    main()
