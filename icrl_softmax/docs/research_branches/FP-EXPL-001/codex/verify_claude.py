"""Independent mathematical acceptance reference for FP-EXPL-001.

No Claude author module is imported.  The command-line adapter compares the
independent reconstruction with the sealed Claude JSON evidence and writes a
small, machine-readable acceptance record.
"""

import argparse
import copy
import hashlib
import platform
import sys
import tempfile
import json
from pathlib import Path

from fractions import Fraction

import numpy as np


BASELINE = "8c915c4bf2bb9533e2374f5d2c91cf34e5c13c77"
GAMMA, ALPHA, SHARPNESS = 0.7, 0.5, 8.0
PI = np.array([[0.75, 0.25], [0.25, 0.75]])
P = np.array([[0.75, 0.25], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25]])
R = np.array([1.0, -0.5, 0.25, 0.25])


def norm(x):
    """Vector infinity norm (matrix infinity norm is explicit elsewhere)."""
    return float(np.max(np.abs(x)))


def reconstruct():
    rng = np.random.Generator(np.random.PCG64(20260911))
    transitions, draws = [], []
    state = 0
    for _ in range(64):
        ua, up = float(rng.random()), float(rng.random())
        action = 0 if ua < 0.5 else 1
        pair = state * 2 + action
        successor = 0 if up < P[pair, 0] else 1
        transitions.append([state, action, float(R[pair]), successor])
        draws.append([ua, up])
        state = successor
    rows = np.array(transitions)
    pairs = (2 * rows[:, 0] + rows[:, 1]).astype(int)
    successors = rows[:, 3].astype(int)
    rewards = rows[:, 2]
    counts = np.bincount(pairs, minlength=4)
    if np.any(counts == 0):
        return {"status": "COVERAGE_FAILURE", "transitions": transitions,
                "draws": draws, "counts": counts.tolist()}

    current0 = np.zeros((64, 4))
    successor0 = np.zeros((64, 4))
    writer0 = np.zeros((4, 64))
    current = np.empty((64, 4))
    successor = np.empty((64, 4))
    writer = np.empty((4, 64))
    exp8 = np.exp(SHARPNESS)
    for t in range(64):
        for y in range(4):
            state_y, action_y = divmod(y, 2)
            current0[t, y] = int(pairs[t] == y)
            successor0[t, y] = int(successors[t] == state_y) * PI[state_y, action_y]
            current[t, y] = (exp8 if pairs[t] == y else 1.0) / (exp8 + 3.0)
            successor[t, y] = ((exp8 if successors[t] == state_y else 1.0)
                               * PI[state_y, action_y] / (exp8 + 1.0))
            writer0[y, t] = int(pairs[t] == y) / counts[y]
            writer[y, t] = ((exp8 if pairs[t] == y else 1.0)
                            / (counts[y] * exp8 + 64 - counts[y]))
    identity = np.eye(4)
    g0 = identity + ALPHA * writer0 @ (GAMMA * successor0 - current0)
    b0 = ALPHA * writer0 @ rewards
    gf = identity + ALPHA * writer @ (GAMMA * successor - current)
    bf = ALPHA * writer @ rewards
    cf = float(np.max(np.sum(np.abs(gf), axis=1)))
    population_t = np.array([[P[x, y // 2] * PI[y // 2, y % 2]
                              for y in range(4)] for x in range(4)])
    q_pi = np.linalg.solve(identity - GAMMA * population_t, R)
    q_hat = np.linalg.solve(identity - g0, b0)
    q_f = np.linalg.solve(identity - gf, bf) if cf < 1 else None
    exact, finite, direct = [np.zeros(4)], [np.zeros(4)], [np.zeros(4)]
    stages, perturbation = [], [0.0]
    for _ in range(64):
        q = exact[-1]
        ec = -ALPHA * writer @ (current - current0) @ q
        es = ALPHA * GAMMA * writer @ (successor - successor0) @ q
        ew = ALPHA * (writer - writer0) @ (rewards + GAMMA * successor0 @ q - current0 @ q)
        delta = (gf @ q + bf) - (g0 @ q + b0)
        stages.append({"current": ec, "successor": es, "write": ew, "total": delta})
        perturbation.append(cf * perturbation[-1] + norm(delta))
        exact.append(g0 @ q + b0)
        finite.append(gf @ finite[-1] + bf)
        qd = direct[-1]
        next_qd = qd.copy()
        for x in range(4):
            group_residuals = []
            for t in range(64):
                if pairs[t] == x:
                    vnext = sum(PI[successors[t], a] * qd[2 * successors[t] + a]
                                for a in range(2))
                    group_residuals.append(rewards[t] + GAMMA * vnext - qd[x])
            next_qd[x] += ALPHA * sum(group_residuals) / len(group_residuals)
        direct.append(next_qd)

    # Conservative analytic certificate for ANY covered batch of length 64.
    # e > 8/3 follows from its first five series terms; (8/3)^8 > 2000.
    # Row L1 routing errors are twice the mass assigned outside the match.
    e_lower_valid = sum(Fraction(1, d) for d in (1, 1, 2, 6, 24)) > Fraction(8, 3)
    exp_lower_valid = Fraction(8, 3) ** 8 > 2000
    uniform_cf_bound = (Fraction(17, 20) + Fraction(7, 10) / 2001
                        + Fraction(3, 2003) + Fraction(17, 10) * Fraction(63, 2063))
    # For this frozen covered batch, bound every entry perturbation away
    # from a strictly positive rational G0. This additionally proves Gf>=0
    # and, since its row sums are exactly 17/20, ||Gf||_inf=17/20.
    minimum_count = int(min(counts))
    batch_perturbation_bound = (
        Fraction(7, 10) / 2001 + Fraction(3, 2003)
        + Fraction(17, 10) * Fraction(64-minimum_count, 2000*minimum_count+64-minimum_count))
    rational_g0 = []
    for x in range(4):
        for y in range(4):
            next_count = sum(int(pairs[t] == x and successors[t] == y // 2) for t in range(64))
            value = (Fraction(int(x == y), 2) + Fraction(7, 20)
                     * Fraction(next_count, int(counts[x])) * Fraction(float(PI[y // 2, y % 2])))
            rational_g0.append(value)
    return {
        "status": "COVERED", "transitions": transitions, "draws": draws,
        "counts": counts, "pairs": pairs, "successors": successors,
        "C0": current0, "S0": successor0, "W0": writer0,
        "C": current, "S": successor, "W": writer,
        "G0": g0, "b0": b0, "Gf": gf, "bf": bf, "c_f": cf,
        "T_pi": population_t, "q_pi": q_pi, "q_hat": q_hat, "q_f_inf": q_f,
        "V_pi": np.sum(q_pi.reshape(2, 2) * PI, axis=1),
        "exact": np.array(exact), "finite": np.array(finite),
        "direct": np.array(direct), "stages": stages, "E": np.array(perturbation),
        "uniform_contraction_proof": {
            "e_lower_valid": bool(e_lower_valid),
            "exp_lower_valid": bool(exp_lower_valid),
            "rational_bound": str(uniform_cf_bound),
            "bound": float(uniform_cf_bound),
            "strictly_less_than_one": bool(uniform_cf_bound < 1),
            "batch_min_exact_G0": str(min(rational_g0)),
            "batch_entry_perturbation_bound": str(batch_perturbation_bound),
            "batch_Gf_strictly_positive": bool(min(rational_g0) > batch_perturbation_bound),
            "batch_exact_cf": "17/20" if min(rational_g0) > batch_perturbation_bound else None,
        },
    }


def strict_load(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON literal: {value}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def finite_float(value):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"nonfinite JSON number: {value}")
        return number

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=reject, parse_float=finite_float, object_pairs_hook=unique)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Audit:
    def __init__(self):
        self.checks = []

    def require(self, label, condition):
        self.checks.append({"label": label, "pass": bool(condition),
                            "mode": "exact/structural", "max_abs_error": None})

    def number(self, label, actual, expected, mode="repeated"):
        # Shape equality prevents broadcasting or silently missing trace entries.
        aa, bb = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        valid = aa.shape == bb.shape and np.all(np.isfinite(aa)) and np.all(np.isfinite(bb))
        if not valid:
            self.require(label + ": finite values and exact shape", False)
            return
        error = np.abs(aa - bb)
        if mode == "absolute":
            tolerance = np.full(aa.shape, 1e-12)
        elif mode == "exact":
            tolerance = np.zeros(aa.shape)
        else:
            tolerance = 1e-10 * (1 + np.maximum(np.abs(aa), np.abs(bb)))
        self.checks.append({"label": label, "pass": bool(np.all(error <= tolerance)),
                            "mode": mode, "max_abs_error": float(error.max(initial=0)),
                            "compared_values": int(aa.size)})

    def bound(self, label, lhs, rhs):
        self.require(label, bool(np.isfinite(lhs) and np.isfinite(rhs)
                     and lhs <= rhs + 1e-10 * (1 + max(abs(lhs), abs(rhs)))))


def expected_layout(ref):
    """Reconstruct fixed parameter sparsity and prompt from the declared layout.

    This is an audit of saved matrices, not an import of the author's operator.
    All subsequent dynamic Q operations run through the generic matrix evaluator.
    """
    h = np.zeros((69, 19))
    h[:, 0] = 1
    for x in range(4):
        s, a = divmod(x, 2)
        h[x, [1, 4 + x, 8 + s, 10 + a]] = 1
        h[x, 14] = np.log(PI[s, a])
    for t, (s, a, reward, u) in enumerate(ref["transitions"]):
        h[t + 4, [2, 4 + 2 * s + a, 8 + u, 10 + a]] = 1
        h[t + 4, 12] = reward
    h[68, 3] = 1
    masks, stages = {}, {}
    names = ("stage1_current_read", "stage2_successor_read", "stage4_writeback")
    for n, name in zip((1, 2, 4), names):
        d = 3 if n == 2 else 4
        wq, wk = np.zeros((d, 19)), np.zeros((d, 19))
        if n == 2:
            wq[0, 8] = wq[1, 9] = np.sqrt(3) * SHARPNESS
            wq[2, 0] = np.sqrt(3)
            wk[0, 8] = wk[1, 9] = wk[2, 14] = 1
        else:
            for j in range(4):
                wq[j, 4 + j], wk[j, 4 + j] = 2 * SHARPNESS, 1
        wv, wo = np.zeros((1, 19)), np.zeros((19, 1))
        wv[0, 17 if n == 4 else 13] = 1
        wo[{1: 15, 2: 16, 4: 18}[n], 0] = 1
        stages[name] = dict(WQ=wq, WK=wk, WV=wv, WO=wo, scale=1 / np.sqrt(d))
        mask = np.zeros((69, 69), dtype=bool)
        mask[:, 68] = True
        active, keys = (slice(0, 4), slice(4, 68)) if n == 4 else (slice(4, 68), slice(0, 4))
        mask[active, 68] = False
        mask[active, keys] = True
        masks[f"stage{n}_allowed"] = mask
    m3, m5, reset = np.eye(19), np.eye(19), np.eye(19)
    m3[17, [12, 16, 15]] = [1, GAMMA, -1]
    m5[13, 18] = ALPHA
    reset[15:, 15:] = 0
    maps = dict(M3_residual=m3, M5_q_update=m5, P_reset_scratch=reset)
    return h, stages, masks, maps


def evaluate_saved(h, network):
    """One generic attention/residual/feedforward pass using saved matrices."""
    h = np.array(h, copy=True)
    captured = {}
    for n, name in ((1, "stage1_current_read"), (2, "stage2_successor_read"),
                    (4, "stage4_writeback")):
        if n == 4:
            h = h @ np.array(network["feedforward_maps"]["M3_residual"]).T
        st = network["stages"][name]
        q, k, v = (h @ np.array(st[key]).T for key in ("WQ", "WK", "WV"))
        scores = q @ k.T * st["scale"]
        masked = np.where(network["masks"][f"stage{n}_allowed"], scores, -np.inf)
        p = np.exp(masked - masked.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        out = p @ v
        captured[name] = dict(queries=q, keys=k, values=v, scores=scores, probs=p, out=out)
        h += out @ np.array(st["WO"]).T
    h = h @ np.array(network["feedforward_maps"]["M5_q_update"]).T
    captured["residual_field_before_scratch_reset"] = h[:, [12, 15, 16, 17, 18, 13]]
    captured["captured_residual_vs_stage4_value_read_max_abs"] = norm(
        h[4:68, 17] - captured["stage4_writeback"]["values"][4:68, 0])
    h = h @ np.array(network["feedforward_maps"]["P_reset_scratch"]).T
    return h, captured


def audit_evidence(e, a, ref):
    a.require("schema/task/version", (e["schema"], e["task_id"], e["task_version"])
              == ("FP-EXPL-001/claude-witness/v1", "FP-EXPL-001", "1.1"))
    frozen = dict(gamma=GAMMA, alpha=ALPHA, target_pi=PI.tolist(),
                  behavior_pi=[[0.5, 0.5], [0.5, 0.5]],
                  rewards_by_pair_00_01_10_11=R.tolist(), P_next_state_by_pair=P.tolist(),
                  canonical_pairs=[[0, 0], [0, 1], [1, 0], [1, 1]], initial_state=0,
                  batch_length=64, seed=20260911, Q0=[0.0]*4, updates=64,
                  xi=8.0, zeta=8.0, tau=8.0, pair_order="00,01,10,11")
    a.require("all frozen inputs exactly", e["frozen_inputs"] == frozen)
    a.require("frozen probability tolerance", e["tolerances"]["one_step_and_probabilities_abs"] == 1e-12)
    sampling = e["sampling"]
    expected_rows = [dict(t=t, s=int(row[0]), a=int(row[1]), r=row[2],
                         s_next=int(row[3]), u_action=ref["draws"][t][0],
                         u_transition=ref["draws"][t][1])
                     for t, row in enumerate(ref["transitions"])]
    a.require("raw transitions and PCG64 draws exactly", sampling["transitions"] == expected_rows)
    a.require("coverage and visit counts", sampling["visit_counts_by_pair_00_01_10_11"]
              == list(ref["counts"]) and sampling["coverage_pass"] is True
              and sampling["min_count"] == int(min(ref["counts"])))
    a.require("128 draws and continuity diagnostics", sampling["total_rng_draws"] == 128
              and sampling["draws_per_transition"] == 2
              and sampling["threshold_reconstruction_max_abs_error"] == 0)
    for route, keys in (("exact", ("C0", "S0", "W0", "G0", "b0")),
                        ("finite", ("C", "S", "W", "Gf", "bf"))):
        for key in keys:
            a.number(key, e["operators"][route][key], ref[key], "absolute")
        a.number(f"{route}: rewards", e["operators"][route]["r"],
                 [row[2] for row in ref["transitions"]], "exact")
    a.number("c0", e["operators"]["exact"]["c0_proved_bound"], 0.85, "absolute")
    a.number("G0 norm", e["operators"]["exact"]["norm_inf_G0"],
             np.linalg.norm(ref["G0"], np.inf), "absolute")
    a.number("cf", e["operators"]["finite"]["c_f"], ref["c_f"], "absolute")
    for key, route in (("G0", "exact"), ("Gf", "finite")):
        a.number(f"{key} spectral diagnostic only", e["operators"][route][f"rho_{key}_numerical_diagnostic"],
                 max(abs(np.linalg.eigvals(ref[key]))))
    a.number("exact identity W0 C0", ref["W0"] @ ref["C0"], np.eye(4), "absolute")
    for key in ("C0", "S0", "W0", "C", "S", "W"):
        a.number(f"{key} row stochastic", ref[key].sum(axis=1), np.ones(ref[key].shape[0]), "absolute")
    s = np.array(e["operators"]["finite"]["S"]).reshape(64, 2, 2)
    a.number("successor target policy conditional", s / s.sum(axis=2, keepdims=True),
             np.broadcast_to(PI, s.shape), "absolute")
    a.require("four mandatory distinct traces", sorted(q["id"] for q in e["sequences"])
              == sorted(["exact-direct", "exact-attention", "finite-literal", "finite-scalar"]))
    for sequence in e["sequences"]:
        kind = {"exact-direct": "direct", "exact-attention": "exact",
                "finite-literal": "finite", "finite-scalar": "finite"}[sequence["id"]]
        a.number("trace:" + sequence["id"], sequence["q_trace_k0_to_k64"], ref[kind])

    net = e["literal_network"]
    h0, stages, masks, maps = expected_layout(ref)
    a.require("token dimensions", net["n_tokens"] == 69 and net["null_token"] == 68)
    a.number("prompt frozen data", net["prompt_H"], h0, "absolute")
    a.require("three saved attention stages", set(net["stages"]) == set(stages))
    for name, weights in stages.items():
        for key, expected in weights.items():
            a.number(f"fixed parameter:{name}/{key}", net["stages"][name][key], expected, "absolute")
    for key, expected in masks.items():
        a.number(f"static role mask:{key}", net["masks"][key], expected, "exact")
    for key, expected in maps.items():
        a.number(f"fixed feedforward:{key}", net["feedforward_maps"][key], expected, "exact")
    h = np.array(net["prompt_H"], dtype=float)
    imm = list(range(13)) + [14]
    # Capture is saved for update 1 only. Reevaluate every update with saved
    # projections to test nonzero Q, frozen fields and scratch clearing as well.
    captures = e["first_step_projections"]["captured"]
    a.require("mandatory first update capture", set(captures) == {"update_1_from_step_0"})
    for k in range(64):
        q_before = h[:4, 13].copy()
        h, cap = evaluate_saved(h, net)
        a.number(f"literal one-step formula k={k}", h[:4, 13],
                 ref["Gf"] @ q_before + ref["bf"], "absolute")
        a.number(f"literal trace k={k+1}", h[:4, 13], ref["finite"][k+1])
        a.number(f"immutable fields k={k+1}", h[:, imm], np.array(net["prompt_H"])[:, imm], "exact")
        a.number(f"scratch cleared k={k+1}", h[:, 15:], np.zeros((69, 4)), "exact")
        a.number(f"only memory Q k={k+1}", h[4:, 13], np.zeros(65), "exact")
        for name, expected in (("stage1_current_read", ref["C"]),
                               ("stage2_successor_read", ref["S"]),
                               ("stage4_writeback", ref["W"])):
            pr = cap[name]["probs"]
            actual = pr[:4, 4:68] if name == "stage4_writeback" else pr[4:68, :4]
            a.number(f"literal probabilities:{name}:k={k}", actual, expected, "absolute")
            if k == 0:
                for field, value in cap[name].items():
                    a.number(f"saved first projection:{name}/{field}",
                             captures["update_1_from_step_0"][name][field], value, "absolute")
        if k == 0:
            for key in ("residual_field_before_scratch_reset", "captured_residual_vs_stage4_value_read_max_abs"):
                a.number("saved capture:" + key, captures["update_1_from_step_0"][key], cap[key], "absolute")
    for key in ("immutable_fields_max_abs_change_over_64_updates",
                "scratch_fields_max_abs_after_each_update", "context_tokens_q_field_max_abs"):
        a.number("recorded integrity:" + key, e["network_integrity"][key], 0, "exact")
    for i, q in enumerate([np.zeros(4), *np.eye(4)]):
        hi = np.array(net["prompt_H"], dtype=float)
        hi[:4, 13] = q  # Allowed independent basis-vector prompt initialization.
        after, _ = evaluate_saved(hi, net)
        a.number(f"finite affine basis/zero {i}", after[:4, 13], ref["Gf"] @ q + ref["bf"], "absolute")
        reward = np.array([row[2] for row in ref["transitions"]])
        direct = q + ALPHA * ref["W0"] @ (reward + GAMMA * ref["S0"] @ q - ref["C0"] @ q)
        a.number(f"exact affine basis/zero {i}", direct, ref["G0"] @ q + ref["b0"], "absolute")

    fp = e["fixed_points"]
    qh, qp, qf = ref["q_hat"], ref["q_pi"], ref["q_f_inf"]
    a.number("q_hat", fp["q_hat_empirical"]["value"], qh)
    a.number("q_pi", fp["q_pi_audit"]["value"], qp)
    a.number("qhat solve residual", fp["q_hat_empirical"]["residual_inf"], norm(ref["G0"] @ qh + ref["b0"] - qh))
    a.number("qpi solve residual", fp["q_pi_audit"]["residual_inf"], norm(R + GAMMA * ref["T_pi"] @ qp - qp))
    a.number("Vpi", fp["q_pi_audit"]["V_pi_by_state"], ref["V_pi"])
    a.number("Vpi difference", fp["q_pi_audit"]["V_pi_state_difference"], ref["V_pi"][0] - ref["V_pi"][1])
    a.number("target immediate rewards", fp["q_pi_audit"]["target_mean_immediate_reward_by_state"], [0.625, 0.25], "exact")
    bias, residual = norm(qh - qp), norm(ref["G0"] @ qp + ref["b0"] - qp)
    rhs = residual / (1 - 0.85)
    for key, val in dict(lhs_norm_inf_q_hat_minus_q_pi=bias, rhs_audit_bound=rhs,
                         F0_q_pi_residual_inf=residual, margin_rhs_minus_lhs=rhs-bias).items():
        a.number("data bias:" + key, fp["data_bias"][key], val)
    a.bound("data bias bound", bias, rhs)

    dec = e["stage_error_decomposition"]
    a.require("64 ordered stage records", [x["k"] for x in dec["per_step"]] == list(range(64)))
    stage_residuals, margins = [], []
    for k, saved in enumerate(dec["per_step"]):
        terms = ref["stages"][k]
        current, successor, write, delta = (terms[key] for key in ("current", "successor", "write", "total"))
        for name, value in (("current", current), ("successor", successor), ("write", write)):
            a.number(f"stage:{k}:{name}", saved[f"e_{name}"], value)
            a.number(f"stage norm:{k}:{name}", saved[f"e_{name}_inf"], norm(value))
        res = norm(current + successor + write - delta)
        actual = norm(ref["finite"][k+1] - ref["exact"][k+1])
        margin = ref["E"][k+1] - actual
        stage_residuals.append(res)
        margins.append(margin)
        fields = dict(delta_inf=norm(delta), triangle_gap=norm(current)+norm(successor)+norm(write)-norm(delta),
                      telescoping_residual=res, E_kplus1=ref["E"][k+1], actual_error_kplus1=actual,
                      bound_margin_kplus1=margin)
        for key, val in fields.items():
            a.number(f"stage:{k}:{key}", saved[key], val)
        a.number(f"signed stage identity k={k}", current + successor + write, delta)
        a.bound(f"perturbation bound k={k+1}", actual, ref["E"][k+1])
    for key, val in dict(max_telescoping_residual=max(stage_residuals), min_bound_margin=min(margins),
                         E_64=ref["E"][-1], actual_error_64=norm(ref["finite"][-1]-ref["exact"][-1])).items():
        a.number("stage summary:" + key, dec[key], val)

    def trajectory_bound(label, saved, trace, fixed, rate):
        a.require(label + " ordered 65 records", [row["k"] for row in saved["per_k"]] == list(range(65)))
        margins = []
        for k, row in enumerate(saved["per_k"]):
            lhs, rhs = norm(trace[k] - fixed), rate**k * norm(fixed)
            for key, val in dict(lhs=lhs, rhs=rhs, margin=rhs-lhs).items():
                a.number(f"{label}:{k}:{key}", row[key], val)
            a.bound(f"{label}:{k}:bound", lhs, rhs)
            margins.append(rhs-lhs)
        a.number(label + " min margin", saved["min_margin"], min(margins))

    trajectory_bound("exact contraction", e["exact_contraction_bound"], ref["exact"], qh, 0.85)
    fa = e["finite_fixed_point_analysis"]
    if qf is None:
        a.require("finite fixed point unavailable", fp["q_f_inf"] is None and fa["status"] != "available")
        return
    a.require("finite fixed point available", fp["q_f_inf_status"] == "available" and fa["status"] == "available")
    a.number("q_f_inf", fp["q_f_inf"]["value"], qf)
    a.number("qf solve residual", fp["q_f_inf"]["residual_inf"], norm(ref["Gf"] @ qf + ref["bf"] - qf))
    shift, rhs = norm(qf-qh), norm(ref["Gf"] @ qh + ref["bf"] - qh) / (1-ref["c_f"])
    for key, val in dict(lhs=shift, rhs=rhs, margin=rhs-shift).items():
        a.number("steady state:" + key, fa["steady_state_bound"][key], val)
    a.bound("finite fixed point shift bound", shift, rhs)
    trajectory_bound("finite contraction", fa["trajectory_bound"], ref["finite"], qf, ref["c_f"])
    saved = fa["signed_decomposition_vs_q_pi"]
    a.require("65 ordered total error decompositions", [x["k"] for x in saved["per_k"]] == list(range(65)))
    residuals, margins = [], []
    for k, row in enumerate(saved["per_k"]):
        q = ref["finite"][k]
        signed = (q-qf) + (qf-qh) + (qh-qp)
        res, gap = norm(signed-(q-qp)), norm(q-qf)+shift+bias-norm(q-qp)
        residuals.append(res)
        margins.append(gap)
        for key, val in dict(transient_inf=norm(q-qf), finite_fixed_point_shift_inf=shift,
                             data_bias_inf=bias, total_error_inf=norm(q-qp), identity_residual=res,
                             norm_triangle_margin=gap).items():
            a.number(f"total decomposition:{k}:{key}", row[key], val)
        a.number(f"signed total identity k={k}", signed, q-qp)
        a.bound(f"total triangle bound k={k}", norm(q-qp), norm(q-qf)+shift+bias)
    a.number("decomp max residual", saved["max_identity_residual"], max(residuals))
    a.number("decomp min norm margin", saved["min_norm_triangle_margin"], min(margins))
    a.require("analytic contraction certificate", all(ref["uniform_contraction_proof"][k]
              for k in ("e_lower_valid", "exp_lower_valid", "strictly_less_than_one")))
    a.require("exact batch cf positivity certificate", ref["uniform_contraction_proof"]["batch_Gf_strictly_positive"])
    a.bound("cf below analytic certificate", ref["c_f"], ref["uniform_contraction_proof"]["bound"])
    a.require("witness summary", e["witness_self_verdict"] == "PASS" and e["failed_runs"] == [])
    hc = e["h_checks"]
    expected_h = {"H1_coverage", "sampling_integrity", "H2a_direct_equals_exact_attention",
                  "H2b_literal_finite_equals_scalar", "H3_stage_errors_and_bounds", "H4_contraction",
                  "H5_state_values_and_data_bias", "affine_basis_reconstruction", "policy_preservation",
                  "grouped_mean_no_frequency_multiplier", "network_integrity", "input_rejection",
                  "finite_fixed_point_bounds"}
    a.require("all hypothesis checks present and PASS", set(hc) == expected_h
              and all(item["pass"] is True for item in hc.values()))
    traces = {x["id"]: np.array(x["q_trace_k0_to_k64"]) for x in e["sequences"]}
    for label, left, right in (("H2a_direct_equals_exact_attention", "exact-direct", "exact-attention"),
                               ("H2b_literal_finite_equals_scalar", "finite-literal", "finite-scalar")):
        lv, rv = traces[left], traces[right]
        scaled = norm(np.abs(lv-rv)/(1e-10*(1+np.maximum(np.abs(lv), np.abs(rv)))))
        a.number(label + " saved scaled trace statistic", hc[label]["max_trace_scaled_error"], scaled)
        a.require(label + " absolute single-step tolerance", 0 <= hc[label]["max_one_step_abs_error"] <= 1e-12)
    a.require("author invalid-input checks", len(hc["input_rejection"]["cases"]) == 8
              and all(x["rejected"] is True for x in hc["input_rejection"]["cases"]))


def accept(results_path):
    results_path = Path(results_path).resolve()
    a, ref = Audit(), reconstruct()
    digests = {"results_sha256": sha256(results_path), "acceptance_source_sha256": sha256(__file__)}
    try:
        e = strict_load(results_path)
        if ref["status"] != "COVERED":
            a.require("construction coverage requirement", False)
        else:
            audit_evidence(e, a, ref)
        vpath = results_path.with_name("verification.json")
        verification = strict_load(vpath)
        digests["author_verification_sha256"] = sha256(vpath)
        groups = verification["groups"]
        a.require("all 12 author verification groups", len(groups) == 12
                  and {x.split("_")[0] for x in groups} == set("ABCDEFGHIJKL"))
        a.require("author verification totals", sum(g["checks"] for g in groups.values())
                  == verification["summary"]["total_checks"]
                  and all(g["checks"] > 0 and g["failures"] == 0 for g in groups.values())
                  and verification["summary"]["total_failures"] == 0
                  and verification["summary"]["verdict"] == "PASS")
        a.number("author verifier counts", verification["summary"]["coverage_counts"], ref["counts"], "exact")
        a.number("author verifier cf", verification["summary"]["c_f"], ref["c_f"], "absolute")
        source_dir = results_path.parents[3] / "docs/research_branches/FP-EXPL-001/claude"
        for name in ("witness", "verify"):
            digest = sha256(source_dir / (name + ".py"))
            a.require(f"source SHA256:{name}", e["code"][name + "_sha256"] == digest)
            digests[name + "_sha256"] = digest
    except (KeyError, ValueError, TypeError, IndexError, OSError, OverflowError) as exc:
        a.require(f"complete valid evidence: {type(exc).__name__}: {exc}", False)
    failures = [item for item in a.checks if not item["pass"]]
    return dict(schema="FP-EXPL-001/codex-acceptance/v2", task_id="FP-EXPL-001", task_version="1.1",
                baseline_commit=BASELINE, results_path=str(results_path), digests=digests,
                environment=dict(python=platform.python_version(), numpy=np.__version__, executable=sys.executable),
                checks=a.checks, analytic_contraction_certificate=ref.get("uniform_contraction_proof"),
                summary=dict(total_checks=len(a.checks), total_failures=len(failures),
                             verdict="PASS" if not failures else "FAIL",
                             coverage_counts=list(map(int, ref["counts"])), c_f=ref.get("c_f")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Sealed Claude results.json")
    parser.add_argument("--output", help="Acceptance JSON output path")
    parser.add_argument("--self-test", action="store_true", help="Reject corrupted evidence; no new research samples")
    args = parser.parse_args()
    if args.self_test:
        report = mutation_checks(args.results)
        if not args.output:
            parser.error("--self-test requires --output to preserve its separate evidence")
    else:
        report = accept(args.results)
    output = Path(args.output) if args.output else (
        Path(__file__).resolve().parents[4] / "results/FP-EXPL-001/codex/verification.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], allow_nan=False))
    for item in report["checks"]:
        if not item["pass"]:
            print(json.dumps(item, allow_nan=False))
    raise SystemExit(0 if report["summary"]["verdict"] == "PASS" else 1)


def mutation_checks(results_path):
    """Meaningful acceptance guard tests, never new MDP runs or Q experiments."""
    evidence, ref = strict_load(results_path), reconstruct()
    cases = []
    for label in ("missing trace", "broadcast shape", "changed stage error", "changed bound",
                  "changed projection", "changed fixed weight", "changed raw draw", "changed late Q"):
        e = copy.deepcopy(evidence)
        if label == "missing trace":
            e["sequences"].pop()
        elif label == "broadcast shape":
            e["operators"]["finite"]["C"] = e["operators"]["finite"]["C"][:1]
        elif label == "changed stage error":
            e["stage_error_decomposition"]["per_step"][40]["e_write"][2] += 0.01
        elif label == "changed bound":
            e["finite_fixed_point_analysis"]["trajectory_bound"]["per_k"][60]["rhs"] += 0.01
        elif label == "changed projection":
            e["first_step_projections"]["captured"]["update_1_from_step_0"]["stage4_writeback"]["values"][5][0] += 0.01
        elif label == "changed fixed weight":
            e["literal_network"]["stages"]["stage1_current_read"]["WV"][0][12] = 0.01
        elif label == "changed raw draw":
            e["sampling"]["transitions"][10]["u_action"] += 1e-14
        else:
            e["sequences"][2]["q_trace_k0_to_k64"][60][0] += 0.01
        audit = Audit()
        try:
            audit_evidence(e, audit, ref)
        except (KeyError, ValueError, TypeError, IndexError):
            audit.require("malformed evidence rejected", False)
        detected = [x["label"] for x in audit.checks if not x["pass"]]
        cases.append(dict(label=label, **{"pass": bool(detected)}, detected_by=detected[:3]))
    for mode, delta in (("absolute", 1.5e-12), ("repeated", 3e-10)):
        audit = Audit()
        audit.number("tolerance boundary", [1+delta, 1e8], [1, 1e8], mode)
        cases.append(dict(label=mode + " elementwise tolerance", **{"pass": not audit.checks[0]["pass"]}))
    with tempfile.TemporaryDirectory(prefix="acceptance_json_") as temp:
        path = Path(temp) / "invalid.json"
        for label, payload in (("NaN rejected", '{"x":NaN}'),
                               ("overflow rejected", '{"x":1e999}'),
                               ("duplicate key rejected", '{"x":1,"x":2}')):
            path.write_text(payload, encoding="utf-8")
            rejected = False
            try:
                strict_load(path)
            except ValueError:
                rejected = True
            cases.append(dict(label=label, **{"pass": rejected}))
    return dict(schema="FP-EXPL-001/acceptance-mutations/v1", checks=cases,
                source_sha256=sha256(__file__), input_sha256=sha256(results_path),
                summary=dict(total_checks=len(cases), total_failures=sum(not x["pass"] for x in cases),
                             verdict="PASS" if all(x["pass"] for x in cases) else "FAIL"))


if __name__ == "__main__":
    main()

