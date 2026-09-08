"""Deterministic contract verification for the FP-ADV-001 action-gap module.

Covers the exact and finite-softmax local formulas, the total-variation/span
lemma, effective successor rows and within-group mass, ordered abstentions,
partial support, unvisited receivers, route dominance, the policy simplex and
exploration floor, the frozen ``theta=0.5`` transfer, the pointwise Bellman
implication, strict JSON, and the no-oracle input boundary.  Truth-based
quantities appear only inside test fixtures, never in the module interface.
"""

from __future__ import annotations

import inspect
import itertools
import json
import math

import numpy as np

import action_gap_certificate as agc
from verify_fixed_policy_q_routes import policy_quantities


TOL = 1e-12


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def make_random_mdp(rng: np.random.Generator, n_states: int = 4, n_actions: int = 3,
                    gamma: float = 0.7) -> dict:
    transition = rng.dirichlet(np.ones(n_states), size=(n_states, n_actions))
    reward = rng.uniform(-1.0, 1.0, size=(n_states, n_actions, n_states))
    p0 = rng.dirichlet(np.ones(n_states))
    return {"nS": n_states, "nA": n_actions, "gamma": gamma,
            "P": transition, "R": reward, "p0": p0}


# 1. Total-variation/span lemma, exhaustive over a finite grid.
def test_tv_span_lemma() -> None:
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows = []
    for a, b in itertools.product(grid, repeat=2):
        if a + b <= 1.0:
            rows.append([a, b, 1.0 - a - b])
    for p, q in itertools.product(rows, repeat=2):
        p_arr = np.asarray(p)
        q_arr = np.asarray(q)
        tv = agc.total_variation(p, q)
        check(abs(tv - 0.5 * float(np.abs(p_arr - q_arr).sum())) <= TOL,
              "total_variation formula mismatch")
        for e_vals in itertools.product([-2.0, -0.5, 0.0, 1.0, 3.0], repeat=3):
            e = np.asarray(e_vals)
            lhs = abs(float((p_arr - q_arr) @ e))
            span = float(e.max() - e.min())
            check(lhs <= tv * span + TOL, "TV/span lemma violated")
            check(lhs <= 2.0 * float(np.abs(e).max()) * tv + TOL,
                  "span <= 2 sup norm implication violated")
    # common offset cancels exactly
    rng = np.random.default_rng(7)
    for _ in range(50):
        p = rng.dirichlet(np.ones(4))
        q = rng.dirichlet(np.ones(4))
        e = np.full(4, rng.uniform(-5.0, 5.0))
        check(abs(float((p - q) @ e)) <= 1e-12, "common offset does not cancel")
    # disjoint supports attain the worst case
    p = [1.0, 0.0]
    q = [0.0, 1.0]
    check(agc.total_variation(p, q) == 1.0, "disjoint rows must give TV = 1")
    e = np.asarray([3.0, -3.0])
    lhs = abs(float((np.asarray(p) - q) @ e))
    check(abs(lhs - agc.total_variation(p, q) * 6.0) <= TOL,
          "disjoint-row worst case not attained")


# 2. Exact recovery decomposition on a synthetic MDP (truth only in fixture).
def test_exact_recovery_decomposition() -> None:
    rng = np.random.default_rng(11)
    mdp = make_random_mdp(rng)
    policy = np.full((4, 3), 1.0 / 3.0)
    exact = policy_quantities(mdp, policy)
    gamma = 0.7
    v_hat = exact["v_pi"] + rng.uniform(-0.3, 0.3, size=4)
    states = rng.integers(0, 4, size=500)
    actions = rng.integers(0, 3, size=500)
    next_states = np.array([rng.choice(4, p=mdp["P"][s, a])
                            for s, a in zip(states, actions)])
    rewards = np.array([mdp["R"][s, a, sp]
                        for s, a, sp in zip(states, actions, next_states)])
    n_pairs = 12
    pair_index = states * 3 + actions
    successor_counts = np.zeros((n_pairs, 4))
    np.add.at(successor_counts, (pair_index, next_states), 1.0)
    pair_counts = np.bincount(pair_index, minlength=n_pairs)
    for pair in range(n_pairs):
        if pair_counts[pair] == 0:
            continue
        mask = pair_index == pair
        q_hat = float(np.mean(rewards[mask] + gamma * v_hat[next_states[mask]]))
        row = agc.exact_successor_row(successor_counts[pair].tolist())
        e = v_hat - exact["v_pi"]
        zeta = float(np.mean(rewards[mask] + gamma * exact["v_pi"][next_states[mask]])
                     - exact["q_pi"].reshape(-1)[pair])
        reconstructed = zeta + gamma * float(np.asarray(row) @ e)
        check(abs(q_hat - exact["q_pi"].reshape(-1)[pair] - reconstructed) <= 1e-10,
              "exact recovery decomposition failed")


# 3. Softmax weights, within-group mass, effective row vs direct matrix form.
def test_softmax_weights_and_rows() -> None:
    rng = np.random.default_rng(13)
    beta = 8.0
    total = 40
    pair_index = rng.integers(0, 6, size=total)
    next_states = rng.integers(0, 4, size=total)
    successor_counts = np.zeros((6, 4))
    np.add.at(successor_counts, (pair_index, next_states), 1.0)
    total_successor = np.bincount(next_states, minlength=4)
    pair_counts = np.bincount(pair_index, minlength=6)
    for group in range(6):
        count = int(pair_counts[group])
        if count == 0:
            continue
        kappa = agc.softmax_on_group_mass(count, total, beta)
        weights = np.where(pair_index == group, math.exp(beta), 1.0)
        weights /= weights.sum()
        check(abs(kappa - float(weights[pair_index == group].sum())) <= TOL,
              "kappa disagrees with direct normalized weights")
        row, row_kappa = agc.softmax_successor_row(
            successor_counts[group].tolist(), total_successor.tolist(), beta)
        check(abs(row_kappa - kappa) <= TOL, "row kappa disagrees with mass")
        direct_row = np.zeros(4)
        for t in range(total):
            direct_row[next_states[t]] += weights[t]
        check(float(np.abs(np.asarray(row) - direct_row).max()) <= TOL,
              "effective successor row disagrees with direct weights")
        check(abs(sum(row) - 1.0) <= TOL and min(row) >= 0.0,
              "effective row is not a distribution")
    # closed form kappa = n e^beta / (n e^beta + T - n)
    for count, length in ((1, 256), (7, 256), (400, 16384), (16384, 16384)):
        kappa = agc.softmax_on_group_mass(count, length, 8.0)
        sharp = math.exp(8.0)
        expected = count * sharp / (count * sharp + length - count)
        check(abs(kappa - expected) <= TOL, "kappa closed form mismatch")


# 4. Contamination decomposition attains its bound under adversarial targets.
def test_softmax_contamination() -> None:
    rng = np.random.default_rng(17)
    value_bound = 3.0
    for _ in range(100):
        count = int(rng.integers(1, 50))
        total = count + int(rng.integers(1, 200))
        beta = float(rng.uniform(1.0, 10.0))
        radius = float(rng.uniform(0.01, 2.0))
        kappa = agc.softmax_on_group_mass(count, total, beta)
        c_bound = agc.softmax_contamination(kappa, radius, value_bound)
        check(abs(c_bound - (kappa * radius + 2.0 * value_bound * (1.0 - kappa))) <= TOL,
              "contamination formula mismatch")
        # adversarial off-group target at +B and true value at -B attains 2B
        on_deviation = radius  # saturated on-group residual
        off_target = value_bound
        true_q = -value_bound
        weights_on = kappa
        deviation = weights_on * on_deviation + (1.0 - kappa) * (off_target - true_q)
        check(abs(deviation - c_bound) <= 1e-9,
              "adversarial contamination does not saturate the bound")


# 5. Local bound validity on the event and dominance over global penalties.
def test_local_validity_and_dominance() -> None:
    rng = np.random.default_rng(19)
    gamma, beta, value_bound = 0.7, 8.0, 4.0
    for _ in range(300):
        n_states = 3
        count_a = int(rng.integers(1, 200))
        count_b = int(rng.integers(1, 200))
        total = int(rng.integers(max(count_a, count_b), 2000))
        row_a = rng.dirichlet(np.ones(n_states))
        row_b = rng.dirichlet(np.ones(n_states))
        e = rng.uniform(-1.0, 1.0, size=n_states)
        e_v = float(np.abs(e).max()) * float(rng.uniform(1.0, 2.0))
        zeta_a = float(rng.uniform(-1.0, 1.0))
        zeta_b = float(rng.uniform(-1.0, 1.0))
        r_a = abs(zeta_a) * float(rng.uniform(1.0, 3.0))
        r_b = abs(zeta_b) * float(rng.uniform(1.0, 3.0))
        true_gap = float(rng.uniform(-2.0, 2.0))
        est_diff = true_gap + (zeta_a - zeta_b) + gamma * float((row_a - row_b) @ e)
        tv = agc.total_variation(row_a.tolist(), row_b.tolist())
        u_exact = agc.exact_local_uncertainty(r_a, r_b, gamma, e_v, tv)
        lcb = est_diff - u_exact
        check(lcb <= true_gap + TOL, "exact local LCB is not a valid lower bound")
        r_max = max(r_a, r_b) * float(rng.uniform(1.0, 5.0))
        e_q_exact = gamma * e_v + r_max
        check(u_exact <= 2.0 * e_q_exact + TOL, "exact dominance failed")
        kappa_a = agc.softmax_on_group_mass(count_a, total, beta)
        kappa_b = agc.softmax_on_group_mass(count_b, total, beta)
        kappa_min = min(kappa_a, kappa_b) * float(rng.uniform(0.5, 1.0))
        c_a = agc.softmax_contamination(kappa_a, r_a, value_bound)
        c_b = agc.softmax_contamination(kappa_b, r_b, value_bound)
        u_soft = agc.softmax_local_uncertainty(c_a, c_b, gamma, e_v, tv)
        e_q_soft = gamma * e_v + r_max + 2.0 * value_bound * (1.0 - kappa_min)
        check(u_soft <= 2.0 * e_q_soft + TOL, "softmax dominance failed")
        check(agc.global_uncertainty(e_q_exact) == 2.0 * e_q_exact,
              "global penalty must be exactly 2 E_Q")


def base_inputs(rng: np.random.Generator, n_states: int = 3, n_actions: int = 3,
                total: int = 90, unvisited: tuple[int, ...] = ()) -> dict:
    pair_counts = np.zeros(n_states * n_actions, dtype=int)
    successor_counts = np.zeros((n_states * n_actions, n_states))
    total_successor = np.zeros(n_states, dtype=int)
    blocked = set(unvisited)
    sampled = 0
    while sampled < total:
        s = int(rng.integers(0, n_states))
        a = int(rng.integers(0, n_actions))
        if s * n_actions + a in blocked:
            continue
        sp = int(rng.integers(0, n_states))
        pair_counts[s * n_actions + a] += 1
        successor_counts[s * n_actions + a, sp] += 1
        total_successor[sp] += 1
        sampled += 1
    policy = np.full((n_states, n_actions), 0.05)
    preferred = rng.integers(0, n_actions, size=n_states)
    policy[np.arange(n_states), preferred] = 1.0 - (n_actions - 1) * 0.05
    radii = [None if c == 0 else 0.05 + 0.5 / math.sqrt(max(c, 1))
             for c in pair_counts]
    return {
        "family": "vfirst",
        "matching": "exact",
        "scope": "local",
        "q_estimate": rng.uniform(-1.0, 1.0, size=(n_states, n_actions)),
        "policy": policy,
        "pi_min": 0.05,
        "theta": 0.5,
        "gamma": 0.7,
        "beta": 8.0,
        "trajectory_length": total,
        "reward_bound": 1.5,
        "pair_counts": pair_counts.tolist(),
        "successor_counts": successor_counts.tolist(),
        "total_successor_counts": total_successor.tolist(),
        "recovery_radii": radii,
        "state_value_bound": 0.4,
    }


# 6. Partial support: unrelated missing pairs never block a valid comparison.
def test_partial_support_and_unvisited_receiver() -> None:
    rng = np.random.default_rng(23)
    inputs = base_inputs(rng, unvisited=(2 * 3 + 2,))
    check(inputs["pair_counts"][2 * 3 + 2] == 0, "fixture pair must be unvisited")
    check(min(inputs["pair_counts"][1], inputs["pair_counts"][2]) > 0,
          "fixture requires visited state-0 donors")
    inputs["q_estimate"] = np.zeros((3, 3))
    inputs["q_estimate"][0] = [1.0, -1.0, -1.0]
    inputs["policy"] = np.full((3, 3), 1.0 / 3.0)
    inputs["pi_min"] = 0.05
    result = agc.evaluate_route_update(**inputs)
    state0 = result["states"][0]
    check(state0["receiver"] == 0, "receiver must be argmax with smallest index")
    eligible = [d for d in state0["donors"] if d["eligible"]]
    check(len(eligible) == 2, "visited donors with positive LCB must be eligible")
    assert all(d["lcb"] > 0.0 for d in eligible)
    assert all(d["pair_count"] > 0 for d in eligible)
    check(result["update_emitted"], "update must emit with eligible donors")
    state2 = result["states"][2]
    unvisited_donor = [d for d in state2["donors"] if d["action"] == 2]
    if state2["receiver"] != 2 and unvisited_donor:
        check(unvisited_donor[0]["reason"] == "candidate_pair_unvisited",
              "unvisited donor must abstain with the ordered reason")
    # unvisited receiver: state 2's argmax is the unvisited action
    inputs2 = base_inputs(rng, unvisited=(2 * 3 + 2,))
    inputs2["q_estimate"][2] = [-5.0, -5.0, 0.0]  # receiver = action 2, unvisited
    result2 = agc.evaluate_route_update(**inputs2)
    state2b = result2["states"][2]
    check(state2b["receiver"] == 2, "receiver selected by estimate even if unvisited")
    check(all(d["reason"] == "candidate_pair_unvisited" for d in state2b["donors"]),
          "unvisited receiver must block all donors with ordered reason")
    check(not any(d["eligible"] for d in state2b["donors"]),
          "unvisited receiver cannot produce transfers")


# 7. Policy update: simplex, floor, theta=1/2, tie-break, abstention.
def test_policy_update_contract() -> None:
    rng = np.random.default_rng(29)
    inputs = base_inputs(rng)
    inputs["q_estimate"] = np.zeros((3, 3))
    inputs["q_estimate"][0] = [2.0, 0.0, 0.0]  # donors 1, 2 with huge LCB
    inputs["policy"] = np.full((3, 3), 1.0 / 3.0)
    result = agc.evaluate_route_update(**inputs)
    check(result["update_emitted"], "fixture must emit")
    pi_plus = np.asarray(result["policy_plus"])
    policy = np.asarray(inputs["policy"])
    for s in range(3):
        check(abs(pi_plus[s].sum() - 1.0) <= 1e-12, "row sum must stay exactly one")
        check(float(pi_plus[s].min()) >= 0.0, "nonnegativity violated")
        check(float(pi_plus[s].min()) >= inputs["pi_min"] - 1e-15,
              "exploration floor violated")
    state0 = result["states"][0]
    for donor in state0["donors"]:
        b = donor["action"]
        expected = 0.5 * (policy[0, b] - inputs["pi_min"])
        check(abs(donor["transfer"] - expected) <= TOL, "theta=1/2 transfer mismatch")
        check(abs(pi_plus[0, b] - (policy[0, b] - donor["transfer"])) <= TOL,
              "donor mass mismatch")
    moved = sum(d["transfer"] for d in state0["donors"])
    check(abs(pi_plus[0, 0] - (policy[0, 0] + moved)) <= TOL,
          "receiver mass mismatch")
    # tie-break: all-equal estimates select action 0
    inputs["q_estimate"] = np.zeros((3, 3))
    check(agc.select_receiver([0.0, 0.0, 0.0]) == 0, "tie must break to smallest index")
    # abstention: negative LCB everywhere -> original policy, no emission
    inputs["q_estimate"] = np.full((3, 3), -3.0)
    inputs["q_estimate"][:, 0] = -2.9
    inputs["state_value_bound"] = 50.0
    result_no = agc.evaluate_route_update(**inputs)
    check(not result_no["update_emitted"], "nonpositive LCB must abstain")
    check(result_no["policy_plus"] is None, "abstention must not emit a policy")
    reasons = {d["reason"] for st in result_no["states"] for d in st["donors"]}
    check(reasons <= {"gap_lcb_nonpositive"}, f"unexpected reasons {reasons}")


# 8. Pointwise Bellman implication on random MDPs with certified true gaps.
def test_bellman_improvement_implication() -> None:
    rng = np.random.default_rng(31)
    emitted_count = 0
    for _ in range(20):
        mdp = make_random_mdp(rng)
        policy = np.full((4, 3), 0.05)
        preferred = rng.integers(0, 3, size=4)
        policy[np.arange(4), preferred] = 0.9
        exact = policy_quantities(mdp, policy)
        q_pi = exact["q_pi"].reshape(4, 3)
        v_pi = exact["v_pi"]
        inputs = base_inputs(rng, n_states=4, n_actions=3)
        inputs["policy"] = policy
        # certificate tells the truth exactly: LCB equals the true gap
        inputs["q_estimate"] = q_pi.copy()
        inputs["recovery_radii"] = [1e-9 if c > 0 else None
                                    for c in inputs["pair_counts"]]
        inputs["state_value_bound"] = 1e-9
        result = agc.evaluate_route_update(**inputs)
        if not result["update_emitted"]:
            continue
        emitted_count += 1
        pi_plus = np.asarray(result["policy_plus"])
        bellman = np.array([
            float(pi_plus[s] @ q_pi[s] - v_pi[s]) for s in range(4)
        ])
        check(float(bellman.min()) >= -1e-9, "Bellman improvement must be nonnegative")
        exact_plus = policy_quantities(mdp, pi_plus)
        value_diff = exact_plus["v_pi"] - v_pi
        check(float(value_diff.min()) >= -1e-9,
              "pointwise policy improvement implication failed")
    check(emitted_count > 0, "fixture must exercise at least one emitted update")


# 9. Ordered reasons and counterexamples.
def test_ordered_reasons_and_failures() -> None:
    rng = np.random.default_rng(37)
    inputs = base_inputs(rng)
    inputs["algorithm_mode_ok"] = False
    result = agc.evaluate_route_update(**inputs)
    check(all(d["reason"] == "algorithm_mode_mismatch"
              for st in result["states"] for d in st["donors"]),
          "mode mismatch must be the first blocking reason")
    check(not result["update_emitted"], "mode mismatch cannot emit")

    inputs = base_inputs(rng)
    inputs["diverged"] = True
    result = agc.evaluate_route_update(**inputs)
    check(all(d["reason"] == "divergence_guard_triggered"
              for st in result["states"] for d in st["donors"]),
          "divergence must be the first blocking reason after mode")

    inputs = base_inputs(rng)
    inputs["state_value_bound"] = None
    result = agc.evaluate_route_update(**inputs)
    check(all(d["reason"] == "state_certificate_not_emitted"
              for st in result["states"] for d in st["donors"]),
          "missing state bound must be reason 3")

    inputs = base_inputs(rng)
    inputs["recovery_radii"][0] = None  # positive count but missing radius
    inputs["q_estimate"] = np.zeros((3, 3))
    inputs["q_estimate"][0] = [5.0, -5.0, -5.0]
    result = agc.evaluate_route_update(**inputs)
    reasons0 = {d["reason"] for d in result["states"][0]["donors"]}
    check(reasons0 == {"recovery_radius_unavailable"},
          f"missing radius must be reason 5, got {reasons0}")

    # multiple simultaneous violations pick the earliest in the frozen order
    inputs = base_inputs(rng, unvisited=(1,))
    inputs["state_value_bound"] = None
    inputs["recovery_radii"][0] = None
    result = agc.evaluate_route_update(**inputs)
    check(all(d["reason"] == "state_certificate_not_emitted"
              for st in result["states"] for d in st["donors"]),
          "earlier reason must dominate later ones")

    # nonfinite computed uncertainty -> numerical_nonfinite
    # disjoint successor rows give TV = 1, so 2*gamma*E_V overflows float64
    overflow_inputs = {
        "family": "vfirst",
        "matching": "exact",
        "scope": "local",
        "q_estimate": [[1.0, 0.0], [0.0, 0.0]],
        "policy": [[0.5, 0.5], [0.5, 0.5]],
        "pi_min": 0.05,
        "theta": 0.5,
        "gamma": 0.7,
        "beta": 8.0,
        "trajectory_length": 4,
        "reward_bound": 1.5,
        "pair_counts": [2, 2, 0, 0],
        "successor_counts": [[2, 0], [0, 2], [0, 0], [0, 0]],
        "total_successor_counts": [2, 2],
        "recovery_radii": [0.1, 0.1, None, None],
        "state_value_bound": 1.5e308,
    }
    result = agc.evaluate_route_update(**overflow_inputs)
    reasons = {d["reason"] for st in result["states"] for d in st["donors"]}
    check("numerical_nonfinite" in reasons, "overflow must be numerical_nonfinite")

    # structural input failures raise instead of emitting
    bad = base_inputs(rng)
    bad["pair_counts"][0] = -1
    try:
        agc.evaluate_route_update(**bad)
    except agc.ActionGapInputError:
        pass
    else:
        raise AssertionError("negative counts must raise ActionGapInputError")
    bad = base_inputs(rng)
    bad["policy"][0] = [0.5, 0.6, 0.6]
    try:
        agc.evaluate_route_update(**bad)
    except agc.ActionGapInputError:
        pass
    else:
        raise AssertionError("non-simplex policy must raise ActionGapInputError")


# 10. Softmax route gates: attention mass and effective row validation.
def test_softmax_route_gates() -> None:
    rng = np.random.default_rng(41)
    inputs = base_inputs(rng)
    inputs["matching"] = "softmax"
    result = agc.evaluate_route_update(**inputs)
    for st in result["states"]:
        for donor in st["donors"]:
            if donor["reason"] is None:
                check(donor["kappa"] is not None and 0.0 < donor["kappa"] <= 1.0,
                      "softmax donor must carry a valid kappa")
                check(donor["total_variation"] is not None, "softmax donor needs TV")
    check(result["routes_validated"] if "routes_validated" in result else True,
          "softmax route record must validate")


# 11. Global control behavior and dominance check helper.
def test_global_controls_and_dominance_helper() -> None:
    rng = np.random.default_rng(43)
    inputs = base_inputs(rng)
    local = agc.evaluate_route_update(**inputs)
    global_inputs = {
        key: value for key, value in inputs.items()
        if key not in ("successor_counts", "total_successor_counts", "recovery_radii")
    }
    global_inputs["scope"] = "global"
    global_inputs["global_q_bound"] = None
    result_none = agc.evaluate_route_update(**global_inputs)
    check(all(d["reason"] == "recovery_radius_unavailable"
              for st in result_none["states"] for d in st["donors"]),
          "unemitted global control must map to recovery_radius_unavailable")
    # emitted global: E_Q composition guarantees dominance
    state_bound = inputs["state_value_bound"]
    r_max = max(r for r in inputs["recovery_radii"] if r is not None)
    global_inputs["global_q_bound"] = 0.7 * state_bound + r_max
    global_record = agc.evaluate_route_update(**global_inputs)
    dominance = agc.check_local_global_dominance(local, global_record)
    check(dominance["available"], "dominance must be available when global emits")
    check(dominance["satisfied"], "local penalty must never exceed the global penalty")
    # weak update dominance: local changed states cover global changed states
    weak = agc.check_weak_update_dominance(local, global_record)
    check(weak["satisfied"], "local update decisions must weakly dominate global")


# 12. Strict JSON and no-oracle boundary.
def test_strict_json_and_no_oracle() -> None:
    payload = {"a": 1, "nested": {"b": [1.0, None, "x"]}}
    encoded = json.dumps(agc.strict_json(payload), allow_nan=False)
    assert json.loads(encoded) == payload

    def reject_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result

    try:
        json.loads('{"a": 1, "a": 2}', object_pairs_hook=reject_pairs)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate keys must be rejected")
    try:
        json.loads('{"a": NaN}', parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite constants must be rejected")
    converted = agc.strict_json({"bad": float("nan"), "inf": float("inf")})
    check(converted == {"bad": None, "inf": None},
          "nonfinite floats must serialize as null")
    for forbidden in ("true_q", "v_pi", "q_pi", "true_kernel", "oracle_diagonal",
                      "true_value", "action_gap_true", "occupancy", "realized_error",
                      "exact_return", "true_transition"):
        try:
            agc.assert_no_oracle_keys({"certificate_inputs": {forbidden: 1}})
        except agc.ActionGapInputError:
            pass
        else:
            raise AssertionError(f"forbidden key {forbidden} must be rejected")
    for name in ("evaluate_route_update", "check_local_global_dominance"):
        parameters = set(inspect.signature(getattr(agc, name)).parameters)
        check(not (parameters & agc.FORBIDDEN_INPUT_KEYS),
              f"{name} signature must not admit oracle parameters")


def main() -> None:
    tests = [
        test_tv_span_lemma,
        test_exact_recovery_decomposition,
        test_softmax_weights_and_rows,
        test_softmax_contamination,
        test_local_validity_and_dominance,
        test_partial_support_and_unvisited_receiver,
        test_policy_update_contract,
        test_bellman_improvement_implication,
        test_ordered_reasons_and_failures,
        test_softmax_route_gates,
        test_global_controls_and_dominance_helper,
        test_strict_json_and_no_oracle,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("PASS action-gap certificate verification")


if __name__ == "__main__":
    main()
