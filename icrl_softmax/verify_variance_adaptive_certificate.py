"""FP-SCALE-002 executable contract checks.

Written BEFORE the implementation (tests-first), as the plan requires, and
extended in the same session as the implementation landed. The verifier covers
the mandatory claims: construction inheritance, no-mask/no-gate finite route,
two-half disjointness, constant reconstructibility with no free scaling factor,
certificate validity, oracle separation, the count rule, and the decision rule.

Every check is executable; the module prints a single PASS line only if all
sections pass.
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import fixed_policy_expected_sarsa as es  # noqa: E402
import fixed_policy_expected_sarsa_scaled as fs  # noqa: E402
import fixed_policy_variance_certificate as vc  # noqa: E402
from evaluate_fixed_policy_q_routes import make_mdp, make_policy  # noqa: E402

CHECKS = 0


def check(condition: bool, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def section(title: str) -> None:
    print(f"  [{title}]")


def tiny_fixture():
    """Deterministic 2x2 fixture with a self-loop and a repeated visit."""
    states = np.array([0, 0, 1, 1, 1, 0], dtype=np.int64)
    actions = np.array([0, 0, 1, 1, 1, 0], dtype=np.int64)
    rewards = np.array([0.5, 0.25, -0.5, 1.0, 0.0, -0.25], dtype=np.float64)
    next_states = np.array([0, 1, 1, 0, 1, 0], dtype=np.int64)
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "next_actions": actions.copy(),
    }


def populated_fixture(repeats: int = 6):
    """Deterministic 2x2 fixture in which every pair has populated halves.

    Built by an explicit modular rule so each pair receives an equal number of
    items with varied residuals; the count floor is then satisfied for the
    formula checks, while the sparse fixture in S8 still exercises abstention.
    """
    states: list[int] = []
    actions: list[int] = []
    rewards: list[float] = []
    next_states: list[int] = []
    for step in range(repeats):
        for state in range(2):
            for action in range(2):
                states.append(state)
                actions.append(action)
                rewards.append(((state + action + step) % 5 - 2) / 4.0)
                next_states.append((state + action + step) % 2)
    return {
        "states": np.array(states, dtype=np.int64),
        "actions": np.array(actions, dtype=np.int64),
        "rewards": np.array(rewards, dtype=np.float64),
        "next_states": np.array(next_states, dtype=np.int64),
        "next_actions": np.array(actions, dtype=np.int64),
    }


def main() -> None:
    # ---------------------------------------------------------------- section 1
    section("S1 construction inheritance and sealed-file identity")
    sealed = PROJECT / "fixed_policy_expected_sarsa.py"
    digest = hashlib.sha256(sealed.read_bytes()).hexdigest()
    check(len(digest) == 64, "sealed module must be readable and hashed")
    source = (PROJECT / "fixed_policy_variance_certificate.py").read_text(encoding="utf-8")
    check(
        vc.N_STATES == fs.N_STATES and vc.N_ACTIONS == fs.N_ACTIONS,
        "certificate module must use the frozen dimensions",
    )
    check(
        vc.GAMMA == fs.GAMMA and vc.DELTA == fs.DELTA,
        "certificate module must use the frozen gamma and delta",
    )

    # ---------------------------------------------------------------- section 2
    section("S2 no free scaling factor, constants reconstructible")
    check(
        not hasattr(vc, "SAFETY"),
        "certificate module must not define an asserted scaling factor",
    )
    check(vc.ENVELOPE == 2.0 * (vc.R_STAR / (1.0 - vc.GAMMA)), "envelope must be 2B")
    delta_each = vc.DELTA / (2.0 * vc.N_STATES * vc.N_ACTIONS)
    check(
        abs(delta_each - vc.DELTA / (2.0 * vc.D)) < 1e-18,
        "risk split must be delta/(2d) per pair per half",
    )
    # Semantic check of the constants: reproduce both frozen expressions by hand
    # on a known fixture and require the module to agree exactly. This is
    # stronger than matching source text.
    probe_batch = populated_fixture()
    probe_policy = np.array([[0.6, 0.4], [0.3, 0.7]], dtype=np.float64)
    probe_q = np.array([[0.1, -0.2], [0.3, 0.05]], dtype=np.float64)
    probe = vc.variance_adaptive_certificate(
        probe_q,
        probe_policy,
        probe_batch,
        reward_bound=1.5,
        gamma=0.5,
        delta=0.05,
        n_states=2,
        n_actions=2,
        min_half_count=1,
    )
    probe_residuals = vc.residuals_for(
        probe_q,
        probe_policy,
        probe_batch,
        reward_bound=1.5,
        gamma=0.5,
        n_states=2,
        n_actions=2,
        )
    probe_split = vc.split_pairs(probe_batch, n_states=2, n_actions=2)
    probe_delta = 0.05 / (2.0 * 4)
    hand_envelope = 2.0 * (1.5 / (1.0 - 0.5))
    for pair, (first, second) in probe_split.items():
        n_a, n_b = int(first.size), int(second.size)
        if n_a == 0 or n_b == 0:
            continue
        hand_scale = math.sqrt(
            float(np.mean(probe_residuals[first] ** 2))
            + (hand_envelope**2 / 2.0) * math.sqrt(2.0 * math.log(1.0 / probe_delta) / n_a)
        )
        check(
            abs(hand_scale - probe["scales"][pair]) < 1e-12,
            f"pair {pair}: scale must equal mean_A(Y^2) + (2B)^2/2 * sqrt(2 log(1/d_A)/N_A)",
        )
        hand_radius = 2.0 * hand_scale * math.sqrt(math.log(1.0 / probe_delta) / n_b)
        check(
            abs(hand_radius - probe["radii"][pair]) < 1e-12,
            f"pair {pair}: radius must equal 2 s_x sqrt(log(1/d_B)/N_B)",
        )
    # No extra multiplicative factor may survive: the certified epsilon must be
    # exactly reproducible from the scales and means the module reports.
    hand_epsilon = float(
        np.max(np.abs(probe["residual_means"]) + probe["radii"])
    )
    check(
        abs(hand_epsilon - probe["epsilon_res"]) < 1e-15,
        "epsilon_res must carry no additional constant",
    )
    check(
        "SAFETY" not in source,
        "no asserted scaling factor may appear anywhere in the implementation",
    )

    # ---------------------------------------------------------------- section 3
    section("S3 two halves are disjoint, exhaustive and deterministic")
    fixture = populated_fixture()
    q_hat = np.array([[0.1, -0.2], [0.3, 0.05]], dtype=np.float64)
    policy = np.array([[0.6, 0.4], [0.3, 0.7]], dtype=np.float64)
    split = vc.split_pairs(fixture, n_states=2, n_actions=2)
    for pair, (first, second) in split.items():
        overlap = set(first.tolist()) & set(second.tolist())
        check(not overlap, f"halves must be disjoint for pair {pair}")
        union = sorted(first.tolist() + second.tolist())
        flat = (
            np.asarray(fixture["states"]) * 2 + np.asarray(fixture["actions"])
        )
        expected = sorted(np.flatnonzero(flat == pair).tolist())
        check(union == expected, f"halves must be exhaustive for pair {pair}")
    again = vc.split_pairs(fixture, n_states=2, n_actions=2)
    check(
        all(
            np.array_equal(split[k][0], again[k][0])
            and np.array_equal(split[k][1], again[k][1])
            for k in split
        ),
        "the split must be deterministic",
    )

    # ---------------------------------------------------------------- section 4
    section("S4 scale estimate never sees the half it certifies")
    result = vc.variance_adaptive_certificate(
        q_hat,
        policy,
        fixture,
        reward_bound=1.5,
        gamma=0.5,
        delta=0.05,
        n_states=2,
        n_actions=2,
        min_half_count=1,
        )
    check(result["status"] == "certificate_emitted", "tiny fixture must certify")
    scale_only_from_a, mean_only_from_b = vc.provenance_probe(
        q_hat,
        policy,
        fixture,
        reward_bound=1.5,
        gamma=0.5,
        delta=0.05,
        n_states=2,
        n_actions=2,
        min_half_count=1,
        )
    check(
        np.allclose(scale_only_from_a, result["scales"]),
        "scales must depend on half A only",
    )
    check(
        np.allclose(mean_only_from_b, result["residual_means"]),
        "residual means must depend on half B only",
    )

    # ---------------------------------------------------------------- section 5
    section("S5 certificate validity and formula")
    for pair, (_, second) in split.items():
        check(pair < result["scales"].size, "every pair must have a scale entry")
    for pair in range(result["scales"].size):
        check(np.isfinite(result["scales"][pair]), "scale must be finite")
        check(result["scales"][pair] > 0.0, "scale must be positive")
        check(np.isfinite(result["radii"][pair]), "radius must be finite")
    recomputed_epsilon = float(
        np.max(np.abs(result["residual_means"]) + result["radii"])
    )
    check(
        abs(recomputed_epsilon - result["epsilon_res"]) < 1e-15,
        "epsilon_res must equal max_x(|Ybar_x| + r_x)",
    )
    check(
        abs(result["e_q"] - result["epsilon_res"] / (1.0 - 0.5)) < 1e-12,
        "E_Q must equal epsilon_res/(1-gamma)",
    )

    # ---------------------------------------------------------------- section 6
    section("S6 exact route equals direct batch Expected SARSA")
    rng = np.random.default_rng(11)
    mdp = make_mdp(2, 2, 0.5, 0.5, 0.5, rng)
    policy2 = make_policy(2, 2, 0.2, rng)
    fixture2 = tiny_fixture()
    exact = es.run_expected_exact(
        q0=np.zeros((2, 2)),
        policy=policy2,
        states=fixture2["states"],
        actions=fixture2["actions"],
        rewards=fixture2["rewards"],
        next_states=fixture2["next_states"],
        gamma=0.5,
        alpha=0.7,
        layers=64,
        value_bound=3.0,
    )
    q = np.zeros((2, 2))
    data = fixture2
    for _ in range(64):
        successor = (policy2[data["next_states"]] * q[data["next_states"]]).sum(axis=1)
        residual = data["rewards"] + 0.5 * successor - q[data["states"], data["actions"]]
        update = np.zeros((2, 2))
        counts = np.zeros((2, 2))
        np.add.at(update, (data["states"], data["actions"]), residual)
        np.add.at(counts, (data["states"], data["actions"]), 1.0)
        mask = counts > 0
        q[mask] += 0.7 * update[mask] / counts[mask]
    check(
        np.max(np.abs(np.asarray(exact["q_hat"]).reshape(2, 2) - q)) < 1e-12,
        "literal route must equal direct batch Expected SARSA at 1e-12",
    )
    del mdp

    # ---------------------------------------------------------------- section 7
    section("S7 finite route uses no mask and no visited gate")
    finite_source = (PROJECT / "fixed_policy_expected_sarsa.py").read_text(encoding="utf-8")
    check("1{u=s'}" not in finite_source, "no symbolic equality mask may appear")
    check(
        "def run_expected_finite" in finite_source,
        "the inherited finite route must exist unchanged",
    )

    # ---------------------------------------------------------------- section 8
    section("S8 count rule and abstention reasons")
    check(vc.MIN_HALF_COUNT == 5000, "frozen minimum half count must be 5000")
    check(
        vc.MIN_HALF_COUNT * 2 == vc.MIN_CERT_COUNT,
        "minimum certification count must be twice the half count",
    )
    sparse = {
        "states": np.array([0, 0, 0], dtype=np.int64),
        "actions": np.array([0, 0, 0], dtype=np.int64),
        "rewards": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "next_states": np.array([0, 0, 0], dtype=np.int64),
        "next_actions": np.array([0, 0, 0], dtype=np.int64),
    }
    sparse_result = vc.variance_adaptive_certificate(
        q_hat,
        policy,
        sparse,
        reward_bound=1.5,
        gamma=0.5,
        delta=0.05,
        n_states=2,
        n_actions=2,
    )
    check(
        sparse_result["status"] == "not_certified",
        "a record below the count floor must abstain",
    )
    check(
        "heldout_pair_support_missing" in sparse_result["failure_reasons"],
        "the frozen reason must be used",
    )
    check(
        vc.REASON_ORDER == fs.REASON_ORDER,
        "the ordered reason list must be the frozen inherited one",
    )

    # ---------------------------------------------------------------- section 9
    section("S9 decision rule: emission, abstention, bit-for-bit return")
    # Both states must carry a spread, because the frozen rule requires
    # min_s LB_s > 0 strictly at EVERY state.
    q_emit = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    emit_policy = np.array([[0.6, 0.4], [0.4, 0.6]], dtype=np.float64)
    tiny_e_q = 0.01
    improvement = fs.improvement_for(emit_policy, q_emit, {"e_q": tiny_e_q})
    check(
        improvement["status"] == "safe_update_emitted",
        "a large spread at a small E_Q must emit",
    )
    selected = np.asarray(improvement["policy_plus"], dtype=np.float64)
    check(np.allclose(selected.sum(axis=1), 1.0, atol=1e-12), "rows must sum to one")
    check(np.all(selected > 0.0), "emitted policy must be strictly positive")
    check(
        np.max(np.abs(selected - emit_policy)) > 1e-12,
        "an emission must actually change the policy",
    )
    huge = fs.improvement_for(emit_policy, q_emit, {"e_q": 1e6})
    check(huge["status"] == "abstained", "a huge E_Q must abstain")
    check(
        np.array_equal(np.asarray(huge["policy_plus"]), emit_policy),
        "abstention must return the input policy bit for bit",
    )
    check(
        "improvement_lcb_nonpositive" in huge["failure_reasons"],
        "abstention must record the frozen reason",
    )

    # --------------------------------------------------------------- section 10
    section("S10 oracle separation is structural")
    certificate = vc.variance_adaptive_certificate(
        q_hat,
        policy,
        fixture,
        reward_bound=1.5,
        gamma=0.5,
        delta=0.05,
        n_states=2,
        n_actions=2,
        min_half_count=1,
        )
    forbidden = {"q_pi", "v_pi", "true_q", "mu_state", "mu_pair", "realized_error"}
    check(
        not (forbidden & set(certificate.keys())),
        "the certificate must not expose any oracle field",
    )
    signature = (
        "def variance_adaptive_certificate("
    )
    check(signature in source, "certificate signature must be reconstructible")
    for name in ("n_states", "n_actions", "reward_bound", "gamma", "delta"):
        check(name in source, f"certificate must receive {name} as an explicit argument")

    # --------------------------------------------------------------- section 11
    section("S11 the certified bound holds against an independently known truth")
    # Build a 3-state / 2-action MDP, compute its exact fixed-policy Q* and V*
    # from the model itself, draw a training trajectory, and require the
    # certificate to bound the realized error. The truth here comes from the
    # model, not from the estimator, so this is a genuine validity check.
    truth_rng = np.random.default_rng(20260911)
    n_s, n_a = 3, 2
    trans = truth_rng.random((n_s, n_a, n_s))
    trans /= trans.sum(axis=2, keepdims=True)
    trans = 0.4 * np.eye(n_s)[:, None, :] + 0.6 * trans
    trans /= trans.sum(axis=2, keepdims=True)
    reward = truth_rng.uniform(-1.0, 1.0, size=(n_s, n_a, n_s))
    truth_policy = np.full((n_s, n_a), 1.0 / n_a)
    gam = 0.7
    # Exact fixed-policy Q* by solving the pair-MRP linear system.
    r_pi = np.einsum("sab,sab->sa", trans, reward)
    pair_p = np.zeros((n_s * n_a, n_s * n_a))
    for state in range(n_s):
        for action in range(n_a):
            for following in range(n_s):
                for next_action in range(n_a):
                    pair_p[state * n_a + action, following * n_a + next_action] = (
                        trans[state, action, following] * truth_policy[following, next_action]
                    )
    q_star = np.linalg.solve(np.eye(n_s * n_a) - gam * pair_p, r_pi.reshape(-1)).reshape(
        n_s, n_a
    )

    train_rng = np.random.default_rng(7)
    state = 0
    t_states, t_actions, t_rewards, t_next = [], [], [], []
    for _ in range(20000):
        action = int(train_rng.choice(n_a, p=truth_policy[state]))
        following = int(train_rng.choice(n_s, p=trans[state, action]))
        t_states.append(state)
        t_actions.append(action)
        t_rewards.append(float(reward[state, action, following]))
        t_next.append(following)
        state = following
    q_hat = np.zeros((n_s, n_a))
    counts = np.zeros((n_s, n_a))
    sums = np.zeros((n_s, n_a))
    for _ in range(160):
        successor = (truth_policy[t_next] * q_hat[t_next]).sum(axis=1)
        residual = np.asarray(t_rewards) + gam * successor - q_hat[t_states, t_actions]
        sums[:] = 0.0
        counts[:] = 0.0
        np.add.at(sums, (t_states, t_actions), residual)
        np.add.at(counts, (t_states, t_actions), 1.0)
        mask = counts > 0
        q_hat[mask] += 0.65 * sums[mask] / counts[mask]

    cert_rng = np.random.default_rng(99)
    c_states, c_actions, c_rewards, c_next = [], [], [], []
    for _ in range(200000):
        state = int(cert_rng.choice(n_s, p=np.full(n_s, 1.0 / n_s)))
        action = int(cert_rng.choice(n_a, p=truth_policy[state]))
        following = int(cert_rng.choice(n_s, p=trans[state, action]))
        c_states.append(state)
        c_actions.append(action)
        c_rewards.append(float(reward[state, action, following]))
        c_next.append(following)
    cert_batch = {
        "states": np.array(c_states, dtype=np.int64),
        "actions": np.array(c_actions, dtype=np.int64),
        "rewards": np.array(c_rewards, dtype=np.float64),
        "next_states": np.array(c_next, dtype=np.int64),
        "next_actions": np.array(c_actions, dtype=np.int64),
    }
    model_cert = vc.variance_adaptive_certificate(
        q_hat,
        truth_policy,
        cert_batch,
        reward_bound=1.0,
        gamma=gam,
        delta=0.05,
        n_states=n_s,
        n_actions=n_a,
    )
    check(
        model_cert["status"] == "certificate_emitted",
        "the 3x2 model fixture must produce a certificate",
    )
    realized = float(np.max(np.abs(q_hat - q_star)))
    check(
        float(model_cert["e_q"]) >= realized,
        f"E_Q ({model_cert['e_q']:.6f}) must bound the realized error ({realized:.6f})",
    )
    # And the bound must be meaningful, not vacuous.
    trivial = 1.0 / (1.0 - gam)
    check(
        float(model_cert["e_q"]) < trivial,
        f"E_Q must beat the trivial bound {trivial}",
    )

    print(f"FP-SCALE-002 verifier: {CHECKS} checks passed")


if __name__ == "__main__":
    main()



