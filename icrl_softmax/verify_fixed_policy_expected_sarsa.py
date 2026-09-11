"""Formula-level verification for fixed-policy Expected SARSA (FP-ESARSA-001).

Deterministic fixture verification of the Claude main route. Covers:

1. canonical Q-memory uniqueness and duplicate rejection;
2. exact grouped attention equals synchronous batch Expected SARSA (1e-12),
   including self-loops, repeated visits, unvisited queries, and synchrony;
3. finite successor/current-read/writeback attention equals its direct
   finite-score formulas (1e-12) with no equality mask or visited gate;
4. exact-residual kernel fixed-point identity and the diagonal-margin
   contraction bound on direct matrices, in both premise regimes;
5. held-out residual certificate: independent high-precision mixture roots,
   empirical event validity at random counts, ordered non-emission reasons,
   filtration-leak counterexample, and strict JSON;
6. relative-softmax exact improvement, adversarial error-box corners, eta
   order, abstention bit-identity, and pointwise oracle value comparisons;
7. literal torch constructions equal the pure references (1e-12).

This script was written before its implementation module, as required by the
task plan; the initial import failure is recorded in the route failure
history.
"""

from __future__ import annotations

import inspect
import json
import math

import numpy as np
import torch
import mpmath as mp

from evaluate_fixed_policy_q_routes import make_mdp, make_policy
from mdps import rollout
from time_uniform_mixture_certificate import (
    build_mixture_grid,
    log_mixture,
    solve_mixture_boundary,
)
from verify_fixed_policy_q_routes import policy_quantities

import fixed_policy_expected_sarsa as es
from model import (
    EndToEndFiniteSoftmaxExpectedSARSA,
    EndToEndMaskedSoftmaxExpectedSARSA,
    FixedPolicyActionExpectation,
)


CHECKS = 0


def check(condition: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(label)


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return bool(abs(float(a) - float(b)) <= tol * (1.0 + max(abs(float(a)), abs(float(b)))))


def section_s1_canonical_memory() -> None:
    print("[S1] canonical Q-memory uniqueness")
    ids = es.canonical_pair_ids(6, 4)
    check(len(ids) == 24 and len(set(ids)) == 24, "canonical ids must be 24 unique pairs")
    check(ids[0] == (0, 0) and ids[-1] == (5, 3), "canonical id order must be row-major")
    for state in range(6):
        for action in range(4):
            check(
                es.canonical_pair_index(state, action, 4) == state * 4 + action,
                "pair index must be state * n_actions + action",
            )
    try:
        es.validate_canonical_memory([(0, 0), (1, 2), (0, 0), (2, 1)])
    except ValueError as error:
        check("duplicate" in str(error), "duplicate rejection must name the duplicate")
    else:
        check(False, "duplicate pair serialization must be rejected")
    accepted = es.validate_canonical_memory([(s, a) for s in range(2) for a in range(3)])
    check(accepted == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)], "unique memory must pass unchanged")
    # Equal numerical values under different identifiers stay distinct.
    q = np.zeros((2, 3))
    accepted_equal_values = es.validate_canonical_memory(
        [(0, 0), (0, 1)], q_values=q
    )
    check(accepted_equal_values == [(0, 0), (0, 1)], "equal values are not duplicates")


FIXTURE_POLICY = np.array([[0.6, 0.4], [0.3, 0.7]])
FIXTURE_STATES = np.array([0, 1, 0, 1])
FIXTURE_ACTIONS = np.array([0, 1, 0, 0])
FIXTURE_REWARDS = np.array([1.0, -0.5, 0.25, 0.75])
FIXTURE_NEXT_STATES = np.array([1, 1, 0, 0])
FIXTURE_NEXT_ACTIONS = np.array([1, 1, 0, 0])
# Hand-computed synchronous Expected SARSA layers (gamma=0.5, alpha=0.25, Q0=0).
EXPECTED_Q1 = np.array([[0.15625, 0.0], [0.1875, -0.125]])
EXPECTED_Q2 = np.array([[0.27734375, 0.0], [0.33984375, -0.22265625]])
EXPECTED_RESIDUALS_L1 = np.array([0.828125, -0.390625, 0.140625, 0.609375])
SAMPLED_Q2 = np.array([[0.275390625, 0.0], [0.34765625, -0.234375]])


def section_s2_exact_grouped() -> None:
    print("[S2] exact grouped Expected SARSA equals direct batch computation")
    result = es.run_expected_exact(
        q0=np.zeros((2, 2)),
        policy=FIXTURE_POLICY,
        states=FIXTURE_STATES,
        actions=FIXTURE_ACTIONS,
        rewards=FIXTURE_REWARDS,
        next_states=FIXTURE_NEXT_STATES,
        gamma=0.5,
        alpha=0.25,
        layers=2,
        value_bound=100.0,
    )
    check(np.allclose(result["layer_snapshots"][0], EXPECTED_Q1, atol=1e-15), "layer-1 Q mismatch")
    check(np.allclose(result["layer_snapshots"][1], EXPECTED_Q2, atol=1e-15), "layer-2 Q mismatch")
    check(
        np.allclose(result["residuals_by_layer"][1], EXPECTED_RESIDUALS_L1, atol=1e-15),
        "layer-1 residuals mismatch",
    )
    # Repeated visits averaged: pair (0,0) visited twice.
    counts = result["train_pair_counts"]
    check(counts[0] == 2 and counts[1] == 0 and counts[2] == 1 and counts[3] == 1, "pair counts wrong")
    # Unvisited exact query unchanged.
    check(result["layer_snapshots"][1][0, 1] == 0.0, "unvisited exact query must stay at its prior value")
    # Synchrony: a within-layer sequential update of pair (0,0) before the
    # residual of transition t=2 would change that residual.
    sequential_delta2 = 0.25 + 0.5 * (0.6 * EXPECTED_Q2[0, 0]) - EXPECTED_Q2[0, 0]
    check(not close(sequential_delta2, EXPECTED_RESIDUALS_L1[2]), "fixture must separate sequential updates")
    check(close(result["residuals_by_layer"][1][2], EXPECTED_RESIDUALS_L1[2]), "route must be synchronous")
    check(not result["diverged"], "bounded fixture must not trigger the guard")


def section_s3_self_loop() -> None:
    print("[S3] self-loop uses one Q entry in two algebraic branches")
    # transition t=1: S=1, A=1, S'=1 (self-loop); the successor expectation
    # includes pi(1|1) * Q(1,1), i.e. the same token the current head reads.
    q = EXPECTED_Q1
    direct = FIXTURE_REWARDS[1] + 0.5 * (
        FIXTURE_POLICY[1, 0] * q[1, 0] + FIXTURE_POLICY[1, 1] * q[1, 1]
    ) - q[1, 1]
    branch_form = FIXTURE_REWARDS[1] - (1.0 - 0.5 * FIXTURE_POLICY[1, 1]) * q[1, 1] + 0.5 * (
        FIXTURE_POLICY[1, 0] * q[1, 0]
    )
    check(close(direct, branch_form), "self-loop branch algebra inconsistent")
    result = es.run_expected_exact(
        q0=np.zeros((2, 2)),
        policy=FIXTURE_POLICY,
        states=FIXTURE_STATES,
        actions=FIXTURE_ACTIONS,
        rewards=FIXTURE_REWARDS,
        next_states=FIXTURE_NEXT_STATES,
        gamma=0.5,
        alpha=0.25,
        layers=2,
        value_bound=100.0,
    )
    check(close(result["residuals_by_layer"][1][1], direct), "self-loop residual mismatch")
    # The exact action-head candidate set for the self-loop query contains each
    # successor action token exactly once.
    candidate_ids = es.exact_action_head_candidates(next_state=1, n_actions=2)
    check(candidate_ids == [(1, 0), (1, 1)], "self-loop action head candidate set wrong")
    check(len(set(candidate_ids)) == 2, "self-loop must not duplicate the logical pair")


def section_s4_sampled_control() -> None:
    print("[S4] sampled SARSA control equals direct sampled computation")
    result = es.run_sampled_exact(
        q0=np.zeros((2, 2)),
        policy=FIXTURE_POLICY,
        states=FIXTURE_STATES,
        actions=FIXTURE_ACTIONS,
        rewards=FIXTURE_REWARDS,
        next_states=FIXTURE_NEXT_STATES,
        next_actions=FIXTURE_NEXT_ACTIONS,
        gamma=0.5,
        alpha=0.25,
        layers=2,
        value_bound=100.0,
    )
    check(np.allclose(result["layer_snapshots"][0], EXPECTED_Q1, atol=1e-15), "sampled layer-1 mismatch")
    check(np.allclose(result["layer_snapshots"][1], SAMPLED_Q2, atol=1e-15), "sampled layer-2 mismatch")
    check(
        not np.allclose(result["layer_snapshots"][1], EXPECTED_Q2, atol=1e-15),
        "fixture must separate sampled from expected",
    )


def section_s5_finite_formulas() -> None:
    print("[S5] finite successor/read/writeback equal direct finite-score formulas")
    q = np.array([[0.5, -0.25], [1.0, 0.75]])
    zeta = xi = tau = 2.0
    e2 = math.exp(2.0)
    row_means = (FIXTURE_POLICY * q).sum(axis=1)
    check(close(row_means[0], 0.2) and close(row_means[1], 0.825), "fixture row means wrong")
    expected_qbar0 = (e2 * 0.2 + 0.825) / (e2 + 1.0)
    expected_qbar1 = (e2 * 0.825 + 0.2) / (e2 + 1.0)
    qbar = es.finite_successor_all(q, FIXTURE_POLICY, zeta)
    check(close(qbar[0], expected_qbar0) and close(qbar[1], expected_qbar1), "finite successor formula wrong")
    total = q.sum()
    expected_read00 = (e2 * 0.5 + (total - 0.5)) / (e2 + 3.0)
    reads = es.finite_read_all(q, xi)
    check(close(reads.reshape(-1)[0], expected_read00), "finite read formula wrong")
    # kappa identities at the frozen sharpness 8 on the 6x4 board.
    e8 = math.exp(8.0)
    check(close(es.kappa_state(6, 8.0), e8 / (e8 + 5.0)), "kappa_state formula wrong")
    check(close(es.kappa_read(24, 8.0), e8 / (e8 + 23.0)), "kappa_read formula wrong")
    # Direct enumeration of the successor softmax at sharpness 8, 6x4 board.
    rng = np.random.default_rng(7)
    q_big = rng.normal(size=(6, 4))
    policy_big = make_policy(6, 4, 0.05, rng)
    state = 3
    scores = np.full((6, 4), 0.0)
    scores[state, :] = 8.0
    scores += np.log(policy_big)
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    direct = float((weights * q_big).sum())
    check(close(es.finite_successor_all(q_big, policy_big, 8.0)[state], direct), "successor softmax mismatch")
    check(close(weights[state, :].sum(), e8 / (e8 + 5.0)), "successor mass must equal kappa_state")
    conditional = weights[state, :] / weights[state, :].sum()
    check(np.allclose(conditional, policy_big[state], atol=1e-12), "conditional action weights must equal pi")
    # Writeback formula including an unvisited query.
    residuals = np.array([0.1, -0.2, 0.3, 0.05])
    current_pairs = np.array([0, 3, 0, 2])
    updates = es.finite_writeback_all(residuals, current_pairs, 4, tau, 0.5)
    sums_match = np.array([0.4, 0.0, 0.05, -0.2])
    counts = np.array([2, 0, 1, 1])
    denom = counts * e2 + (4 - counts)
    expected_updates = 0.5 * (e2 * sums_match + (residuals.sum() - sums_match)) / denom
    check(np.allclose(updates, expected_updates, atol=1e-15), "finite writeback formula wrong")
    check(updates[1] != 0.0, "unvisited finite query must receive a reported leakage update")
    # Full finite route layer-1 values on the exact fixture.
    result = es.run_expected_finite(
        q0=np.zeros((2, 2)),
        policy=FIXTURE_POLICY,
        states=FIXTURE_STATES,
        actions=FIXTURE_ACTIONS,
        rewards=FIXTURE_REWARDS,
        next_states=FIXTURE_NEXT_STATES,
        gamma=0.5,
        alpha=0.25,
        layers=1,
        value_bound=100.0,
        zeta=zeta,
        xi=xi,
        tau=tau,
    )
    q1 = result["layer_snapshots"][0]
    # Layer 0 residuals: qbar^fin(0)=read^fin=0 because Q0=0, so delta=R;
    # Q1(x) = alpha * (e^tau sum_match + (total - sum_match)) / denom.
    sums0 = np.array([1.25, 0.0, 0.75, -0.5])
    counts0 = np.array([2, 0, 1, 1])
    residual_total = float(FIXTURE_REWARDS.sum())  # layer-0 residuals equal the rewards
    denom0 = counts0 * e2 + (4 - counts0)
    expected_q1 = 0.25 * (e2 * sums0 + (residual_total - sums0)) / denom0
    check(np.allclose(q1.reshape(-1), expected_q1, atol=1e-15), "finite route layer-1 mismatch")
    check(q1[0, 1] != 0.0, "finite route must update the unvisited query (leakage reported)")
    check(not result["diverged"], "bounded fixture must not trigger the guard")


def section_s6_operator() -> None:
    print("[S6] exact-residual kernel fixed point and diagonal-margin contraction")
    rng = np.random.default_rng(20260911)
    d = 24
    gamma, alpha = 0.7, 0.65
    # Satisfied premise: M = 0.9 I + 0.1 W with W row-stochastic.
    w = rng.random((d, d))
    w /= w.sum(axis=1, keepdims=True)
    m_ok = 0.9 * np.eye(d) + 0.1 * w
    check(abs(m_ok.sum(axis=1) - 1.0).max() < 1e-12, "M must be row-stochastic")
    premise_c = float(np.diag(m_ok).min() - (1.0 + gamma) / 2.0)
    check(premise_c >= 0.05 - 1e-15, "fixture premise margin wrong")
    p = rng.random((d, d))
    p /= p.sum(axis=1, keepdims=True)
    r = rng.uniform(-1.5, 1.5, size=d)
    q_pi = np.linalg.solve(np.eye(d) - gamma * p, r)
    bellman = r + gamma * (p @ q_pi)
    check(np.max(np.abs(bellman - q_pi)) < 1e-10, "direct solve must be a fixed point")
    f_at_fixed = q_pi + alpha * (m_ok @ (bellman - q_pi))
    check(np.max(np.abs(f_at_fixed - q_pi)) < 1e-10, "F_pi must preserve Q^pi")
    linear = np.eye(d) - alpha * m_ok + alpha * gamma * m_ok @ p
    declared = 1.0 - 2.0 * alpha * premise_c
    check(declared < 1.0, "declared bound must be a contraction")
    check(np.abs(linear).sum(axis=1).max() <= declared + 1e-12, "operator norm exceeds declared bound")
    q_iter = rng.normal(size=d)
    previous = np.max(np.abs(q_iter - q_pi))
    for _ in range(10):
        q_iter = q_iter + alpha * (m_ok @ (r + gamma * (p @ q_iter) - q_iter))
        current = np.max(np.abs(q_iter - q_pi))
        check(current <= declared * previous + 1e-10, "iteration must contract at the declared rate")
        previous = current
    # Violated premise: uniform kernel, min diagonal 1/24 << (1+gamma)/2.
    m_bad = np.full((d, d), 1.0 / d)
    margin_bad = float(np.diag(m_bad).min() - (1.0 + gamma) / 2.0)
    check(margin_bad < 0.0, "uniform kernel must violate the premise")
    declared_bad = 1.0 - 2.0 * alpha * margin_bad
    check(declared_bad > 1.0, "violated premise must not yield a contraction bound")
    check(not es.contraction_premise_satisfied(m_bad, gamma), "premise check must reject the uniform kernel")
    check(es.contraction_premise_satisfied(m_ok, gamma), "premise check must accept the fixture kernel")


def section_s7_mixture_roots() -> None:
    print("[S7] independent high-precision mixture roots (d=24, delta=0.05)")
    mp.mp.dps = 50
    n_groups, delta = 24, 0.05
    grid = build_mixture_grid(n_groups=n_groups, delta=delta)
    target = math.log(n_groups / delta)
    # Local formula recomputation of the grid.
    normalizer = sum((index + 1) ** -2 for index in range(15))
    for index, row in enumerate(grid):
        weight = (index + 1) ** -2 / normalizer
        log_level = math.log(2.0 * n_groups / (delta * weight))
        rate = math.sqrt(2.0 * log_level / 2**index)
        check(close(float(row["weight"]), weight, 1e-15), "grid weight mismatch")
        check(close(float(row["rate"]), rate, 1e-15), "grid rate mismatch")
        check(int(row["target_count"]) == 2**index, "grid target count mismatch")
    rows_mp = [(mp.mpf(str(row["weight"])), mp.mpf(str(row["rate"]))) for row in grid]
    target_mp = mp.log(mp.mpf(24) / mp.mpf("0.05"))

    def log_m_mp(k: int, q: "mp.mpf") -> "mp.mpf":
        terms = [mp.log(w) - (a * a) * k / 2 + mp.log(mp.cosh(a * q)) for w, a in rows_mp]
        peak = max(terms)
        return peak + mp.log(sum(mp.exp(t - peak) for t in terms))

    for count in (1, 2, 3, 5, 17, 100, 1000, 8192):
        certified = solve_mixture_boundary(count, grid, n_groups=n_groups, delta=delta)
        root = mp.findroot(
            lambda q: log_m_mp(count, q) - target_mp,
            mp.mpf(str(certified["boundary"])),
        )
        boundary = mp.mpf(str(certified["boundary"]))
        slack = mp.mpf("1e-9") * (1 + root)
        check(boundary >= root - slack, f"certified boundary below the true root at k={count}")
        check(boundary <= root + 4 * slack, f"certified boundary too loose at k={count}")
        check(
            log_mixture(count, certified["boundary"], grid) >= target - 1e-9,
            f"certified boundary must remain conservative at k={count}",
        )
        check(certified["lower_boundary"] < certified["boundary"], "bracket must be nondegenerate")
    # Radius formula: r_x = 2B q_mix(N_x) / N_x with B = R_star/(1-gamma).
    boundary = solve_mixture_boundary(37, grid, n_groups=n_groups, delta=delta)
    radius = es.mixture_pair_radius(37, reward_bound=1.5, gamma=0.7, n_groups=24, delta=0.05)
    check(close(radius, 2.0 * 5.0 * float(boundary["boundary"]) / 37.0), "radius formula mismatch")


def _train_and_certify(rng: np.random.Generator, length: int) -> dict:
    mdp = make_mdp(6, 4, 0.7, 0.3, 0.5, rng)
    policy = make_policy(6, 4, 0.05, rng)
    exact = policy_quantities(mdp, policy)
    start = int(rng.choice(6, p=exact["mu_state"]))
    states, actions, rewards = rollout(mdp, policy, start=start, n=length, rng=rng)
    half = length // 2
    train = dict(
        states=states[:half],
        actions=actions[:half],
        rewards=rewards[1 : half + 1].astype(np.float64),
        next_states=states[1 : half + 1],
    )
    heldout = dict(
        states=states[half:length],
        actions=actions[half:length],
        rewards=rewards[half + 1 : length + 1].astype(np.float64),
        next_states=states[half + 1 : length + 1],
    )
    route = es.run_expected_exact(
        q0=np.zeros((6, 4)),
        policy=policy,
        gamma=0.7,
        alpha=0.65,
        layers=160,
        value_bound=5.0,
        **train,
    )
    certificate = es.build_residual_certificate(
        q_hat=route["q_hat"],
        policy=policy,
        gamma=0.7,
        reward_bound=1.5,
        delta=0.05,
        **heldout,
    )
    return {
        "mdp": mdp,
        "policy": policy,
        "exact": exact,
        "route": route,
        "certificate": certificate,
        "heldout": heldout,
    }


def section_s8_certificate_validity() -> None:
    print("[S8] certificate validity at random counts (Monte Carlo, fixed seed)")
    rng = np.random.default_rng(20260911)
    trials = 400
    emitted = 0
    q_violations = 0
    residual_violations = 0
    worst_ratio = 0.0
    for _ in range(trials):
        bundle = _train_and_certify(rng, 512)
        certificate = bundle["certificate"]
        if certificate["status"] != "certificate_emitted":
            check(
                "heldout_pair_support_missing" in certificate["failure_reasons"],
                "non-emission must come from the ordered reason list",
            )
            continue
        emitted += 1
        q_hat = bundle["route"]["q_hat"]
        true_q = np.asarray(bundle["exact"]["q_pi"], dtype=np.float64)
        oracle_error = float(np.max(np.abs(q_hat - true_q)))
        if oracle_error > float(certificate["e_q"]) + 1e-9:
            q_violations += 1
        residual_operator = (
            bundle["exact"]["reward_sa"].reshape(-1)
            + 0.7 * bundle["exact"]["p_pair"] @ q_hat.reshape(-1)
            - q_hat.reshape(-1)
        )
        means = np.asarray(certificate["residual_means"], dtype=np.float64)
        radii = np.asarray(certificate["radii"], dtype=np.float64)
        excess = np.abs(means - residual_operator) - radii
        if float(excess.max()) > 1e-9:
            residual_violations += 1
        worst_ratio = max(worst_ratio, oracle_error / float(certificate["e_q"]))
    check(emitted > 0, "at least one record must emit a certificate")
    check(
        residual_violations / trials <= 0.05,
        f"residual event violation rate {residual_violations}/{trials} exceeds delta",
    )
    check(
        q_violations / trials <= 0.05,
        f"Q certificate violation rate {q_violations}/{trials} exceeds delta",
    )
    print(f"  emitted={emitted}/{trials} residual_violations={residual_violations} "
          f"q_violations={q_violations} worst_error/E_Q={worst_ratio:.4f}")


def section_s9_certificate_reasons() -> None:
    print("[S9] ordered non-emission reasons and guard paths")
    policy = FIXTURE_POLICY
    heldout = dict(
        states=FIXTURE_STATES,
        actions=FIXTURE_ACTIONS,
        rewards=FIXTURE_REWARDS,
        next_states=FIXTURE_NEXT_STATES,
    )
    good_q = np.zeros((2, 2))
    # Missing held-out support (pair (0,1) never appears held out).
    certificate = es.build_residual_certificate(
        q_hat=good_q, policy=policy, gamma=0.5, reward_bound=1.5, delta=0.05, **heldout
    )
    check(certificate["status"] == "not_certified", "missing support must not certify")
    check(
        certificate["failure_reasons"] == ["heldout_pair_support_missing"],
        "missing support reason mismatch",
    )
    check(certificate["e_q"] is None, "non-emission must not report E_Q")
    # Divergence guard dominates support when both fail: frozen order puts the
    # guard first.
    big_q = np.full((2, 2), 5.01)
    certificate = es.build_residual_certificate(
        q_hat=big_q, policy=policy, gamma=0.5, reward_bound=1.5, delta=0.05, **heldout
    )
    check(
        certificate["failure_reasons"][:2]
        == ["divergence_guard_triggered", "heldout_pair_support_missing"],
        "frozen reason order must place the guard before support",
    )
    # Mode mismatch precedes every other reason.
    certificate = es.build_residual_certificate(
        q_hat=big_q,
        policy=policy,
        gamma=0.5,
        reward_bound=1.5,
        delta=0.05,
        algorithm_mode="off_policy_async",
        **heldout,
    )
    check(
        certificate["failure_reasons"][0] == "algorithm_mode_mismatch",
        "mode mismatch must be the first reason",
    )
    # Duplicate memory rejection.
    try:
        es.build_residual_certificate(
            q_hat=good_q,
            policy=policy,
            gamma=0.5,
            reward_bound=1.5,
            delta=0.05,
            pair_ids=[(0, 0), (1, 0), (0, 0), (1, 1)],
            **heldout,
        )
    except ValueError as error:
        check("duplicate" in str(error), "duplicate memory must raise")
    else:
        check(False, "duplicate memory must raise")
    # Full held-out support emits with the exact formula.
    full_states = np.array([0, 0, 1, 1])
    full_actions = np.array([0, 1, 0, 1])
    full_next = np.array([1, 0, 0, 1])
    full_rewards = np.array([0.5, -0.25, 1.0, -0.75])
    q_hat = np.array([[0.1, -0.2], [0.3, 0.05]])
    certificate = es.build_residual_certificate(
        q_hat=q_hat,
        policy=policy,
        states=full_states,
        actions=full_actions,
        rewards=full_rewards,
        next_states=full_next,
        gamma=0.5,
        reward_bound=1.5,
        delta=0.05,
    )
    check(certificate["status"] == "certificate_emitted", "full support must emit")
    # Manual recomputation of every field.
    n_groups, delta = 4, 0.05
    grid = build_mixture_grid(n_groups=n_groups, delta=delta)
    value_bound = 1.5 / 0.5
    qbar = (policy[full_next] * q_hat[full_next]).sum(axis=1)
    residuals = full_rewards + 0.5 * qbar - q_hat[full_states, full_actions]
    counts = np.bincount(full_states * 2 + full_actions, minlength=4)
    means = np.bincount(full_states * 2 + full_actions, weights=residuals, minlength=4) / counts
    radii = np.array([
        2.0 * value_bound
        * float(solve_mixture_boundary(int(c), grid, n_groups=n_groups, delta=delta)["boundary"])
        / int(c)
        for c in counts
    ])
    epsilon = float(np.max(np.abs(means) + radii))
    check(np.allclose(certificate["residual_means"], means, atol=1e-15), "residual means mismatch")
    check(np.allclose(certificate["radii"], radii, atol=1e-12), "radii mismatch")
    check(close(certificate["epsilon_res"], epsilon), "epsilon_res mismatch")
    check(close(certificate["e_q"], epsilon / 0.5), "E_Q mismatch")
    check(certificate["n_groups"] == 4, "group count must equal d")


def section_s10_filtration_leak() -> None:
    print("[S10] train/held-out leak counterexample")
    # One-state one-action chain, gamma=0, R = +/-1 with equal probability.
    # Split-respecting construction: Qhat from the training half only.
    rng = np.random.default_rng(5)
    trials = 2000
    split_means = []
    leaked_means = []
    for _ in range(trials):
        rewards = rng.choice([-1.0, 1.0], size=8)
        q_train = float(rewards[:4].mean())  # training-half-only estimate
        split_means.append(float(np.mean(rewards[4:] - 0.0)) - (0.0 - q_train))
        # Leaked construction: Qhat equals the FIRST held-out reward, i.e.
        # Qhat depends on held-out data; then Y_t = R_{t+1} - Qhat is centered
        # at the conditional mean of R, not at (T Qhat - Qhat) with Qhat fixed.
        q_leak = rewards[4]
        leaked_means.append(float(np.mean(rewards[4:] - q_leak)))
    # For the split-respecting estimate, the training-only Qhat is independent
    # of the held-out rewards, so the aggregated residual concentrates at zero.
    check(abs(float(np.mean(split_means))) < 0.08, "split-respecting martingale must be centered")
    # When Qhat is built from held-out data (q_leak = R_4) the residual is no
    # longer a martingale difference sequence: its per-trial held-out mean
    # carries a non-vanishing random offset of order |q_leak| = 1, so it fails
    # to concentrate. The spread across trials is the observable signature.
    leaked_offset = np.array(leaked_means)
    check(
        float(np.std(leaked_offset)) > 0.5,
        "leaked construction must visibly break the martingale centering",
    )
    # The certificate's signature admits no oracle or future data fields.
    parameters = set(inspect.signature(es.build_residual_certificate).parameters)
    allowed = {
        "q_hat", "policy", "states", "actions", "rewards", "next_states",
        "reward_bound", "gamma", "delta", "algorithm_mode", "pair_ids",
        "mixture_tolerance",
    }
    check(parameters <= allowed, f"certificate signature exceeds observable inputs: {parameters - allowed}")


def section_s11_policy_improvement() -> None:
    print("[S11] exact improvement, adversarial corners, emission safety, abstention")
    rng = np.random.default_rng(11)
    etas = es.ETA_GRID
    check(tuple(etas) == (1.0, 0.5, 0.2, 0.1, 0.05), "eta grid must be the frozen descending grid")
    # Exact improvement identity on random fixtures.
    for _ in range(200):
        mdp = make_mdp(6, 4, 0.7, 0.3, 0.5, rng)
        policy = make_policy(6, 4, 0.05, rng)
        exact = policy_quantities(mdp, policy)
        q_pi = np.asarray(exact["q_pi"], dtype=np.float64)
        for eta in etas:
            candidate = es.relative_softmax_candidate(policy, q_pi, eta)
            improvement = ((candidate - policy) * q_pi).sum(axis=1)
            check(np.all(improvement >= -1e-12), "exact tilt must not decrease any state")
            z = float(np.sum(policy * np.exp(eta * q_pi), axis=1)[0])
            kl_forward = float(np.sum(candidate * (eta * q_pi - math.log(z)), axis=1)[0])
            kl_reverse = float(np.sum(policy * (math.log(z) - eta * q_pi), axis=1)[0])
            check(
                close(improvement[0], (kl_forward + kl_reverse) / eta, 1e-9),
                "KL identity mismatch",
            )
    # Adversarial corners: LB_s valid against the worst box corner per state.
    for _ in range(100):
        policy = make_policy(6, 4, 0.05, rng)
        q_hat = rng.normal(size=(6, 4))
        e_q = float(rng.choice([0.05, 0.3, 1.0]))
        eta = float(rng.choice(etas))
        candidate = es.relative_softmax_candidate(policy, q_hat, eta)
        delta_pi = candidate - policy
        i_hat = (delta_pi * q_hat).sum(axis=1)
        lb = i_hat - e_q * np.abs(delta_pi).sum(axis=1)
        for state in range(6):
            signs = np.sign(delta_pi[state])
            signs[signs == 0.0] = 1.0
            worst_q = q_hat[state] - e_q * signs
            worst_i = float(np.dot(delta_pi[state], worst_q))
            check(worst_i >= lb[state] - 1e-12, "adversarial corner violates the lower bound")
            for mask in range(16):
                corner = np.array([(mask >> bit) & 1 for bit in range(4)], dtype=np.float64) * 2.0 - 1.0
                corner_q = q_hat[state] + e_q * corner
                check(
                    float(np.dot(delta_pi[state], corner_q)) >= lb[state] - 1e-12,
                    "error-box corner violates the lower bound",
                )
    # Emission safety with an oracle value comparison.
    emitted_cases = 0
    for _ in range(120):
        mdp = make_mdp(6, 4, 0.7, 0.3, 0.5, rng)
        policy = make_policy(6, 4, 0.05, rng)
        exact = policy_quantities(mdp, policy)
        q_pi = np.asarray(exact["q_pi"], dtype=np.float64)
        e_q = 1e-6  # near-exact estimate: certificate event holds by construction
        decision = es.decide_policy_update(policy, q_pi, e_q)
        if decision["status"] != "safe_update_emitted":
            check(
                decision["failure_reasons"][-1] in ("improvement_lcb_nonpositive", "policy_unchanged"),
                "non-emission must use the ordinary abstention reasons",
            )
            check(
                np.array_equal(decision["policy_plus"], policy)
                and decision["policy_plus"].tobytes() == policy.tobytes(),
                "abstention must return the policy bit-for-bit",
            )
            continue
        emitted_cases += 1
        new_policy = np.asarray(decision["policy_plus"], dtype=np.float64)
        check(np.all(new_policy > 0.0), "emitted policy must be strictly positive")
        check(np.max(np.abs(new_policy.sum(axis=1) - 1.0)) <= 1e-12, "rows must be normalized")
        check(not np.array_equal(new_policy, policy), "emitted policy must be changed")
        new_exact = policy_quantities(mdp, new_policy)
        old_v = np.asarray(exact["v_pi"], dtype=np.float64)
        new_v = np.asarray(new_exact["v_pi"], dtype=np.float64)
        check(np.all(new_v >= old_v - 1e-9), "emitted update must not decrease any state value")
        check(np.all(np.asarray(decision["lb_by_state"]) >= -1e-12), "emitted LB must be nonnegative")
        check(np.max(np.asarray(decision["lb_by_state"])) > 0.0, "emission needs a positive LB")
    check(emitted_cases > 0, "near-exact fixtures must emit at least once")
    # Eta order: construct a case where eta=1.0 fails but a smaller eta passes.
    for _ in range(2000):
        mdp = make_mdp(6, 4, 0.7, 0.3, 0.5, rng)
        policy = make_policy(6, 4, 0.05, rng)
        exact = policy_quantities(mdp, policy)
        q_pi = np.asarray(exact["q_pi"], dtype=np.float64)
        gap = float(np.max(q_pi) - np.min(q_pi))
        if gap < 1e-3:
            continue
        # An error budget of 0.1 * gap is large enough that the largest eta can
        # fail its lower bound while an intermediate eta still passes; a much
        # smaller budget lets eta=1.0 pass always and never exercises the order.
        e_q = 0.1 * gap
        decision = es.decide_policy_update(policy, q_pi, e_q)
        if decision["status"] == "safe_update_emitted" and decision["eta_selected"] < 1.0:
            etas_seen = [c["eta"] for c in decision["candidates"]]
            check(etas_seen == sorted(etas_seen, reverse=True), "candidates must be scanned descending")
            first_pass = next(
                c["eta"] for c in decision["candidates"] if c["passes_all_checks"]
            )
            check(first_pass == decision["eta_selected"], "emission must pick the first passing eta")
            larger = [c for c in decision["candidates"] if c["eta"] > decision["eta_selected"]]
            check(all(not c["passes_all_checks"] for c in larger), "larger etas must have failed")
            break
    else:
        check(False, "fixture search must find a smaller-eta emission")
    # Overflow resistance: extreme but finite Q.
    extreme_q = np.full((6, 4), 1e6)
    extreme_q[:, 0] = -1e6
    policy = make_policy(6, 4, 0.05, np.random.default_rng(3))
    decision = es.decide_policy_update(policy, extreme_q, 1.0)
    serialized = json.dumps(es.strict_json_ready(decision))
    check("NaN" not in serialized and "Infinity" not in serialized, "extreme inputs must stay finite")
    # Zero-Q row: candidate equals pi -> policy_unchanged path.
    zero_q = np.zeros((6, 4))
    decision = es.decide_policy_update(policy, zero_q, 0.0)
    check(decision["status"] == "abstained", "zero Q must abstain")
    check("policy_unchanged" in decision["failure_reasons"], "zero Q must record policy_unchanged")
    check(decision["policy_plus"].tobytes() == policy.tobytes(), "abstention must be bit-identical")
    # Invalid policy rejection.
    bad_policy = policy.copy()
    bad_policy[0, 0] = 0.0
    try:
        es.decide_policy_update(bad_policy, zero_q, 0.0)
    except ValueError:
        pass
    else:
        check(False, "zero-probability input policy must be rejected")
    bad_row = policy.copy()
    bad_row[1] *= 2.0
    try:
        es.decide_policy_update(bad_row, zero_q, 0.0)
    except ValueError:
        pass
    else:
        check(False, "misnormalized input policy must be rejected")


def section_s12_strict_json() -> None:
    print("[S12] strict JSON integrity")
    rng = np.random.default_rng(17)
    bundle = _train_and_certify(rng, 512)
    ready = es.strict_json_ready(bundle["certificate"])
    text = json.dumps(ready)

    def reject_duplicate(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError("nonfinite constant")

    json.loads(text, object_pairs_hook=reject_duplicate, parse_constant=reject_constant)
    poisoned = json.dumps({"value": float("nan")})
    try:
        json.loads(poisoned, parse_constant=reject_constant)
    except ValueError:
        pass
    else:
        check(False, "nonfinite JSON must be rejected")
    try:
        json.loads('{"a": 1, "a": 2}', object_pairs_hook=reject_duplicate)
    except ValueError:
        pass
    else:
        check(False, "duplicate keys must be rejected")
    decision = es.decide_policy_update(
        bundle["policy"],
        bundle["route"]["q_hat"],
        bundle["certificate"]["e_q"] if bundle["certificate"]["e_q"] is not None else 1.0,
    )
    json.dumps(es.strict_json_ready(decision))


def section_s13_literal_networks() -> None:
    print("[S13] literal torch constructions equal the pure references")
    dtype = torch.float64
    q0 = torch.zeros((2, 2), dtype=dtype)
    policy_t = torch.tensor(FIXTURE_POLICY, dtype=dtype)
    states_t = torch.tensor(FIXTURE_STATES)
    actions_t = torch.tensor(FIXTURE_ACTIONS)
    rewards_t = torch.tensor(FIXTURE_REWARDS, dtype=dtype)
    next_states_t = torch.tensor(FIXTURE_NEXT_STATES)

    # Exact action-expectation head.
    head = FixedPolicyActionExpectation()
    q1_t = torch.tensor(EXPECTED_Q1, dtype=dtype)
    qbar, action_attention = head(q1_t, next_states_t, policy_t)
    expected_qbar = np.array([
        FIXTURE_POLICY[int(ns)] @ EXPECTED_Q1[int(ns)] for ns in FIXTURE_NEXT_STATES
    ])
    check(
        torch.allclose(qbar, torch.tensor(expected_qbar, dtype=dtype), atol=1e-12),
        "action head output mismatch",
    )
    attention = action_attention.numpy()
    for t, ns in enumerate(FIXTURE_NEXT_STATES):
        for s in range(2):
            for a in range(2):
                pair = s * 2 + a
                expected_w = FIXTURE_POLICY[ns, a] if s == ns else 0.0
                check(close(attention[t, pair], expected_w), "action attention weight mismatch")

    # Exact end-to-end masked route.
    masked = EndToEndMaskedSoftmaxExpectedSARSA(gamma=0.5, alpha=0.25)
    q_new, diagnostics = masked(q0, states_t, actions_t, rewards_t, next_states_t, policy_t)
    check(
        torch.allclose(q_new, torch.tensor(EXPECTED_Q1, dtype=dtype), atol=1e-12),
        "masked route output mismatch",
    )
    residuals = diagnostics["residuals"].numpy()
    check(np.allclose(residuals, FIXTURE_REWARDS, atol=1e-12), "masked layer-0 residuals mismatch")
    write_attention = diagnostics["write_attention"].numpy()
    # Columns are queries (4 pairs); rows are 4 transitions + 1 null token.
    check(write_attention.shape == (5, 4), "write attention shape mismatch")
    for pair, expected_rows in ((0, {0, 2}), (2, {3}), (3, {1})):
        column = write_attention[:, pair]
        expected_w = 1.0 / len(expected_rows)
        for row in range(4):
            check(
                close(column[row], expected_w if row in expected_rows else 0.0),
                "exact write weight mismatch",
            )
        check(close(column[4], 0.0), "visited query must not touch the null token")
    check(close(write_attention[4, 1], 1.0), "unvisited query must attend only the null token")
    check(
        torch.allclose(
            diagnostics["update"].reshape(-1)[1],
            torch.zeros((), dtype=dtype),
            atol=1e-12,
        ),
        "unvisited exact query must not move",
    )

    # Finite end-to-end route.
    finite = EndToEndFiniteSoftmaxExpectedSARSA(gamma=0.5, alpha=0.25, zeta=2.0, xi=2.0, tau=2.0)
    q_new_f, diagnostics_f = finite(q0, states_t, actions_t, rewards_t, next_states_t, policy_t)
    reference = es.run_expected_finite(
        q0=np.zeros((2, 2)),
        policy=FIXTURE_POLICY,
        states=FIXTURE_STATES,
        actions=FIXTURE_ACTIONS,
        rewards=FIXTURE_REWARDS,
        next_states=FIXTURE_NEXT_STATES,
        gamma=0.5,
        alpha=0.25,
        layers=1,
        value_bound=100.0,
        zeta=2.0,
        xi=2.0,
        tau=2.0,
    )
    check(
        torch.allclose(
            q_new_f, torch.tensor(reference["layer_snapshots"][0], dtype=dtype), atol=1e-12
        ),
        "finite route output mismatch vs reference",
    )
    # No equality mask, no visited gate: every attention entry strictly positive.
    for name in ("action_attention", "read_attention", "write_attention"):
        matrix = diagnostics_f[name].numpy()
        check(np.all(np.isfinite(matrix)), f"{name} must be finite")
        check(np.all(matrix > 0.0), f"{name} must have full support (no mask/gate)")
        check(
            np.allclose(matrix.sum(axis=0 if name == "write_attention" else 1), 1.0, atol=1e-12),
            f"{name} must be row/column normalized",
        )
    check(
        float(diagnostics_f["update"].reshape(-1)[1]) != 0.0,
        "finite unvisited query must receive a reported leakage update",
    )
    # Sharpness-8 finite network equals the direct formula on a 6x4 board.
    rng = np.random.default_rng(23)
    q_big = rng.normal(size=(6, 4))
    policy_big = make_policy(6, 4, 0.05, rng)
    states_b = rng.integers(0, 6, size=40)
    actions_b = rng.integers(0, 4, size=40)
    rewards_b = rng.uniform(-1.5, 1.5, size=40)
    next_b = rng.integers(0, 6, size=40)
    finite8 = EndToEndFiniteSoftmaxExpectedSARSA(gamma=0.7, alpha=0.65, zeta=8.0, xi=8.0, tau=8.0)
    q_out8, _ = finite8(
        torch.tensor(q_big, dtype=dtype),
        torch.tensor(states_b),
        torch.tensor(actions_b),
        torch.tensor(rewards_b, dtype=dtype),
        torch.tensor(next_b),
        torch.tensor(policy_big, dtype=dtype),
    )
    reference8 = es.run_expected_finite(
        q0=q_big,
        policy=policy_big,
        states=states_b,
        actions=actions_b,
        rewards=rewards_b,
        next_states=next_b,
        gamma=0.7,
        alpha=0.65,
        layers=1,
        value_bound=100.0,
        zeta=8.0,
        xi=8.0,
        tau=8.0,
    )
    check(
        torch.allclose(
            q_out8, torch.tensor(reference8["layer_snapshots"][0], dtype=dtype), atol=1e-12
        ),
        "sharpness-8 finite network mismatch",
    )
    # Masked network at 6x4 equals direct batch Expected SARSA.
    masked8 = EndToEndMaskedSoftmaxExpectedSARSA(gamma=0.7, alpha=0.65)
    q_out_m, _ = masked8(
        torch.tensor(q_big, dtype=dtype),
        torch.tensor(states_b),
        torch.tensor(actions_b),
        torch.tensor(rewards_b, dtype=dtype),
        torch.tensor(next_b),
        torch.tensor(policy_big, dtype=dtype),
    )
    reference_m = es.run_expected_exact(
        q0=q_big,
        policy=policy_big,
        states=states_b,
        actions=actions_b,
        rewards=rewards_b,
        next_states=next_b,
        gamma=0.7,
        alpha=0.65,
        layers=1,
        value_bound=100.0,
    )
    check(
        torch.allclose(
            q_out_m, torch.tensor(reference_m["layer_snapshots"][0], dtype=dtype), atol=1e-12
        ),
        "masked network mismatch vs direct batch Expected SARSA",
    )


def section_s14_source_boundary() -> None:
    print("[S14] source boundary of certificate and finite route")
    certificate_source = inspect.getsource(es.build_residual_certificate)
    for forbidden in ("q_pi", "true_", "v_pi", "p_pair", "oracle"):
        check(forbidden not in certificate_source, f"certificate source mentions {forbidden}")
    finite_source = inspect.getsource(EndToEndFiniteSoftmaxExpectedSARSA)
    for forbidden in ('float("-inf")', "masked_fill", "unique(", "-inf"):
        check(forbidden not in finite_source, f"finite route uses a forbidden mask/gate: {forbidden}")
    decision_source = inspect.getsource(es.decide_policy_update)
    for forbidden in ("q_pi", "true_", "v_pi", "oracle"):
        check(forbidden not in decision_source, f"decision source mentions {forbidden}")


def main() -> None:
    section_s1_canonical_memory()
    section_s2_exact_grouped()
    section_s3_self_loop()
    section_s4_sampled_control()
    section_s5_finite_formulas()
    section_s6_operator()
    section_s7_mixture_roots()
    section_s8_certificate_validity()
    section_s9_certificate_reasons()
    section_s10_filtration_leak()
    section_s11_policy_improvement()
    section_s12_strict_json()
    section_s13_literal_networks()
    section_s14_source_boundary()
    print(f"PASS fixed-policy Expected SARSA verification ({CHECKS} checks)")


if __name__ == "__main__":
    main()
