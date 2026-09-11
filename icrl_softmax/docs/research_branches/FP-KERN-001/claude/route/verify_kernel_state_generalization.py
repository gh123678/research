"""Deterministic contract verification for FP-KERN-001 kernel generalization.

This verifier pins the frozen task contract before any formal output exists:

1. the primary leave-one-action-out route never reads the target action's
   signature-trajectory coordinates;
2. two common observed non-target actions gate every cross-state comparison;
3. distances, sorted-float64 median bandwidths, Gaussian weights, and
   observation-weighted effective sample sizes match hand computation;
4. zero-count target recovery uses only other-state target observations, and
   the self-only case reduces exactly to the unpooled pair mean;
5. the same-action anchor control abstains at zero target count or missing
   target-action signature and never claims zero-count coverage;
6. every malformed input and ordinary abstention follows the frozen ordered
   reason contract with no silent fallback;
7. state-label permutation leaves estimates equivariant;
8. the aggregate V-first replica matches the unchanged iterative estimator;
9. the hidden-cluster generator is deterministic, balanced, simplex-valid,
   reward-bounded, sticky-mixed, and structurally separated from observables;
10. oracle or unknown input keys are rejected at the pure boundary.
"""

from __future__ import annotations

import numpy as np

from kernel_state_generalization import (
    REASON_BANDWIDTH,
    REASON_COUNT,
    REASON_DENOMINATOR,
    REASON_ESTIMATE,
    REASON_ORDER,
    REASON_POLICY,
    REASON_SHAPE,
    REASON_SUPPORT,
    REASON_TARGET,
    REASON_VALUE,
    ROUTES,
    KernelInputError,
    evaluate_all_routes,
    greedy_with_floor_policy,
    validate_observable_inputs,
    vfirst_value_from_aggregates,
)


GAMMA = 0.70
REWARD_BOUND = 1.0
VALUE_SCALE = REWARD_BOUND / (1.0 - GAMMA)
N_STATES = 3
N_ACTIONS = 4


def build_inputs(
    sig_q: np.ndarray,
    sig_counts: np.ndarray,
    tgt_sums: np.ndarray,
    tgt_counts: np.ndarray,
    value: np.ndarray,
    policy: np.ndarray | None = None,
) -> dict:
    sig_q = np.asarray(sig_q, dtype=np.float64)
    sig_counts = np.asarray(sig_counts, dtype=np.int64)
    sig_sums = np.where(sig_counts > 0, sig_q * sig_counts, 0.0)
    if policy is None:
        policy = np.full((N_STATES, N_ACTIONS), 0.05, dtype=np.float64)
        policy[:, 0] = 1.0 - (N_ACTIONS - 1) * 0.05
    return {
        "n_states": int(sig_counts.shape[0]),
        "n_actions": int(sig_counts.shape[1]),
        "gamma": GAMMA,
        "reward_bound": REWARD_BOUND,
        "policy": np.asarray(policy, dtype=np.float64),
        "value_estimate": np.asarray(value, dtype=np.float64),
        "signature_counts": sig_counts,
        "signature_sums": sig_sums,
        "target_counts": np.asarray(tgt_counts, dtype=np.int64),
        "target_sums": np.asarray(tgt_sums, dtype=np.float64),
    }


def base_signature() -> tuple[np.ndarray, np.ndarray]:
    sig_q = np.array(
        [
            [0.9, 1.0, 0.8, -0.2],
            [0.1, 0.5, 0.4, 0.1],
            [-0.7, -0.5, -0.4, 0.9],
        ],
        dtype=np.float64,
    )
    sig_counts = np.full((N_STATES, N_ACTIONS), 5, dtype=np.int64)
    return sig_q, sig_counts


def expected_distances(
    sig_q: np.ndarray, sig_counts: np.ndarray, action: int
) -> tuple[np.ndarray, np.ndarray]:
    """Independent hand loop for the frozen distance and common-support mask."""
    n_states, n_actions = sig_counts.shape
    common = np.zeros((n_states, n_states), dtype=np.int64)
    distance = np.full((n_states, n_states), np.nan)
    for s in range(n_states):
        for sp in range(n_states):
            shared = [
                b
                for b in range(n_actions)
                if b != action and sig_counts[s, b] > 0 and sig_counts[sp, b] > 0
            ]
            common[s, sp] = len(shared)
            if s == sp:
                distance[s, sp] = 0.0
            elif len(shared) >= 2:
                sq = np.mean(
                    [
                        ((sig_q[s, b] - sig_q[sp, b]) / (2.0 * VALUE_SCALE)) ** 2
                        for b in shared
                    ]
                )
                distance[s, sp] = float(np.sqrt(sq))
    return common, distance


def hand_primary_estimate(
    weights_row: np.ndarray, tgt_sums: np.ndarray, tgt_counts: np.ndarray, action: int
) -> tuple[float, float, float]:
    numerator = float(
        np.sum(weights_row * tgt_sums[:, action])
    )
    denominator = float(np.sum(weights_row * tgt_counts[:, action]))
    ess = denominator**2 / float(np.sum(weights_row**2 * tgt_counts[:, action]))
    return numerator / denominator, denominator, ess


def primary_outputs(result: dict) -> dict:
    return result["routes"]["leave_one_action_out_kernel"]


def test_target_action_excluded() -> None:
    sig_q, sig_counts = base_signature()
    value = np.array([0.3, -0.1, 0.2])
    tgt_counts = np.array(
        [[0, 4, 2, 1], [3, 0, 5, 2], [2, 1, 0, 6]], dtype=np.int64
    )
    tgt_sums = 0.4 * tgt_counts.astype(np.float64)
    inputs = build_inputs(sig_q, sig_counts, tgt_sums, tgt_counts, value)
    baseline = evaluate_all_routes(inputs)
    perturbed_q = sig_q.copy()
    perturbed_q[:, 2] = [55.0, -77.0, 123.0]
    perturbed = build_inputs(perturbed_q, sig_counts, tgt_sums, tgt_counts, value)
    changed = evaluate_all_routes(perturbed)
    action = 2
    assert baseline["primary"]["bandwidth"][action] == changed["primary"]["bandwidth"][action]
    assert (
        baseline["primary"]["distances"][str(action)] == changed["primary"]["distances"][str(action)]
    )
    assert baseline["primary"]["weights"][str(action)] == changed["primary"]["weights"][str(action)]
    for state in range(N_STATES):
        assert (
            primary_outputs(baseline)["estimates"][state][action]
            == primary_outputs(changed)["estimates"][state][action]
        )
        assert (
            primary_outputs(baseline)["reasons"][state][action]
            == primary_outputs(changed)["reasons"][state][action]
        )
    other_action = 1
    assert (
        baseline["primary"]["distances"][str(other_action)]
        != changed["primary"]["distances"][str(other_action)]
    )
    print("[K1] target action excluded from primary signature")


def test_two_common_action_gate() -> None:
    sig_q, sig_counts = base_signature()
    sig_counts[2, :] = 0
    sig_counts[2, 0] = 4
    sig_counts[2, 1] = 4
    value = np.zeros(N_STATES)
    tgt_counts = np.zeros((N_STATES, N_ACTIONS), dtype=np.int64)
    tgt_counts[0, 0] = 3
    tgt_counts[1, 0] = 2
    tgt_sums = np.where(tgt_counts > 0, 0.5 * tgt_counts, 0.0)
    inputs = build_inputs(sig_q, sig_counts, tgt_sums, tgt_counts, value)
    result = evaluate_all_routes(inputs)
    out = primary_outputs(result)
    assert out["reasons"][2][0] == REASON_SUPPORT
    assert out["estimates"][2][0] is None
    assert out["estimates"][0][0] is not None
    assert out["estimates"][1][0] is not None
    common = result["primary"]["common_counts"]["0"]
    assert common[2][0] == 1 and common[2][1] == 1
    print("[K2] two-common-action support gate")


def test_identical_signatures_bandwidth() -> None:
    sig_q, sig_counts = base_signature()
    sig_q[:] = sig_q[0]
    value = np.zeros(N_STATES)
    tgt_counts = np.array(
        [[0, 4, 2, 1], [3, 0, 5, 2], [2, 1, 0, 6]], dtype=np.int64
    )
    tgt_sums = 0.3 * tgt_counts.astype(np.float64)
    inputs = build_inputs(sig_q, sig_counts, tgt_sums, tgt_counts, value)
    result = evaluate_all_routes(inputs)
    assert result["primary"]["bandwidth"][0] is None
    out = primary_outputs(result)
    assert out["reasons"][0][0] == REASON_BANDWIDTH
    assert out["estimates"][0][0] is None
    assert out["reasons"][1][1] == REASON_BANDWIDTH
    assert out["estimates"][1][1] is None
    local = result["routes"]["local_unpooled"]
    assert local["estimates"][1][1] is None
    assert local["reasons"][1][1] == REASON_TARGET
    pool = result["routes"]["action_only_pool"]
    assert pool["estimates"][1][1] is not None
    print("[K3] degenerate distances force bandwidth abstention without fallback")


def test_median_bandwidth_hand() -> None:
    for n_states in (3, 4):
        rng = np.random.default_rng(991 + n_states)
        sig_q = rng.uniform(-1.0, 1.0, size=(n_states, N_ACTIONS))
        sig_counts = np.full((n_states, N_ACTIONS), 7, dtype=np.int64)
        value = rng.uniform(-0.5, 0.5, size=n_states)
        tgt_counts = rng.integers(0, 6, size=(n_states, N_ACTIONS)).astype(np.int64)
        tgt_sums = rng.uniform(-1.0, 1.0, size=(n_states, N_ACTIONS)) * tgt_counts
        policy = np.full((n_states, N_ACTIONS), 0.05)
        policy[:, 1] = 1.0 - (N_ACTIONS - 1) * 0.05
        inputs = build_inputs(sig_q, sig_counts, tgt_sums, tgt_counts, value, policy)
        result = evaluate_all_routes(inputs)
        for action in range(N_ACTIONS):
            common, distance = expected_distances(sig_q, sig_counts, action)
            eligible = [
                distance[s, sp]
                for s in range(n_states)
                for sp in range(s + 1, n_states)
                if common[s, sp] >= 2 and distance[s, sp] > 0.0
            ]
            expected = float(np.median(np.asarray(sorted(eligible), dtype=np.float64)))
            observed = result["primary"]["bandwidth"][action]
            assert observed is not None and abs(observed - expected) < 1e-15
            assert result["primary"]["common_counts"][str(action)] == common.tolist()
            for s in range(n_states):
                for sp in range(n_states):
                    stored = result["primary"]["distances"][str(action)][s][sp]
                    if np.isnan(distance[s, sp]):
                        assert stored is None
                    else:
                        assert abs(stored - distance[s, sp]) < 1e-15
    print("[K4] sorted-float64 median bandwidth (odd and even counts)")


def test_gaussian_weights() -> None:
    sig_q, sig_counts = base_signature()
    value = np.zeros(N_STATES)
    tgt_counts = np.full((N_STATES, N_ACTIONS), 2, dtype=np.int64)
    tgt_sums = 0.2 * tgt_counts.astype(np.float64)
    inputs = build_inputs(sig_q, sig_counts, tgt_sums, tgt_counts, value)
    result = evaluate_all_routes(inputs)
    action = 0
    bandwidth = result["primary"]["bandwidth"][action]
    _, distance = expected_distances(sig_q, sig_counts, action)
    weights = result["primary"]["weights"][str(action)]
    for s in range(N_STATES):
        assert weights[s][s] == 1.0
        for sp in range(N_STATES):
            if s == sp:
                continue
            expected = float(
                np.exp(-(distance[s, sp] ** 2) / (2.0 * bandwidth**2))
            )
            assert abs(weights[s][sp] - expected) < 1e-15
    assert weights[0][2] < weights[0][1] < weights[0][0]
    print("[K5] Gaussian weights exact and monotone in distance")


def test_effective_sample_size() -> None:
    sig_q, sig_counts = base_signature()
    value = np.zeros(N_STATES)
    tgt_counts = np.array(
        [[0, 4, 2, 1], [3, 0, 5, 2], [2, 1, 0, 6]], dtype=np.int64
    )
    rng = np.random.default_rng(77)
    tgt_sums = rng.uniform(-0.8, 0.8, size=tgt_counts.shape) * tgt_counts
    inputs = build_inputs(sig_q, sig_counts, tgt_sums, tgt_counts, value)
    result = evaluate_all_routes(inputs)
    out = primary_outputs(result)
    action = 3
    state = 0
    weights_row = np.asarray(result["primary"]["weights"][str(action)][state])
    estimate, denominator, ess = hand_primary_estimate(
        weights_row, tgt_sums, tgt_counts, action
    )
    assert abs(out["estimates"][state][action] - estimate) < 1e-15
    assert abs(out["denominators"][state][action] - denominator) < 1e-15
    assert abs(out["ess"][state][action] - ess) < 1e-12
    print("[K6] observation-weighted effective sample size")


def test_zero_count_recovery() -> None:
    sig_q, sig_counts = base_signature()
    value = np.zeros(N_STATES)
    tgt_counts = np.zeros((N_STATES, N_ACTIONS), dtype=np.int64)
    tgt_counts[1, 0] = 4
    tgt_counts[2, 0] = 6
    tgt_sums = np.zeros_like(tgt_counts, dtype=np.float64)
    tgt_sums[1, 0] = 2.0
    tgt_sums[2, 0] = -3.0
    inputs = build_inputs(sig_q, sig_counts, tgt_sums, tgt_counts, value)
    result = evaluate_all_routes(inputs)
    out = primary_outputs(result)
    weights_row = np.asarray(result["primary"]["weights"]["0"][0])
    estimate, denominator, _ = hand_primary_estimate(
        weights_row, tgt_sums, tgt_counts, 0
    )
    assert out["reasons"][0][0] is None
    assert abs(out["estimates"][0][0] - estimate) < 1e-15
    assert denominator > 0.0

    gated_q, gated_counts = base_signature()
    gated_counts[2, :] = 0
    gated_counts[2, 0] = 3
    gated_counts[2, 1] = 3
    single_inputs = build_inputs(
        gated_q, gated_counts, tgt_sums, tgt_counts, value
    )
    single = evaluate_all_routes(single_inputs)
    single_out = primary_outputs(single)
    assert single_out["reasons"][0][0] is None
    neighbor_mean = tgt_sums[1, 0] / tgt_counts[1, 0]
    assert abs(single_out["estimates"][0][0] - neighbor_mean) < 1e-15
    local = single["routes"]["local_unpooled"]
    assert local["estimates"][0][0] is None
    assert local["reasons"][0][0] == REASON_TARGET
    print("[K7] zero-count recovery uses only other-state target observations")


def test_self_only_reduction() -> None:
    sig_q, sig_counts = base_signature()
    value = np.array([0.1, 0.2, -0.3])
    tgt_counts = np.zeros((N_STATES, N_ACTIONS), dtype=np.int64)
    tgt_counts[0, 1] = 5
    tgt_sums = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
    tgt_sums[0, 1] = 1.5
    inputs = build_inputs(sig_q, sig_counts, tgt_sums, tgt_counts, value)
    result = evaluate_all_routes(inputs)
    out = primary_outputs(result)
    local_mean = tgt_sums[0, 1] / tgt_counts[0, 1]
    assert out["reasons"][0][1] is None
    assert abs(out["estimates"][0][1] - local_mean) < 1e-15
    assert abs(out["ess"][0][1] - 5.0) < 1e-12
    print("[K8] self-only weight reduces exactly to the unpooled pair mean")


def test_anchor_abstention_and_estimate() -> None:
    sig_q, sig_counts = base_signature()
    value = np.zeros(N_STATES)
    tgt_counts = np.array(
        [[0, 4, 2, 1], [3, 0, 5, 2], [2, 1, 0, 6]], dtype=np.int64
    )
    tgt_sums = 0.25 * tgt_counts.astype(np.float64)
    inputs = build_inputs(sig_q, sig_counts, tgt_sums, tgt_counts, value)
    result = evaluate_all_routes(inputs)
    anchor = result["routes"]["same_action_anchor_kernel"]
    assert anchor["estimates"][0][0] is None
    assert anchor["reasons"][0][0] == REASON_TARGET

    no_sig_counts = sig_counts.copy()
    no_sig_counts[0, 1] = 0
    no_sig = build_inputs(sig_q, no_sig_counts, tgt_sums, tgt_counts, value)
    no_sig_result = evaluate_all_routes(no_sig)
    no_sig_anchor = no_sig_result["routes"]["same_action_anchor_kernel"]
    assert no_sig_anchor["estimates"][0][1] is None
    assert no_sig_anchor["reasons"][0][1] == REASON_SUPPORT

    both_missing_counts = sig_counts.copy()
    both_missing_counts[0, 2] = 0
    both_tgt_counts = tgt_counts.copy()
    both_tgt_counts[0, 2] = 0
    both_tgt_sums = tgt_sums.copy()
    both_tgt_sums[0, 2] = 0.0
    both = build_inputs(
        sig_q, both_missing_counts, both_tgt_sums, both_tgt_counts, value
    )
    both_anchor = evaluate_all_routes(both)["routes"]["same_action_anchor_kernel"]
    assert both_anchor["reasons"][0][2] == REASON_SUPPORT

    action = 3
    state = 1
    anchor_bandwidth = result["anchor"]["bandwidth"][action]
    assert anchor_bandwidth is not None
    expected_anchor = []
    for s in range(N_STATES):
        for sp in range(s + 1, N_STATES):
            d = abs(sig_q[s, action] - sig_q[sp, action]) / (2.0 * VALUE_SCALE)
            if d > 0.0:
                expected_anchor.append(d)
    assert abs(anchor_bandwidth - float(np.median(sorted(expected_anchor)))) < 1e-15
    weights_row = np.asarray(result["anchor"]["weights"][str(action)][state])
    estimate, denominator, _ = hand_primary_estimate(
        weights_row, tgt_sums, tgt_counts, action
    )
    assert anchor["reasons"][state][action] is None
    assert abs(anchor["estimates"][state][action] - estimate) < 1e-15
    assert denominator > 0.0
    print("[K9] same-action anchor gating, ordering, and estimate")


def test_permutation_equivariance() -> None:
    sig_q, sig_counts = base_signature()
    value = np.array([0.3, -0.1, 0.2])
    rng = np.random.default_rng(55)
    tgt_counts = rng.integers(0, 7, size=(N_STATES, N_ACTIONS)).astype(np.int64)
    tgt_sums = rng.uniform(-1.0, 1.0, size=(N_STATES, N_ACTIONS)) * tgt_counts
    inputs = build_inputs(sig_q, sig_counts, tgt_sums, tgt_counts, value)
    base = evaluate_all_routes(inputs)
    permutation = np.array([2, 0, 1])
    perm_inputs = build_inputs(
        sig_q[permutation],
        sig_counts[permutation],
        tgt_sums[permutation],
        tgt_counts[permutation],
        value[permutation],
        np.asarray(inputs["policy"])[permutation],
    )
    permuted = evaluate_all_routes(perm_inputs)
    inverse = np.argsort(permutation)
    for route in ROUTES:
        base_estimates = np.asarray(
            [
                [np.nan if v is None else v for v in row]
                for row in base["routes"][route]["estimates"]
            ]
        )
        perm_estimates = np.asarray(
            [
                [np.nan if v is None else v for v in row]
                for row in permuted["routes"][route]["estimates"]
            ]
        )
        reordered = perm_estimates[inverse]
        assert np.allclose(
            base_estimates, reordered, rtol=1e-12, atol=1e-12, equal_nan=True
        )
        base_reasons = base["routes"][route]["reasons"]
        perm_reasons = permuted["routes"][route]["reasons"]
        for s in range(N_STATES):
            assert base_reasons[s] == perm_reasons[int(permutation[s])]
    print("[K10] state-label permutation equivariance")


def test_malformed_inputs() -> None:
    sig_q, sig_counts = base_signature()
    value = np.zeros(N_STATES)
    tgt_counts = np.full((N_STATES, N_ACTIONS), 2, dtype=np.int64)
    tgt_sums = 0.1 * tgt_counts.astype(np.float64)
    good = build_inputs(sig_q, sig_counts, tgt_sums, tgt_counts, value)

    def expect(reason: str, mutate) -> None:
        mutated = {key: np.asarray(v).copy() if isinstance(v, np.ndarray) else v for key, v in good.items()}
        mutate(mutated)
        try:
            validate_observable_inputs(mutated)
        except KernelInputError as error:
            assert error.reason == reason, (reason, error.reason)
            return
        raise AssertionError(f"expected KernelInputError {reason}")

    expect(REASON_SHAPE, lambda m: m.__setitem__("target_counts", m["target_counts"][:, :-1]))
    expect(REASON_SHAPE, lambda m: m.__setitem__("target_counts", m["target_counts"].astype(np.float64)))
    expect(REASON_COUNT, lambda m: m["target_counts"].__setitem__((0, 0), -1))
    expect(REASON_COUNT, lambda m: m["signature_sums"].__setitem__((0, 0), np.inf))
    expect(REASON_COUNT, lambda m: m["target_counts"].__setitem__((1, 3), 0))
    expect(REASON_VALUE, lambda m: m["value_estimate"].__setitem__(2, np.nan))
    expect(REASON_SHAPE, lambda m: m.__setitem__("q_pi", np.zeros((N_STATES, N_ACTIONS))))
    expect(REASON_SHAPE, lambda m: m.__setitem__("cluster_assignment", [0, 0, 0, 1, 1, 1]))
    expect(REASON_SHAPE, lambda m: m.__setitem__("unknown_field", 1))
    expect(REASON_SHAPE, lambda m: m.__setitem__("policy", np.full((N_STATES, N_ACTIONS), 0.3)))
    expect(REASON_SHAPE, lambda m: m.__setitem__("gamma", 1.0))
    expect(REASON_SHAPE, lambda m: m.__setitem__("reward_bound", 0.0))
    print("[K11] malformed input and oracle-key rejection at the pure boundary")


def test_reason_ordering() -> None:
    assert REASON_ORDER == (
        REASON_SHAPE,
        REASON_COUNT,
        REASON_VALUE,
        REASON_SUPPORT,
        REASON_BANDWIDTH,
        REASON_TARGET,
        REASON_DENOMINATOR,
        REASON_ESTIMATE,
        REASON_POLICY,
    )
    sig_q, sig_counts = base_signature()
    sig_q[:] = sig_q[0]
    value = np.zeros(N_STATES)
    tgt_counts = np.zeros((N_STATES, N_ACTIONS), dtype=np.int64)
    tgt_sums = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
    inputs = build_inputs(sig_q, sig_counts, tgt_sums, tgt_counts, value)
    result = evaluate_all_routes(inputs)
    out = primary_outputs(result)
    assert out["reasons"][0][0] == REASON_BANDWIDTH
    assert out["estimates"][0][0] is None

    unsupported_counts = sig_counts.copy()
    unsupported_counts[2, :] = 0
    unsupported_counts[2, 0] = 2
    unsupported_counts[2, 1] = 2
    ordered = build_inputs(sig_q, unsupported_counts, tgt_sums, tgt_counts, value)
    ordered_out = primary_outputs(evaluate_all_routes(ordered))
    assert ordered_out["reasons"][2][0] == REASON_SUPPORT
    print("[K12] frozen abstention reason ordering")


def test_denominator_invalid() -> None:
    sig_q, sig_counts = base_signature()
    sig_counts[2, :] = 0
    sig_counts[2, 0] = 3
    sig_counts[2, 1] = 3
    value = np.zeros(N_STATES)
    tgt_counts = np.zeros((N_STATES, N_ACTIONS), dtype=np.int64)
    tgt_counts[2, 0] = 4
    tgt_sums = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
    tgt_sums[2, 0] = 2.0
    inputs = build_inputs(sig_q, sig_counts, tgt_sums, tgt_counts, value)
    result = evaluate_all_routes(inputs)
    out = primary_outputs(result)
    assert out["reasons"][0][0] == REASON_DENOMINATOR
    assert out["estimates"][0][0] is None
    assert result["routes"]["action_only_pool"]["estimates"][0][0] is not None
    print("[K13] kernel denominator failure stays explicit")


def test_local_and_pool_hand() -> None:
    sig_q, sig_counts = base_signature()
    value = np.zeros(N_STATES)
    tgt_counts = np.array(
        [[0, 4, 2, 1], [3, 0, 5, 2], [2, 1, 0, 6]], dtype=np.int64
    )
    tgt_sums = np.array(
        [
            [0.0, 2.0, -1.0, 0.5],
            [1.5, 0.0, 2.5, -1.0],
            [-0.5, 0.25, 0.0, 3.0],
        ]
    )
    inputs = build_inputs(sig_q, sig_counts, tgt_sums, tgt_counts, value)
    result = evaluate_all_routes(inputs)
    local = result["routes"]["local_unpooled"]
    pool = result["routes"]["action_only_pool"]
    assert local["estimates"][0][0] is None and local["reasons"][0][0] == REASON_TARGET
    assert abs(local["estimates"][0][1] - 0.5) < 1e-15
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            pooled = float(tgt_sums[:, a].sum()) / float(tgt_counts[:, a].sum())
            assert abs(pool["estimates"][s][a] - pooled) < 1e-15
            assert pool["reasons"][s][a] is None
    empty_counts = np.zeros((N_STATES, N_ACTIONS), dtype=np.int64)
    empty = build_inputs(
        sig_q, sig_counts, np.zeros_like(tgt_sums), empty_counts, value
    )
    empty_pool = evaluate_all_routes(empty)["routes"]["action_only_pool"]
    assert empty_pool["estimates"][0][0] is None
    assert empty_pool["reasons"][0][0] == REASON_TARGET
    print("[K14] local and unconditional pooling hand values")


def test_vfirst_aggregate_replica() -> None:
    from evaluate_fixed_policy_q_routes import (
        iterative_state_evaluation,
        make_mdp,
        make_policy,
    )
    from mdps import rollout
    from verify_fixed_policy_q_routes import policy_quantities

    rng = np.random.default_rng(31415)
    mdp = make_mdp(6, 4, GAMMA, 0.08, 0.5, rng)
    policy = make_policy(6, 4, 0.05, rng)
    exact = policy_quantities(mdp, policy)
    value_limit = 1.5 / (1.0 - GAMMA)
    for length in (256, 1024):
        start = int(rng.choice(6, p=exact["mu_state"]))
        states, _, rewards = rollout(mdp, policy, start=start, n=length, rng=rng)
        transition_rewards = rewards[1:].astype(np.float64)
        reference, ref_diag = iterative_state_evaluation(
            states,
            transition_rewards,
            6,
            GAMMA,
            0.65,
            160,
            beta=None,
            value_limit=value_limit,
        )
        current = states[:-1]
        following = states[1:]
        counts = np.bincount(current, minlength=6).astype(np.int64)
        reward_sums = np.bincount(current, weights=transition_rewards, minlength=6)
        transition_counts = np.zeros((6, 6), dtype=np.float64)
        np.add.at(transition_counts, (current, following), 1.0)
        replica, rep_diag = vfirst_value_from_aggregates(
            counts,
            reward_sums,
            transition_counts,
            GAMMA,
            0.65,
            160,
            value_limit,
        )
        assert np.array_equal(reference, replica)
        assert ref_diag["iterations_used"] == rep_diag["iterations_used"]
    print("[K15] aggregate V-first replica matches the unchanged estimator bitwise")


def test_hidden_cluster_generator() -> None:
    from kernel_generalization_mdps import (
        make_current_mdp,
        make_hidden_cluster_mdp,
    )

    seed = np.random.SeedSequence(20260909)
    first_mdp, first_hidden = make_hidden_cluster_mdp(
        6, 4, GAMMA, 0.08, 0.5, np.random.default_rng(seed)
    )
    second_mdp, second_hidden = make_hidden_cluster_mdp(
        6, 4, GAMMA, 0.08, 0.5, np.random.default_rng(np.random.SeedSequence(20260909))
    )
    for key in ("P", "R", "p0"):
        assert np.array_equal(first_mdp[key], second_mdp[key])
    assert np.array_equal(
        first_hidden["cluster_labels"], second_hidden["cluster_labels"]
    )
    assert set(first_mdp) == {"nS", "nA", "gamma", "P", "R", "p0"}
    labels = first_hidden["cluster_labels"]
    assert sorted(np.bincount(labels, minlength=2).tolist()) == [3, 3]
    row_sums = first_mdp["P"].sum(axis=2)
    assert np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-12)
    assert np.all(first_mdp["P"] >= 0.0)
    diagonal = np.asarray(
        [first_mdp["P"][s, a, s] for s in range(6) for a in range(4)]
    )
    assert np.all(diagonal >= 1.0 - 0.08 - 1e-12)
    assert float(np.max(np.abs(first_mdp["R"]))) <= 1.5 + 1e-12
    proto = first_hidden["p_proto"]
    struct = 0.9 * proto[labels] + 0.1 * first_hidden["p_independent"]
    struct /= struct.sum(axis=2, keepdims=True)
    expected_p = np.zeros_like(first_mdp["P"])
    for s in range(6):
        for a in range(4):
            expected_p[s, a] = 0.92 * np.eye(6)[s] + 0.08 * struct[s, a]
    assert np.allclose(first_mdp["P"], expected_p, rtol=0.0, atol=1e-12)

    varied = set()
    for draw in range(6):
        mdp_v, hidden_v = make_hidden_cluster_mdp(
            6, 4, GAMMA, 0.5, 0.0, np.random.default_rng(np.random.SeedSequence(draw))
        )
        varied.add(tuple(int(v) for v in hidden_v["cluster_labels"]))
    assert len(varied) > 1

    current_seed = np.random.SeedSequence(4242)
    wrapped = make_current_mdp(6, 4, GAMMA, 0.08, 0.25, np.random.default_rng(current_seed))
    from evaluate_fixed_policy_q_routes import make_mdp, make_policy

    reference_rng = np.random.default_rng(np.random.SeedSequence(4242))
    reference = make_mdp(6, 4, GAMMA, 0.08, 0.25, reference_rng)
    assert np.array_equal(wrapped["P"], reference["P"])
    assert np.array_equal(wrapped["R"], reference["R"])
    policy_a = make_policy(6, 4, 0.05, reference_rng)
    wrapped_policy = make_policy(6, 4, 0.05, np.random.default_rng(np.random.SeedSequence(7)))
    assert policy_a.shape == wrapped_policy.shape == (6, 4)
    print("[K16] hidden-cluster generator structure and determinism")


def test_greedy_with_floor_policy() -> None:
    base_policy = np.full((N_STATES, N_ACTIONS), 0.05)
    base_policy[:, 2] = 0.85
    estimates = np.array(
        [
            [0.1, 0.4, -0.2, 0.3],
            [np.nan, 0.0, 0.1, 0.2],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    updated, statuses = greedy_with_floor_policy(estimates, base_policy, 0.05)
    assert statuses[0] == "updated"
    assert np.allclose(updated[0], [0.05, 0.85, 0.05, 0.05])
    assert statuses[1] == REASON_POLICY
    assert np.array_equal(updated[1], base_policy[1])
    assert statuses[2] == "updated"
    assert np.allclose(updated[2], [0.85, 0.05, 0.05, 0.05])
    print("[K17] greedy-with-floor diagnostic policy and incomplete-row retention")


def main() -> None:
    test_target_action_excluded()
    test_two_common_action_gate()
    test_identical_signatures_bandwidth()
    test_median_bandwidth_hand()
    test_gaussian_weights()
    test_effective_sample_size()
    test_zero_count_recovery()
    test_self_only_reduction()
    test_anchor_abstention_and_estimate()
    test_permutation_equivariance()
    test_malformed_inputs()
    test_reason_ordering()
    test_denominator_invalid()
    test_local_and_pool_hand()
    test_vfirst_aggregate_replica()
    test_hidden_cluster_generator()
    test_greedy_with_floor_policy()
    print("PASS kernel state generalization contract verification")


if __name__ == "__main__":
    main()
