"""Contract tests for the visit-indexed martingale certificate (Claude route).

Covers the frozen FP-MART-001 plan Task 4 list: risk allocation, radius
constants and monotonicity, visit-count selection, support fixtures,
random-count lookup without data-dependent risk, zero-initialization bound B,
Direct-Q / state-value / V-first no-split composition, margin rejection,
invalid delta, mode mismatch, nonfinite arithmetic, deterministic failure
ordering, selective status semantics, strict JSON, certificate/audit
separation, and the two proof-level counterexample fixtures (wrong state
filtration; unadjusted post-hoc minimum).
"""

from __future__ import annotations

import inspect
import json
import math

import numpy as np

from fixed_policy_finite_sample_certificate import strict_json_ready
import visit_indexed_martingale_certificate as vimc


GAMMA = 0.70
ALPHA = 0.65
BETA = 8.0
DELTA = 0.05
N_STATES = 6
N_ACTIONS = 4
N_PAIRS = N_STATES * N_ACTIONS
G_GROUPS = N_STATES + 2 * N_PAIRS
VALUE_BOUND = 5.0


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def approx_equal(left: float, right: float, tol: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=tol, abs_tol=tol)


def make_event(
    state_counts: list[int] | None = None,
    pair_counts: list[int] | None = None,
    trajectory_length: int = 256,
    delta: float = DELTA,
    value_bound: float = VALUE_BOUND,
    **flags: object,
) -> dict:
    if state_counts is None:
        state_counts = [trajectory_length // N_STATES] * N_STATES
    if pair_counts is None:
        pair_counts = [trajectory_length // N_PAIRS] * N_PAIRS
    return vimc.shared_visit_event(
        trajectory_length=trajectory_length,
        n_states=N_STATES,
        n_pairs=N_PAIRS,
        state_counts=state_counts,
        pair_counts=pair_counts,
        delta=delta,
        value_bound=value_bound,
        **flags,
    )


def test_risk_allocation_and_radius_constant() -> None:
    n = 256
    for k in (1, 2, 7, 50, n):
        radius = vimc.hoeffding_radius(
            k,
            n_groups=G_GROUPS,
            trajectory_length=n,
            delta=DELTA,
            value_bound=VALUE_BOUND,
        )
        expected = VALUE_BOUND * math.sqrt(
            2.0 * math.log(2.0 * G_GROUPS * n / DELTA) / k
        )
        check(approx_equal(radius, expected), "radius must use the proved constant")
        tail = 2.0 * math.exp(-(radius * k) ** 2 / (2.0 * VALUE_BOUND**2 * k))
        target = DELTA / (G_GROUPS * n)
        check(
            approx_equal(tail, target, 1e-9),
            f"per-cell tail {tail} must equal the allocation {target}",
        )
        total = G_GROUPS * n * tail
        check(
            approx_equal(total, DELTA, 1e-9),
            "union allocation must exhaust delta exactly",
        )


def test_radius_monotonicity_and_scaling() -> None:
    n = 1024
    radii = [
        vimc.hoeffding_radius(
            k,
            n_groups=G_GROUPS,
            trajectory_length=n,
            delta=DELTA,
            value_bound=VALUE_BOUND,
        )
        for k in range(1, 17)
    ]
    for earlier, later in zip(radii, radii[1:]):
        check(later < earlier, "radius must strictly decrease in k")
    base = radii[0]
    for index, radius in enumerate(radii, start=1):
        check(
            approx_equal(radius * math.sqrt(index), base, 1e-9),
            "radius must scale exactly as 1/sqrt(k)",
        )


def test_radius_input_validation() -> None:
    for bad_k in (0, -1, 1025, 2.5):
        try:
            vimc.hoeffding_radius(
                bad_k,
                n_groups=G_GROUPS,
                trajectory_length=1024,
                delta=DELTA,
                value_bound=VALUE_BOUND,
            )
        except ValueError:
            continue
        raise AssertionError(f"invalid count {bad_k} accepted")
    for bad_delta in (0.0, 1.0, -0.1, math.nan):
        try:
            vimc.hoeffding_radius(
                3,
                n_groups=G_GROUPS,
                trajectory_length=1024,
                delta=bad_delta,
                value_bound=VALUE_BOUND,
            )
        except ValueError:
            continue
        raise AssertionError(f"invalid delta {bad_delta} accepted")


def test_support_summary_selection() -> None:
    summary = vimc.support_summary([3, 0, 5, 2])
    check(summary["min_count"] == 0, "min count selection failed")
    check(summary["missing"] == 1, "missing count failed")
    check(not summary["full_support"], "full support detection failed")
    full = vimc.support_summary([3, 4, 5, 2])
    check(full["full_support"] and full["min_count"] == 2, "full fixture failed")


def test_random_count_lookup_has_no_data_dependent_risk() -> None:
    n = 512
    event_a = make_event(
        state_counts=[n // N_STATES] * N_STATES,
        pair_counts=[n // N_PAIRS] * N_PAIRS,
        trajectory_length=n,
    )
    skewed_pairs = [3] + [(n - 3) // (N_PAIRS - 1)] * (N_PAIRS - 1)
    skewed_pairs[-1] += n - sum(skewed_pairs)
    event_b = make_event(
        state_counts=[n // N_STATES] * N_STATES,
        pair_counts=skewed_pairs,
        trajectory_length=n,
    )
    check(
        approx_equal(event_a["log_factor"], event_b["log_factor"]),
        "log factor must not depend on observed counts",
    )
    check(
        approx_equal(event_a["delta"], event_b["delta"]),
        "risk budget must not be reallocated from data",
    )
    direct = vimc.hoeffding_radius(
        event_b["pair_min_count"],
        n_groups=G_GROUPS,
        trajectory_length=n,
        delta=DELTA,
        value_bound=VALUE_BOUND,
    )
    check(
        approx_equal(event_b["pair_radius"], direct),
        "radius at the random observed min count must equal the "
        "simultaneous-bound radius evaluated at that count",
    )


def test_full_and_missing_support_emission() -> None:
    event = make_event()
    check(event["event_valid"], "valid event expected")
    direct = vimc.direct_q_certificate(
        event,
        matching="exact",
        gamma=GAMMA,
        alpha=ALPHA,
        iterations_used=25,
    )
    check(
        direct["status"] == vimc.STATUS_SELECTIVE,
        "full-support exact route must emit the selective status",
    )
    check(direct["failure_reasons"] == [], "emission must carry no failures")

    sparse_pairs = [0] * N_PAIRS
    sparse_pairs[0] = 256
    event_missing = make_event(pair_counts=sparse_pairs)
    direct_missing = vimc.direct_q_certificate(
        event_missing,
        matching="exact",
        gamma=GAMMA,
        alpha=ALPHA,
        iterations_used=25,
    )
    check(
        direct_missing["status"] == vimc.STATUS_NOT_EMITTED,
        "missing support must not emit",
    )
    check(
        direct_missing["failure_reasons"] == ["missing_support"],
        "missing support rejection mismatch",
    )
    check(direct_missing["total_bound"] is None, "non-emission must null the bound")


def test_zero_initialization_bound() -> None:
    event = make_event()
    direct = vimc.direct_q_certificate(
        event,
        matching="exact",
        gamma=GAMMA,
        alpha=ALPHA,
        iterations_used=0,
    )
    check(
        approx_equal(direct["initial_error_bound"], VALUE_BOUND),
        "zero initialization must use the computable bound B",
    )
    check(
        approx_equal(direct["total_bound"], VALUE_BOUND),
        "at zero iterations the total bound must equal B",
    )
    signature = inspect.signature(vimc.direct_q_certificate)
    forbidden = {
        "initial_error",
        "true_q",
        "true_value",
        "q_pi",
        "v_pi",
        "occupancy",
        "transition",
        "spectral",
    }
    check(
        not (set(signature.parameters) & forbidden),
        "certificate signature must not accept oracle quantities",
    )


def test_direct_q_exact_composition_arithmetic() -> None:
    n = 256
    event = make_event(trajectory_length=n)
    iterations = 37
    direct = vimc.direct_q_certificate(
        event,
        matching="exact",
        gamma=GAMMA,
        alpha=ALPHA,
        iterations_used=iterations,
    )
    rho = 1.0 - ALPHA * (1.0 - GAMMA)
    radius = VALUE_BOUND * math.sqrt(
        2.0 * math.log(2.0 * G_GROUPS * n / DELTA) / (n // N_PAIRS)
    )
    expected = rho**iterations * VALUE_BOUND + (1.0 - rho**iterations) * radius / (
        1.0 - GAMMA
    )
    check(approx_equal(direct["rho"], rho), "exact rho mismatch")
    check(approx_equal(direct["residual_radius"], radius), "exact radius mismatch")
    check(
        approx_equal(direct["total_bound"], expected, 1e-9),
        "exact composition arithmetic mismatch",
    )


def test_direct_q_softmax_margin_gate() -> None:
    event = make_event()
    diagonal = 0.95
    direct = vimc.direct_q_certificate(
        event,
        matching="softmax",
        gamma=GAMMA,
        alpha=ALPHA,
        iterations_used=11,
        beta=BETA,
        kernel_diagonal_min=diagonal,
    )
    margin = diagonal - (1.0 + GAMMA) / 2.0
    check(approx_equal(direct["margin"], margin), "softmax margin mismatch")
    check(direct["status"] == vimc.STATUS_SELECTIVE, "softmax route should emit")

    flat_diagonal = (1.0 + GAMMA) / 2.0 - 0.01
    rejected = vimc.direct_q_certificate(
        event,
        matching="softmax",
        gamma=GAMMA,
        alpha=ALPHA,
        iterations_used=11,
        beta=BETA,
        kernel_diagonal_min=flat_diagonal,
    )
    check(
        rejected["failure_reasons"] == ["pair_kernel_margin_nonpositive"],
        "nonpositive pair margin must be rejected deterministically",
    )
    check(rejected["status"] == vimc.STATUS_NOT_EMITTED, "margin failure emits")


def test_state_value_and_vfirst_composition() -> None:
    n = 1024
    event = make_event(
        state_counts=[n // N_STATES] * N_STATES,
        pair_counts=[n // N_PAIRS] * N_PAIRS,
        trajectory_length=n,
    )
    state = vimc.state_value_certificate(
        event,
        matching="exact",
        gamma=GAMMA,
        alpha=ALPHA,
        iterations_used=20,
    )
    check(state["status"] == vimc.STATUS_SELECTIVE, "state stage must emit")
    vfirst = vimc.vfirst_nosplit_certificate(
        event,
        state,
        recovery_matching="exact",
        gamma=GAMMA,
    )
    radius = VALUE_BOUND * math.sqrt(
        2.0 * math.log(2.0 * G_GROUPS * n / DELTA) / (n // N_PAIRS)
    )
    expected = GAMMA * state["total_bound"] + radius
    check(
        approx_equal(vfirst["total_bound"], expected, 1e-9),
        "V-first exact composition mismatch",
    )
    check(
        approx_equal(vfirst["softmax_leakage"], 0.0),
        "exact recovery must have zero leakage",
    )

    recovery_diagonal = 0.6
    vfirst_soft = vimc.vfirst_nosplit_certificate(
        event,
        state,
        recovery_matching="softmax",
        gamma=GAMMA,
        beta=BETA,
        recovery_diagonal_min=recovery_diagonal,
    )
    leakage = 2.0 * VALUE_BOUND * (1.0 - recovery_diagonal)
    check(
        approx_equal(vfirst_soft["softmax_leakage"], leakage),
        "softmax recovery leakage mismatch",
    )
    check(
        approx_equal(
            vfirst_soft["total_bound"], GAMMA * state["total_bound"] + radius + leakage
        ),
        "V-first softmax composition mismatch",
    )

    state_soft_rejected = vimc.state_value_certificate(
        event,
        matching="softmax",
        gamma=GAMMA,
        alpha=ALPHA,
        iterations_used=20,
        beta=BETA,
        kernel_diagonal_min=(1.0 + GAMMA) / 2.0 - 0.01,
    )
    check(
        state_soft_rejected["failure_reasons"] == ["state_kernel_margin_nonpositive"],
        "nonpositive state margin must be rejected",
    )


def test_vfirst_softmax_needs_no_pair_margin() -> None:
    event = make_event()
    state = vimc.state_value_certificate(
        event,
        matching="exact",
        gamma=GAMMA,
        alpha=ALPHA,
        iterations_used=5,
    )
    vfirst_soft = vimc.vfirst_nosplit_certificate(
        event,
        state,
        recovery_matching="softmax",
        gamma=GAMMA,
        beta=BETA,
        recovery_diagonal_min=0.4,
    )
    check(
        vfirst_soft["status"] == vimc.STATUS_SELECTIVE,
        "softmax recovery must not be gated by the pair contraction margin",
    )


def test_risk_budget_and_mode_rejection() -> None:
    bad_delta = make_event(delta=1.5)
    check(
        bad_delta["failure_reasons"] == ["risk_budget_invalid"],
        "invalid delta must yield risk_budget_invalid",
    )
    check(not bad_delta["event_valid"], "invalid risk budget must invalidate")
    direct = vimc.direct_q_certificate(
        bad_delta,
        matching="exact",
        gamma=GAMMA,
        alpha=ALPHA,
        iterations_used=3,
    )
    check(direct["status"] == vimc.STATUS_NOT_EMITTED, "invalid delta emitted")
    check(
        "risk_budget_invalid" in direct["failure_reasons"],
        "risk failure must propagate to routes",
    )

    bad_mode = make_event(fixed_context=False)
    check(
        bad_mode["failure_reasons"] == ["algorithm_mode_mismatch"],
        "non-fixed context must be rejected",
    )
    bad_mode_sync = make_event(synchronous_update=False)
    check(
        bad_mode_sync["failure_reasons"] == ["algorithm_mode_mismatch"],
        "asynchronous update must be rejected",
    )


def test_nonfinite_arithmetic_rejection() -> None:
    unit_pairs = [1] * (N_PAIRS - 1)
    unit_pairs.append(256 - (N_PAIRS - 1))
    huge = make_event(value_bound=1e308, pair_counts=unit_pairs)
    check(
        "numerical_nonfinite" in huge["failure_reasons"],
        "overflowing radius must be rejected as nonfinite",
    )
    nan_event = make_event(value_bound=math.nan)
    check(
        "numerical_nonfinite" in nan_event["failure_reasons"],
        "NaN value bound must be rejected as nonfinite",
    )
    for event in (huge, nan_event):
        direct = vimc.direct_q_certificate(
            event,
            matching="exact",
            gamma=GAMMA,
            alpha=ALPHA,
            iterations_used=3,
        )
        check(
            direct["status"] == vimc.STATUS_NOT_EMITTED,
            "nonfinite arithmetic must not emit",
        )
        check(
            direct["total_bound"] is None,
            "nonfinite route must null numeric outputs",
        )
    nan_diagonal_event = make_event()
    rejected = vimc.direct_q_certificate(
        nan_diagonal_event,
        matching="softmax",
        gamma=GAMMA,
        alpha=ALPHA,
        iterations_used=3,
        beta=BETA,
        kernel_diagonal_min=math.inf,
    )
    check(
        "numerical_nonfinite" in rejected["failure_reasons"],
        "nonfinite diagonal must be rejected as nonfinite",
    )


def test_divergence_guard_rejection() -> None:
    event = make_event()
    direct = vimc.direct_q_certificate(
        event,
        matching="exact",
        gamma=GAMMA,
        alpha=ALPHA,
        iterations_used=3,
        divergence_guard_triggered=True,
    )
    check(
        direct["failure_reasons"] == ["divergence_guard_triggered"],
        "divergence guard must reject deterministically",
    )


def test_deterministic_failure_ordering() -> None:
    sparse_pairs = [0] * N_PAIRS
    sparse_pairs[0] = 256
    event = make_event(pair_counts=sparse_pairs, fixed_context=False)
    direct = vimc.direct_q_certificate(
        event,
        matching="softmax",
        gamma=GAMMA,
        alpha=ALPHA,
        iterations_used=3,
        beta=BETA,
        kernel_diagonal_min=0.1,
        divergence_guard_triggered=True,
    )
    check(
        direct["failure_reasons"]
        == [
            "algorithm_mode_mismatch",
            "divergence_guard_triggered",
            "missing_support",
            "pair_kernel_margin_nonpositive",
        ],
        f"failure ordering is not deterministic: {direct['failure_reasons']}",
    )
    repeat = vimc.direct_q_certificate(
        event,
        matching="softmax",
        gamma=GAMMA,
        alpha=ALPHA,
        iterations_used=3,
        beta=BETA,
        kernel_diagonal_min=0.1,
        divergence_guard_triggered=True,
    )
    check(
        repeat["failure_reasons"] == direct["failure_reasons"],
        "failure ordering must be reproducible",
    )


def test_status_semantics_do_not_reuse_legacy_names() -> None:
    check(
        vimc.STATUS_SELECTIVE == "selective_high_probability_certified",
        "selective status name mismatch",
    )
    check(
        vimc.STATUS_SELECTIVE not in {"high_probability_certified", "pathwise_bound_verified"},
        "new status must not overwrite legacy status meanings",
    )
    check(
        vimc.STATUS_NOT_EMITTED not in {"high_probability_certified", "pathwise_bound_verified"},
        "non-emission status must be distinct",
    )


def test_strict_json_and_audit_separation() -> None:
    event = make_event()
    state = vimc.state_value_certificate(
        event,
        matching="exact",
        gamma=GAMMA,
        alpha=ALPHA,
        iterations_used=9,
    )
    payload = {
        "certificate": {
            "event": event,
            "direct": vimc.direct_q_certificate(
                event,
                matching="exact",
                gamma=GAMMA,
                alpha=ALPHA,
                iterations_used=9,
            ),
            "state": state,
            "vfirst": vimc.vfirst_nosplit_certificate(
                event, state, recovery_matching="exact", gamma=GAMMA
            ),
            "variance_adaptive": vimc.variance_adaptive_certificate(DELTA / 2.0),
        },
        "oracle_audit": {"true_q_sup_error": 0.123},
    }
    text = json.dumps(strict_json_ready(payload), allow_nan=False)
    reparsed = json.loads(text)
    check("oracle_audit" in reparsed, "audit namespace lost")
    certificate_blob = json.dumps(reparsed["certificate"])
    for forbidden_key in (
        "oracle_audit",
        "true_q",
        "true_value",
        "q_pi",
        "v_pi",
        "true_residual",
        "true_occupancy",
    ):
        check(
            forbidden_key not in certificate_blob,
            f"certificate namespace leaks oracle field {forbidden_key}",
        )
    check(
        reparsed["certificate"]["variance_adaptive"]["status"] == "unavailable",
        "variance-adaptive constituent must be labelled unavailable",
    )


def test_variance_adaptive_unavailable_and_risk_validation() -> None:
    adaptive = vimc.variance_adaptive_certificate(0.025)
    check(
        adaptive["failure_reasons"] == ["variance_adaptive_unavailable"],
        "variance-adaptive route must report its documented unavailability",
    )
    check(adaptive["radius"] is None, "unavailable radius must be null")
    check(
        adaptive["status"] != vimc.STATUS_SELECTIVE,
        "unavailable constituent must not emit",
    )
    invalid = vimc.variance_adaptive_certificate(1.5)
    check(
        "risk_budget_invalid" in invalid["failure_reasons"],
        "invalid preallocated share must be rejected",
    )


def test_fixture_wrong_state_filtration() -> None:
    """F1: the state residual is not centered given the post-action field."""
    gamma = 0.5
    transition = np.array(
        [
            [[0.9, 0.1], [0.2, 0.8]],
            [[0.3, 0.7], [0.6, 0.4]],
        ]
    )
    reward = np.array(
        [
            [[1.0, -0.5], [0.25, -1.0]],
            [[0.5, 0.75], [-0.25, 1.0]],
        ]
    )
    policy = np.array([[0.3, 0.7], [0.6, 0.4]])
    p_pi = np.einsum("sa,san->sn", policy, transition)
    reward_sa = np.sum(transition * reward, axis=2)
    reward_pi = np.sum(policy * reward_sa, axis=1)
    v_pi = np.linalg.solve(np.eye(2) - gamma * p_pi, reward_pi)

    post_action_mean = reward_sa + gamma * transition @ v_pi - v_pi[:, None]
    check(
        float(np.max(np.abs(post_action_mean))) > 1e-6,
        "fixture must exhibit a nonzero advantage: the state residual is not "
        "centered under the post-action filtration",
    )
    pre_action_mean = np.sum(policy * post_action_mean, axis=1)
    check(
        float(np.max(np.abs(pre_action_mean))) < 1e-12,
        "state residual must be centered under the pre-action filtration",
    )
    bound = float(np.max(np.abs(reward))) / (1.0 - gamma)
    widths = []
    for state in range(2):
        for action in range(2):
            samples = reward[state, action] + gamma * v_pi
            widths.append(float(samples.max() - samples.min()))
    check(
        max(widths) <= 2.0 * bound + 1e-12,
        "conditional range width must respect the proved 2B constant",
    )


def test_fixture_unadjusted_posthoc_minimum() -> None:
    """F2: min of two delta-level bounds certifies at 2 delta, not delta."""
    k, n = 8, 256
    radius = vimc.hoeffding_radius(
        k,
        n_groups=G_GROUPS,
        trajectory_length=n,
        delta=DELTA,
        value_bound=VALUE_BOUND,
    )
    per_bound_tail = 2.0 * math.exp(
        -(radius * k) ** 2 / (2.0 * VALUE_BOUND**2 * k)
    )
    union_two = 2.0 * G_GROUPS * n * per_bound_tail
    check(
        union_two > DELTA,
        "taking the post-hoc minimum of two delta-level bounds must exceed "
        "the declared risk budget",
    )
    preallocated = vimc.hoeffding_radius(
        k,
        n_groups=G_GROUPS,
        trajectory_length=n,
        delta=DELTA / 2.0,
        value_bound=VALUE_BOUND,
    )
    check(
        preallocated > radius,
        "preallocated delta/2 constituents must use a larger radius",
    )
    tail_pre = 2.0 * math.exp(
        -(preallocated * k) ** 2 / (2.0 * VALUE_BOUND**2 * k)
    )
    check(
        2.0 * G_GROUPS * n * tail_pre <= DELTA * (1.0 + 1e-9),
        "preallocated delta/2 constituents must restore the budget",
    )


def main() -> None:
    tests = [
        test_risk_allocation_and_radius_constant,
        test_radius_monotonicity_and_scaling,
        test_radius_input_validation,
        test_support_summary_selection,
        test_random_count_lookup_has_no_data_dependent_risk,
        test_full_and_missing_support_emission,
        test_zero_initialization_bound,
        test_direct_q_exact_composition_arithmetic,
        test_direct_q_softmax_margin_gate,
        test_state_value_and_vfirst_composition,
        test_vfirst_softmax_needs_no_pair_margin,
        test_risk_budget_and_mode_rejection,
        test_nonfinite_arithmetic_rejection,
        test_divergence_guard_rejection,
        test_deterministic_failure_ordering,
        test_status_semantics_do_not_reuse_legacy_names,
        test_strict_json_and_audit_separation,
        test_variance_adaptive_unavailable_and_risk_validation,
        test_fixture_wrong_state_filtration,
        test_fixture_unadjusted_posthoc_minimum,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS visit-indexed martingale certificate contract ({len(tests)} tests)")


if __name__ == "__main__":
    main()
