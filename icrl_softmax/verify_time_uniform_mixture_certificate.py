"""Contract checks for the time-uniform geometric-mixture certificate.

This verifier is written before the module under test exists (plan Task 4).
It covers the frozen FP-TU-001 contract: grid/weights/rates, stable
log-domain arithmetic, conditional-MGF fixtures, mixture initial value and
risk allocation, conservative inversion, exhaustive count audits through
16384, high-precision root comparisons, count validation, route composition,
ordered failure paths, strict JSON and oracle separation, the post-hoc-minimum
counterexample, and the transition-variance uniform obstruction.
"""

from __future__ import annotations

import json
import math

import numpy as np
from mpmath import mp, mpf

from time_uniform_mixture_certificate import (
    INVERSION_MAX_ITERATIONS,
    INVERSION_TOL,
    MIXTURE_GRID_SIZE,
    STATUS_NOT_CERTIFIED,
    STATUS_SELECTIVE_HIGH_PROBABILITY,
    MixtureInversionError,
    build_time_uniform_certificate,
    log_cosh,
    log_mixture,
    mixture_grid,
    mixture_radius,
    mixture_root,
    stitch_boundary,
    stitch_radius,
    strict_json_ready,
)
from visit_indexed_martingale_certificate import (
    build_visit_indexed_certificate,
    simultaneous_hoeffding_radius,
)

FROZEN_GROUPS = 54  # m + 2d = 6 + 2*24
FROZEN_DELTA = 0.05
FROZEN_LENGTHS = (256, 1024, 4096, 16384)
MAX_COUNT = 16384
# Pre-registered feasibility anchors from the frozen task's pre-review closure
# (recomputed here from the frozen formulas, never copied into the module).
EXPECTED_MAX_RATIO = {256: 0.9573574, 1024: 0.9263356, 4096: 0.8975201, 16384: 0.8793887}
EXPECTED_GLOBAL_MIN_RATIO = 0.676575
EXPECTED_GLOBAL_MAX_RATIO = 0.879389


def expect_value_error(function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def expect_inversion_error(function, reason, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except MixtureInversionError as error:
        assert error.reason == reason, (error.reason, reason)
        return
    raise AssertionError(f"expected MixtureInversionError({reason})")


def fixture(
    *,
    state_counts: list[int] | None = None,
    pair_counts: list[int] | None = None,
    beta: float = 8.0,
    fixed_context: bool = True,
    synchronous_update: bool = True,
    algorithm_mode: str = "fixed_policy_synchronous",
    direct_exact_diverged: bool = False,
    direct_softmax_diverged: bool = False,
) -> dict:
    states = [4, 4] if state_counts is None else state_counts
    pairs = [2, 2, 2, 2] if pair_counts is None else pair_counts
    return build_time_uniform_certificate(
        state_counts=states,
        pair_counts=pairs,
        trajectory_length=sum(states),
        reward_bound=1.5,
        gamma=0.7,
        alpha=0.65,
        beta=beta,
        delta=0.05,
        direct_exact_iterations=7,
        direct_softmax_iterations=6,
        state_exact_iterations=5,
        state_softmax_iterations=4,
        direct_exact_diverged=direct_exact_diverged,
        direct_softmax_diverged=direct_softmax_diverged,
        fixed_context=fixed_context,
        synchronous_update=synchronous_update,
        algorithm_mode=algorithm_mode,
    )


def verify_grid_weights_and_risk() -> None:
    grid = mixture_grid(FROZEN_GROUPS, FROZEN_DELTA)
    assert grid["n_groups"] == FROZEN_GROUPS
    assert grid["delta"] == FROZEN_DELTA
    counts = grid["count_grid"]
    weights = grid["weights"]
    log_terms = grid["log_terms"]
    rates = grid["rates"]
    assert len(counts) == MIXTURE_GRID_SIZE == 15
    assert counts == [2**j for j in range(MIXTURE_GRID_SIZE)]
    raw = [(j + 1) ** -2 for j in range(MIXTURE_GRID_SIZE)]
    norm = math.fsum(raw)
    for j in range(MIXTURE_GRID_SIZE):
        assert math.isclose(weights[j], raw[j] / norm, rel_tol=1e-15, abs_tol=1e-15)
        assert 0.0 < weights[j] < 1.0 and math.isfinite(weights[j])
        expected_log = math.log(2.0 * FROZEN_GROUPS / (FROZEN_DELTA * weights[j]))
        assert math.isclose(log_terms[j], expected_log, rel_tol=1e-15, abs_tol=1e-15)
        expected_rate = math.sqrt(2.0 * expected_log / counts[j])
        assert math.isclose(rates[j], expected_rate, rel_tol=1e-15, abs_tol=1e-15)
        assert rates[j] > 0.0 and math.isfinite(rates[j])
    assert abs(math.fsum(weights) - 1.0) <= 1e-15
    # Risk allocation: the mandatory mixture spends delta/G per group, and the
    # separate signed-line audit would spend delta*w_j/(2G) per (group, sign,
    # component); both total exactly delta.
    assert math.isclose(FROZEN_GROUPS * (FROZEN_DELTA / FROZEN_GROUPS), FROZEN_DELTA)
    signed_total = (
        FROZEN_GROUPS
        * 2
        * math.fsum(FROZEN_DELTA * w / (2.0 * FROZEN_GROUPS) for w in weights)
    )
    assert math.isclose(signed_total, FROZEN_DELTA, rel_tol=1e-15, abs_tol=1e-15)

    expect_value_error(mixture_grid, 0, FROZEN_DELTA)
    expect_value_error(mixture_grid, -3, FROZEN_DELTA)
    expect_value_error(mixture_grid, True, FROZEN_DELTA)
    expect_value_error(mixture_grid, 2.5, FROZEN_DELTA)
    expect_value_error(mixture_grid, FROZEN_GROUPS, 0.0)
    expect_value_error(mixture_grid, FROZEN_GROUPS, 1.0)
    expect_value_error(mixture_grid, FROZEN_GROUPS, math.nan)


def verify_log_domain_stability() -> None:
    assert log_cosh(0.0) == 0.0
    assert math.isclose(log_cosh(1.5), math.log(math.cosh(1.5)), rel_tol=1e-15)
    assert math.isclose(log_cosh(-1.5), log_cosh(1.5), rel_tol=1e-15, abs_tol=1e-15)
    large = log_cosh(1000.0)
    assert math.isfinite(large)
    assert math.isclose(large, 1000.0 - math.log(2.0), rel_tol=1e-15, abs_tol=1e-12)

    grid = mixture_grid(FROZEN_GROUPS, FROZEN_DELTA)
    for count, q in ((1, 0.0), (3, 0.7), (17, 2.25), (1000, 5.5)):
        direct_terms = [
            weights * math.exp(-0.5 * rate * rate * count) * math.cosh(rate * q)
            for weights, rate in zip(grid["weights"], grid["rates"], strict=True)
        ]
        direct = math.log(math.fsum(direct_terms))
        stable = log_mixture(count, q, grid=grid)
        assert math.isclose(stable, direct, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isfinite(log_mixture(1, 1e6, grid=grid))
    assert math.isfinite(log_mixture(MAX_COUNT, 1e6, grid=grid))
    # Mixture initial value: at k = 0, q = 0 every term is log w_j.
    assert math.isclose(
        math.exp(log_mixture(1, 0.0, grid=grid)),
        math.fsum(
            w * math.exp(-0.5 * a * a) for w, a in zip(grid["weights"], grid["rates"], strict=True)
        ),
        rel_tol=1e-12,
    )
    expect_value_error(log_mixture, 0, 1.0, grid=grid)
    expect_value_error(log_mixture, 1, math.nan, grid=grid)


def verify_conditional_mgf_fixtures() -> None:
    # Interval-form conditional Hoeffding: centered Y on any [c, d] with
    # d - c <= 2 has MGF at most exp(a^2/2); symmetry about 0 is not assumed.
    # Exhaustive centered two-point fixtures, which are the extreme points of
    # centered distributions on an interval.
    for width in (0.25, 0.5, 1.0, 1.5, 2.0):
        proxy = (0.5 * width) ** 2
        for fraction in np.arange(0.05, 1.0, 0.05):
            hi = width * float(fraction)
            lo = hi - width
            if not (-1.0 <= lo < 0.0 < hi <= 1.0):
                continue
            p_hi = -lo / width
            for a in np.arange(-3.0, 3.01, 0.25):
                mgf = p_hi * math.exp(float(a) * hi) + (1.0 - p_hi) * math.exp(float(a) * lo)
                bound = math.exp(0.5 * float(a) ** 2 * proxy)
                assert mgf <= bound * (1.0 + 1e-14) + 1e-15, (width, lo, a, mgf, bound)
    # Asymmetric, zero-mean, width-exactly-2 two-point fixtures whose support
    # exceeds [-1, 1]: width alone must still deliver proxy (d - c)^2/4 = 1.
    for lo, hi in ((-1.5, 0.5), (-0.5, 1.5), (-1.25, 0.75), (-0.75, 1.25)):
        assert hi - lo == 2.0
        assert hi > 1.0 or lo < -1.0  # support escapes [-1, 1]
        assert lo != -hi  # asymmetric
        p_hi = -lo / (hi - lo)
        assert math.isclose(p_hi * hi + (1.0 - p_hi) * lo, 0.0, abs_tol=1e-15)
        for a in np.arange(-6.0, 6.01, 0.125):
            mgf = p_hi * math.exp(float(a) * hi) + (1.0 - p_hi) * math.exp(float(a) * lo)
            bound = math.exp(0.5 * float(a) ** 2)
            assert mgf <= bound * (1.0 + 1e-14) + 1e-15, (lo, hi, a, mgf, bound)
    # The chord argument's terminal inequality cosh(a) <= exp(a^2/2).
    for a in np.arange(-6.0, 6.01, 0.125):
        assert math.cosh(float(a)) <= math.exp(0.5 * float(a) ** 2) * (1.0 + 1e-15)
    # One-step supermartingale factor on a width-2B residual, normalized.
    gamma, reward = 0.7, 1.5
    bound_b = reward / (1.0 - gamma)
    lo, hi = (-reward - gamma * bound_b) / bound_b, (reward + gamma * bound_b) / bound_b
    assert math.isclose(hi - lo, 2.0, rel_tol=1e-15)
    center = 0.5 * (hi + lo)
    p_hi = (0.0 - lo) / (hi - lo)
    for a in np.arange(-3.0, 3.01, 0.5):
        mgf = p_hi * math.exp(float(a) * (hi - center)) + (1.0 - p_hi) * math.exp(
            float(a) * (lo - center)
        )
        assert mgf <= math.exp(0.5 * float(a) ** 2) * (1.0 + 1e-14) + 1e-15


def verify_initial_value_and_allocation() -> None:
    grid = mixture_grid(FROZEN_GROUPS, FROZEN_DELTA)
    # Every component process starts at exp(0)*cosh(0) = 1, and the convex
    # mixture starts at sum_j w_j = 1.
    assert math.fsum(grid["weights"]) == 1.0 or abs(math.fsum(grid["weights"]) - 1.0) <= 1e-15
    target = math.log(FROZEN_GROUPS / FROZEN_DELTA)
    # Lower bracket endpoint is always below the target because M(k,0) <= 1.
    for count in (1, 2, 7, 256, 4096, MAX_COUNT):
        log_at_zero = log_mixture(count, 0.0, grid=grid)
        assert math.exp(log_at_zero) <= 1.0 + 1e-15
        assert log_at_zero < target


def verify_exhaustive_counts() -> None:
    grid = mixture_grid(FROZEN_GROUPS, FROZEN_DELTA)
    target = math.log(FROZEN_GROUPS / FROZEN_DELTA)
    q_mix = np.empty(MAX_COUNT + 1, dtype=np.float64)
    q_stitch = np.empty(MAX_COUNT + 1, dtype=np.float64)
    for count in range(1, MAX_COUNT + 1):
        result = mixture_root(count, n_groups=FROZEN_GROUPS, delta=FROZEN_DELTA)
        assert result["count"] == count
        assert math.isfinite(result["q_mix"])
        assert math.isfinite(result["q_stitch"])
        assert result["iterations"] <= INVERSION_MAX_ITERATIONS
        assert result["bracket_width"] <= INVERSION_TOL * (1.0 + result["q_mix"]) * 1.5
        residual = result["log_mixture_at_root"] - target
        assert 0.0 <= residual <= INVERSION_TOL * (1.0 + abs(target)) * 1.5
        # Conservative root: M(k, q_mix) >= target, and the stitch brackets it.
        assert result["q_mix"] <= result["q_stitch"]
        assert math.isclose(result["q_stitch"], stitch_boundary(count, grid=grid), rel_tol=0.0, abs_tol=0.0)
        q_mix[count] = result["q_mix"]
        q_stitch[count] = result["q_stitch"]

    # Radius monotonicity through 16384 (B cancels in the ratio checks below;
    # use B = 1 via reward_bound = 1 - gamma).
    gamma = 0.7
    reward = 1.0 - gamma
    r_mix = np.empty(MAX_COUNT + 1, dtype=np.float64)
    r_mix[0] = np.inf
    r_mix[1:] = q_mix[1:] / np.arange(1, MAX_COUNT + 1, dtype=np.float64)
    for count in range(2, MAX_COUNT + 1):
        assert r_mix[count] <= r_mix[count - 1] * (1.0 + 1e-12), count
        radius = mixture_radius(
            count, reward_bound=reward, gamma=gamma, n_groups=FROZEN_GROUPS, delta=FROZEN_DELTA
        )
        assert math.isclose(radius, r_mix[count], rel_tol=1e-15)
        stitch = stitch_radius(
            count, reward_bound=reward, gamma=gamma, n_groups=FROZEN_GROUPS, delta=FROZEN_DELTA
        )
        assert radius <= stitch

    # Per-record dominance against the exact legacy radius for every frozen n.
    strict_seen = False
    for length in FROZEN_LENGTHS:
        log_term = math.log(2.0 * FROZEN_GROUPS * length / FROZEN_DELTA)
        max_ratio = 0.0
        for count in range(1, length + 1):
            r_old = math.sqrt(2.0 * log_term / count)
            ratio = r_mix[count] / r_old
            assert ratio <= 1.0 + 1e-9, (length, count, ratio)
            max_ratio = max(max_ratio, ratio)
            if ratio < 1.0 - 1e-9:
                strict_seen = True
        assert math.isclose(
            max_ratio, EXPECTED_MAX_RATIO[length], rel_tol=0.0, abs_tol=1e-6
        ), (length, max_ratio)
    assert strict_seen

    # Separate global n_max = 16384 design audit.
    log_term = math.log(2.0 * FROZEN_GROUPS * MAX_COUNT / FROZEN_DELTA)
    ratios = np.array(
        [r_mix[k] / math.sqrt(2.0 * log_term / k) for k in range(1, MAX_COUNT + 1)]
    )
    assert math.isclose(float(ratios.min()), EXPECTED_GLOBAL_MIN_RATIO, abs_tol=1e-6)
    assert math.isclose(float(ratios.max()), EXPECTED_GLOBAL_MAX_RATIO, abs_tol=1e-6)
    assert int(np.argmin(ratios)) + 1 == 1
    assert int(np.argmax(ratios)) + 1 == MAX_COUNT
    assert bool(np.all(ratios < 1.0))
    assert np.all(np.diff(r_mix[1:]) <= 1e-12 * r_mix[1:-1] + 1e-15)


def verify_high_precision_roots() -> None:
    mp.dps = 60
    grid = mixture_grid(FROZEN_GROUPS, FROZEN_DELTA)
    weights = [mpf(w) for w in grid["weights"]]
    rates = [mpf(a) for a in grid["rates"]]
    target = mp.log(mpf(FROZEN_GROUPS) / mpf(FROZEN_DELTA))

    def mp_log_mixture(count: int, q: mpf) -> mpf:
        terms = [
            mp.log(w) - mpf("0.5") * a * a * count + mp.log(mp.cosh(a * q))
            for w, a in zip(weights, rates, strict=True)
        ]
        peak = max(terms)
        return peak + mp.log(mp.fsum([mp.exp(t - peak) for t in terms]))

    counts = [1, 2, 3, 7, 100, 256, 1000, 1024, 4096, 8192, 12345, 16383, MAX_COUNT]
    counts += [2**j for j in range(MIXTURE_GRID_SIZE)]
    for count in sorted(set(counts)):
        lo, hi = mpf(0), mpf(stitch_boundary(count, grid=grid))
        assert mp_log_mixture(count, lo) < target <= mp_log_mixture(count, hi)
        for _ in range(300):
            mid = (lo + hi) / 2
            if mp_log_mixture(count, mid) >= target:
                hi = mid
            else:
                lo = mid
        true_root = hi
        result = mixture_root(count, n_groups=FROZEN_GROUPS, delta=FROZEN_DELTA)
        q_float = mpf(result["q_mix"])
        assert q_float >= true_root * (1 - mpf("1e-9")), (count, result["q_mix"], float(true_root))
        assert q_float <= true_root * (1 + mpf("1e-9")) + mpf("1e-12"), count


def verify_count_validation() -> None:
    common = dict(reward_bound=1.0, gamma=0.5, n_groups=5, delta=0.1)
    expect_value_error(mixture_radius, 0, **common)
    expect_value_error(mixture_radius, -1, **common)
    expect_value_error(mixture_radius, True, **common)
    expect_value_error(mixture_radius, 2.5, **common)
    expect_value_error(mixture_radius, 1, reward_bound=0.0, gamma=0.5, n_groups=5, delta=0.1)
    expect_value_error(mixture_radius, 1, reward_bound=-1.0, gamma=0.5, n_groups=5, delta=0.1)
    expect_value_error(mixture_radius, 1, reward_bound=math.nan, gamma=0.5, n_groups=5, delta=0.1)
    expect_value_error(mixture_radius, 1, reward_bound=1.0, gamma=0.0, n_groups=5, delta=0.1)
    expect_value_error(mixture_radius, 1, reward_bound=1.0, gamma=1.0, n_groups=5, delta=0.1)
    expect_value_error(mixture_radius, 1, reward_bound=1.0, gamma=0.5, n_groups=0, delta=0.1)
    expect_value_error(mixture_radius, 1, reward_bound=1.0, gamma=0.5, n_groups=5, delta=0.0)
    expect_value_error(mixture_radius, 1, reward_bound=1.0, gamma=0.5, n_groups=5, delta=1.0)
    expect_value_error(mixture_root, 0, n_groups=5, delta=0.1)
    expect_value_error(mixture_root, 1, n_groups=5, delta=0.1, tol=0.0)
    expect_value_error(mixture_root, 1, n_groups=5, delta=0.1, max_iterations=0 - 1)
    expect_value_error(stitch_boundary, 0, grid=mixture_grid(5, 0.1))

    expect_value_error(fixture, pair_counts=[2, 3, 3])
    expect_value_error(fixture, pair_counts=[1, 1, 3, 3])
    expect_value_error(fixture, state_counts=[4, 5], pair_counts=[2, 2, 2, 2])
    expect_value_error(fixture, state_counts=[4, 4], pair_counts=[2, 2, 2, 3])
    expect_value_error(fixture, state_counts=[4, 4], pair_counts=[2, 2, 2, True])
    expect_value_error(fixture, fixed_context="yes")
    expect_value_error(
        build_time_uniform_certificate,
        state_counts=[5, 5],
        pair_counts=[2, 2, 3, 3],
        trajectory_length=10,
        reward_bound=math.nan,
        gamma=0.7,
        alpha=0.65,
        beta=8.0,
        delta=0.05,
        direct_exact_iterations=1,
        direct_softmax_iterations=1,
        state_exact_iterations=1,
        state_softmax_iterations=1,
    )


def verify_grid_isolation_and_iteration_cap() -> None:
    # Callers cannot pollute the internally cached frozen grid.
    grid = mixture_grid(FROZEN_GROUPS, FROZEN_DELTA)
    original = {key: (list(v) if isinstance(v, list) else v) for key, v in grid.items()}
    grid["weights"][0] = 999.0
    grid["rates"][0] = -1.0
    grid["count_grid"][0] = -7
    grid["log_terms"][0] = math.nan
    grid["log_weights"][0] = math.nan
    grid["half_squared_rates"][0] = math.nan
    grid["n_groups"] = 1
    grid["delta"] = 0.99
    fresh = mixture_grid(FROZEN_GROUPS, FROZEN_DELTA)
    assert fresh == original
    assert fresh is not grid
    assert fresh["weights"] is not grid["weights"]
    # Downstream certificates are unaffected by caller mutation.
    radius = mixture_radius(2, reward_bound=1.5, gamma=0.7, n_groups=10, delta=0.05)
    assert radius == mixture_radius(2, reward_bound=1.5, gamma=0.7, n_groups=10, delta=0.05)

    # A caller-supplied grid must match the frozen computation field by
    # field: key set, lengths, the fifteen k_j, weights, L_j, a_j,
    # log_weights, and half_squared_rates.
    mismatched_groups = mixture_grid(FROZEN_GROUPS + 1, FROZEN_DELTA)
    expect_value_error(
        mixture_root, 3, n_groups=FROZEN_GROUPS, delta=FROZEN_DELTA, grid=mismatched_groups
    )
    mismatched_delta = mixture_grid(FROZEN_GROUPS, 0.1)
    expect_value_error(
        mixture_root, 3, n_groups=FROZEN_GROUPS, delta=FROZEN_DELTA, grid=mismatched_delta
    )

    def tampered(field, mutate):
        candidate = mixture_grid(FROZEN_GROUPS, FROZEN_DELTA)
        mutate(candidate[field])
        expect_value_error(
            mixture_root,
            3,
            n_groups=FROZEN_GROUPS,
            delta=FROZEN_DELTA,
            grid=candidate,
        )

    for field in ("weights", "log_weights", "log_terms", "rates", "half_squared_rates"):
        tampered(field, lambda values: values.__setitem__(0, values[0] * 1.000001))
        tampered(field, lambda values: values.__setitem__(7, values[7] + 1e-9))
        tampered(field, lambda values: values.pop())
        tampered(field, lambda values: values.append(values[-1]))
    tampered("count_grid", lambda values: values.__setitem__(3, 9))
    tampered("count_grid", lambda values: values.__setitem__(0, True))
    tampered("weights", lambda values: values.__setitem__(5, True))
    scalar_delta = mixture_grid(FROZEN_GROUPS, FROZEN_DELTA)
    scalar_delta["delta"] = FROZEN_DELTA * (1.0 + 1e-9)
    expect_value_error(
        mixture_root, 3, n_groups=FROZEN_GROUPS, delta=FROZEN_DELTA, grid=scalar_delta
    )
    bool_groups = mixture_grid(FROZEN_GROUPS, FROZEN_DELTA)
    bool_groups["grid_size"] = True
    expect_value_error(
        mixture_root, 3, n_groups=FROZEN_GROUPS, delta=FROZEN_DELTA, grid=bool_groups
    )
    expect_value_error(
        mixture_root, 3, n_groups=FROZEN_GROUPS, delta=FROZEN_DELTA, grid="not a mapping"
    )
    extra_key = mixture_grid(FROZEN_GROUPS, FROZEN_DELTA)
    extra_key["injected"] = 1.0
    expect_value_error(
        mixture_root, 3, n_groups=FROZEN_GROUPS, delta=FROZEN_DELTA, grid=extra_key
    )
    missing_key = mixture_grid(FROZEN_GROUPS, FROZEN_DELTA)
    del missing_key["rates"]
    expect_value_error(
        mixture_root, 3, n_groups=FROZEN_GROUPS, delta=FROZEN_DELTA, grid=missing_key
    )
    wrong_size = mixture_grid(FROZEN_GROUPS, FROZEN_DELTA)
    wrong_size["grid_size"] = 14
    expect_value_error(
        mixture_root, 3, n_groups=FROZEN_GROUPS, delta=FROZEN_DELTA, grid=wrong_size
    )
    # An exact copy of the frozen grid is accepted.
    consistent = mixture_grid(FROZEN_GROUPS, FROZEN_DELTA)
    result = mixture_root(3, n_groups=FROZEN_GROUPS, delta=FROZEN_DELTA, grid=consistent)
    assert math.isfinite(result["q_mix"])

    # max_iterations above the frozen cap of 200 is rejected; the cap itself
    # is accepted and suffices.
    expect_value_error(
        mixture_root,
        3,
        n_groups=FROZEN_GROUPS,
        delta=FROZEN_DELTA,
        max_iterations=INVERSION_MAX_ITERATIONS + 1,
    )
    capped = mixture_root(
        3,
        n_groups=FROZEN_GROUPS,
        delta=FROZEN_DELTA,
        max_iterations=INVERSION_MAX_ITERATIONS,
    )
    assert capped["iterations"] <= INVERSION_MAX_ITERATIONS


def verify_route_composition() -> None:
    certificate = fixture()
    event = certificate["event"]
    assert event["method"] == "time_uniform_geometric_cosh_mixture"
    assert event["n_groups"] == 2 + 2 * 4
    assert event["n_groups"] == 10
    bound = event["value_bound"]
    assert bound == 1.5 / (1.0 - 0.7)
    assert event["conditional_interval_width"] == 2.0 * bound
    risk = event["risk_allocation"]
    assert math.isclose(risk["per_group_failure_probability"], 0.05 / 10.0)
    assert math.isclose(risk["total_failure_probability"], 0.05)
    assert risk["route_level_resplit"] is False

    pair_family = event["pair_bellman"]
    state_family = event["state_bellman"]
    assert pair_family["full_support"] and state_family["full_support"]
    expected_pair_radius = mixture_radius(
        2, reward_bound=1.5, gamma=0.7, n_groups=10, delta=0.05
    )
    expected_state_radius = mixture_radius(
        4, reward_bound=1.5, gamma=0.7, n_groups=10, delta=0.05
    )
    assert pair_family["max_radius"] == expected_pair_radius
    assert state_family["max_radius"] == expected_state_radius
    for family in (pair_family, state_family, event["recovery"]):
        for mix_r, stitch_r, old_r in zip(
            family["mixture_radius_by_group"],
            family["stitch_radius_by_group"],
            family["old_radius_by_group"],
            strict=True,
        ):
            if mix_r is None:
                continue
            assert mix_r <= stitch_r <= old_r or mix_r <= old_r
            assert mix_r <= stitch_r
            assert mix_r <= old_r
        assert family["max_radius_ratio_vs_old"] < 1.0
        assert family["max_radius_ratio_vs_stitch"] <= 1.0
        assert not family["inversion_failure_reasons"]
    assert event["recovery"]["mixture_radius_by_group"] == pair_family[
        "mixture_radius_by_group"
    ]

    direct = certificate["routes"]["direct_exact"]
    assert direct["residual_radius"] == expected_pair_radius
    rho = 1.0 - 0.65 * (1.0 - 0.7)
    expected_total = rho**7 * bound + (1.0 - rho**7) * expected_pair_radius / (1.0 - 0.7)
    assert math.isclose(direct["total_bound"], expected_total, rel_tol=1e-14)
    old_radius = simultaneous_hoeffding_radius(
        2, reward_bound=1.5, gamma=0.7, n_groups=10, horizon=8, delta=0.05
    )
    assert direct["old_residual_radius"] == old_radius
    expected_old_total = rho**7 * bound + (1.0 - rho**7) * old_radius / (1.0 - 0.7)
    assert math.isclose(direct["old_total_bound"], expected_old_total, rel_tol=1e-14)
    legacy = build_visit_indexed_certificate(
        state_counts=[4, 4],
        pair_counts=[2, 2, 2, 2],
        trajectory_length=8,
        reward_bound=1.5,
        gamma=0.7,
        alpha=0.65,
        beta=8.0,
        delta=0.05,
        direct_exact_iterations=7,
        direct_softmax_iterations=6,
        state_exact_iterations=5,
        state_softmax_iterations=4,
    )
    assert math.isclose(
        direct["old_total_bound"],
        legacy["routes"]["direct_exact"]["total_bound"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    assert direct["total_bound"] < direct["old_total_bound"]
    assert math.isclose(
        direct["radius_ratio_vs_old"], expected_pair_radius / old_radius, rel_tol=1e-15
    )
    assert math.isclose(
        direct["radius_reduction_vs_old"], old_radius - expected_pair_radius, rel_tol=1e-14
    )
    assert math.isclose(
        direct["total_bound_ratio_vs_old"],
        direct["total_bound"] / direct["old_total_bound"],
        rel_tol=1e-15,
    )
    assert direct["status"] == STATUS_SELECTIVE_HIGH_PROBABILITY
    assert direct["failure_reasons"] == []

    soft = certificate["routes"]["direct_softmax"]
    assert soft["margin"] > 0.0
    assert soft["status"] == STATUS_SELECTIVE_HIGH_PROBABILITY
    assert soft["total_bound"] < soft["old_total_bound"]

    state = certificate["state_value"]["exact"]
    vfirst = certificate["routes"]["vfirst_nosplit_exact"]
    recovery_radius = event["recovery"]["max_radius"]
    assert math.isclose(vfirst["total_bound"], 0.7 * state["total_bound"] + recovery_radius)
    assert math.isclose(
        vfirst["old_total_bound"],
        0.7 * state["old_total_bound"] + event["recovery"]["max_old_radius"],
    )
    soft_vfirst = certificate["routes"]["vfirst_nosplit_softmax"]
    soft_state = certificate["state_value"]["softmax"]
    pair_diagonal = soft_vfirst["recovery_kernel_diagonal_lower_bound"]
    leakage = 2.0 * bound * (1.0 - pair_diagonal)
    assert math.isclose(soft_vfirst["softmax_leakage"], leakage)
    assert math.isclose(
        soft_vfirst["total_bound"],
        0.7 * soft_state["total_bound"] + recovery_radius + leakage,
    )
    assert soft_vfirst["total_bound"] < soft_vfirst["old_total_bound"]

    lower_beta = fixture(beta=2.0)
    assert lower_beta["routes"]["direct_exact"]["total_bound"] == direct["total_bound"]
    assert lower_beta["routes"]["vfirst_nosplit_exact"]["total_bound"] == vfirst["total_bound"]


def verify_rejections_and_ordering() -> None:
    bad_margin = fixture(beta=0.0)
    assert bad_margin["routes"]["direct_softmax"]["status"] == STATUS_NOT_CERTIFIED
    assert bad_margin["routes"]["direct_softmax"]["failure_reasons"] == [
        "pair_kernel_margin_nonpositive"
    ]
    assert bad_margin["routes"]["direct_softmax"]["total_bound"] is None
    assert bad_margin["routes"]["direct_softmax"]["old_total_bound"] is None
    assert bad_margin["state_value"]["softmax"]["failure_reasons"] == [
        "state_kernel_margin_nonpositive"
    ]
    assert bad_margin["routes"]["vfirst_nosplit_softmax"]["failure_reasons"] == [
        "state_kernel_margin_nonpositive"
    ]

    multi = fixture(beta=0.0, fixed_context=False)
    assert multi["routes"]["direct_softmax"]["failure_reasons"] == [
        "algorithm_mode_mismatch",
        "pair_kernel_margin_nonpositive",
    ]
    mode = fixture(algorithm_mode="changing_policy")
    assert mode["routes"]["direct_exact"]["failure_reasons"] == [
        "algorithm_mode_mismatch"
    ]
    diverged = fixture(direct_exact_diverged=True)
    assert diverged["routes"]["direct_exact"]["failure_reasons"] == [
        "divergence_guard_triggered"
    ]

    missing = fixture(state_counts=[2, 6], pair_counts=[0, 2, 3, 3])
    assert missing["event"]["pair_bellman"]["mixture_radius_by_group"][0] is None
    assert missing["event"]["pair_bellman"]["old_radius_by_group"][0] is None
    for route in ("direct_exact", "direct_softmax", "vfirst_nosplit_exact"):
        assert missing["routes"][route]["status"] == STATUS_NOT_CERTIFIED
        assert "pair_support_missing" in missing["routes"][route]["failure_reasons"]
        assert missing["routes"][route]["total_bound"] is None

    # Deterministic inversion failures propagate as ordered non-emission
    # reasons and never fall back to the old radius.
    import time_uniform_mixture_certificate as module

    original = module.mixture_radius

    def boom(*args, **kwargs):
        raise MixtureInversionError("mixture_inversion_not_converged", "injected")

    try:
        module.mixture_radius = boom
        broken = fixture()
    finally:
        module.mixture_radius = original
    for route in ("direct_exact", "vfirst_nosplit_exact"):
        assert broken["routes"][route]["status"] == STATUS_NOT_CERTIFIED
        assert broken["routes"][route]["failure_reasons"] == [
            "mixture_inversion_not_converged"
        ]
        assert broken["routes"][route]["total_bound"] is None
        assert broken["routes"][route]["old_total_bound"] is None

    # Low-level inversion failure branches are deterministic, exercised by
    # local monkeypatches rather than by tampering with the frozen grid
    # (caller-supplied grids are validated field-by-field against it).
    grid = mixture_grid(FROZEN_GROUPS, FROZEN_DELTA)
    target = math.log(FROZEN_GROUPS / FROZEN_DELTA)
    original_log_mixture = module.log_mixture

    def inflated_log_mixture(count: int, q: float, *, grid: dict) -> float:
        return target + 1.0

    try:
        module.log_mixture = inflated_log_mixture
        expect_inversion_error(
            mixture_root,
            "mixture_inversion_unbracketed",
            3,
            n_groups=FROZEN_GROUPS,
            delta=FROZEN_DELTA,
            grid=grid,
        )
    finally:
        module.log_mixture = original_log_mixture
    original_stitch = module.stitch_boundary

    def tiny_stitch(count: int, *, grid: dict) -> float:
        return 1e-9

    try:
        module.stitch_boundary = tiny_stitch
        expect_inversion_error(
            mixture_root,
            "mixture_inversion_unbracketed",
            3,
            n_groups=FROZEN_GROUPS,
            delta=FROZEN_DELTA,
        )
    finally:
        module.stitch_boundary = original_stitch
    expect_inversion_error(
        mixture_root,
        "mixture_inversion_not_converged",
        3,
        n_groups=FROZEN_GROUPS,
        delta=FROZEN_DELTA,
        max_iterations=1,
    )


def verify_strict_json_and_oracle_separation() -> None:
    converted = strict_json_ready({"array": np.asarray([1.0, np.nan]), "inf": math.inf})
    assert converted == {"array": [1.0, None], "inf": None}
    json.dumps(converted, allow_nan=False)

    certificate = fixture()
    json.dumps(strict_json_ready(certificate), allow_nan=False)
    banned = {
        "occupancy",
        "transition_matrix",
        "spectral_gap",
        "q_pi",
        "v_pi",
        "true_residual",
        "true_initial_error",
        "oracle_audit",
    }

    def keys(value) -> set[str]:
        if isinstance(value, dict):
            result = {str(key).lower() for key in value}
            for child in value.values():
                result.update(keys(child))
            return result
        if isinstance(value, list):
            result: set[str] = set()
            for child in value:
                result.update(keys(child))
            return result
        return set()

    assert not (banned & keys(certificate))
    for route in ("direct_exact", "direct_softmax", "vfirst_nosplit_exact", "vfirst_nosplit_softmax"):
        item = certificate["routes"][route]
        assert item["status"] in (STATUS_SELECTIVE_HIGH_PROBABILITY, STATUS_NOT_CERTIFIED)
        assert item["selective_high_probability_certified"] == (
            item["status"] == STATUS_SELECTIVE_HIGH_PROBABILITY
        )

    def strict_loads(text: str):
        def no_duplicates(pairs):
            seen: set[str] = set()
            result = {}
            for key, value in pairs:
                if key in seen:
                    raise ValueError(f"duplicate key {key}")
                seen.add(key)
                result[key] = value
            return result

        def reject_constant(value: str) -> None:
            raise ValueError(f"non-finite constant {value}")

        return json.loads(
            text, object_pairs_hook=no_duplicates, parse_constant=reject_constant
        )

    strict_loads('{"a": 1, "b": [2, 3]}')
    for bad in ('{"a": 1, "a": 2}', '{"a": NaN}', '{"a": Infinity}', '{"a": -Infinity}'):
        try:
            strict_loads(bad)
        except ValueError:
            continue
        raise AssertionError(f"strict loader accepted {bad}")


def verify_unadjusted_posthoc_min_counterexample() -> None:
    # The mixture and the stitch are each valid at level delta. Selecting
    # their pointwise minimum after seeing the trajectory fails on the union
    # of the two failure events, whose mass can reach 2*delta.
    delta = 0.05
    failure_probabilities = np.asarray([delta, delta, 1.0 - 2.0 * delta])
    assert math.isclose(float(failure_probabilities.sum()), 1.0)
    selected_min_failure = float(failure_probabilities[:2].sum())
    assert selected_min_failure > delta
    assert math.isclose(selected_min_failure, 2.0 * delta)


def verify_transition_variance_obstruction() -> None:
    # Feasibility-only study (plan Task 8), narrowed conclusion: without extra
    # structure beyond the width-2B range, the uniform worst case over
    # data-consistent laws P and all targets V in [-B, B]^m attains B^2
    # (aligned corners saturate Popoviciu). This is a statement about the
    # structure-free uniform supremum only; it does NOT claim that
    # data-dependent adaptive tightening is impossible in general -- a
    # confidence-set plus Bellman-coupling construction is a separate,
    # currently uncompleted proof.
    gamma, reward = 0.7, 1.0
    bound_b = reward / (1.0 - gamma)
    values = np.asarray([reward + gamma * bound_b, -reward - gamma * bound_b])
    assert math.isclose(float(np.max(np.abs(values))), bound_b, rel_tol=1e-15)
    variance = float(np.var(values))
    assert math.isclose(variance, bound_b**2, rel_tol=1e-12)
    # The same saturation needs an unseen-successor variant: any mass on an
    # unobserved successor can carry an aligned corner value while every
    # observed successor is identical, so the data show zero variance.
    p_unseen = 0.01
    mixed = np.asarray([0.0, 0.0, bound_b])
    probs = np.asarray([0.5 * (1 - p_unseen), 0.5 * (1 - p_unseen), p_unseen])
    mean = float(probs @ mixed)
    mixed_variance = float(probs @ (mixed - mean) ** 2)
    assert 0.0 < mixed_variance <= bound_b**2 + 1e-15
    observed_only = float(np.var(mixed[:2]))
    assert observed_only == 0.0
    assert observed_only < mixed_variance  # unseen mass is invisible to data
    # But the unseen mass itself shrinks with sample confidence: a successor
    # of mass p stays unseen with probability (1 - p)^n, and the standard
    # high-probability ceiling on total unseen mass, log(1/delta)/n,
    # decreases in n. Unseen mass therefore cannot by itself rule out
    # data-dependent tightening at large n; the binding obstruction is the
    # fully observed aligned-corner configuration above.
    delta = 0.05
    sample_sizes = (10, 100, 1000, 10000)
    escape = [(1.0 - p_unseen) ** n for n in sample_sizes]
    ceilings = [math.log(1.0 / delta) / n for n in sample_sizes]
    for index in range(len(sample_sizes) - 1):
        assert escape[index + 1] < escape[index]
        assert ceilings[index + 1] < ceilings[index]
    assert escape[-1] < 1e-40
    assert ceilings[-1] < p_unseen
    # Popoviciu: every assignment within the width-2B range stays below B^2.
    rng = np.random.default_rng(20260904)
    for _ in range(2000):
        assignment = rng.uniform(-bound_b, bound_b, size=6)
        probs = rng.dirichlet(np.ones(6))
        mean = float(probs @ assignment)
        assert float(probs @ (assignment - mean) ** 2) <= bound_b**2 * (1.0 + 1e-12)


def main() -> None:
    verify_grid_weights_and_risk()
    verify_log_domain_stability()
    verify_conditional_mgf_fixtures()
    verify_initial_value_and_allocation()
    verify_exhaustive_counts()
    verify_high_precision_roots()
    verify_count_validation()
    verify_grid_isolation_and_iteration_cap()
    verify_route_composition()
    verify_rejections_and_ordering()
    verify_strict_json_and_oracle_separation()
    verify_unadjusted_posthoc_min_counterexample()
    verify_transition_variance_obstruction()
    print("PASS time-uniform mixture certificate contracts")


if __name__ == "__main__":
    main()
