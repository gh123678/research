"""Deterministic contract checks for the FP-TU-001 mixture certificate."""

from __future__ import annotations

import json
import math
from decimal import Decimal, localcontext

from time_uniform_mixture_certificate import (
    DEFAULT_COMPONENTS,
    DEFAULT_MAX_COUNT,
    DEFAULT_TOLERANCE,
    build_mixture_grid,
    build_time_uniform_certificate,
    log_cosh,
    log_mixture,
    logsumexp,
    mixture_radius,
    old_visit_indexed_radius,
    solve_mixture_boundary,
    stitch_boundary,
    strict_json_ready,
)


def _expect_value_error(function: object, *args: object, **kwargs: object) -> None:
    try:
        function(*args, **kwargs)  # type: ignore[operator]
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _base_certificate(**overrides: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "state_counts": [40, 60],
        "pair_counts": [20, 20, 30, 30],
        "trajectory_length": 100,
        "reward_bound": 1.0,
        "gamma": 0.7,
        "alpha": 0.65,
        "beta": 8.0,
        "delta": 0.05,
        "direct_exact_iterations": 20,
        "direct_softmax_iterations": 20,
        "state_exact_iterations": 20,
        "state_softmax_iterations": 20,
        "iteration_cap": 20,
    }
    arguments.update(overrides)
    return build_time_uniform_certificate(**arguments)  # type: ignore[arg-type]


def test_grid_and_stable_math() -> None:
    grid = build_mixture_grid(n_groups=54, delta=0.05)
    assert len(grid) == DEFAULT_COMPONENTS == 15
    assert [row["target_count"] for row in grid] == [2**j for j in range(15)]
    assert math.isclose(sum(float(row["weight"]) for row in grid), 1.0)
    assert all(float(row["weight"]) > 0.0 for row in grid)
    for j, row in enumerate(grid):
        expected_weight = (j + 1) ** -2 / sum((ell + 1) ** -2 for ell in range(15))
        assert math.isclose(float(row["weight"]), expected_weight, rel_tol=1e-15)
        expected_log = math.log(2.0 * 54.0 / (0.05 * expected_weight))
        assert math.isclose(float(row["log_level"]), expected_log, rel_tol=1e-15)
        assert math.isclose(
            float(row["rate"]),
            math.sqrt(2.0 * expected_log / (2**j)),
            rel_tol=1e-15,
        )

    for value in (0.0, 1e-12, 0.5, 40.0, -40.0, 1000.0):
        stable = log_cosh(value)
        assert math.isfinite(stable)
        if abs(value) < 40.0:
            assert math.isclose(stable, math.log(math.cosh(value)), abs_tol=1e-14)
    assert math.isclose(logsumexp([math.log(0.25), math.log(0.75)]), 0.0, abs_tol=1e-15)
    assert math.isfinite(logsumexp([-10000.0, -10001.0]))

    target = math.log(54.0 / 0.05)
    assert log_mixture(1, 0.0, grid) < target
    assert math.isclose(log_mixture(0, 0.0, grid), 0.0, abs_tol=1e-15)

    # Exhaustive bounded two-point fixtures: width two and conditional mean zero.
    for probability_index in range(1, 20):
        probability = probability_index / 20.0
        low = -2.0 * probability
        high = 2.0 * (1.0 - probability)
        for rate in (-4.0, -1.0, -0.1, 0.1, 1.0, 4.0):
            mgf = (1.0 - probability) * math.exp(rate * low) + probability * math.exp(rate * high)
            assert mgf <= math.exp(rate * rate / 2.0) * (1.0 + 1e-14)


def _decimal_log_mixture(
    count: int, boundary: Decimal, grid: list[dict[str, float | int]]
) -> Decimal:
    terms = []
    two = Decimal(2)
    for row in grid:
        weight = Decimal(str(row["weight"]))
        rate = Decimal(str(row["rate"]))
        cosh = ((rate * boundary).exp() + (-rate * boundary).exp()) / two
        terms.append(weight * (-(rate * rate) * Decimal(count) / two).exp() * cosh)
    return sum(terms, Decimal(0)).ln()


def test_independent_high_precision_roots() -> None:
    groups = 54
    delta = 0.05
    grid = build_mixture_grid(n_groups=groups, delta=delta)
    with localcontext() as context:
        context.prec = 60
        target = (Decimal(groups) / Decimal(str(delta))).ln()
        for count in (1, 2, 3, 7, 16, 31, 255, 256, 1000, 4096, 8191, 16384):
            lower = Decimal(0)
            upper = Decimal(str(stitch_boundary(count, grid)))
            for _ in range(220):
                midpoint = (lower + upper) / Decimal(2)
                if _decimal_log_mixture(count, midpoint, grid) >= target:
                    upper = midpoint
                else:
                    lower = midpoint
            production = float(
                solve_mixture_boundary(
                    count, grid, n_groups=groups, delta=delta
                )["boundary"]
            )
            reference = float(upper)
            assert production >= reference - 2e-12 * (1.0 + reference)
            assert abs(production - reference) <= 3e-12 * (1.0 + reference)


def test_solver_and_all_count_contract() -> None:
    groups = 54
    delta = 0.05
    grid = build_mixture_grid(n_groups=groups, delta=delta)
    target = math.log(groups / delta)
    previous_radius = math.inf
    strict = False
    selected = {1, 2, 3, 7, 16, 31, 256, 1000, 4096, 8191, 16384}
    for count in range(1, DEFAULT_MAX_COUNT + 1):
        stitch = stitch_boundary(count, grid)
        solution = solve_mixture_boundary(count, grid, n_groups=groups, delta=delta)
        q = float(solution["boundary"])
        radius = mixture_radius(
            count,
            reward_bound=1.0,
            gamma=0.7,
            n_groups=groups,
            delta=delta,
        )
        assert math.isfinite(q) and math.isfinite(radius)
        assert q <= stitch * (1.0 + 2e-12)
        assert float(solution["upper_gap"]) >= 0.0
        assert float(solution["upper_gap"]) <= DEFAULT_TOLERANCE * (1.0 + abs(target))
        assert float(solution["bracket_width"]) <= DEFAULT_TOLERANCE * (1.0 + q)
        assert int(solution["iterations"]) <= 200
        assert radius <= previous_radius * (1.0 + 2e-12)
        previous_radius = radius
        for horizon in (256, 1024, 4096, 16384):
            if count <= horizon:
                old = old_visit_indexed_radius(
                    count,
                    reward_bound=1.0,
                    gamma=0.7,
                    n_groups=groups,
                    horizon=horizon,
                    delta=delta,
                )
                assert radius <= old * (1.0 + 2e-12)
                strict = strict or radius < old * (1.0 - 1e-12)
        if count in selected:
            direct = sum(
                float(row["weight"])
                * math.exp(-float(row["rate"]) ** 2 * count / 2.0)
                * math.cosh(float(row["rate"]) * q)
                for row in grid
                if float(row["rate"]) * q < 700.0
            )
            if direct > 0.0 and math.isfinite(direct):
                assert math.log(direct) >= target - 1e-10
    assert strict


def test_builder_composition_and_preservation() -> None:
    certificate = _base_certificate()
    assert set(certificate) == {
        "certificate_inputs",
        "event",
        "state_value",
        "routes",
        "optional_variance_adaptive",
    }
    event = certificate["event"]
    assert isinstance(event, dict)
    assert event["method"] == "finite_geometric_time_uniform_cosh_mixture"
    assert event["n_groups"] == 10
    risk = event["risk_allocation"]
    assert isinstance(risk, dict)
    assert math.isclose(float(risk["per_group_failure_probability"]), 0.005)
    assert math.isclose(float(risk["total_failure_probability"]), 0.05)
    assert risk["route_level_resplit"] is False

    routes = certificate["routes"]
    state = certificate["state_value"]
    assert isinstance(routes, dict) and isinstance(state, dict)
    for route in (*routes.values(), *state.values()):
        assert isinstance(route, dict)
        assert route["status"] == "selective_high_probability_certified"
        assert route["failure_reasons"] == []
        assert route["finite_bound_emitted"] is True
        assert math.isfinite(float(route["total_bound"]))

    missing = _base_certificate(
        state_counts=[100, 0], pair_counts=[50, 50, 0, 0]
    )
    missing_routes = missing["routes"]
    assert isinstance(missing_routes, dict)
    assert missing_routes["direct_exact"]["failure_reasons"] == ["pair_support_missing"]
    assert missing_routes["vfirst_nosplit_exact"]["failure_reasons"] == [
        "state_support_missing",
        "pair_support_missing",
    ]

    mismatch = _base_certificate(fixed_context=False)
    mismatch_routes = mismatch["routes"]
    assert isinstance(mismatch_routes, dict)
    assert mismatch_routes["direct_exact"]["failure_reasons"] == [
        "algorithm_mode_mismatch"
    ]

    payload = strict_json_ready(certificate)
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
    assert "NaN" not in encoded and "Infinity" not in encoded
    assert "oracle" not in encoded.lower()


def test_input_failures_and_posthoc_minimum_counterexample() -> None:
    _expect_value_error(build_mixture_grid, n_groups=0, delta=0.05)
    _expect_value_error(build_mixture_grid, n_groups=54, delta=1.0)
    _expect_value_error(mixture_radius, 0, reward_bound=1.0, gamma=0.7, n_groups=54, delta=0.05)
    _expect_value_error(_base_certificate, trajectory_length=100.5)
    _expect_value_error(_base_certificate, state_counts=[50, 49])
    _expect_value_error(_base_certificate, pair_counts=[20, 20, 30, 29])
    _expect_value_error(_base_certificate, gamma=1.0)
    _expect_value_error(_base_certificate, delta=float("nan"))
    _expect_value_error(_base_certificate, iteration_cap=19)

    # Two independently valid level-delta rules cannot be minimized without
    # allocating risk: under independence their union exceeds delta.
    delta = 0.05
    assert 1.0 - (1.0 - delta) ** 2 > delta


def main() -> None:
    test_grid_and_stable_math()
    test_independent_high_precision_roots()
    test_solver_and_all_count_contract()
    test_builder_composition_and_preservation()
    test_input_failures_and_posthoc_minimum_counterexample()
    print("time-uniform mixture certificate checks passed")


if __name__ == "__main__":
    main()
