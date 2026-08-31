"""Exact-model Markov, coverage, and one-hot kernel diagnostics.

The high-probability coverage field implements the stationary, time-independent
Hoeffding bound of Fan, Jiang, and Sun (2021). Mixing-gap fields are dependency
diagnostics; they are not presented as a completed cross-fitting theorem.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def validate_transition(transition: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transition, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition must be a square matrix")
    if not np.all(np.isfinite(matrix)) or np.min(matrix) < -1e-12:
        raise ValueError("transition must be finite and nonnegative")
    matrix = np.maximum(matrix, 0.0)
    row_sums = matrix.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("every transition row must have positive mass")
    matrix /= row_sums
    return matrix


def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    matrix = validate_transition(transition)
    n_states = matrix.shape[0]
    system = matrix.T - np.eye(n_states)
    right_hand_side = np.zeros(n_states)
    system[-1] = 1.0
    right_hand_side[-1] = 1.0
    stationary = np.linalg.solve(system, right_hand_side)
    stationary = np.maximum(stationary, 0.0)
    stationary /= stationary.sum()
    residual = float(np.max(np.abs(stationary @ matrix - stationary)))
    if residual > 1e-9:
        raise ValueError(f"stationary solve residual too large: {residual}")
    return stationary


def stationary_time_reversal(
    transition: np.ndarray, stationary: np.ndarray | None = None
) -> np.ndarray:
    matrix = validate_transition(transition)
    mu = (
        stationary_distribution(matrix)
        if stationary is None
        else np.asarray(stationary, dtype=np.float64)
    )
    if mu.shape != (matrix.shape[0],) or np.min(mu) <= 0.0:
        raise ValueError("stationary distribution must have full positive support")
    reverse = matrix.T * mu[None, :] / mu[:, None]
    reverse = validate_transition(reverse)
    detailed_balance_residual = float(
        np.max(
            np.abs(
                mu[:, None] * matrix
                - mu[None, :] * reverse.T
            )
        )
    )
    if detailed_balance_residual > 1e-9:
        raise AssertionError("time-reversal stationary-flow identity failed")
    return reverse


def edge_chain_transition(
    transition: np.ndarray,
    stationary: np.ndarray | None = None,
    support_tolerance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the positive-support chain E_t=(X_t,X_{t+1})."""
    matrix = validate_transition(transition)
    if not np.isfinite(support_tolerance) or support_tolerance < 0.0:
        raise ValueError("support_tolerance must be finite and nonnegative")
    mu = (
        stationary_distribution(matrix)
        if stationary is None
        else np.asarray(stationary, dtype=np.float64)
    )
    edge_mass = mu[:, None] * matrix
    edges = np.argwhere(edge_mass > support_tolerance)
    if edges.size == 0:
        raise ValueError("edge chain has empty stationary support")
    edge_index = {
        (int(source), int(target)): index
        for index, (source, target) in enumerate(edges)
    }
    edge_transition = np.zeros((len(edges), len(edges)), dtype=np.float64)
    for source_index, (_, middle) in enumerate(edges):
        for target in np.flatnonzero(matrix[middle] > support_tolerance):
            target_index = edge_index.get((int(middle), int(target)))
            if target_index is not None:
                edge_transition[source_index, target_index] = matrix[middle, target]
    edge_transition = validate_transition(edge_transition)
    edge_stationary = edge_mass[edges[:, 0], edges[:, 1]]
    edge_stationary /= edge_stationary.sum()
    residual = float(
        np.max(
            np.abs(edge_stationary @ edge_transition - edge_stationary)
        )
    )
    if residual > 1e-9:
        raise AssertionError("edge-chain stationary formula failed")
    return edge_transition, edge_stationary, edges.astype(np.int64)


def edge_support_metadata(
    transition: np.ndarray,
    stationary: np.ndarray | None = None,
    support_tolerance: float = 0.0,
) -> dict[str, float | int | bool]:
    """Describe whether a numerical threshold truncates positive edge mass."""
    matrix = validate_transition(transition)
    mu = (
        stationary_distribution(matrix)
        if stationary is None
        else np.asarray(stationary, dtype=np.float64)
    )
    if not np.isfinite(support_tolerance) or support_tolerance < 0.0:
        raise ValueError("support_tolerance must be finite and nonnegative")
    edge_mass = mu[:, None] * matrix
    positive = edge_mass > 0.0
    retained = edge_mass > support_tolerance
    discarded = positive & ~retained
    return {
        "support_tolerance": float(support_tolerance),
        "positive_edge_count": int(np.sum(positive)),
        "retained_edge_count": int(np.sum(retained)),
        "discarded_positive_edge_count": int(np.sum(discarded)),
        "discarded_positive_mass": float(np.sum(edge_mass[discarded])),
        "numerical_support_truncated": bool(np.any(discarded)),
    }


def dobrushin_coefficient(transition: np.ndarray) -> float:
    matrix = validate_transition(transition)
    row_distances = 0.5 * np.abs(
        matrix[:, None, :] - matrix[None, :, :]
    ).sum(axis=2)
    return float(np.max(row_distances))


def _orthogonal_basis(stationary_vector: np.ndarray) -> np.ndarray:
    n_states = stationary_vector.size
    if n_states == 1:
        return np.empty((1, 0), dtype=np.float64)
    candidates = np.column_stack(
        [stationary_vector, np.eye(n_states, dtype=np.float64)]
    )
    orthogonal, _ = np.linalg.qr(candidates)
    return orthogonal[:, 1:]


def spectral_certificate(
    transition: np.ndarray,
    stationary: np.ndarray | None = None,
    include_absolute: bool = True,
) -> dict[str, float | None]:
    """Compute finite-state right and optional absolute spectral factors."""
    matrix = validate_transition(transition)
    mu = (
        stationary_distribution(matrix)
        if stationary is None
        else np.asarray(stationary, dtype=np.float64)
    )
    if np.min(mu) <= 0.0:
        raise ValueError("spectral certificate needs positive stationary support")
    root = np.sqrt(mu)
    similarity = root[:, None] * matrix / root[None, :]
    if matrix.shape[0] == 1:
        lambda_right = 0.0
        lambda_absolute = 0.0
    elif matrix.shape[0] > 128 and not include_absolute:
        reversiblization = csr_matrix(0.5 * (similarity + similarity.T))
        leading = eigsh(
            reversiblization,
            k=2,
            which="LA",
            return_eigenvectors=False,
        )
        lambda_right = float(np.sort(leading)[-2])
        lambda_absolute = None
    else:
        basis = _orthogonal_basis(root)
        reversiblization = 0.5 * (similarity + similarity.T)
        restricted_right = basis.T @ reversiblization @ basis
        lambda_right = float(np.linalg.eigvalsh(restricted_right)[-1])
        if include_absolute:
            restricted = basis.T @ similarity @ basis
            lambda_absolute = float(np.linalg.svd(restricted, compute_uv=False)[0])
        else:
            lambda_absolute = None

    lambda_right = float(np.clip(lambda_right, -1.0, 1.0))
    positive_right = max(lambda_right, 0.0)
    right_inflation = (
        float("inf")
        if positive_right >= 1.0 - 1e-14
        else (1.0 + positive_right) / (1.0 - positive_right)
    )
    if include_absolute:
        assert lambda_absolute is not None
        lambda_absolute = float(np.clip(lambda_absolute, 0.0, 1.0))
        absolute_inflation = (
            float("inf")
            if lambda_absolute >= 1.0 - 1e-14
            else (1.0 + lambda_absolute) / (1.0 - lambda_absolute)
        )
    else:
        absolute_inflation = None
    return {
        "lambda_right": lambda_right,
        "right_spectral_gap": 1.0 - lambda_right,
        "right_hoeffding_inflation": right_inflation,
        "lambda_absolute": lambda_absolute,
        "absolute_spectral_gap": (
            None if not include_absolute else 1.0 - lambda_absolute
        ),
        "absolute_hoeffding_inflation": absolute_inflation,
    }


def total_variation_mixing_curve(
    transition: np.ndarray,
    stationary: np.ndarray | None = None,
    max_steps: int = 4096,
) -> np.ndarray:
    matrix = validate_transition(transition)
    mu = (
        stationary_distribution(matrix)
        if stationary is None
        else np.asarray(stationary, dtype=np.float64)
    )
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    distribution = np.eye(matrix.shape[0], dtype=np.float64)
    curve = np.empty(max_steps + 1, dtype=np.float64)
    for step in range(max_steps + 1):
        curve[step] = float(
            np.max(0.5 * np.abs(distribution - mu[None, :]).sum(axis=1))
        )
        if step < max_steps:
            distribution = distribution @ matrix
    return curve


def mixing_time(curve: np.ndarray, threshold: float = 0.25) -> int | None:
    indices = np.flatnonzero(np.asarray(curve) <= threshold)
    return None if indices.size == 0 else int(indices[0])


def total_variation_at_step(
    transition: np.ndarray, stationary: np.ndarray, step: int
) -> float:
    if step < 0:
        raise ValueError("step must be nonnegative")
    powered = np.linalg.matrix_power(transition, step)
    return float(
        np.max(0.5 * np.abs(powered - stationary[None, :]).sum(axis=1))
    )


def _first_step_below(
    distance_at_step: Any, threshold: float, max_steps: int
) -> int | None:
    if distance_at_step(0) <= threshold:
        return 0
    upper = 1
    while upper < max_steps and distance_at_step(upper) > threshold:
        upper = min(2 * upper, max_steps)
    if distance_at_step(upper) > threshold:
        return None
    lower = upper // 2
    while lower + 1 < upper:
        middle = (lower + upper) // 2
        if distance_at_step(middle) <= threshold:
            upper = middle
        else:
            lower = middle
    return upper


def suggest_crossfit_gap(
    state_transition: np.ndarray,
    pair_transition: np.ndarray,
    trajectory_length: int,
    delta: float = 0.05,
    search_steps: int | None = None,
) -> dict[str, Any]:
    """Suggest a gap from exact forward/reverse TV curves.

    Edge-chain curves equal the corresponding pair-chain curve shifted by one
    step, so no large edge transition powers are materialized here.
    """
    if trajectory_length < 4:
        raise ValueError("trajectory_length must be at least 4")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0,1)")
    state = validate_transition(state_transition)
    pair = validate_transition(pair_transition)
    mu_state = stationary_distribution(state)
    mu_pair = stationary_distribution(pair)
    reverse_state = stationary_time_reversal(state, mu_state)
    reverse_pair = stationary_time_reversal(pair, mu_pair)
    cap = trajectory_length // 8
    steps = search_steps or max(4096, 2 * cap + 1)

    def state_forward(step: int) -> float:
        return total_variation_at_step(state, mu_state, step)

    def state_reverse_distance(step: int) -> float:
        return total_variation_at_step(reverse_state, mu_state, step)

    def pair_forward(step: int) -> float:
        return total_variation_at_step(pair, mu_pair, step)

    def pair_reverse_distance(step: int) -> float:
        return total_variation_at_step(reverse_pair, mu_pair, step)

    def edge_forward(step: int) -> float:
        return 1.0 if step == 0 else pair_forward(step - 1)

    def edge_reverse_distance(step: int) -> float:
        return 1.0 if step == 0 else pair_reverse_distance(step - 1)

    distances = (
        state_forward,
        state_reverse_distance,
        pair_forward,
        pair_reverse_distance,
        edge_forward,
        edge_reverse_distance,
    )

    def worst_distance(step: int) -> float:
        return max(distance(step) for distance in distances)

    threshold = delta / (4.0 * trajectory_length)
    gap_raw = _first_step_below(worst_distance, threshold, steps)
    gap_capped = gap_raw is None or gap_raw > cap
    gap_used = cap if gap_capped else int(gap_raw)
    return {
        "delta": delta,
        "dependency_tv_threshold": threshold,
        "gap_raw": gap_raw,
        "gap_used": gap_used,
        "gap_cap": cap,
        "gap_capped": gap_capped,
        "dependency_tv_at_gap": worst_distance(gap_used),
        "state_mixing_time_quarter": _first_step_below(
            state_forward, 0.25, steps
        ),
        "pair_mixing_time_quarter": _first_step_below(
            pair_forward, 0.25, steps
        ),
        "edge_mixing_time_quarter": _first_step_below(
            edge_forward, 0.25, steps
        ),
        "state_reverse_mixing_time_quarter": _first_step_below(
            state_reverse_distance, 0.25, steps
        ),
        "pair_reverse_mixing_time_quarter": _first_step_below(
            pair_reverse_distance, 0.25, steps
        ),
        "edge_reverse_mixing_time_quarter": _first_step_below(
            edge_reverse_distance, 0.25, steps
        ),
    }


def kernel_coverage_certificate(
    pair_stationary: np.ndarray,
    pair_counts: np.ndarray,
    trajectory_length: int,
    beta: float,
    gamma: float,
    right_hoeffding_inflation: float,
    delta: float = 0.05,
) -> dict[str, Any]:
    mu = np.asarray(pair_stationary, dtype=np.float64)
    counts = np.asarray(pair_counts, dtype=np.int64)
    if mu.ndim != 1 or counts.shape != mu.shape:
        raise ValueError("pair stationary distribution and counts must align")
    if trajectory_length <= 0 or counts.sum() != trajectory_length:
        raise ValueError("pair counts must sum to trajectory_length")
    if np.min(mu) <= 0.0:
        raise ValueError("pair stationary distribution must be positive")
    sharp = float(np.exp(beta))
    population_diagonal = mu * sharp / (mu * sharp + 1.0 - mu)
    empirical_diagonal = counts * sharp / (
        counts * sharp + trajectory_length - counts
    )
    threshold = (1.0 + gamma) / 2.0
    required_beta = np.log(
        threshold * (1.0 - mu) / ((1.0 - threshold) * mu)
    )
    if np.isfinite(right_hoeffding_inflation):
        coverage_radius = float(
            np.sqrt(
                right_hoeffding_inflation
                * np.log(2.0 * mu.size / delta)
                / (2.0 * trajectory_length)
            )
        )
    else:
        coverage_radius = float("inf")
    population_slack = float(np.min(population_diagonal) - threshold)
    empirical_slack = float(np.min(empirical_diagonal) - threshold)
    return {
        "pair_stationary_min": float(np.min(mu)),
        "coverage_radius": coverage_radius,
        "coverage_certified": bool(np.min(mu) > coverage_radius),
        "certified_pair_count_lower": float(
            trajectory_length * max(float(np.min(mu)) - coverage_radius, 0.0)
        ),
        "missing_pairs": int(np.sum(counts == 0)),
        "min_pair_count": int(np.min(counts)),
        "required_beta_max": float(np.max(required_beta)),
        "population_kernel_diagonal_min": float(
            np.min(population_diagonal)
        ),
        "empirical_kernel_diagonal_min": float(np.min(empirical_diagonal)),
        "population_kernel_threshold_slack": population_slack,
        "empirical_kernel_threshold_slack": empirical_slack,
        "kernel_certified": bool(
            population_slack > 0.0 and empirical_slack > 0.0
        ),
        "direct_operator_diagnostic": True,
    }


def build_markov_base_certificate(
    state_transition: np.ndarray,
    pair_transition: np.ndarray,
    edge_support_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Compute trajectory-length-independent exact-model diagnostics."""
    state = validate_transition(state_transition)
    pair = validate_transition(pair_transition)
    mu_state = stationary_distribution(state)
    mu_pair = stationary_distribution(pair)
    edge, mu_edge, _ = edge_chain_transition(
        pair, mu_pair, support_tolerance=edge_support_tolerance
    )
    support = edge_support_metadata(
        pair, mu_pair, support_tolerance=edge_support_tolerance
    )
    return {
        "state_dobrushin": dobrushin_coefficient(state),
        "pair_dobrushin": dobrushin_coefficient(pair),
        # Different middle states have disjoint outgoing edge supports.
        "edge_dobrushin": 0.0 if edge.shape[0] == 1 else 1.0,
        "state": spectral_certificate(state, mu_state),
        "pair": spectral_certificate(pair, mu_pair),
        "edge": spectral_certificate(
            edge, mu_edge, include_absolute=False
        ),
        "stationary_support": {
            "state_stationary_min": float(np.min(mu_state)),
            "pair_stationary_min": float(np.min(mu_pair)),
            "edge_stationary_min": float(np.min(mu_edge)),
        },
        "edge_support": support,
    }


def build_markov_coverage_certificate(
    state_transition: np.ndarray,
    pair_transition: np.ndarray,
    pair_counts: np.ndarray,
    trajectory_length: int,
    beta: float,
    gamma: float,
    delta: float = 0.05,
    base_certificate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = validate_transition(state_transition)
    pair = validate_transition(pair_transition)
    mu_pair = stationary_distribution(pair)
    base = (
        build_markov_base_certificate(state, pair)
        if base_certificate is None
        else base_certificate
    )
    gap = suggest_crossfit_gap(
        state, pair, trajectory_length, delta=delta
    )
    kernel = kernel_coverage_certificate(
        mu_pair,
        pair_counts,
        trajectory_length,
        beta,
        gamma,
        base["pair"]["right_hoeffding_inflation"],
        delta=delta,
    )
    return {
        **base,
        "gap": gap,
        "kernel_coverage": kernel,
        "certificate_scope": (
            "stationary pair-count Hoeffding plus kernel diagnostics; "
            "not a complete iterative or cross-fitting theorem"
        ),
    }
