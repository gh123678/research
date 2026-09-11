"""Reference implementation of fixed-policy grouped Expected SARSA (FP-ESARSA-001).

Pure-numpy reference for the three frozen routes plus the residual certificate
and the relative-softmax improvement decision rule. This module is the
implementation half of the tests-first pair; the executable contract is
`verify_fixed_policy_expected_sarsa.py`.

Routes
------
- ``run_expected_exact``   exact grouped Expected SARSA (primary);
- ``run_expected_finite``  finite-logit grouped Expected SARSA (primary);
- ``run_sampled_exact``    sampled SARSA control.

The exact route replaces the sampled next action by the exact policy
expectation ``sum_b pi(b|s') Q(s',b)``. The finite route replaces every exact
one-hot retrieval with a finite-sharpness softmax over the full token memory,
with no equality mask and no visited gate, so unvisited queries receive a
reported leakage update.
"""

from __future__ import annotations

import math

import numpy as np

from time_uniform_mixture_certificate import (
    DEFAULT_TOLERANCE,
    MixtureInversionError,
    build_mixture_grid,
    mixture_radius,
    solve_mixture_boundary,
)

ALGORITHM_MODE = "on_policy_synchronous_expected_sarsa"
ETA_GRID = (1.0, 0.5, 0.2, 0.1, 0.05)

_REASON_ORDER = (
    "algorithm_mode_mismatch",
    "duplicate_q_memory",
    "divergence_guard_triggered",
    "heldout_pair_support_missing",
    "mixture_inversion_unbracketed",
    "mixture_inversion_not_converged",
    "mixture_root_not_conservative",
    "numerical_nonfinite",
    "policy_invalid",
    "improvement_lcb_nonpositive",
    "policy_unchanged",
)
_REASON_RANK = {reason: index for index, reason in enumerate(_REASON_ORDER)}


# --------------------------------------------------------------------------- #
# Canonical Q-memory
# --------------------------------------------------------------------------- #
def canonical_pair_ids(n_states: int, n_actions: int) -> list[tuple[int, int]]:
    """Canonical row-major list of state-action identifiers."""
    return [(state, action) for state in range(n_states) for action in range(n_actions)]


def canonical_pair_index(state: int, action: int, n_actions: int) -> int:
    """Flat pair index ``state * n_actions + action``."""
    return int(state) * int(n_actions) + int(action)


def _as_pair(entry) -> tuple[int, int]:
    values = tuple(entry)
    if len(values) != 2:
        raise ValueError(f"pair identifier must have two entries, got {entry!r}")
    return (int(values[0]), int(values[1]))


def validate_canonical_memory(pairs, q_values=None) -> list[tuple[int, int]]:
    """Return the identifiers unchanged, rejecting duplicate logical pairs.

    Duplicate detection is by identifier, never by numerical value: two pairs
    that happen to carry equal Q entries remain distinct memory slots.
    """
    ordered: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for entry in pairs:
        pair = _as_pair(entry)
        if pair in seen:
            raise ValueError(f"duplicate pair {pair} in Q memory")
        seen.add(pair)
        ordered.append(pair)
    if q_values is not None and not isinstance(q_values, np.ndarray):
        raise ValueError("q_values must be a numpy array when supplied")
    return ordered


# --------------------------------------------------------------------------- #
# Finite-logit retrieval formulas (pure functions, vectorised)
# --------------------------------------------------------------------------- #
def kappa_state(n_states: int, zeta: float) -> float:
    """Successor softmax mass on the matched next-state action block."""
    e = math.exp(float(zeta))
    return e / (e + float(n_states) - 1.0)


def kappa_read(n_pairs: int, xi: float) -> float:
    """Read softmax mass on the matched memory token."""
    e = math.exp(float(xi))
    return e / (e + float(n_pairs) - 1.0)


def finite_successor_all(q: np.ndarray, policy: np.ndarray, zeta: float) -> np.ndarray:
    """Finite successor expectation per next state.

    For query state ``s`` the score of memory token ``(u, b)`` is
    ``zeta * 1{u = s} + log pi(b|u)`` and the softmax runs over the full
    ``n_states * n_actions`` token set.
    """
    q = np.asarray(q, dtype=np.float64)
    policy = np.asarray(policy, dtype=np.float64)
    n_states, n_actions = q.shape
    log_pi = np.log(policy).reshape(-1)
    values = q.reshape(-1)
    result = np.zeros(n_states, dtype=np.float64)
    for state in range(n_states):
        scores = log_pi.copy()
        scores[state * n_actions : (state + 1) * n_actions] += float(zeta)
        scores -= scores.max()
        weights = np.exp(scores)
        result[state] = float((weights * values).sum() / weights.sum())
    return result


def finite_read_all(q: np.ndarray, xi: float) -> np.ndarray:
    """Finite read of the full memory for every pair token.

    Read score of token ``y`` at query pair ``x`` is ``xi * 1{x = y}``; the
    softmax runs over the full ``n_pairs`` token set.
    """
    flat = np.asarray(q, dtype=np.float64).reshape(-1)
    n = flat.size
    e = math.exp(float(xi))
    total = flat.sum()
    reads = (e * flat + (total - flat)) / (e + n - 1.0)
    return reads.reshape(np.asarray(q).shape)


def finite_writeback_all(
    residuals: np.ndarray,
    current_pairs: np.ndarray,
    n_pairs: int,
    tau: float,
    alpha: float,
) -> np.ndarray:
    """Finite write-back over all queries, including unvisited ones.

    ``w_t(x) = e^{tau 1{X_t = x}} / (N_x e^tau + (m - N_x))`` with ``m`` the
    number of transitions; the update is ``alpha * sum_t w_t(x) delta_t``. No
    visited gate is applied, so an unvisited query still moves by the reported
    leakage term.
    """
    residuals = np.asarray(residuals, dtype=np.float64).reshape(-1)
    current_pairs = np.asarray(current_pairs, dtype=np.int64).reshape(-1)
    m = residuals.size
    e_tau = math.exp(float(tau))
    counts = np.bincount(current_pairs, minlength=n_pairs).astype(np.float64)
    sums = np.bincount(current_pairs, weights=residuals, minlength=n_pairs)
    total = residuals.sum()
    denom = counts * e_tau + (m - counts)
    return float(alpha) * (e_tau * sums + (total - sums)) / denom


def exact_action_head_candidates(next_state: int, n_actions: int) -> list[tuple[int, int]]:
    """Action-head candidate tokens for a successor query, each pair once."""
    return [(int(next_state), action) for action in range(int(n_actions))]


# --------------------------------------------------------------------------- #
# Exact grouped Expected SARSA and the sampled control
# --------------------------------------------------------------------------- #
def _validate_route_inputs(q0, policy, states, actions, rewards, next_states):
    q0 = np.asarray(q0, dtype=np.float64)
    policy = np.asarray(policy, dtype=np.float64)
    if q0.ndim != 2:
        raise ValueError("q0 must have shape (n_states, n_actions)")
    if policy.shape != q0.shape:
        raise ValueError("policy must share the Q shape")
    states = np.asarray(states, dtype=np.int64).reshape(-1)
    actions = np.asarray(actions, dtype=np.int64).reshape(-1)
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    next_states = np.asarray(next_states, dtype=np.int64).reshape(-1)
    lengths = {states.size, actions.size, rewards.size, next_states.size}
    if len(lengths) != 1:
        raise ValueError("transition fields must have equal length")
    return q0, policy, states, actions, rewards, next_states


def run_expected_exact(
    q0,
    policy,
    states,
    actions,
    rewards,
    next_states,
    gamma,
    alpha,
    layers,
    value_bound,
) -> dict:
    """Synchronous grouped Expected SARSA on the exact policy expectation."""
    q, policy, states, actions, rewards, next_states = _validate_route_inputs(
        q0, policy, states, actions, rewards, next_states
    )
    n_states, n_actions = q.shape
    n_pairs = n_states * n_actions
    current = states * n_actions + actions
    counts = np.bincount(current, minlength=n_pairs).astype(np.float64)

    snapshots: list[np.ndarray] = []
    residuals_by_layer: list[np.ndarray] = []
    diverged = False
    for _ in range(int(layers)):
        qbar = (policy[next_states] * q[next_states]).sum(axis=1)
        residuals = rewards + float(gamma) * qbar - q[states, actions]
        accumulated = np.bincount(current, weights=residuals, minlength=n_pairs)
        update = np.zeros(n_pairs, dtype=np.float64)
        visited = counts > 0
        update[visited] = float(alpha) * accumulated[visited] / counts[visited]
        q = q + update.reshape(n_states, n_actions)
        snapshots.append(q.copy())
        residuals_by_layer.append(residuals)
        if not np.all(np.isfinite(q)) or float(np.max(np.abs(q))) > float(value_bound):
            diverged = True
            break

    return {
        "layer_snapshots": snapshots,
        "residuals_by_layer": residuals_by_layer,
        "q_hat": q,
        "diverged": diverged,
        "train_pair_counts": np.bincount(current, minlength=n_pairs),
    }


def run_sampled_exact(
    q0,
    policy,
    states,
    actions,
    rewards,
    next_states,
    next_actions,
    gamma,
    alpha,
    layers,
    value_bound,
) -> dict:
    """Sampled SARSA control: the successor uses the recorded next action."""
    q, policy, states, actions, rewards, next_states = _validate_route_inputs(
        q0, policy, states, actions, rewards, next_states
    )
    next_actions = np.asarray(next_actions, dtype=np.int64).reshape(-1)
    n_states, n_actions = q.shape
    n_pairs = n_states * n_actions
    current = states * n_actions + actions
    following = next_states * n_actions + next_actions
    counts = np.bincount(current, minlength=n_pairs).astype(np.float64)

    snapshots: list[np.ndarray] = []
    residuals_by_layer: list[np.ndarray] = []
    diverged = False
    for _ in range(int(layers)):
        flat = q.reshape(-1)
        residuals = rewards + float(gamma) * flat[following] - flat[current]
        accumulated = np.bincount(current, weights=residuals, minlength=n_pairs)
        update = np.zeros(n_pairs, dtype=np.float64)
        visited = counts > 0
        update[visited] = float(alpha) * accumulated[visited] / counts[visited]
        q = q + update.reshape(n_states, n_actions)
        snapshots.append(q.copy())
        residuals_by_layer.append(residuals)
        if not np.all(np.isfinite(q)) or float(np.max(np.abs(q))) > float(value_bound):
            diverged = True
            break

    return {
        "layer_snapshots": snapshots,
        "residuals_by_layer": residuals_by_layer,
        "q_hat": q,
        "diverged": diverged,
        "train_pair_counts": np.bincount(current, minlength=n_pairs),
    }


def run_expected_finite(
    q0,
    policy,
    states,
    actions,
    rewards,
    next_states,
    gamma,
    alpha,
    layers,
    value_bound,
    zeta,
    xi,
    tau,
) -> dict:
    """Finite-logit grouped Expected SARSA: finite retrieval and write-back."""
    q, policy, states, actions, rewards, next_states = _validate_route_inputs(
        q0, policy, states, actions, rewards, next_states
    )
    n_states, n_actions = q.shape
    n_pairs = n_states * n_actions
    current = states * n_actions + actions

    snapshots: list[np.ndarray] = []
    residuals_by_layer: list[np.ndarray] = []
    diverged = False
    for _ in range(int(layers)):
        reads = finite_read_all(q, xi).reshape(-1)[current]
        successor = finite_successor_all(q, policy, zeta)[next_states]
        residuals = rewards + float(gamma) * successor - reads
        update = finite_writeback_all(residuals, current, n_pairs, tau, alpha)
        q = q + update.reshape(n_states, n_actions)
        snapshots.append(q.copy())
        residuals_by_layer.append(residuals)
        if not np.all(np.isfinite(q)) or float(np.max(np.abs(q))) > float(value_bound):
            diverged = True
            break

    return {
        "layer_snapshots": snapshots,
        "residuals_by_layer": residuals_by_layer,
        "q_hat": q,
        "diverged": diverged,
        "train_pair_counts": np.bincount(current, minlength=n_pairs),
    }


# --------------------------------------------------------------------------- #
# Residual certificate
# --------------------------------------------------------------------------- #
def contraction_premise_satisfied(kernel: np.ndarray, gamma: float) -> bool:
    """Whether the diagonal margin ``min_x M_xx > (1 + gamma) / 2`` holds."""
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be square")
    margin = float(np.diag(kernel).min()) - 0.5 * (1.0 + float(gamma))
    return bool(margin > 0.0)


def mixture_pair_radius(
    count: int,
    *,
    reward_bound: float,
    gamma: float,
    n_groups: int,
    delta: float,
) -> float:
    """Per-pair radius ``2 B q_mix(N_x) / N_x`` with ``B = R* / (1 - gamma)``."""
    return 2.0 * mixture_radius(
        count,
        reward_bound=reward_bound,
        gamma=gamma,
        n_groups=n_groups,
        delta=delta,
    )


def build_residual_certificate(
    q_hat,
    policy,
    states,
    actions,
    rewards,
    next_states,
    reward_bound,
    gamma,
    delta,
    algorithm_mode=ALGORITHM_MODE,
    pair_ids=None,
    mixture_tolerance=DEFAULT_TOLERANCE,
) -> dict:
    """Held-out residual certificate for a fixed-policy routed Q estimate.

    The certificate only observes ``q_hat``, the input policy, the held-out
    transition fields, the reward bound, the discount, and the mixture
    tolerance. It never reads the model transition kernel or the exact value
    function.
    """
    q_hat = np.asarray(q_hat, dtype=np.float64)
    policy = np.asarray(policy, dtype=np.float64)
    if q_hat.ndim != 2:
        raise ValueError("q_hat must have shape (n_states, n_actions)")
    if policy.shape != q_hat.shape:
        raise ValueError("policy must share the Q shape")
    n_states, n_actions = q_hat.shape
    n_groups = n_states * n_actions

    if pair_ids is not None:
        validate_canonical_memory(pair_ids, q_values=q_hat)

    states = np.asarray(states, dtype=np.int64).reshape(-1)
    actions = np.asarray(actions, dtype=np.int64).reshape(-1)
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    next_states = np.asarray(next_states, dtype=np.int64).reshape(-1)

    value_bound = float(reward_bound) / (1.0 - float(gamma))
    reasons: list[str] = []

    if algorithm_mode != ALGORITHM_MODE:
        reasons.append("algorithm_mode_mismatch")

    finite = bool(np.all(np.isfinite(q_hat)))
    if finite and float(np.max(np.abs(q_hat))) > value_bound:
        reasons.append("divergence_guard_triggered")

    current = states * n_actions + actions
    counts = np.bincount(current, minlength=n_groups)
    support_ok = bool(counts.size == n_groups and np.all(counts > 0))
    if not support_ok:
        reasons.append("heldout_pair_support_missing")

    if states.size:
        successor_mean = (policy[next_states] * q_hat[next_states]).sum(axis=1)
        residuals = rewards + float(gamma) * successor_mean - q_hat[states, actions]
    else:
        residuals = np.zeros(0, dtype=np.float64)

    residual_means = np.zeros(n_groups, dtype=np.float64)
    radii = np.zeros(n_groups, dtype=np.float64)
    grid = build_mixture_grid(n_groups=n_groups, delta=float(delta))
    for pair in range(n_groups):
        count = int(counts[pair]) if counts.size == n_groups else 0
        if count <= 0:
            continue
        residual_means[pair] = float(residuals[current == pair].mean())
        try:
            solution = solve_mixture_boundary(
                count,
                grid,
                n_groups=n_groups,
                delta=float(delta),
                tolerance=mixture_tolerance,
            )
        except MixtureInversionError as error:
            reasons.append(error.reason)
        else:
            radii[pair] = 2.0 * value_bound * float(solution["boundary"]) / count

    if not finite:
        reasons.append("numerical_nonfinite")

    ordered = sorted(
        set(reasons), key=lambda reason: (_REASON_RANK.get(reason, len(_REASON_ORDER)), reason)
    )
    if ordered:
        return {
            "status": "not_certified",
            "failure_reasons": ordered,
            "e_q": None,
            "epsilon_res": None,
            "residual_means": residual_means,
            "radii": radii,
            "n_groups": n_groups,
        }

    epsilon_res = float(np.max(np.abs(residual_means) + radii))
    return {
        "status": "certificate_emitted",
        "failure_reasons": [],
        "e_q": epsilon_res / (1.0 - float(gamma)),
        "epsilon_res": epsilon_res,
        "residual_means": residual_means,
        "radii": radii,
        "n_groups": n_groups,
    }


# --------------------------------------------------------------------------- #
# Relative-softmax improvement decision
# --------------------------------------------------------------------------- #
def relative_softmax_candidate(policy: np.ndarray, q_values: np.ndarray, eta: float) -> np.ndarray:
    """``pi_eta^+ ∝ pi * exp(eta * q)`` row-normalised, overflow safe."""
    policy = np.asarray(policy, dtype=np.float64)
    q_values = np.asarray(q_values, dtype=np.float64)
    if policy.shape != q_values.shape:
        raise ValueError("policy and q_values must share a shape")
    logits = np.log(policy) + float(eta) * q_values
    logits = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def _validate_input_policy(policy: np.ndarray) -> np.ndarray:
    policy = np.asarray(policy, dtype=np.float64)
    if policy.ndim != 2:
        raise ValueError("policy must have shape (n_states, n_actions)")
    if not np.all(policy > 0.0):
        raise ValueError("input policy must be strictly positive")
    if float(np.max(np.abs(policy.sum(axis=1) - 1.0))) > 1e-9:
        raise ValueError("input policy rows must be normalised")
    return policy


def decide_policy_update(policy: np.ndarray, q_hat: np.ndarray, e_q: float) -> dict:
    """First-passage relative-softmax emission over the frozen descending eta grid.

    A candidate eta is accepted when every state's certified improvement lower
    bound ``Ihat_s - E_Q ||pi+ - pi||_1`` is strictly positive.
    """
    policy = _validate_input_policy(policy)
    q_hat = np.asarray(q_hat, dtype=np.float64)
    if q_hat.shape != policy.shape:
        raise ValueError("q_hat must share the policy shape")
    e_q = float(e_q)
    n_states = policy.shape[0]

    candidates: list[dict] = []
    selected_eta = None
    selected_lb = None
    selected_policy = None
    for eta in ETA_GRID:
        candidate = relative_softmax_candidate(policy, q_hat, eta)
        delta_pi = candidate - policy
        i_hat = (delta_pi * q_hat).sum(axis=1)
        lb = i_hat - e_q * np.abs(delta_pi).sum(axis=1)
        passes = bool(float(np.min(lb)) > 0.0)
        candidates.append(
            {
                "eta": float(eta),
                "passes_all_checks": passes,
                "lb_by_state": lb.tolist(),
            }
        )
        if passes and selected_eta is None:
            selected_eta = float(eta)
            selected_lb = lb
            selected_policy = candidate

    if selected_eta is not None:
        return {
            "status": "safe_update_emitted",
            "failure_reasons": [],
            "eta_selected": selected_eta,
            "lb_by_state": selected_lb,
            "policy_plus": selected_policy,
            "candidates": candidates,
        }

    reasons = ["improvement_lcb_nonpositive"]
    unchanged = all(
        float(np.max(np.abs(relative_softmax_candidate(policy, q_hat, eta) - policy))) <= 1e-12
        for eta in ETA_GRID
    )
    if unchanged:
        reasons.append("policy_unchanged")
    return {
        "status": "abstained",
        "failure_reasons": reasons,
        "eta_selected": None,
        "lb_by_state": np.zeros(n_states, dtype=np.float64),
        "policy_plus": policy.copy(),
        "candidates": candidates,
    }


# --------------------------------------------------------------------------- #
# Strict JSON
# --------------------------------------------------------------------------- #
def _strict_scalar(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite value cannot be serialised as strict JSON")
        return number
    if isinstance(value, str) or value is None:
        return value
    raise ValueError(f"unsupported strict JSON scalar {type(value).__name__}")


def strict_json_ready(obj):
    """Recursively convert numpy containers to strict, finite JSON values."""
    if isinstance(obj, dict):
        return {str(key): strict_json_ready(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [strict_json_ready(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return [strict_json_ready(item) for item in obj.reshape(-1).tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    return _strict_scalar(obj)
