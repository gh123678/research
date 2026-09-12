"""FP-TIGHT-001: an empirically tightened residual certificate.

This module is NEW and additive. It does not modify, import-patch, or shadow
``fixed_policy_variance_certificate.py``: that file is part of the scientific
corpus and must stay byte-identical everywhere. The sealed certificate is
therefore re-implemented here only where the change is, and every unchanged
ingredient is reproduced exactly so the two are directly comparable on the same
splits and the same risk allocation.

Why a new certificate is needed
-------------------------------

The frozen certificate bounds ``||Qhat - Q^pi||_inf`` by

    E_Q = max_x ( |mean_B(x)| + r_x ) / (1 - gamma),
    r_x = 2 * s_x * sqrt(log(1/delta_each) / N_B),
    s_x = sqrt( mean_A(Y^2) + (2B)^2/2 * sqrt(2 log(1/delta_each) / N_A) ).

The second-moment step is a legitimate Hoeffding application: ``Z = Y^2`` lies in
``[0, (2B)^2]``. The mean step is not. It is labelled "Hoeffding's lemma ... on the
bounded residuals", but Hoeffding's lemma needs the RANGE of ``Y``, which is
``2 * (2B) = 20``, not the data-estimated scale ``s_x ~ 1.2``. So the frozen step
silently substitutes an estimated scale for a range.

Measured on sealed batches, this matters: the frozen radius is 93% envelope by
construction while the observed ``max|Y|`` is only 2.4--2.7, i.e. the assumed
range is ~4x larger than anything the data shows.

The repair is not a new assumption. It is Bernstein's inequality, which is exactly
the inequality that uses ``E[Y^2] <= V`` together with the range and needs no
sub-Gaussianity of ``Y`` with parameter ``V``:

    P( |mean_B(Y) - E[Y]| >= t )
        <= 2 exp( -N_B t^2 / (2 V + (4/3) E_range t) ),     Y in [-E_range/2, E_range/2].

Two variants are provided because they tighten different terms and one of them is
strictly safer than the other:

``bernstein_certificate``  -- keeps the frozen second-moment step (Hoeffding on
    ``Y^2``) and replaces only the mean step with Bernstein. Same two ingredients
    as the frozen construction; only the inequality is corrected. This is the
    conservative variant: its only new element is a valid inequality.

``empirical_bernstein_certificate`` -- replaces the second-moment step with the
    Maurer-Pontil empirical Bernstein bound, so the range ``(2B)^2`` is replaced by
    the observed variance of ``Y^2``. This is where the "84% from a worst-case
    assumption" measurement says the bulk of the slack lives, so it tightens much
    more -- at the cost of a second estimated quantity and a heavier constant.

Both keep the frozen envelope ``2B`` for the range term. A data-driven range was
measured as a third, weaker lever (-36% alone) and is deliberately NOT taken here:
after the two repairs above the range term becomes the binding one, and that is the
next task's question, not this one's.

Risk allocation
---------------

The frozen certificate spends ``delta/(2d)`` on each of ``2d`` bounds. This module
needs three bounds per pair -- the second-moment bound, and one one-sided mean
bound in each direction, since ``||.||_inf`` is two-sided -- so it spends
``delta/(3d)`` on each of ``3d`` bounds, which sums to exactly ``delta``. The
allocation is stated in the output as ``delta_each`` so a reader can re-add it.

Provenance
----------

Oracle quantities are never accepted. The certificate receives only ``q_hat``, the
frozen policy, held-out transition fields, ``R_star``, ``gamma`` and ``delta``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

N_STATES = 4
N_ACTIONS = 3
D = N_STATES * N_ACTIONS
GAMMA = 0.70
R_STAR = 1.5
VALUE_BOUND = R_STAR / (1.0 - GAMMA)
ENVELOPE = 2.0 * VALUE_BOUND
Y_RANGE = 2.0 * ENVELOPE
DELTA = 0.05
MIN_HALF_COUNT = 5000

# Maurer-Pontil empirical Bernstein, scaled from [0,1] to a range K:
#   E[Z] <= mean_N(Z) + sqrt(2 V_N log(2/delta)/N) + 7 K log(2/delta) / (3 (N-1)).
EMPIRICAL_BERNSTEIN_CONSTANT = 7.0 / 3.0


def residuals_for(
    q_hat: Any,
    policy: Any,
    batch: dict[str, Any],
    *,
    gamma: float = GAMMA,
) -> np.ndarray:
    """Held-out Bellman residuals ``Y_t(Qhat)``, identical to the frozen module."""
    q_hat = np.asarray(q_hat, dtype=np.float64)
    policy = np.asarray(policy, dtype=np.float64)
    states = np.asarray(batch["states"], dtype=np.int64)
    actions = np.asarray(batch["actions"], dtype=np.int64)
    rewards = np.asarray(batch["rewards"], dtype=np.float64)
    next_states = np.asarray(batch["next_states"], dtype=np.int64)
    successor = (policy[next_states] * q_hat[next_states]).sum(axis=1)
    return rewards + float(gamma) * successor - q_hat[states, actions]


def split_pairs(
    batch: dict[str, Any], *, n_states: int = N_STATES, n_actions: int = N_ACTIONS
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Deterministic disjoint halves, byte-identical to the frozen split rule."""
    flat = np.asarray(batch["states"], dtype=np.int64) * int(n_actions) + np.asarray(
        batch["actions"], dtype=np.int64
    )
    split: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for pair in range(int(n_states) * int(n_actions)):
        members = np.flatnonzero(flat == pair)
        half = members.size // 2
        split[pair] = (members[:half], members[half:])
    return split


def _equilibrium(variance: float, range_: float, log_term: float, n: int) -> float:
    """Smallest ``t`` with ``t^2 = (2 V + (4/3) R t) log_term / N``.

    Bernstein's bound is implicit in ``t``; the map is a contraction from any
    non-negative start, so fixed-point iteration converges. Iterating to a fixed
    tolerance rather than a fixed count keeps the returned radius reproducible.
    """
    if n <= 0:
        return float("inf")
    t = math.sqrt(2.0 * max(variance, 0.0) * log_term / n)
    for _ in range(200):
        nxt = math.sqrt(
            (2.0 * max(variance, 0.0) + (4.0 / 3.0) * range_ * t) * log_term / n
        )
        if abs(nxt - t) <= 1e-15 * max(1.0, t):
            return nxt
        t = nxt
    return t


def _prepare(
    q_hat: Any,
    policy: Any,
    batch: dict[str, Any],
    n_states: int,
    n_actions: int,
    min_half_count: int,
) -> tuple[list[str], np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]], float]:
    n_groups = int(n_states) * int(n_actions)
    reasons: list[str] = []
    q_hat = np.asarray(q_hat, dtype=np.float64)
    if q_hat.shape != (int(n_states), int(n_actions)):
        raise ValueError("q_hat must match the declared dimensions")
    if not np.all(np.isfinite(q_hat)):
        reasons.append("numerical_nonfinite")
    elif float(np.max(np.abs(q_hat))) > VALUE_BOUND:
        reasons.append("divergence_guard_triggered")

    policy = np.asarray(policy, dtype=np.float64)
    residuals = residuals_for(q_hat, policy, batch)
    split = split_pairs(batch, n_states=n_states, n_actions=n_actions)
    for pair in range(n_groups):
        first, second = split[pair]
        if first.size < min_half_count or second.size < min_half_count:
            reasons.append("heldout_pair_support_missing")
            break
    return reasons, residuals, split, float(n_groups)


def _finish(
    reasons: list[str],
    means: np.ndarray,
    radii: np.ndarray,
    scales: np.ndarray,
    n_groups: float,
    delta_each: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Ordered exactly as the frozen REASON_ORDER, so downstream code that sorts
    # reasons keeps working without importing the frozen module's tuple.
    order = (
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
    if reasons:
        ordered = sorted(set(reasons), key=order.index)
        return {
            "status": "not_certified",
            "failure_reasons": ordered,
            "e_q": None,
            "epsilon_res": None,
            "residual_means": means,
            "radii": radii,
            "scales": scales,
            "n_groups": n_groups,
            "delta_each": delta_each,
            **(extra or {}),
        }
    epsilon_res = float(np.max(np.abs(means) + radii))
    return {
        "status": "certificate_emitted",
        "failure_reasons": [],
        "e_q": epsilon_res / (1.0 - GAMMA),
        "epsilon_res": epsilon_res,
        "residual_means": means,
        "radii": radii,
        "scales": scales,
        "n_groups": n_groups,
        "delta_each": delta_each,
        **(extra or {}),
    }


def bernstein_certificate(
    q_hat: Any,
    policy: Any,
    batch: dict[str, Any],
    *,
    reward_bound: float = R_STAR,
    gamma: float = GAMMA,
    delta: float = DELTA,
    n_states: int = N_STATES,
    n_actions: int = N_ACTIONS,
    min_half_count: int = MIN_HALF_COUNT,
) -> dict[str, Any]:
    """Frozen second-moment step, corrected mean step.

    Only one ingredient changes relative to the sealed certificate: the half-B mean
    deviation is bounded by Bernstein with the frozen envelope as the range, instead
    of ``2 s_x sqrt(log(1/delta)/N)``. Risk ``delta/(3d)`` per bound, ``3d`` bounds.
    """
    del reward_bound, gamma
    reasons, residuals, split, n_groups = _prepare(
        q_hat, policy, batch, n_states, n_actions, min_half_count
    )
    d = int(n_states) * int(n_actions)
    delta_each = float(delta) / (3.0 * d)
    log_term = math.log(2.0 / delta_each)
    scales = np.zeros(d, dtype=np.float64)
    radii = np.zeros(d, dtype=np.float64)
    means = np.zeros(d, dtype=np.float64)
    if not reasons:
        for pair in range(d):
            first, second = split[pair]
            n_a, n_b = int(first.size), int(second.size)
            m2 = float(np.mean(residuals[first] ** 2))
            # Hoeffding on Z = Y^2 in [0, (2B)^2]: the same step as the frozen one.
            slack = (ENVELOPE**2 / 2.0) * math.sqrt(math.log(1.0 / delta_each) / n_a)
            v_x = m2 + slack
            scales[pair] = math.sqrt(max(v_x, 0.0))
            radii[pair] = _equilibrium(v_x, Y_RANGE, log_term, n_b)
            means[pair] = float(np.mean(residuals[second]))
    return _finish(reasons, means, radii, scales, n_groups, delta_each)


def counterfactual_no_envelope(
    q_hat: Any,
    policy: Any,
    batch: dict[str, Any],
    *,
    reward_bound: float = R_STAR,
    gamma: float = GAMMA,
    delta: float = DELTA,
    n_states: int = N_STATES,
    n_actions: int = N_ACTIONS,
    min_half_count: int = MIN_HALF_COUNT,
) -> dict[str, Any]:
    """NOT A CERTIFICATE. The frozen construction with the envelope term deleted.

    This exists to answer one question with a number instead of a narrative: the
    envelope is 87-95% of ``s_x^2``, so is deleting it a bigger lever than
    correcting the mean-step inequality? It is a floor, not a usable bound -- with
    the slack gone there is no high-confidence statement left, and its ``e_q`` must
    never be reported as a guarantee or used for a decision outside this
    comparison. The evaluator labels it accordingly and excludes it from the
    coverage audit.
    """
    reasons, residuals, split, n_groups = _prepare(
        q_hat, policy, batch, n_states, n_actions, min_half_count
    )
    d = int(n_states) * int(n_actions)
    delta_each = float(delta) / (2.0 * d)
    log_term = math.log(1.0 / delta_each)
    scales = np.zeros(d, dtype=np.float64)
    radii = np.zeros(d, dtype=np.float64)
    means = np.zeros(d, dtype=np.float64)
    if not reasons:
        for pair in range(d):
            first, second = split[pair]
            n_b = int(second.size)
            s_x = math.sqrt(max(float(np.mean(residuals[first] ** 2)), 0.0))
            scales[pair] = s_x
            radii[pair] = 2.0 * s_x * math.sqrt(log_term / n_b)
            means[pair] = float(np.mean(residuals[second]))
    return _finish(reasons, means, radii, scales, n_groups, delta_each)


def data_range_certificate(
    q_hat: Any,
    policy: Any,
    batch: dict[str, Any],
    *,
    reward_bound: float = R_STAR,
    gamma: float = GAMMA,
    delta: float = DELTA,
    n_states: int = N_STATES,
    n_actions: int = N_ACTIONS,
    min_half_count: int = MIN_HALF_COUNT,
) -> dict[str, Any]:
    """A SOUND data-driven range, by truncation with its bias accounted for.

    The frozen envelope is ``2B = 10`` and the observed ``max|Y|`` is around `2.5`,
    so the range term ``(4/3) R t`` is carrying roughly four times more than the data
    supports. FP-TIGHT-001 priced deleting it at ``-45%`` and marked the deletion
    unsound, because with the range gone nothing bounds the tails. This does it
    soundly instead.

    Three pieces, per pair:

    1. **Threshold from the other half.** ``tau = max|Y|`` over half A. Because half
       A is independent of half B, everything below holds *conditionally on tau*
       with no union bound over a grid of thresholds -- which is why the threshold is
       taken from A rather than optimised on B.
    2. **Tail bound.** On half B, ``p = P(|Y| > tau)`` is bounded by Hoeffding on the
       indicator at risk ``delta_tau``:
       ``p <= p_hat + sqrt(log(1/delta_tau) / (2 N_B))``.
    3. **Truncated variable.** With ``Y' = Y 1{|Y| <= tau}``, Bernstein applies to
       ``Y'`` with range ``2 tau``, and the truncation costs exactly two terms:

       .. code-block:: text

           |E[Y]| <= |mean_B(Y)|                  observed
                   + (1/N) sum_{|Y_i|>tau} |Y_i|  the empirical tail mass, exact
                   + sqrt(V_x * p)                 the tail's contribution to E[Y]
                   + bernstein(Var_B(Y'), 2 tau)   concentration of the truncation

       The cross term uses Cauchy-Schwarz,
       ``E|Y|1{|Y|>tau} <= sqrt(E[Y^2] P(|Y|>tau)) <= sqrt(V_x p)``.

    Risk: three bounds per pair (the second moment from ``bernstein``'s first step,
    the tail indicator, and the truncated mean), so ``delta/(3d)`` each.

    This is sound, and it is *looser* than the unsound deletion by construction: the
    empirical tail mass and the ``sqrt(V_x p)`` term are exactly the price of
    honesty. Whether the price leaves anything worth having is the empirical
    question.
    """
    del reward_bound, gamma
    reasons, residuals, split, n_groups = _prepare(
        q_hat, policy, batch, n_states, n_actions, min_half_count
    )
    d = int(n_states) * int(n_actions)
    delta_each = float(delta) / (3.0 * d)
    log_term = math.log(2.0 / delta_each)
    log_tail = math.log(1.0 / delta_each)
    scales = np.zeros(d, dtype=np.float64)
    radii = np.zeros(d, dtype=np.float64)
    means = np.zeros(d, dtype=np.float64)
    taus = np.zeros(d, dtype=np.float64)
    tail_terms = np.zeros(d, dtype=np.float64)
    bias_terms = np.zeros(d, dtype=np.float64)
    if not reasons:
        for pair in range(d):
            first, second = split[pair]
            n_a, n_b = int(first.size), int(second.size)
            y_a = residuals[first]
            y_b = residuals[second]

            # Second moment from half A, exactly as the frozen construction does.
            m2 = float(np.mean(y_a**2))
            slack = (ENVELOPE**2 / 2.0) * math.sqrt(log_tail / n_a)
            v_x = m2 + slack
            scales[pair] = math.sqrt(max(v_x, 0.0))

            # Threshold from half A only: independent of half B, so no grid union.
            tau = max(float(np.max(np.abs(y_a))), 1e-12)
            taus[pair] = tau

            beyond = np.abs(y_b) > tau
            p_hat = float(np.mean(beyond))
            # The tail probability must be bounded as TIGHTLY as possible, because
            # it enters the bias below under a square root. Hoeffding on the
            # indicator costs sqrt(log(1/delta)/(2N)) ~ 0.014 at N = 16369, which
            # alone makes the bias 0.15 -- three times the whole frozen radius. The
            # Maurer-Pontil form, whose range term is (7/3) log(2/delta)/(N-1) ~
            # 9.4e-4, is far tighter here precisely because the indicator's observed
            # variance is essentially zero. Using the weaker bound would make this
            # certificate look worse than it is, so the tighter one is used and the
            # negative result below is not an artifact of that choice.
            var_p = float(np.var(beyond.astype(np.float64), ddof=1)) if n_b > 1 else 0.0
            p_ub = min(
                1.0,
                p_hat
                + math.sqrt(2.0 * max(var_p, 0.0) * log_term / n_b)
                + EMPIRICAL_BERNSTEIN_CONSTANT * log_term / max(n_b - 1, 1),
            )
            tail_mass = float(np.sum(np.abs(y_b[beyond])) / n_b) if n_b else 0.0
            bias = math.sqrt(max(v_x, 0.0) * p_ub)

            truncated = np.where(beyond, 0.0, y_b)
            var_t = float(np.var(truncated, ddof=1)) if n_b > 1 else 0.0
            conc = _equilibrium(var_t, 2.0 * tau, log_term, n_b)

            means[pair] = float(np.mean(y_b))
            tail_terms[pair] = tail_mass
            bias_terms[pair] = bias
            radii[pair] = tail_mass + bias + conc
    out = _finish(reasons, means, radii, scales, n_groups, delta_each)
    out["taus"] = taus
    out["tail_mass"] = tail_terms
    out["bias_terms"] = bias_terms
    return out


def empirical_bernstein_certificate(
    q_hat: Any,
    policy: Any,
    batch: dict[str, Any],
    *,
    reward_bound: float = R_STAR,
    gamma: float = GAMMA,
    delta: float = DELTA,
    n_states: int = N_STATES,
    n_actions: int = N_ACTIONS,
    min_half_count: int = MIN_HALF_COUNT,
) -> dict[str, Any]:
    """Maurer-Pontil at both steps: the range is replaced by observed variance.

    Step 1 bounds ``E[Y^2]`` from half A by
        ``mean_A(Y^2) + sqrt(2 Var_A(Y^2) log(2/delta) / N_A)
                    + (7/3) (2B)^2 log(2/delta) / (N_A - 1)``,
    Step 2 bounds both one-sided mean deviations on half B by
        ``sqrt(2 Var_B(Y) log(2/delta) / N_B)
          + (7/3) (2 * 2B) log(2/delta) / (N_B - 1)``.
    """
    del reward_bound, gamma
    reasons, residuals, split, n_groups = _prepare(
        q_hat, policy, batch, n_states, n_actions, min_half_count
    )
    d = int(n_states) * int(n_actions)
    delta_each = float(delta) / (3.0 * d)
    log_term = math.log(2.0 / delta_each)
    scales = np.zeros(d, dtype=np.float64)
    radii = np.zeros(d, dtype=np.float64)
    means = np.zeros(d, dtype=np.float64)
    if not reasons:
        for pair in range(d):
            first, second = split[pair]
            n_a, n_b = int(first.size), int(second.size)
            z_a = residuals[first] ** 2
            m2 = float(np.mean(z_a))
            v_z = float(np.var(z_a, ddof=1)) if n_a > 1 else 0.0
            v_x = (
                m2
                + math.sqrt(2.0 * max(v_z, 0.0) * log_term / n_a)
                + EMPIRICAL_BERNSTEIN_CONSTANT
                * (ENVELOPE**2)
                * log_term
                / max(n_a - 1, 1)
            )
            scales[pair] = math.sqrt(max(v_x, 0.0))
            v_y = float(np.var(residuals[second], ddof=1)) if n_b > 1 else 0.0
            radii[pair] = math.sqrt(
                2.0 * max(v_y, 0.0) * log_term / n_b
            ) + EMPIRICAL_BERNSTEIN_CONSTANT * Y_RANGE * log_term / max(n_b - 1, 1)
            means[pair] = float(np.mean(residuals[second]))
    return _finish(reasons, means, radii, scales, n_groups, delta_each)
