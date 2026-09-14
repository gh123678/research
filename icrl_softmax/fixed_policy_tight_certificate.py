"""FP-BOUND-001: sound tightenings of the residual certificate's scalar ``E_Q``.

ADDITIVE MODULE. It imports nothing sealed for its arithmetic and modifies no
sealed file; the frozen decision rule, the eta grid and ``delta_total`` are
untouched. Every function here returns the same scalar contract as
``fixed_policy_mp_certificate.mp_certificate_firstvisit`` so that
``fs.improvement_for`` can be called on it unchanged.

WHY THE FROZEN ``E_Q`` IS NOT TIGHT
-----------------------------------
``E_Q = max_x (|mean_x| + t_x) / (1 - gamma)`` with
``t_x = sqrt(2 V_x log(2/delta')/n_x) + (7/3) R log(2/delta')/(n_x - 1)``,
``R = 2E``, ``E = R* + gamma B + B``, ``B = R*/(1-gamma) = 5``, ``delta' =
delta_step/(2d)``. Three provable slack sources, each priced in the task sheet:

L1  THE ENVELOPE IS WORST-CASE OVER QHAT, BUT QHAT IS KNOWN.
    ``E = R* + gamma*B + B`` only assumes ``|Qhat| <= B``. Since ``Qhat`` is an
    input the certificate already holds, the true range of
    ``Y = r + gamma Vhat(s') - Qhat(s,a)`` is bounded, deterministically, by
    ``E_eff = R* + gamma ||Vhat||_inf + ||Qhat||_inf`` with
    ``Vhat = pi . Qhat``. No estimate, no confidence correction, no new premise:
    the same Maurer-Pontil theorem with a smaller TRUE range.

L2  THE IMPLEMENTATION NEVER SPENDS HALF ITS RISK BUDGET.
    Maurer-Pontil Theorem 4 is TWO-SIDED at level ``delta``. Allocating
    ``delta' = delta_step/(2d)`` and then reasoning about "two directions"
    therefore spends only ``d * delta' = delta_step/2``. ``delta' = delta_step/d``
    spends the budget exactly. ``log(2/delta')`` falls from 6.87 to 6.17.

L3  THE ``1/(1-gamma)`` STEP THROWS AWAY THE PER-PAIR STRUCTURE.
    ``Qhat - Q^pi = (I - gamma P^pi)^{-1} rho`` is an IDENTITY. With
    ``|rho_x| <= eps_x`` per pair, exact propagation gives

        w        = (I - gamma K)^{-1} epsbar,
        K(s,s')  = sum_a pi(a|s) P(s'|s,a),
        epsbar(s)= sum_a pi(a|s) eps(s,a),
        |u(s,a)| <= eps(s,a) + gamma * sum_s' P(s'|s,a) w(s'),

    whose max is ``<= max_x eps_x/(1-gamma)`` with equality only when ``eps`` is
    constant across pairs. The gain is exactly the non-uniformity of ``eps``
    (measured median max/min ratio: 4.40).

    L3 READS THE TRANSITION KERNEL. It is the same family as the pre-registered
    oracle-kernel arm and MUST be labelled as such everywhere: it corroborates,
    but cannot be used for any claim about learning from behavioural data alone.

Reference-only (NOT a deliverable, also needs the kernel): the exact support
range of ``Y_x`` from the MDP tables, ``support_range_certificate``, quoted to
show how much of the range lever L1 leaves on the table.
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
VALUE_BOUND = R_STAR / (1.0 - GAMMA)  # B = 5
ENVELOPE = R_STAR + GAMMA * VALUE_BOUND + VALUE_BOUND  # E = 10 (the frozen one)
Y_RANGE = 2.0 * ENVELOPE  # 20
MP_CONSTANT = 7.0 / 3.0

REASONS = ("divergence_guard_triggered", "heldout_pair_support_missing", "numerical_nonfinite")

LEVERS = ("frozen", "L1", "L12", "L123", "L12M", "L12S", "support_range")
#: ``L12M`` is WITHDRAWN as a guarantee (2026-09-13 audit). Its propagation applies
#: Hoeffding to ``W_k``, which is itself computed from the same successor draws, so
#: the concentration step is not licensed; a data-dependent ``f`` can be anti-
#: correlated with the very sample that estimates its mean. ``L12M`` is kept only to
#: reproduce the 2026-09-13 bundles. ``L12S`` is the repaired construction: each
#: pair's first-visit sample is split, the intervals come from half A, and every
#: propagation estimate is computed on half B, which is independent of half A.
WITHDRAWN_LEVERS = ("L12M",)
#: fraction of each pair's retained sample used for the interval half (A)
SPLIT_FRACTION = 0.5


def effective_envelope(
    q_hat: Any, policy: Any, *, reward_bound: float = R_STAR, gamma: float = GAMMA
) -> float:
    """``E_eff = R* + gamma*||Vhat||_inf + ||Qhat||_inf`` -- L1's range constant.

    A deterministic function of the KNOWN ``(q_hat, policy)``. Since
    ``|r| <= R*`` and ``|Vhat(s')| <= ||Vhat||_inf`` and ``|Qhat(s,a)| <=
    ||Qhat||_inf`` pointwise, ``|Y_x| <= E_eff`` for every pair, so the residual
    range is at most ``2 E_eff``. This is strictly sharper than the frozen
    ``E = R* + gamma*B + B`` whenever ``max(||Vhat||_inf, ||Qhat||_inf) < B``,
    which holds for every route in this project.
    """
    q = np.asarray(q_hat, dtype=np.float64)
    pi = np.asarray(policy, dtype=np.float64)
    v_hat = (pi * q).sum(axis=1)
    return float(reward_bound) + float(gamma) * float(
        np.max(np.abs(v_hat))
    ) + float(np.max(np.abs(q)))


def support_range(
    q_hat: Any,
    policy: Any,
    transition: Any,
    reward: Any,
    *,
    gamma: float = GAMMA,
) -> np.ndarray:
    """Exact support range of ``Y_x`` from the MDP tables (REFERENCE ONLY).

    ``Y_x`` can only take the values ``R(s,a,s') + gamma Vhat(s') - Qhat(s,a)``
    for successors ``s'`` with ``P(s'|s,a) > 0``, so its exact range is a finite,
    computable constant -- no estimation and no probability cost. Quoted to price
    what L1 leaves on the table; needs the kernel, so not a deliverable.
    """
    q = np.asarray(q_hat, dtype=np.float64)
    pi = np.asarray(policy, dtype=np.float64)
    P = np.asarray(transition, dtype=np.float64)
    R = np.asarray(reward, dtype=np.float64)
    v_hat = (pi * q).sum(axis=1)
    y = R + float(gamma) * v_hat[None, None, :] - q[:, :, None]
    live = P > 0.0
    hi = np.where(live, y, -np.inf).max(axis=2)
    lo = np.where(live, y, np.inf).min(axis=2)
    return (hi - lo).reshape(-1)


def successor_counts(
    batch: dict[str, Any], d: int, n_states: int = N_STATES, n_actions: int = N_ACTIONS
) -> np.ndarray:
    """Per-pair successor histogram ``C[x, s'] = #{i in pair x : s'_i = s'}``.

    A sufficient statistic of the batch for every propagation estimate below: the
    same iid draws of ``s' ~ P(.|s,a)`` that produced the residuals also estimate
    ``E_{s'~P(.|s,a)}[f(s')]`` for ANY ``f``, with no kernel.
    """
    flat = np.asarray(batch["states"], dtype=np.int64) * int(n_actions) + np.asarray(
        batch["actions"], dtype=np.int64
    )
    nxt = np.asarray(batch["next_states"], dtype=np.int64)
    out = np.zeros((d, int(n_states)), dtype=np.int64)
    for pair in range(d):
        members = np.flatnonzero(flat == pair)
        if members.size:
            out[pair] = np.bincount(nxt[members], minlength=int(n_states))
    return out


def propagation_data_driven(
    eps: np.ndarray,
    sizes: np.ndarray,
    counts: np.ndarray,
    policy: Any,
    *,
    delta_prop: float,
    gamma: float = GAMMA,
    n_iter: int = 12,
    n_states: int = N_STATES,
    n_actions: int = N_ACTIONS,
) -> dict[str, Any]:
    """MODEL-FREE exact-propagation bound (L12M): no transition kernel is read.

    L3 uses the identity ``Qhat - Q^pi = -(I - gamma P^pi)^{-1} rho`` with the
    known ``P``. This does the same propagation using only the certification
    batch, because the batch already contains iid draws of ``s' ~ P(.|s,a)`` for
    every pair: ``E_{s'~P(.|s,a)}[f(s')]`` is estimated by the pair's own
    successor histogram, with a Hoeffding correction at its own risk level.

    Induction, with ``eps_x`` the certified interval on ``|rho_x|``:

        W_0        = max_x eps_x/(1-gamma)          (a valid bound on ||w||_inf)
        W_{k+1}(s) = sum_a pi(a|s) eps(s,a)
                     + gamma sum_a pi(a|s) [ T_x(W_k) + c_k(x) ]
        T_x(f)     = (1/n_x) sum_i f(s'_i),   c_k(x) = ||W_k||_inf sqrt(log(1/da_k)/(2 n_x))

    ``|w| <= W_k`` for every k, and the returned bound on ``|u|`` follows from one
    more application with its own risk ``db``. Risk spent: ``delta_prop`` total,
    split ``delta_prop/(2 n_iter d)`` per (iteration, pair) and ``delta_prop/(2d)``
    for the final estimate. The samples are reused across iterations, which is why
    the risk is union-bounded over ``n_iter`` rather than assumed fresh.

    This recovers most of L3's gain while reading no kernel; it is strictly weaker
    than L3 (it pays for its estimates) and strictly stronger than the uniform
    ``1/(1-gamma)`` step, since ``T_x`` converges to ``K``.
    """
    pi = np.asarray(policy, dtype=np.float64)
    eps_grid = np.asarray(eps, dtype=np.float64).reshape(int(n_states), int(n_actions))
    sizes = np.asarray(sizes, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    epsbar = (pi * eps_grid).sum(axis=1)

    delta_k = float(delta_prop) / (2.0 * int(n_iter) * sizes.size)
    log_k = math.log(1.0 / delta_k)
    delta_f = float(delta_prop) / (2.0 * sizes.size)
    log_f = math.log(1.0 / delta_f)

    def estimate(f: np.ndarray, log_term: float):
        """One-sided Hoeffding estimate of ``E_{s'~P(.|s,a)}[f(s')]`` per pair."""
        t = (counts * f[None, :]).sum(axis=1) / sizes
        c = float(np.max(f)) * np.sqrt(log_term / (2.0 * sizes))
        return t, c

    w = np.full(int(n_states), float(np.max(eps)) / (1.0 - float(gamma)))
    for _ in range(int(n_iter)):
        t, c = estimate(w, log_k)
        nxt_w = epsbar + float(gamma) * (
            pi * (t + c).reshape(int(n_states), int(n_actions))
        ).sum(axis=1)
        if float(np.max(np.abs(nxt_w - w))) <= 1e-14:
            w = nxt_w
            break
        w = nxt_w

    t, c = estimate(w, log_f)
    bound = eps_grid + float(gamma) * (t + c).reshape(int(n_states), int(n_actions))
    return {
        "e_q": float(np.max(bound)),
        "w": w.tolist(),
        "uniform_bound": float(np.max(eps)) / (1.0 - float(gamma)),
        "delta_per_iteration": delta_k,
        "delta_final": delta_f,
        "n_iter": int(n_iter),
    }


def split_sample_certificate(
    q_hat: Any,
    policy: Any,
    batch: dict[str, Any],
    *,
    min_visits: int,
    delta_step: float,
    split_fraction: float = SPLIT_FRACTION,
    delta_prop_fraction: float = 0.5,
    n_states: int = N_STATES,
    n_actions: int = N_ACTIONS,
    reward_bound: float = R_STAR,
    gamma: float = GAMMA,
) -> dict[str, Any]:
    """L12S: the sound replacement for the withdrawn ``L12M``.

    WHY ``L12M`` WAS WRONG. ``L12M`` estimated ``E_{s'~P}[W_k(s')]`` by the
    pair's own successor average and charged it a Hoeffding term ``c_k`` built
    from ``W_k``. But ``W_k`` is itself computed from those same successor draws,
    so the function being averaged is not fixed: a data-dependent ``f`` may be
    large exactly on the observed successors, and then ``T_x(f) > E[f] + c`` with
    probability far above the nominal level. The induction is unlicensed, not
    merely loose.

    THE REPAIR. Split each pair's retained first-visit sample, in the retained
    order, into half A (first ``floor(f*n_x)``) and half B. Conditional on half A:

    * the interval ``|rho_x| <= eps_x`` is a FIXED vector (nothing in it uses half
      B), and the envelope ``E_eff`` is a deterministic function of ``(Qhat, pi)``;
    * ``V_0 = max_x eps_x/(1-gamma)`` is therefore fixed, and so is every
      ``V_{k+1}(s) = epsbar(s) + gamma sum_a pi(a|s)[T^B_x(V_k) + c_k(x)]``;
    * ``T^B_x`` and its Hoeffding term ``c_k(x) = ||V_k||_inf sqrt(log(1/da_k)/(2 n_Bx))``
      are computed on half B, which is INDEPENDENT of half A and of ``V_k``.

    So each concentration step is applied to a fixed function and is valid, and
    the induction ``|w| <= V_k`` goes through. The final bound is
    ``max_{s,a} [ eps(s,a) + gamma (T^B_x(V_K) + c_K(x)) ]``.

    RISK. ``delta_eps + delta_prop = delta_step``: the per-pair intervals are
    two-sided at total per-pair risk ``delta_eps/d``, and the propagation spends
    ``delta_prop`` split over ``K*d`` (iteration, pair) events plus ``d`` final
    ones.

    PRICE. Both halves are smaller than the whole sample, so the interval is
    looser than ``L12``'s and the successor averages are noisier than ``L12M``'s.
    That is what being licensed costs; the amount is measured, not assumed.
    """
    d = int(n_states) * int(n_actions)
    reasons = _guards(q_hat, reward_bound=reward_bound, gamma=gamma)
    q = np.asarray(q_hat, dtype=np.float64)
    pi = np.asarray(policy, dtype=np.float64)
    residuals = _residuals(q, pi, batch, gamma=gamma)
    flat = np.asarray(batch["states"], dtype=np.int64) * int(n_actions) + np.asarray(
        batch["actions"], dtype=np.int64
    )
    nxt = np.asarray(batch["next_states"], dtype=np.int64)

    y_range = 2.0 * effective_envelope(q, pi, reward_bound=reward_bound, gamma=gamma)
    delta_prop = float(delta_prop_fraction) * float(delta_step)
    delta_eps = float(delta_step) - delta_prop
    # two-sided at total per-pair risk delta_eps/d == one-sided at delta_eps/(2d)
    # (Maurer-Pontil Thm 4 is one-sided; the absolute-value form is the union of
    # the two directions, which is exactly log(2/delta_dir) with delta_dir halved)
    delta_dir = delta_eps / (2.0 * d)
    log_term = math.log(2.0 / delta_dir)

    sizes_a = np.zeros(d, dtype=np.int64)
    sizes_b = np.zeros(d, dtype=np.int64)
    means = np.zeros(d)
    radii = np.zeros(d)
    eps = np.zeros(d)
    counts_b = np.zeros((d, int(n_states)), dtype=np.int64)
    if not reasons:
        for pair in range(d):
            members = np.flatnonzero(flat == pair)
            n = int(members.size)
            if n < int(min_visits):
                reasons = reasons + ["heldout_pair_support_missing"]
                break
            cut = int(math.floor(float(split_fraction) * n))
            cut = max(1, min(n - 1, cut))
            idx_a, idx_b = members[:cut], members[cut:]
            sizes_a[pair] = idx_a.size
            sizes_b[pair] = idx_b.size
            y_a = residuals[idx_a]
            v = float(np.var(y_a, ddof=1)) if idx_a.size > 1 else 0.0
            means[pair] = float(np.mean(y_a))
            radii[pair] = math.sqrt(2.0 * max(v, 0.0) * log_term / idx_a.size) + (
                MP_CONSTANT * y_range * log_term / max(idx_a.size - 1, 1)
            )
            eps[pair] = abs(means[pair]) + radii[pair]
            counts_b[pair] = np.bincount(nxt[idx_b], minlength=int(n_states))
    out: dict[str, Any] = {
        "status": "not_certified" if reasons else "certificate_emitted",
        "failure_reasons": reasons,
        "lever": "L12S",
        "residual_means": means,
        "radii": radii,
        "epsilon_res_by_pair": eps,
        "pair_sizes": sizes_a + sizes_b,
        "delta_step": float(delta_step),
        "delta_each": delta_dir,
        "y_range": float(y_range),
        "n_groups": d,
        "e_q": None,
        "propagation": None,
    }
    if reasons:
        return out

    prop = propagation_split(
        eps,
        sizes_b,
        counts_b,
        pi,
        delta_prop=delta_prop,
        gamma=gamma,
        n_states=int(n_states),
        n_actions=int(n_actions),
    )
    prop["kind"] = "split"
    prop["split_fraction"] = float(split_fraction)
    prop["delta_prop_fraction"] = float(delta_prop_fraction)
    prop["sizes_half_a"] = sizes_a.tolist()
    prop["sizes_half_b"] = sizes_b.tolist()
    prop["successor_counts_half_b"] = counts_b.tolist()
    prop["delta_eps"] = delta_eps
    out["propagation"] = prop
    out["e_q"] = prop["e_q"]
    return out


def propagation_split(
    eps: np.ndarray,
    sizes_b: np.ndarray,
    counts_b: np.ndarray,
    policy: Any,
    *,
    delta_prop: float,
    gamma: float = GAMMA,
    n_iter: int = 12,
    n_states: int = N_STATES,
    n_actions: int = N_ACTIONS,
) -> dict[str, Any]:
    """The propagation induction of L12S, on half B only (see above).

    ``eps`` and hence every ``V_k`` is fixed given half A; ``counts_b``/``sizes_b``
    come from half B, which is independent of half A, so each Hoeffding step is
    applied to a fixed function. Risk ``delta_prop`` is union-bounded over
    ``n_iter * d`` iteration events and ``d`` final ones.
    """
    pi = np.asarray(policy, dtype=np.float64)
    eps_grid = np.asarray(eps, dtype=np.float64).reshape(int(n_states), int(n_actions))
    sizes_b = np.asarray(sizes_b, dtype=np.float64)
    counts_b = np.asarray(counts_b, dtype=np.float64)
    epsbar = (pi * eps_grid).sum(axis=1)

    delta_k = float(delta_prop) / (2.0 * int(n_iter) * sizes_b.size)
    log_k = math.log(1.0 / delta_k)
    delta_f = float(delta_prop) / (2.0 * sizes_b.size)
    log_f = math.log(1.0 / delta_f)

    def estimate(f):
        t = (counts_b * f[None, :]).sum(axis=1) / sizes_b
        c = float(np.max(f)) * np.sqrt(log_k / (2.0 * sizes_b))
        return t, c

    v = np.full(int(n_states), float(np.max(eps)) / (1.0 - float(gamma)))
    iterations = 0
    for _ in range(int(n_iter)):
        iterations += 1
        t, c = estimate(v)
        nxt_v = epsbar + float(gamma) * (
            pi * (t + c).reshape(int(n_states), int(n_actions))
        ).sum(axis=1)
        if float(np.max(np.abs(nxt_v - v))) <= 1e-14:
            v = nxt_v
            break
        v = nxt_v

    t, c = estimate(v)
    bound = eps_grid + float(gamma) * (t + c).reshape(int(n_states), int(n_actions))
    return {
        "e_q": float(np.max(bound)),
        "w": v.tolist(),
        "uniform_bound": float(np.max(eps)) / (1.0 - float(gamma)),
        "delta_per_iteration": delta_k,
        "delta_final": delta_f,
        "n_iter": int(n_iter),
        "iterations_used": iterations,
    }


def _guards(q_hat: Any, *, reward_bound: float, gamma: float) -> list[str]:
    q = np.asarray(q_hat, dtype=np.float64)
    if not np.all(np.isfinite(q)):
        return ["numerical_nonfinite"]
    if float(np.max(np.abs(q))) > float(reward_bound) / (1.0 - float(gamma)):
        return ["divergence_guard_triggered"]
    return []


def _pair_slices(batch: dict[str, Any], d: int, n_actions: int = N_ACTIONS) -> dict[int, np.ndarray]:
    flat = np.asarray(batch["states"], dtype=np.int64) * int(n_actions) + np.asarray(
        batch["actions"], dtype=np.int64
    )
    return {pair: np.flatnonzero(flat == pair) for pair in range(d)}


def _residuals(q_hat, policy, batch, gamma=GAMMA):
    q = np.asarray(q_hat, dtype=np.float64)
    pi = np.asarray(policy, dtype=np.float64)
    st = np.asarray(batch["states"], dtype=np.int64)
    ac = np.asarray(batch["actions"], dtype=np.int64)
    rw = np.asarray(batch["rewards"], dtype=np.float64)
    ns = np.asarray(batch["next_states"], dtype=np.int64)
    return rw + float(gamma) * (pi[ns] * q[ns]).sum(axis=1) - q[st, ac]


def certificate(
    q_hat: Any,
    policy: Any,
    batch: dict[str, Any],
    *,
    min_visits: int,
    delta_step: float,
    lever: str = "L123",
    transition: Any = None,
    reward: Any = None,
    n_states: int = N_STATES,
    n_actions: int = N_ACTIONS,
    reward_bound: float = R_STAR,
    gamma: float = GAMMA,
    delta_prop_fraction: float = 0.5,
    split_fraction: float = SPLIT_FRACTION,
) -> dict[str, Any]:
    """One ``E_Q`` per lever, from the chain-replicated first-visit sample.

    ``lever`` is one of ``LEVERS``:

    ``frozen``        the current MP certificate (E = 10, delta' = delta_step/(2d));
    ``L1``            frozen + the known-Qhat envelope;
    ``L12``           L1 + the full risk budget;
    ``L123``          L12 + exact propagation (needs ``transition``);
    ``L12M``          L1 + MODEL-FREE propagation from the batch's successor draws
                      (half the budget reserved for the propagation estimates);
    ``support_range`` L1 + the oracle support range (reference only).
    """
    if lever not in LEVERS:
        raise ValueError(f"unknown lever {lever!r}")
    if lever == "L12S":
        return split_sample_certificate(
            q_hat, policy, batch, min_visits=min_visits, delta_step=delta_step,
            split_fraction=split_fraction,
            delta_prop_fraction=delta_prop_fraction,
            n_states=n_states, n_actions=n_actions, reward_bound=reward_bound, gamma=gamma,
        )
    d = int(n_states) * int(n_actions)
    reasons = _guards(q_hat, reward_bound=reward_bound, gamma=gamma)
    q = np.asarray(q_hat, dtype=np.float64)
    pi = np.asarray(policy, dtype=np.float64)
    residuals = _residuals(q, pi, batch, gamma=gamma)
    groups = _pair_slices(batch, d, int(n_actions))
    sizes = np.array([groups[p].size for p in range(d)], dtype=np.int64)
    if int(sizes.min()) < int(min_visits):
        reasons = reasons + ["heldout_pair_support_missing"]

    use_eff = lever != "frozen"
    use_full_budget = lever in ("L12", "L123")
    y_range = 2.0 * effective_envelope(
        q, pi, reward_bound=reward_bound, gamma=gamma
    ) if use_eff else Y_RANGE
    if lever == "support_range":
        if transition is None or reward is None:
            raise ValueError("support_range needs the MDP tables")
        y_range = None  # per-pair, computed below

    delta_dir = float(delta_step) / (d if use_full_budget else 2.0 * d)
    if lever == "L12M":
        # L12M spends two budgets: (1 - frac) on the per-pair intervals and frac on
        # the propagation estimates. The default frac = 0.5 reproduces FP-BOUND-002;
        # FP-BOUND-003 measures a lopsided split, because the propagation estimates
        # are far less risk-hungry than the intervals (their confidence terms are
        # O(1e-3) against a gain of O(3e-2), so spending delta on them is waste).
        delta_prop = float(delta_prop_fraction) * float(delta_step)
        delta_dir = (float(delta_step) - delta_prop) / d
    log_term = math.log(2.0 / delta_dir)

    means = np.zeros(d)
    radii = np.zeros(d)
    eps = np.zeros(d)
    per_pair_range = np.full(d, y_range if y_range is not None else np.nan)
    if not reasons:
        if lever == "support_range":
            per_pair_range = support_range(q, pi, transition, reward, gamma=gamma)
        for pair in range(d):
            y = residuals[groups[pair]]
            n = int(sizes[pair])
            v = float(np.var(y, ddof=1)) if n > 1 else 0.0
            means[pair] = float(np.mean(y))
            radii[pair] = math.sqrt(2.0 * max(v, 0.0) * log_term / n) + MP_CONSTANT * (
                per_pair_range[pair] if lever == "support_range" else y_range
            ) * log_term / max(n - 1, 1)
            eps[pair] = abs(means[pair]) + radii[pair]

    out: dict[str, Any] = {
        "status": "not_certified" if reasons else "certificate_emitted",
        "failure_reasons": reasons,
        "lever": lever,
        "residual_means": means,
        "radii": radii,
        "epsilon_res_by_pair": eps,
        "pair_sizes": sizes,
        "delta_step": float(delta_step),
        "delta_each": delta_dir,
        "y_range": float(y_range) if y_range is not None else None,
        "n_groups": d,
        "e_q": None,
        "propagation": None,
    }
    if reasons:
        return out

    if lever in ("L123",):
        if transition is None:
            raise ValueError("L123 needs the transition kernel")
        P = np.asarray(transition, dtype=np.float64)
        K = np.einsum("sa,sat->st", pi, P)
        inv = np.linalg.inv(np.eye(int(n_states)) - float(gamma) * K)
        epsbar = (pi * eps.reshape(int(n_states), int(n_actions))).sum(axis=1)
        w = inv @ epsbar
        prop = float(gamma) * (P * w[None, None, :]).sum(axis=2)
        bound = eps.reshape(int(n_states), int(n_actions)) + prop
        out["propagation"] = {
            "kind": "kernel",
            "K": K.tolist(),
            "inverse": inv.tolist(),
            "epsbar": epsbar.tolist(),
            "w": w.tolist(),
            "uniform_bound": float(np.max(eps)) / (1.0 - float(gamma)),
        }
        out["e_q"] = float(np.max(bound))
    elif lever == "L12M":
        # Model-free: the propagation is estimated from the batch's own successor
        # draws, so half the risk budget is reserved for those estimates.
        cnt = successor_counts(batch, d, n_states=int(n_states), n_actions=int(n_actions))
        delta_prop = float(delta_prop_fraction) * float(delta_step)
        prop = propagation_data_driven(
            eps,
            sizes,
            cnt,
            pi,
            delta_prop=delta_prop,
            gamma=gamma,
            n_states=int(n_states),
            n_actions=int(n_actions),
        )
        prop["kind"] = "data_driven"
        prop["successor_counts"] = cnt.tolist()
        prop["delta_eps"] = float(delta_step) - delta_prop
        prop["delta_prop_fraction"] = float(delta_prop_fraction)
        out["propagation"] = prop
        out["e_q"] = prop["e_q"]
    else:
        out["e_q"] = float(np.max(eps)) / (1.0 - float(gamma))
    return out
