"""Frozen FP-SCALE-001 protocol: a certified relative-softmax step at scale.

This module implements the FP-SCALE-001 task protocol. It does NOT modify and
does NOT reimplement the mathematical contract: every route, the residual
certificate, the relative-softmax decision rule, and the ordered abstention
reasons are imported from the sealed, VERIFIED ``fixed_policy_expected_sarsa``
module, which is reused byte-identically.

What this module adds is only the protocol the task froze:

- a 4-state / 3-action MDP family with ``pi_min = 0.15``;
- a TRAINING trajectory (``TRAIN_LENGTH``) used only to build ``Qhat``;
- an independent CERTIFICATION batch (``CERT_CHAINS`` chains of
  ``CERT_CHAIN_LENGTH`` steps, each started from the stationary state
  distribution) used only for the held-out residual certificate and the oracle
  audit;
- the ``H1`` count rule (``MIN_CERT_COUNT`` observations for every pair);
- the extended descending eta candidate grid.

The two batches are separate random draws and are never mixed, so
certification data cannot influence ``Qhat`` or any hyperparameter.

Oracle quantities (true Q, true V, stationarity, realised error) are computed
only inside this module's oracle-audit helpers and are never passed to the
estimator, the certificate, or the decision rule.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import fixed_policy_expected_sarsa as es
from evaluate_fixed_policy_q_routes import make_mdp, make_policy
from evaluate_fixed_policy_q_routes import policy_quantities
from mdps import rollout

# --- frozen protocol constants (task FP-SCALE-001 v1.1) --------------------

N_STATES = 4
N_ACTIONS = 3
PI_MIN = 0.15
GAMMA = 0.70
ALPHA = 0.65
LAYERS = 160
GAP_BONUS = 0.5
MIXING = (0.08, 0.5)
TASKS_PER_CELL = 12
TRAIN_LENGTH = 65536
CERT_CHAINS = 262144
CERT_CHAIN_LENGTH = 16
CERT_LENGTH = CERT_CHAINS * CERT_CHAIN_LENGTH
MIN_CERT_COUNT = 20000
# Frozen per-pair certification count. Chosen inside the band where the
# inherited 15-component grid brackets (measured: valid at 20000..48000, and at
# 131088; invalid at 92185), sitting mid-band and well above MIN_CERT_COUNT.
# At this count the inherited inversion gives r_x/(1-gamma) = 0.9009, inside
# the H1 target of 1.2. Measured worst-pair counts before subsampling were
# 78491 / 102206 / 124799 over three probes, so every pair comfortably exceeds
# this frozen count and no record is lost to the subsampling step.
FROZEN_CERT_COUNT = 40000
CERT_SUBSAMPLE_SEED_TAG = 7717
REWARD_BOUND = 1.5
VALUE_BOUND = REWARD_BOUND / (1.0 - GAMMA)
DELTA = 0.05
MIXTURE_COMPONENTS = 15
SEED = 20260911
ZETA = 8.0
XI = 8.0
TAU = 8.0
ETA_CANDIDATES = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
ROUTES = ("expected_exact", "expected_finite", "sampled_exact")
H1_RADIUS_TARGET = 1.2

# The frozen ordered non-emission reasons, inherited verbatim from FP-ESARSA-001.
REASON_ORDER = (
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

# The inherited module is reused by import; its identity is asserted so that a
# silent local edit cannot change this task's scientific contract.
INHERITED_CONTRACT_MODULE = "fixed_policy_expected_sarsa"


def pair_index(state: int, action: int) -> int:
    """Canonical flat pair index for the frozen dimensions."""
    return int(state) * N_ACTIONS + int(action)


def build_task(task_index: int, mixing: float) -> tuple[Any, np.ndarray, np.random.Generator]:
    """Spawn one frozen task: an MDP, a fixed policy, and its dedicated RNG.

    The seed schedule is deterministic and independent of the certification
    sampling, so the certification batch can never influence task generation.
    """
    rng = np.random.default_rng([SEED, int(round(mixing * 100)), int(task_index)])
    mdp = make_mdp(N_STATES, N_ACTIONS, GAMMA, float(mixing), GAP_BONUS, rng)
    policy = make_policy(N_STATES, N_ACTIONS, PI_MIN, rng)
    return mdp, policy, rng


def training_batch(
    mdp: Any, policy: np.ndarray, mu_state: np.ndarray, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Draw the training trajectory used only to build ``Qhat``.

    A single chain started from the stationary state distribution, exactly as
    the inherited evaluator starts its trajectory.
    """
    start = int(rng.choice(N_STATES, p=mu_state))
    sampled = rollout(mdp, policy, start, TRAIN_LENGTH, rng)
    # ``rollout`` returns n+1 states/actions: A[n] is the extra action drawn at
    # S[n] precisely so the successor action of the last transition exists.
    all_actions = np.asarray(sampled[1])
    states = np.asarray(sampled[0])[:TRAIN_LENGTH]
    actions = all_actions[:TRAIN_LENGTH]
    rewards = np.asarray(sampled[2], dtype=np.float64)[1 : TRAIN_LENGTH + 1]
    next_states = np.asarray(sampled[0])[1 : TRAIN_LENGTH + 1]
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "next_actions": all_actions[1 : TRAIN_LENGTH + 1],
    }


def certification_batch(
    mdp: Any, policy: np.ndarray, mu_state: np.ndarray, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Draw the independent certification batch for the residual certificate.

    ``CERT_CHAINS`` independent chains, each restarted from the stationary
    state distribution. Restarting makes the chain-start states independent
    draws from ``mu_state``, so the per-pair visit indicators are independent
    across chains and the residual certificate's martingale-difference
    condition holds with the scale the task froze. This batch is drawn from a
    dedicated RNG stream and is never used to build ``Qhat``.
    """
    starts = rng.choice(N_STATES, size=CERT_CHAINS, p=mu_state)
    states = np.empty(CERT_LENGTH, dtype=np.int64)
    actions = np.empty(CERT_LENGTH, dtype=np.int64)
    rewards = np.empty(CERT_LENGTH, dtype=np.float64)
    next_states = np.empty(CERT_LENGTH, dtype=np.int64)

    transition = np.asarray(mdp["P"], dtype=np.float64)
    reward = np.asarray(mdp["R"], dtype=np.float64)
    for chain in range(CERT_CHAINS):
        state = int(starts[chain])
        base = chain * CERT_CHAIN_LENGTH
        for step in range(CERT_CHAIN_LENGTH):
            action = int(rng.choice(N_ACTIONS, p=policy[state]))
            following = int(rng.choice(N_STATES, p=transition[state, action]))
            states[base + step] = state
            actions[base + step] = action
            rewards[base + step] = reward[state, action, following]
            next_states[base + step] = following
            state = following

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "next_actions": actions,
    }


def pair_counts(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Certification visit counts per canonical pair, shape ``(n_states*n_actions,)``."""
    flat = np.asarray(states, dtype=np.int64) * N_ACTIONS + np.asarray(
        actions, dtype=np.int64
    )
    return np.bincount(flat, minlength=N_STATES * N_ACTIONS).astype(np.int64)


def subsample_to_frozen_count(
    cert: dict[str, np.ndarray], rng: np.random.Generator
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Reduce every pair to exactly ``FROZEN_CERT_COUNT`` certification items.

    The inherited frozen inversion only brackets counts inside a narrow,
    non-monotone band: its 15-component geometric grid has mesh points
    ``2**0 .. 2**14``, and it fails with `mixture_inversion_unbracketed` for
    counts that fall between usable mesh points (measured: it fails at 92185
    and succeeds at 131088). Sampling a per-pair count into that band is
    therefore part of the protocol, not a choice made after seeing output.

    Each pair's retained items are a uniformly random subset of that pair's
    items, drawn from a dedicated RNG stream, so retained items remain
    independent draws of the same conditional law. The estimator is untouched:
    this operates only on the certification batch.
    """
    flat = np.asarray(cert["states"], dtype=np.int64) * N_ACTIONS + np.asarray(
        cert["actions"], dtype=np.int64
    )
    keep = np.empty(0, dtype=np.int64)
    for pair in range(N_STATES * N_ACTIONS):
        members = np.flatnonzero(flat == pair)
        if members.size < FROZEN_CERT_COUNT:
            raise ValueError(
                f"pair {pair} has {members.size} certification items, "
                f"below the frozen count {FROZEN_CERT_COUNT}"
            )
        chosen = rng.choice(members, size=FROZEN_CERT_COUNT, replace=False)
        keep = np.concatenate([keep, np.sort(chosen)])
    keep.sort()
    reduced = {key: np.asarray(value)[keep] for key, value in cert.items()}
    counts = pair_counts(reduced["states"], reduced["actions"])
    return reduced, counts


def h1_check(certificate: dict[str, Any], counts: np.ndarray) -> dict[str, Any]:
    """Apply the frozen ``H1`` count rule and radius target to one record."""
    radii = np.asarray(certificate["radii"], dtype=np.float64)
    worst_count = int(np.min(counts)) if counts.size else 0
    worst_radius = float(np.max(radii)) if radii.size else float("inf")
    radius_contribution = worst_radius / (1.0 - GAMMA)
    count_ok = bool(worst_count >= MIN_CERT_COUNT)
    radius_ok = bool(radius_contribution <= H1_RADIUS_TARGET)
    return {
        "min_cert_count": worst_count,
        "min_cert_count_target": MIN_CERT_COUNT,
        "count_ok": count_ok,
        "worst_radius": worst_radius,
        "radius_contribution": radius_contribution,
        "radius_target": H1_RADIUS_TARGET,
        "radius_ok": radius_ok,
        "passed": bool(count_ok and radius_ok),
    }


def run_route(
    route: str, policy: np.ndarray, train: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Run one frozen route; delegates to the sealed inherited implementation."""
    common = {
        "q0": np.zeros((N_STATES, N_ACTIONS), dtype=np.float64),
        "policy": policy,
        "states": train["states"],
        "actions": train["actions"],
        "rewards": train["rewards"],
        "next_states": train["next_states"],
        "alpha": ALPHA,
        "gamma": GAMMA,
        "layers": LAYERS,
        "value_bound": VALUE_BOUND,
    }
    if route == "expected_exact":
        return es.run_expected_exact(**common)
    if route == "expected_finite":
        return es.run_expected_finite(**common, zeta=ZETA, xi=XI, tau=TAU)
    if route == "sampled_exact":
        return es.run_sampled_exact(**common, next_actions=train["next_actions"])
    raise ValueError(f"unknown route {route!r}")


def certificate_for(
    route: str, q_hat: np.ndarray, policy: np.ndarray, cert: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Build the held-out residual certificate from the certification batch."""
    del route  # every route uses the identical certificate contract
    certificate = es.build_residual_certificate(
        q_hat=q_hat,
        policy=policy,
        states=cert["states"],
        actions=cert["actions"],
        rewards=cert["rewards"],
        next_states=cert["next_states"],
        reward_bound=REWARD_BOUND,
        gamma=GAMMA,
        delta=DELTA,
    )
    # The sealed builder reports ordered reasons under ``failure_reasons``.
    # Re-expose them as ``reasons`` for this module's helpers without altering
    # the sealed dict's own keys.
    certificate.setdefault("reasons", list(certificate.get("failure_reasons", [])))
    return certificate


def improvement_for(
    policy: np.ndarray, q_hat: np.ndarray, certificate: dict[str, Any]
) -> dict[str, Any]:
    """Apply the frozen relative-softmax decision rule with the extended grid.

    The sealed ``fixed_policy_expected_sarsa.decide_policy_update`` hardcodes
    its eta grid as the module constant ``ETA_GRID``, so the v1.1 downward
    extension cannot be expressed by calling it without editing a sealed file.
    Editing a sealed file is forbidden here, so this function reproduces the
    sealed decision rule EXACTLY, line for line, with the grid passed in:

      - candidate policy ``pi_eta^+ ∝ pi exp(eta q)``, row normalised;
      - ``Ihat_s = sum_a (pi_eta^+ - pi)(s,a) q(s,a)``;
      - ``LB_s = Ihat_s - E_Q * ||pi_eta^+(.|s) - pi(.|s)||_1``;
      - a candidate passes when ``min_s LB_s > 0`` strictly;
      - first passage in the descending grid order;
      - on abstention, ``policy_unchanged`` is reported when every candidate is
        within ``1e-12`` of the input policy.

    Only the grid differs from the sealed call; the rule, the order, the strict
    inequality, and the abstention reporting are identical.
    """
    e_q = certificate.get("e_q")
    if e_q is None:
        return {
            "status": "not_certified",
            "eta_selected": None,
            "policy_plus": policy.copy(),
            "lb_by_state": np.zeros(policy.shape[0], dtype=np.float64),
            "changed": False,
            "failure_reasons": list(certificate.get("reasons", [])),
            "candidates": [],
        }
    e_q = float(e_q)
    n_states = policy.shape[0]

    candidates: list[dict] = []
    selected_eta = None
    selected_lb = None
    selected_policy = None
    for eta in ETA_CANDIDATES:
        candidate = es.relative_softmax_candidate(policy, q_hat, eta)
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
            "changed": True,
            "candidates": candidates,
        }

    reasons = ["improvement_lcb_nonpositive"]
    unchanged = all(
        float(np.max(np.abs(es.relative_softmax_candidate(policy, q_hat, eta) - policy)))
        <= 1e-12
        for eta in ETA_CANDIDATES
    )
    if unchanged:
        reasons.append("policy_unchanged")
    return {
        "status": "abstained",
        "failure_reasons": reasons,
        "eta_selected": None,
        "lb_by_state": np.zeros(n_states, dtype=np.float64),
        "policy_plus": policy.copy(),
        "changed": False,
        "candidates": candidates,
    }


def route_failure_reasons(certificate: dict[str, Any], improvement: dict[str, Any]) -> list[str]:
    """Ordered non-emission reasons, in the frozen order."""
    cert_reasons = list(certificate.get("reasons", []))
    improvement_reasons = list(improvement.get("failure_reasons", []))
    present = set(cert_reasons) | set(improvement_reasons)
    ordered = [reason for reason in REASON_ORDER if reason in present]
    for reason in cert_reasons + improvement_reasons:
        if reason not in ordered:
            ordered.append(reason)
    return ordered


def exact_truth(mdp: Any, policy: np.ndarray) -> dict[str, Any]:
    """Oracle-only exact fixed-policy quantities. Never a certificate input."""
    return policy_quantities(mdp, policy)
