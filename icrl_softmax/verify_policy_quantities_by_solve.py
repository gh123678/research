"""Check `policy_quantities` against a method that shares none of its code.

Every same-actor check in this repository compares numbers produced through
`evaluate_fixed_policy_q_routes.policy_quantities`. That routine defines Q^pi, v^pi
and mu_state, and it is simultaneously the audit's ground truth and the census's
yardstick. If it were wrong, all 29 checkers would be wrong in the same direction.

This recomputes the same quantities by two routes that share no code with it:

  1. a DIRECT LINEAR SOLVE of the policy-evaluation Bellman system,
     (I - gamma * P_pi) Q = R_pi, solved as a (d*A) x (d*A) linear system;
  2. VALUE ITERATION of v under pi to a tight tolerance.

and compares against `policy_quantities`. It also checks the two identities that
tie the three quantities together (Q from v, and v as the policy-weighted Q), so a
self-consistent but wrong convention is caught too.

This is a DIFFERENT METHOD, SAME ACTOR. It rules out a coding error in
`policy_quantities`; it does not make the line independently verified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent / "icrl_softmax"
sys.path.insert(0, str(PROJECT))

import fixed_policy_expected_sarsa_scaled as fs  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402

CASES = [(0.08, 0), (0.08, 5), (0.08, 11), (0.5, 0), (0.5, 4), (0.5, 11)]
TOL = 1e-10


def solve_directly(mdp, policy, gamma: float):
    """Q^pi by one linear solve of the (d*A) Bellman system."""
    transition = np.asarray(mdp["P"], dtype=np.float64)
    reward = np.asarray(mdp["R"], dtype=np.float64)
    n_states, n_actions = policy.shape
    size = n_states * n_actions

    # Expected immediate reward per (s,a): R(s,a) = sum_s' P(s'|s,a) R(s,a,s').
    r_sa = np.einsum("sap,sap->sa", transition, reward)

    # I - gamma * P_pi on the flattened (s,a) index. The coefficient of Q(s',a')
    # in sum_s' P(s'|s,a) v(s') with v(s') = sum_a' pi(a'|s') Q(s',a') is
    # P(s'|s,a) * pi(a'|s').
    operator = np.eye(size)
    for s in range(n_states):
        for a in range(n_actions):
            row = s * n_actions + a
            for s_next in range(n_states):
                prob = transition[s, a, s_next]
                if prob == 0.0:
                    continue
                for a_next in range(n_actions):
                    operator[row, s_next * n_actions + a_next] -= (
                        gamma * prob * policy[s_next, a_next]
                    )
    q = np.linalg.solve(operator, r_sa.reshape(-1)).reshape(n_states, n_actions)
    return q


def value_iteration(mdp, policy, gamma: float, iterations: int = 200000):
    """v^pi by iterating v <- sum_a pi(a|s) [R + gamma P v] to convergence."""
    transition = np.asarray(mdp["P"], dtype=np.float64)
    reward = np.asarray(mdp["R"], dtype=np.float64)
    n_states = policy.shape[0]
    r_sa = np.einsum("sap,sap->sa", transition, reward)
    v = np.zeros(n_states)
    for _ in range(iterations):
        q = r_sa + gamma * np.einsum("sap,p->sa", transition, v)
        v_new = np.einsum("sa,sa->s", policy, q)
        if np.max(np.abs(v_new - v)) < 1e-14:
            return v_new
        v = v_new
    return v


def stationary_distribution(policy, transition):
    """mu by solving mu = mu P_pi with the normalisation row."""
    n_states = policy.shape[0]
    p_pi = np.einsum("sa,sap->sp", policy, transition)
    system = (p_pi.T - np.eye(n_states))
    system[-1, :] = 1.0
    rhs = np.zeros(n_states)
    rhs[-1] = 1.0
    return np.linalg.solve(system, rhs)


def main() -> int:
    failures = 0
    print("policy_quantities vs two independent methods")
    print("=" * 78)
    print(f"{'mixing':>7} {'task':>5}  {'Q gap':>11} {'v gap':>11} "
          f"{'Q-from-v gap':>13} {'v-from-Q gap':>13} {'mu gap':>11}")
    for mixing, task_index in CASES:
        mdp, policy, _ = fs.build_task(task_index=task_index, mixing=mixing)
        exact = policy_quantities(mdp, policy)
        q_ref = np.asarray(exact["q_pi"], dtype=np.float64)
        v_ref = np.asarray(exact["v_pi"], dtype=np.float64)

        q_solved = solve_directly(mdp, policy, fs.GAMMA)
        v_iterated = value_iteration(mdp, policy, fs.GAMMA)

        transition = np.asarray(mdp["P"], dtype=np.float64)
        reward = np.asarray(mdp["R"], dtype=np.float64)
        r_sa = np.einsum("sap,sap->sa", transition, reward)
        # Q from v: R(s,a) + gamma * sum_s' P(s'|s,a) v(s').
        q_from_v = r_sa + fs.GAMMA * np.einsum("sap,p->sa", transition, v_ref)
        # v from Q: policy-weighted Q.
        v_from_q = np.einsum("sa,sa->s", policy, q_ref)
        # mu: the stationary law of the chain induced by pi.
        mu_solved = stationary_distribution(policy, transition)
        mu_ref = np.asarray(exact["mu_state"], dtype=np.float64)

        gaps = (
            float(np.max(np.abs(q_ref - q_solved))),
            float(np.max(np.abs(v_ref - v_iterated))),
            float(np.max(np.abs(q_ref - q_from_v))),
            float(np.max(np.abs(v_ref - v_from_q))),
            float(np.max(np.abs(mu_ref - mu_solved))),
        )
        ok = all(g < TOL for g in gaps)
        if not ok:
            failures += 1
        print(
            f"{mixing:>7} {task_index:>5}  {gaps[0]:>11.3e} {gaps[1]:>11.3e} "
            f"{gaps[2]:>13.3e} {gaps[3]:>13.3e} {gaps[4]:>11.3e}   "
            f"{'OK' if ok else 'MISMATCH'}"
        )

    print("=" * 78)
    if failures:
        print(f"FAIL: {failures} of {len(CASES)} records disagree beyond {TOL:.0e}")
        return 1
    print(
        f"PASS: all {len(CASES)} records agree with a direct linear solve, with "
        f"value iteration, on the stationary law, and satisfy both cross-identities, "
        f"to < {TOL:.0e}."
    )
    print(
        "LIMITATION: different method, SAME ACTOR. This rules out a coding error in\n"
        "policy_quantities. It does not make the line independently verified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
