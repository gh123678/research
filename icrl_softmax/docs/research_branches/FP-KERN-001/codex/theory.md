# FP-KERN-001 Codex method note

## Claim boundary

This route tests empirical cross-state generalization. It does not prove a
confidence bound, safe policy update, or non-degradation theorem. Exact MDP
quantities are generated only after observable route outputs freeze and are
used solely for audit metrics.

## Observable construction

Three independent stationary-start trajectories are generated for each fixed
MDP and policy. The value trajectory yields the unchanged exact-matching
V-first nuisance estimate `V_hat`. The signature trajectory yields

```text
q_sig(s,b) = mean[R + gamma V_hat(S_next) | (S,A)=(s,b)].
```

The target trajectory yields counts `N_target(s,a)` and recovery-target sums.
The target action is excluded from every primary state signature. For target
action `a`, states `s,s_prime` need at least two commonly observed actions
`b != a`, and

```text
d_a(s,s_prime)^2
  = mean_b [(q_sig(s,b)-q_sig(s_prime,b))/(2B)]^2.
```

The Gaussian bandwidth is the sorted-float64 positive median of all eligible
cross-state distances for that record and action. Missing or degenerate
bandwidths cause abstention. With

```text
K_a(s,s_prime) = exp(-d_a(s,s_prime)^2/(2 h_a^2)),
```

the pooled target estimate is

```text
Q_hat(s,a)
  = sum_s_prime K_a(s,s_prime) target_sum(s_prime,a)
    / sum_s_prime K_a(s,s_prime) N_target(s_prime,a).
```

The observation-weighted effective sample size is

```text
ESS = (sum K*N)^2 / sum K^2*N.
```

The route can estimate a zero-target-count pair only when another state has
target-action observations and the non-target signatures support a finite
kernel. It cannot create information when every source count is zero.

## Shared nuisance boundary

`V_hat` is estimated from an independent trajectory and shared by all routes.
It represents fixed-policy continuation and therefore depends on the complete
policy, including target-action probabilities at successor states. “Leave one
action out” means that action `a` is absent from the state-signature
coordinates; it does not mean the nuisance value is independent of action
`a`. This is a common dependence, not target-trajectory leakage, and no safety
claim is derived from it.

## Why the positive control is necessary

The current generator independently samples transition and reward rows for
each state-action pair. State labels have no metric meaning and there is no
guaranteed relationship between the other-action signature and the omitted
target action. A failure there can therefore mean the environment lacks
transferable structure.

The positive control hides two balanced state clusters. For every cluster and
action, transition and reward rows share a 0.90 prototype component and retain
0.10 independent state variation. The existing sticky transition and action
zero reward bonus are then applied. Cluster labels and prototypes are withheld
from the kernel. Success only in this family establishes conditional, rather
than general, feasibility.

## Baselines

- `local_unpooled` uses only the target state's observed target-action values.
- `action_only_pool` pools that action over all states without similarity.
- `same_action_anchor_kernel` uses target-action signature information and is
  restricted to positive target counts.
- `leave_one_action_out_kernel` is the primary route.

The primary route must improve both missing-pair estimation and downstream
action ordering under the pre-registered screen. Coverage failures remain in
the denominator, and empty per-record count bins cannot be silently treated as
successes.

## Interpretation

`GENERAL_FEASIBLE` means the primary screen passes both the hidden-cluster and
current families. `STRUCTURE_CONDITIONAL` means the method works only when the
environment contains shared structure. `NOT_SUPPORTED` means it fails even the
positive-control screen. `INVALID` is reserved for implementation,
provenance, isolation, or reconstruction failure.

