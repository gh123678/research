# Fixed-policy action-gap certificates and one safe policy update

Task: `FP-ADV-001`

Status: `VERIFIED`

Frozen scientific baseline: `c579047950dfabb2600020cd2e53dd24b3e39c84`

## Scope of the result

This result answers a narrow question.  Starting from one fixed policy and the
simultaneous fixed-policy confidence event already verified by `FP-TU-001`, it
shows how to certify only the action differences actually used by one policy
update.  It does not claim repeated policy iteration, convergence to an
optimal policy, or useful update emission.

The distinction is important:

- a positive certificate is sufficient for a pointwise non-degrading policy
  update;
- an absent certificate means abstention, not evidence that the candidate
  action is worse;
- the frozen experiment produced no positive certificates, so the theorem is
  valid but empirically inactive on that matrix.

## Reused simultaneous event

Let `E` be the `FP-TU-001` event, with `P(E^c) <= delta`, simultaneous over all
pre-registered state Bellman, pair Bellman, and recovery groups.  Every bound,
receiver choice, donor choice, and update below is a deterministic consequence
of `E`.  No new union bound or risk split is introduced after observing the
trajectory.

Consequently, the probability statement is selective:

```text
P(EmitUpdate and (
    any used action ordering is false
    or exists s: V^{pi_plus}(s) < V^pi(s)
)) <= delta.
```

There is no claim that an update is emitted with probability at least
`1-delta`, and no coverage claim conditional on emission.

## Exact V-first local action gap

For a visited pair `g=(s,a)`, define

```text
q_hat_g = mean_{t:g_t=g}[R_{t+1} + gamma V_hat(S_{t+1})]
P_bar_g = empirical successor row for those visits
r_g     = the inherited recovery radius at the observed count
```

Suppose the inherited exact state route emits
`||V_hat-V^pi||_infinity <= E_V`.  For two actions `a,b` in the same state,
the recovery terms contribute at most `r_sa+r_sb`.  With
`e=V_hat-V^pi`, the shared value-estimation error is

```text
gamma |(P_bar_sa-P_bar_sb)^T e|.
```

Because

```text
|(p-q)^T e| <= TV(p,q) span(e)
span(e) <= 2 ||e||_infinity,
```

the observable pair-specific uncertainty is

```text
U_exact(s,a,b)
  = r_sa + r_sb
    + 2 gamma E_V TV(P_bar_sa,P_bar_sb).

LCB_exact(s,a,b)
  = q_hat(s,a)-q_hat(s,b)-U_exact(s,a,b).
```

On `E`, `LCB_exact>0` certifies the strict true ordering
`Q^pi(s,a)>Q^pi(s,b)`.

The transition-row difference is the source of localization.  If two actions
have similar empirical successor rows, their common value-function error
partly cancels.  Replacing total variation by its worst-case value one recovers
the corresponding global complete-Q penalty, so the local penalty cannot be
wider when that global control emits.

## Finite-softmax V-first local action gap

For each pair query, normalize the frozen one-hot softmax weights.  Let
`kappa_g` be their total mass on the queried group and let
`P_bar_g^beta` be the associated normalized effective successor row.  With
`B=R_star/(1-gamma)`, every fixed-policy recovery target lies in `[-B,B]`.

The on-group recovery error is at most `kappa_g r_g`.  Off-group mass can
differ from the queried target by at most `2B`, giving

```text
C_g = kappa_g r_g + 2B(1-kappa_g).
```

If the finite-softmax state route emits
`||V_hat_beta-V^pi||_infinity <= E_V^beta`, the same span argument yields

```text
U_soft(s,a,b)
  = C_sa + C_sb
    + 2 gamma E_V^beta
        TV(P_bar_sa^beta,P_bar_sb^beta).

LCB_soft(s,a,b)
  = q_hat_beta(s,a)-q_hat_beta(s,b)-U_soft(s,a,b).
```

This construction needs neither a true transition kernel nor a true Q table.
Its price is the explicit `2B(1-kappa_g)` contamination term.  As in the exact
case, total variation is at most one, so the local penalty is no wider than the
matching complete-Q V-first control under the frozen definitions.

## Safe single-step update

For each state, select the smallest-index maximizer

```text
a_star(s) = argmax_a q_hat(s,a).
```

A donor `b != a_star` is eligible only if both selected pair counts are
positive, the applicable LCB is strictly positive, and
`pi(b|s)>pi_min`.  With the frozen transfer fraction `theta=1/2`, move

```text
eta(s,b) = 0.5[pi(b|s)-pi_min]
```

from every eligible donor to `a_star`.  If no donor is eligible, return the
original policy exactly.

The transfer preserves row sums, nonnegativity, and the exploration floor.
On `E`, every transferred donor satisfies

```text
Q^pi(s,a_star)-Q^pi(s,b) >= LCB(s,a_star,b) > 0.
```

Therefore

```text
(T_{pi_plus}V^pi)(s)-V^pi(s)
  = sum_b eta(s,b)[Q^pi(s,a_star)-Q^pi(s,b)]
  >= sum_b eta(s,b)LCB(s,a_star,b)
  >= 0.
```

Bellman monotonicity and contraction then give
`V^{pi_plus}>=V^pi` componentwise.

## What is and is not proved

The task proves that an emitted update is safe on the reused event.  It also
proves the exact and softmax local penalties and their deterministic dominance
over the matching global V-first penalties.  It does not prove that the
certificates are strong enough to emit, and it does not turn this one-step
result into an iterative model-free algorithm.

The frozen formal experiment is therefore a valid negative usefulness result:
all six routes abstained in all 480 records.  The next research question is not
whether this proof can be restated, but whether a fixed-policy Expected-SARSA
construction can obtain tighter action-level information while retaining the
same separation between evaluation and policy improvement.
