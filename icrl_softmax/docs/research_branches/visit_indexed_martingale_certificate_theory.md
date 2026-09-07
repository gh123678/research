# Visit-indexed martingale certificate: verified shared theory

Status: `VERIFIED` under task `FP-MART-001` on 2026-09-04. This document is
the reconciled theorem accepted after independent GPT and Claude construction
and reciprocal reproduction.

## 1. Setting and claim

Let the state space have size `m`, let the target state-action support have
size `d`, and fix a Markov policy `pi`. One trajectory of `n` transitions is
sampled in the order

```text
S_t -> A_t -> (R_{t+1}, S_{t+1}) -> A_{t+1}.
```

Rewards are deterministic functions of transition edges and satisfy the
declared pre-sampling bound `|R| <= R_star`. For `0 < gamma < 1`, define

```text
B = R_star / (1 - gamma).
```

Then `||V^pi||_infinity <= B` and `||Q^pi||_infinity <= B`. No stationary
occupancy, transition kernel, spectral gap, `V^pi`, `Q^pi`, true residual, or
true initial error is an input to the emitted certificate.

The verified probability statement is selective:

```text
P(Emit and an emitted bound is violated) <= delta.
```

It is not a claim that full support or emission occurs with probability at
least `1-delta`, and it is not conditional coverage given emission.

## 2. Filtrations and centered residuals

Use the pre-action filtration

```text
G_t = sigma(S_0, A_0, R_1, S_1, ..., A_{t-1}, R_t, S_t)
```

and the post-action filtration `H_t = sigma(G_t, A_t)`. Thus
`G_t subset H_t subset G_{t+1}`.

For `X_t = (S_t,A_t)`, define three fixed-target residuals:

```text
xi^V_t = R_{t+1} + gamma V^pi(S_{t+1}) - V^pi(S_t),
xi^Q_t = R_{t+1} + gamma Q^pi(X_{t+1}) - Q^pi(X_t),
xi^R_t = R_{t+1} + gamma V^pi(S_{t+1}) - Q^pi(X_t).
```

The state residual is centered given `G_t`; the pair Bellman and recovery
residuals are centered given `H_t`. Conditioning the state residual on `H_t`
would generally expose the advantage and is therefore invalid. The verifier
contains an explicit two-state counterexample for this wrong filtration.

Given the appropriate filtration, the subtracted current-target term is
fixed. The varying term lies in
`[-R_star-gamma B, R_star+gamma B]`, so every conditional range has width at
most

```text
2 R_star + 2 gamma B = 2B.
```

This conditional width, rather than a crude `4B` width inferred from an
absolute residual bound, determines the verified constant.

## 3. Predictable visit selection and the simultaneous event

For a state group use `I_t^s = 1{S_t=s}`, which is `G_t`-measurable. For a
pair group use `I_t^x = 1{X_t=x}`, which is `H_t`-measurable. For each fixed
count `k <= n`, retain only the first `k` visits with the predictable selector

```text
J_t^{g,k} = I_t^g 1{sum_{u<=t} I_u^g <= k}.
```

Then `D_t^{g,k} = J_t^{g,k} xi_t` is a martingale difference and its
conditional range width is at most `2B J_t^{g,k}`. The GPT stopped-process
presentation and the Claude predictable-transform presentation are equivalent
ways to establish this fact.

For fixed `g`, `k`, and `lambda`, define the compensated exponential process

```text
M_t = exp(lambda sum_{u<t} D_u^{g,k}
          - lambda^2 B^2/2 sum_{u<t} J_u^{g,k}).
```

Conditional Hoeffding's lemma makes `M_t` a positive supermartingale, so
`E[M_n] <= 1`. Since `sum J <= k` pathwise, exponential Markov optimization
gives

```text
P(|sum_{j=1}^{min(k,N_g)} xi_{tau_g(j)}| >= epsilon)
  <= 2 exp(-epsilon^2 / (2 B^2 k)).
```

There are `G = m + 2d` groups: `m` state residual groups, `d` pair Bellman
groups, and `d` recovery groups. A union bound over every group and every
`1 <= k <= n` yields, with probability at least `1-delta`,

```text
|mean residual for the first k visits of group g|
  <= r_H(k,delta)
  = B sqrt(2 log(2 G n / delta) / k)
```

simultaneously for all `(g,k)`. Substituting the random final observed count
`N_g` is valid because that count was already included in this simultaneous
event; no conditioning on `N_g` or independence between counts and residuals
is used.

## 4. Deterministic certificate composition

On the simultaneous event, freeze the trajectory. Let `r_X` and `r_S` be the
worst pair and state radii obtained from their observed minimum positive
counts. Zero initialization gives the computable initial bound `B`.

For Direct-Q exact matching,

```text
rho = 1 - alpha(1-gamma),
U_Q(L) = rho^L B + (1-rho^L) r_X/(1-gamma).
```

For Direct-Q finite softmax, let `d_min` be the minimum observed empirical
one-hot diagonal and

```text
c_Q = d_min - (1+gamma)/2.
```

When `c_Q > 0`, use `rho = 1-2 alpha c_Q` and replace the statistical
denominator by `2c_Q`.

The state-value stage has the same recurrence with state counts and state
diagonals. Its bound `U_V(L)` composes into V-first no-split recovery as

```text
U_VF_exact(L) = gamma U_V(L) + r_X,
U_VF_soft(L)  = gamma U_V(L) + r_X + 2B(1-d_min).
```

The same trajectory may be used for the value and recovery stages because the
recovery analysis is written around the fixed ghost target `V^pi`; no false
independence is invoked. V-first softmax recovery is not an iterated pair
operator and therefore does not require the Direct-Q pair contraction-margin
gate.

## 5. Emission gates and machine contract

Emission requires observed support for every group used by the route, positive
route-relevant empirical margins, fixed-context synchronous mode, a valid risk
budget, no divergence-guard trigger, and finite arithmetic. Failures are
reported deterministically and no missing-support fallback uses true
occupancy.

Counts must be original integers, have the declared dimensions, sum to the
trajectory horizon, and have state totals equal to the state aggregation of
pair counts. Machine outputs are strict JSON, use `null` for unavailable
quantities, place new values under `visit_indexed_certificate`, and place all
truth-based diagnostics under a separate `oracle_audit` namespace.

## 6. Optional variance-adaptive route

Neither independent route proved or implemented a nontrivial observable
variance proxy from the frozen allowed inputs. The optional constituent is
therefore emitted as `variance_adaptive_unavailable` and receives no risk
share. This is a construction gap, not an impossibility theorem. No post-hoc
minimum of separately calibrated bounds is taken.

## 7. Scope and sources

The result covers a fixed policy, fixed context, synchronous updates, a finite
state-action space, deterministic bounded edge rewards, and the frozen
stationary-start evaluation. It does not establish online policy improvement,
nonstationary-start guarantees, or general stochastic-reward guarantees.

The concentration step is proved above; attribution follows Hoeffding's
bounded-sum inequality and Azuma's martingale extension:

- W. Hoeffding (1963), https://doi.org/10.1080/01621459.1963.10500830.
- K. Azuma (1967), https://doi.org/10.2748/tmj/1178243286.
- D. Freedman (1975), https://doi.org/10.1214/aop/1176996452, consulted for
  the optional route but not used in the mandatory certificate.
