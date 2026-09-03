# FP-MART-001 Claude route: visit-indexed martingale certificate theory

> Author: Claude Code (auxiliary researcher), independent construction.
> Date: 2026-09-03. Branch `claude/FP-MART-001`.
> Frozen contract: `docs/research_tasks/FP-MART-001.md` v1.0,
> design `2026-09-03-visit-indexed-martingale-certificate-design.md`.
> This document is written before any implementation. It derives every step of
> the mandatory Hoeffding route from the frozen sampling order; the
> concentration proof is fully self-contained so that no imported theorem's
> assumptions are load-bearing (web fetch of primary PDFs was blocked in this
> environment; bibliographic records are cited for attribution).

## 1. Setting

Finite state space `|S| = m`, action space `|A|`, fixed Markov policy
`pi(a|s)` with full support on the `d` target pairs `X_pi` (in the frozen
matrix `d = m * n_actions = 24`). One trajectory of `n` transitions sampled in
the frozen order

```
S_t -> A_t -> (R_{t+1}, S_{t+1}) -> A_{t+1},
```

where `A_t ~ pi(.|S_t)`, `(R_{t+1}, S_{t+1}) ~ (r(S_t,A_t,.), P(S_t,A_t,.))`,
the edge reward is deterministic with `|R| <= R_star`, and

```
B = R_star / (1 - gamma),   ||V^pi||_inf <= B,   ||Q^pi||_inf <= B.
```

## 2. Filtrations

For `t = 0, ..., n` define

- pre-action:  `G_t = sigma(S_0, A_0, R_1, S_1, ..., A_{t-1}, R_t, S_t)`;
- post-action: `H_t = sigma(G_t, A_t)`.

`G_t` contains the history through `S_t`; `H_t` adds `A_t`. Both are
increasing, `G_t subset H_t subset G_{t+1}`.

## 3. Centered residual families and conditional ranges

Define for each transition time `t = 0, ..., n-1`:

- state Bellman residual (per state group `s`):
  `xi^V_t = R_{t+1} + gamma * V^pi(S_{t+1}) - V^pi(S_t)`;
- pair Bellman residual (per pair group `z`):
  `xi^Q_t = R_{t+1} + gamma * Q^pi(X_{t+1}) - Q^pi(X_t)`;
- fixed-`V^pi` recovery residual (per pair group `z`):
  `xi^R_t = R_{t+1} + gamma * V^pi(S_{t+1}) - Q^pi(X_t)`,

with `X_t = (S_t, A_t)`. These match the frozen code's
`observed_ghost_residuals` definitions (`rewards[t] = R_{t+1}` alignment).

**Centering.** `xi^V_t` is `G_{t+1}`-measurable and

```
E[xi^V_t | G_t]
  = E_{A_t~pi(.|S_t)} E[ r(S_t,A_t,S') + gamma V^pi(S') | S_t, A_t ] - V^pi(S_t)
  = r^pi(S_t) + gamma (P^pi V^pi)(S_t) - V^pi(S_t) = 0
```

by the Bellman equation for `V^pi`; the outer expectation over `A_t` is valid
because `pi` is a fixed Markov policy, so `A_t` depends on the `G_t`-history
only through `S_t in G_t`. `xi^Q_t` and `xi^R_t` are `H_{t+1}`-measurable and

```
E[xi^Q_t | H_t] = r_bar(S_t,A_t) + gamma sum_s' P(S_t,A_t,s') V^pi(s')
                - Q^pi(S_t,A_t) = 0,
E[xi^R_t | H_t] = same conditional mean - Q^pi(S_t,A_t) = 0,
```

using `V^pi(s') = sum_a' pi(a'|s') Q^pi(s',a')` and the Bellman equation for
`Q^pi`. Conditioning a state residual on `H_t` instead gives the advantage
`A^pi(S_t,A_t)`, which is not zero in general (Section 11 gives a numerical
counterexample fixture); the pre-action centering is essential.

**Exact conditional ranges.** Given `G_t` (resp. `H_t`), the frozen-target
term `V^pi(S_t)` (resp. `Q^pi(X_t)`) is a constant. The varying part
`R_{t+1} + gamma V^pi(S_{t+1})` (resp. `... + gamma Q^pi(X_{t+1})`) lies in
`[-R_star - gamma B, R_star + gamma B]`, so every conditional range has width
at most

```
c = 2 R_star + 2 gamma B = 2 R_star (1 + gamma/(1-gamma)) = 2B.
```

This is the proved constant used everywhere below; it is derived, not copied
from the design's candidate form. (Using the crude `|xi| <= 2B` two-sided
bound directly would give width `4B`; the centering of the frozen target under
the conditioning field halves the width to `2B`.)

## 4. Visit indexing, measurability, optional skipping

For state group `s` let `I^s_t = 1{S_t = s}`; for pair group `z` let
`I^z_t = 1{X_t = z}`. Then `I^s_t` is `G_t`-measurable and `I^z_t` is
`H_t`-measurable: each indicator is known *before* the corresponding residual
is observed. The visit index `nu^g_t = sum_{j<=t} I^g_j` inherits the same
measurability. For each deterministic `k in {1..n}` define the truncated
selection `J^{g,k}_t = I^g_t 1{nu^g_t <= k}`, again measurable in the same
pre-observation field.

Let `tau_g(j) = min{ t : nu^g_t = j }` be the `j`-th visit time of group `g`.
For the state (resp. pair) families `(xi_t, F_{t+1})` with `F in {G, H}` form
a martingale difference sequence (MDS). Because `J^{g,k}_t` is bounded and
`F_t`-measurable, the martingale transform

```
D^{g,k}_t = J^{g,k}_t xi_t
```

satisfies `E[D^{g,k}_t | F_t] = J^{g,k}_t E[xi_t | F_t] = 0`, is
`F_{t+1}`-measurable, and has conditional range width at most `2B J^{g,k}_t`.
Moreover the telescoping identification

```
sum_{t=0}^{n-1} D^{g,k}_t = sum_{j=1}^{min(k, N_g)} xi_{tau_g(j)},
   N_g = nu^g_{n-1} = final observed count of group g,
```

is a pathwise identity. This is the optional-skipping step: we never assume an
arbitrary stopping time is predictable; we only use the verified
pre-observation measurability of `I^g_t` and `nu^g_t`, and the transform MDS
property replaces any sampling of partial sums at stopping times. (Classical
optional-skipping statements identify the subsequence along increasing
stopping times as a martingale; here the equivalent transform form is proved
directly and needs nothing beyond the tower property.)

## 5. Hoeffding-Azuma step (self-contained)

**Hoeffding's lemma.** If `Y` is a random variable with `E[Y|F] = 0` and
`a <= Y <= b` a.s. with `a, b` `F`-measurable, then
`E[e^{lambda Y} | F] <= exp(lambda^2 (b-a)^2 / 8)`. Proof: the exponential is
convex, so `e^{lambda y}` lies below the chord on `[a,b]`; taking conditional
expectations and optimizing the resulting one-variable function
`log(1 - p + p e^{lambda(b-a)}) - p lambda (b-a)` with `p = -a/(b-a)` gives
the bound `lambda^2 (b-a)^2/8`.

**Application.** Fix group `g` and count `k`. Iterate the conditional lemma
along `t = n-1, ..., 0` for `D^{g,k}_t`:

```
E[ exp(lambda sum_t D^{g,k}_t) ]
  <= E[ exp(lambda^2 sum_t (2B J^{g,k}_t)^2 / 8) ]
  <= exp(lambda^2 B^2 k / 2),
```

because `sum_t J^{g,k}_t <= k` pathwise. Exponential Markov and optimization
at `lambda = epsilon / (B^2 k)` give, for every `epsilon > 0`,

```
P( |sum_{j=1}^{min(k,N_g)} xi_{tau_g(j)}| >= epsilon )
  <= 2 exp( - epsilon^2 / (2 B^2 k) ).
```

This is the Azuma-Hoeffding bound specialized to our MDS with conditional
range width `2B`; the two-sided factor `2` is explicit.

## 6. The simultaneous visit-indexed event

There are `G = m + 2d` residual groups (`m` state Bellman, `d` pair Bellman,
`d` recovery). Allocate risk uniformly over groups and counts:
`delta_{g,k} = delta / (G n)`. Choose

```
epsilon_k = B sqrt(2 k log(2 G n / delta)),
r_H(k, delta) = epsilon_k / k = B sqrt(2 log(2 G n / delta) / k).
```

Then each `(g,k)` failure probability is at most `2 exp(-epsilon_k^2/(2 B^2
k)) = delta/(G n)`, and one union bound gives an event `E_sim` with
`P(E_sim) >= 1 - delta` on which, simultaneously for every group `g` and every
`k in {1..n}`,

```
|sum_{j=1}^{min(k,N_g)} xi_{tau_g(j)}| <= epsilon_k.
```

No stationarity, occupancy, transition-matrix, or spectral quantity appears.

## 7. Random-count substitution

On `E_sim`, for every group with observed final count `N_g >= 1`, plugging
`k = N_g` (a choice inside the simultaneous event) yields

```
|bar_delta_g| = |N_g^{-1} sum_{j=1}^{N_g} xi_{tau_g(j)}| <= r_H(N_g, delta).
```

Because the event covers all `k` at once, substituting the *random* count
`N_g` requires no conditioning on `N_g` and no independence between counts and
residuals. Dependence across groups is irrelevant to the union bound.

## 8. Route composition on `E_sim` (deterministic)

Fix the trajectory. All empirical operators are then frozen; every claim below
is pathwise deterministic on `E_sim`. Zero initialization gives the computable
initial-error bounds `||Q_0 - Q^pi||_inf <= B` and `||V_0 - V^pi||_inf <= B`.
Data-dependent early stopping (layer count `L` chosen by the frozen update
rule) is covered because the recurrences hold for every `L >= 0`. Let
`N^X_min = min_z N_z` over target pairs and `N^S_min = min_s N_s` over states;
`r_H` is nonincreasing in `k`, so worst-group radii are `r_H(N^X_min)` and
`r_H(N^S_min)`.

1. **Direct-Q, exact matching.** On observed full pair support, the frozen
   empirical operator satisfies
   `||Fhat_{Q,1}(Q^pi) - Q^pi||_inf <= max_z |bar_delta^Q_z| <=
   r_H(N^X_min)`, hence with `rho = 1 - alpha(1-gamma)`,

   ```
   ||Q_L - Q^pi||_inf <= rho^L B + (1 - rho^L) r_H(N^X_min) / (1 - gamma).
   ```

2. **Direct-Q, finite softmax.** With the observed one-hot empirical diagonal
   `d_hat(x) = m_beta(N_x/n)`, `m_beta(u) = e^{beta}u/(e^{beta}u + 1 - u)`,
   `c_Q = min_x d_hat(x) - (1+gamma)/2`; if `c_Q > 0`, `rho = 1 - 2 alpha
   c_Q` and

   ```
   ||Q_L - Q^pi||_inf <= rho^L B + (1 - rho^L) r_H(N^X_min) / (2 c_Q).
   ```

3. **State-value stage.** Same form with state counts and state residuals:
   radius `r_H(N^S_min)`, margin `c_V` from the observed state diagonals
   (exact: `c_V = (1-gamma)/2`), clipping to `[-B,B]` nonexpansive.

4. **V-first no-split, exact.** Fixed-`V^pi` ghost recovery residual
   `|bar_delta^R_z| <= r_H(N^X_min)` plus the deterministic gamma-Lipschitz
   recovery argument give

   ```
   ||Qhat_ns,L - Q^pi||_inf <= gamma * eps_{V,L} + r_H(N^X_min),
   ```

   with `eps_{V,L}` the state-stage total bound.

5. **V-first no-split, softmax.** Adds the explicit observed leakage
   `L_beta = 2B(1 - min_x d_hat(x))` computed from observed counts; the
   recovery gate needs no pair contraction margin.

These are exactly the completed deterministic recurrences of
`shared_fixed_policy_finite_sample_theory.md` with the visit-indexed radii in
place of the occupancy-denominator residuals.

## 9. Selective emission semantics

Let `Emit` be the trajectory-based event that all required observed counts are
positive, all route-relevant empirical margins are positive, the algorithm
mode matches the frozen contract (fixed context, synchronous update, no
divergence-guard trigger), the risk budget is valid, and all arithmetic is
finite. `Emit` is `sigma(trajectory)`-measurable and is checked before
emission. On `E_sim`, whenever `Emit` holds, Section 8 shows every emitted
bound is valid. Therefore

```
P(Emit and certified bound violated) <= P(E_sim^c) <= delta.
```

No claim `P(Emit) >= 1 - delta` is made or needed; missing support is honest
non-emission; no conditional-coverage division by `P(Emit)` is used.

## 10. Optional variance-adaptive route: recorded as unavailable

A Freedman or empirical-Bernstein radius at the fixed targets `V^pi, Q^pi`
needs the conditional variance (or an observable surrogate) of
`xi_{tau_g(j)}`. The conditional variance depends on the true values and
kernel; the realized residuals are not observable without the true targets.
Any data-only variance proxy would therefore be an oracle quantity, which the
frozen contract forbids. The only observable bound on the conditional variance
is Popoviciu's `Var <= (b-a)^2/4`, which recovers exactly the Hoeffding
constant already proved. Consequently the variance-adaptive constituent is
reported as `variance_adaptive_unavailable` on every record: hypothesis 7 is a
negative result within the oracle-free constraint, recorded accurately, with
no post-hoc minimum across bounds (Section 11, fixture F2, demonstrates why an
unadjusted minimum would break the risk budget). Hypotheses 1-6 are
unaffected.

## 11. Proof-level counterexample fixtures (used by the verifier)

- **F1, wrong filtration for state residuals.** Explicit 2-state, 2-action
  MDP with deterministic rewards and `pi` non-uniform: direct computation
  shows `E[xi^V_t | H_t] = A^pi(S_t, A_t) != 0` on a positive-probability
  `(s,a)`, while `E[xi^V_t | G_t] = 0`. Hence state residuals must be centered
  pre-action; the post-action centering used for pairs is invalid for states.
- **F2, unadjusted post-hoc minimum.** Two valid radii at level `delta` each:
  the event "at least one fails" has probability up to `2 delta`, so selecting
  the smaller radius after seeing data certifies at level `2 delta`, not
  `delta`. Preallocating `delta/2` to each constituent restores the budget at
  the price of the additive `log 2` inside the radius. The fixture asserts
  both the failure inflation and the exact preallocated radius.

## 12. Sources and assumptions

All probabilistic steps are proved in-line above (Sections 4-6), so no
imported statement is load-bearing. Attribution and bibliographic record:

- Hoeffding's lemma and the two-sided bounded-sum tail bound originate in
  W. Hoeffding, "Probability Inequalities for Sums of Bounded Random
  Variables", JASA 58(301):13-30, 1963,
  https://doi.org/10.1080/01621459.1963.10500830. Assumption used there:
  independent summands in intervals; the martingale relaxation is Azuma's.
- K. Azuma, "Weighted sums of certain dependent random variables", Tohoku
  Math. J. 19(3):357-367, 1967, https://doi.org/10.2748/tmj/1178243286.
  Assumption: martingale differences with bounded ranges; we use only the
  conditional-range form proved in Section 5.
- C. McDiarmid, "On the method of bounded differences", Surveys in
  Combinatorics, LMS Lecture Notes 141:148-188, 1989,
  https://doi.org/10.1017/CBO9781107359949.008 (conditional-range martingale
  form).
- Optional sampling/skipping background: R. Durrett, "Probability: Theory and
  Examples", 5th ed., Cambridge University Press, 2019, ch. 4 (optional
  sampling for stopped filtrations). Our transform proof in Section 4 uses
  only the tower property and verified predictability.
- Completed in-project deterministic recurrences:
  `docs/research_branches/shared_fixed_policy_finite_sample_theory.md`
  (Sections 6-8), which this document reuses unchanged.

Web fetch of these URLs was blocked in this execution environment on
2026-09-03; metadata was cross-checked via web search. Since every imported
idea is re-proved in-line with explicit assumptions, the certificate does not
depend on any unverifiable external claim.

## 13. Limitations

- Finite-horizon union bound: the `log(2Gn/delta)` factor incurs a `log n`
  overhead relative to a fixed-count statement; time-uniform stitched bounds
  are out of scope.
- The radius is range-based; at the frozen scale (`B ~ 5`, `G = 54`) emitted
  bounds are valid but typically numerically loose. Emission validity and
  numerical nontriviality are reported separately.
- Scope remains: one fixed policy, frozen trajectory, deterministic edge
  reward, finite state-action space, no stochastic reward noise, no
  nonstationary start, no online control.
- The variance-adaptive refinement is unavailable under the oracle-free
  input contract (Section 10).
