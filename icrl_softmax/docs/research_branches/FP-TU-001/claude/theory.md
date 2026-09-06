# FP-TU-001 Claude route: time-uniform mixture certificate theory

Route: Claude independent construction on `claude/FP-TU-001` from the common
activation commit `0ce18b4676f70ca0556496e804aa65563efa63da`.

Frozen contract: `docs/research_tasks/FP-TU-001.md` (version 1.0, `ACTIVE`),
design `docs/superpowers/specs/2026-09-04-time-uniform-mixture-certificate-design.md`,
plan `docs/superpowers/plans/2026-09-04-time-uniform-mixture-certificate-plan.md`.

This document is the Claude route's independent proof of the mandatory
time-uniform mixture certificate, written before any implementation. It follows
the twelve proof obligations of plan Task 3. Items inherited from the verified
`FP-MART-001` theory are restated here with exact references to
`docs/research_branches/visit_indexed_martingale_certificate_theory.md`
(hereafter `[MART]`) so the inheritance is auditable.

## 1. Residual families and filtrations (inherited)

Sampling order is `S_t -> A_t -> (R_{t+1}, S_{t+1}) -> A_{t+1}` with
deterministic edge rewards `|R| <= R_star` declared before MDP sampling, and
`B = R_star/(1-gamma)`. The pre-action filtration is

```text
G_t = sigma(S_0, A_0, R_1, S_1, ..., A_{t-1}, R_t, S_t),
H_t = sigma(G_t, A_t),        G_t subset H_t subset G_{t+1}.
```

The three fixed-target residual families of `[MART]` section 2 are

```text
xi^V_t = R_{t+1} + gamma V^pi(S_{t+1}) - V^pi(S_t),   centered given G_t,
xi^Q_t = R_{t+1} + gamma Q^pi(X_{t+1}) - Q^pi(X_t),   centered given H_t,
xi^R_t = R_{t+1} + gamma V^pi(S_{t+1}) - Q^pi(X_t),   centered given H_t,
```

where `X_t = (S_t, A_t)`. The state residual is centered given `G_t` only;
conditioning it on `H_t` exposes the advantage and is invalid (the
`[MART]` verifier's two-state counterexample). This task changes nothing about
the residual definitions, targets, or filtrations.

## 2. Predictable visit selection (inherited)

For a state group `s`, `I_t^s = 1{S_t = s}` is `G_t`-measurable; for a pair
group `x`, `I_t^x = 1{X_t = x}` is `H_t`-measurable. For each fixed count cap
`k`, `[MART]` section 3 retains the first `k` visits with the predictable
selector `J_t^{g,k} = I_t^g 1{sum_{u<=t} I_u^g <= k}` and shows
`D_t^{g,k} = J_t^{g,k} xi_t` is a martingale difference in the appropriate
filtration (`G_t` for state groups, `H_t` for pair groups). The stopped-process
and predictable-transform presentations are equivalent (`[MART]` section 3).

This task uses the uncapped transform `D_t^g = I_t^g xi_t` and the cumulative
visit count `C_t^g = sum_{u<=t} I_u^g`, both predictable in the same sense:
`I_t^g` is measurable with respect to the conditioning field (`G_t` or `H_t`).

## 3. Conditional centering and range width (inherited)

By `[MART]` section 2, given the appropriate field the subtracted
current-target term is fixed and the varying term lies in
`[-R_star - gamma B, R_star + gamma B]`. Writing `mu_t` for the conditional
mean of that varying term, the residual `xi_t` is conditionally centered and
supported in the shifted interval
`[-R_star - gamma B - mu_t, R_star + gamma B - mu_t]`, whose width is at most
`2B` and which contains `0` (the conditional mean of a centered variable lies
in its support interval). Width alone does NOT imply that this interval
equals `[-B, B]`: the conditional mean `mu_t` need not vanish, so the support
can be asymmetric or even partly outside `[-B, B]`. The normalized increment
`Y = xi_t / B` is therefore conditionally centered with support in an
interval `[c, d]` satisfying `d - c <= 2` and `c <= 0 <= d`. Section 4 uses
exactly these two properties and nothing stronger.

## 4. Fixed-rate exponential supermartingale

Conditional Hoeffding lemma, interval form: if `Y` is conditionally centered
given a field `F` and `Y in [c, d]` almost surely with `d - c <= 2`, then for
every real `a`,

```text
E[exp(a Y) | F] <= exp(a^2 (d - c)^2 / 8) <= exp(a^2 / 2).
```

Proof: `y -> exp(a y)` is convex, so on `[c, d]` it lies below the chord

```text
exp(a y) <= ((d - y) exp(a c) + (y - c) exp(a d)) / (d - c).
```

Taking conditional expectations kills the `y`-terms by centering and leaves

```text
E[exp(a Y) | F] <= (d exp(a c) - c exp(a d)) / (d - c).
```

With `p = -c / (d - c) in [0, 1]` and `u = a (d - c)` the right side is
`exp(phi(u))` where

```text
phi(u) = -p u + log(1 - p + p exp(u)).
```

Now `phi(0) = phi'(0) = 0` and
`phi''(u) = p (1 - p) exp(u) / (1 - p + p exp(u))^2 <= 1/4`, because `phi''`
is the variance of a `Bernoulli`-tilted variable taking values in `{0, 1}`.
Taylor's theorem then gives `phi(u) <= u^2 / 8`, i.e.
`E[exp(a Y) | F] <= exp(a^2 (d - c)^2 / 8) <= exp(a^2 / 2)` whenever
`d - c <= 2`. This is Hoeffding (1963) lemma 1 in conditional form; the
variance proxy is `sigma^2 = (d - c)^2 / 4 <= 1 = (range/2)^2` (Howard et al.
2020, equation 2.1). No symmetry of `[c, d]` about zero is assumed or used;
the asymmetric, mean-zero, width-2 two-point distributions with support
partly outside `[-1, 1]` are exactly the extremal cases, and the verifier
checks them explicitly.

For one residual group `g`, define in the original time index `t`

```text
Y_t = sum_{u<=t} I_u^g xi_u / B,        C_t = sum_{u<=t} I_u^g,
M_t(a) = exp(a Y_t - a^2 C_t / 2).
```

Because `I_t^g` is measurable with respect to the conditioning field, one step
gives

```text
E[M_t(a) | F_t] = M_{t-1}(a) exp(-a^2 I_t^g / 2) E[exp(a I_t^g xi_t / B) | F_t]
                <= M_{t-1}(a),
```

since the inner expectation is `1` when `I_t^g = 0` and at most
`exp(a^2/2)` when `I_t^g = 1` (section 3 plus the conditional Hoeffding
lemma). Thus `M_t(a)` is a nonnegative supermartingale in the original time
filtration with `M_0(a) = 1`. This is hypothesis 1 of the frozen task: for the
visit-indexed sum `Z_k = S_k / B`, the process `exp(a Z_k - a^2 k / 2)` is a
nonnegative supermartingale, because `Y_t = Z_{C_t}` pathwise.

Working in the original time index avoids any optional-sampling argument:
uniformity over visit counts is obtained from uniformity over `t` below.

## 5. Two-sided cosh mixture

For each grid index `j`, `cosh(a_j z) = (exp(a_j z) + exp(-a_j z))/2`, so

```text
k -> exp(-a_j^2 k/2) cosh(a_j Z_k)
```

is the equal average of the `+a_j` and `-a_j` positive supermartingales of
section 4 (equivalently, of the time-`t` processes `M_t(+a_j)` and
`M_t(-a_j)`), hence itself a nonnegative supermartingale with initial value
`cosh(0) = 1`. The frozen convex mixture

```text
M(k, z) = sum_j w_j exp(-a_j^2 k/2) cosh(a_j z),   w_j > 0, sum_j w_j = 1,
```

is again a nonnegative supermartingale (nonnegative convex combinations
preserve the supermartingale property) with initial value
`M(0, 0) = sum_j w_j = 1`. This is hypothesis 2, first part.

## 6. Ville crossing control and group allocation

Ville's inequality (Ville 1939; Howard et al. 2020, Theorem 1) for a
nonnegative supermartingale with initial value one states
`P(exists t: M_t >= 1/alpha) <= alpha` for every `alpha > 0`. Applying it to
group `g`'s mixture with `alpha = delta / G`,

```text
P(exists t: M(C_t^g, Y_t^g) >= G/delta) <= delta/G.
```

A union bound over the `G = m + 2d` groups gives one event, uniform over all
groups and all times (hence all visit counts), with total failure probability
at most `delta`:

```text
P(exists g, t: M(C_t^g, Y_t^g) >= G/delta) <= delta.
```

The full declared `delta` is spent on this single mandatory mixture; the
analytic line stitching of section 10 is a separate audit object and receives
no share of the reported certificate.

## 7. Root uniqueness and the crossing-to-radius equivalence

For integer `k >= 1`, `M(k, q)` as a function of `q >= 0` is continuous,
strictly increasing (its derivative
`sum_j w_j exp(-a_j^2 k/2) a_j sinh(a_j q)` is positive for `q > 0`), and
satisfies

```text
M(k, 0) = sum_j w_j exp(-a_j^2 k/2) <= sum_j w_j = 1 < G/delta,
```

because `G >= 1` and `0 < delta < 1`, while `M(k, q) -> infinity` as
`q -> infinity` (every summand diverges since `a_j > 0`). Hence
`M(k, q) = G/delta` has a unique nonnegative root `q_mix(k)` (hypothesis 3,
existence and uniqueness). Since `M(k, .)` is even and increasing on
`[0, infinity)`,

```text
M(k, |Z_k|) < G/delta   <=>   |Z_k| < q_mix(k).
```

On the section-6 good event, simultaneously for all groups and counts,

```text
|mean of the first k residuals of group g| = B |Z_k| / k < B q_mix(k) / k
= r_mix(k).
```

## 8. Random final counts and selective emission

The good event is uniform over all times `t`, so it includes the terminal
time `t = n` at which `C_n^g = N_g` is the random final observed count and
`Y_n^g = Z_{N_g}`. Substituting `k = N_g` uses no conditioning on `N_g` and no
independence between counts and residuals, exactly as in `[MART]` section 3.
The only probability claim remains the selective one,

```text
P(Emit and an emitted bound is violated) <= delta,
```

with no claim that support or emission has probability at least `1 - delta`
and no division by the emission probability (frozen probability contract).
This is hypothesis 2, second part, and the semantics required by hypotheses
5 and 6.

## 9. Conservative numerical inversion

The implementation computes `log M(k, q)` by `logsumexp` over the fifteen
terms `log w_j - a_j^2 k/2 + logcosh(a_j q)` with the stable identity
`logcosh(x) = |x| + log1p(exp(-2 |x|)) - log 2`. For each `k`:

- lower bracket `q_lo = 0`, with `log M(k, 0) < log(G/delta)` by section 7;
- upper bracket `q_hi = q_stitch(k)`, with
  `log M(k, q_stitch(k)) >= log(G/delta)` by section 10;
- deterministic bisection maintaining the invariant
  `log M(k, q_lo) < target <= log M(k, q_hi)`: at midpoint `qm`, if
  `log M(k, qm) >= target` then `q_hi = qm`, else `q_lo = qm`;
- stop only when both frozen conditions hold with `tol = 1e-12`, within 200
  iterations:

  ```text
  q_hi - q_lo <= tol * (1 + q_hi),
  0 <= log M(k, q_hi) - target <= tol * (1 + |target|);
  ```

- return the upper endpoint `q_hi`, which satisfies
  `M(k, q_hi) >= G/delta`, hence `q_hi >= q_mix(k)`: the returned root is
  conservative. Bracketing, convergence, finiteness, or conservativeness
  failure produces an ordered deterministic non-emission reason and never a
  silent fallback to the old radius (hypothesis 3, computation part).

## 10. Line stitching: independent validity and the bracket inequality

For component `j` and sign `s in {+1, -1}`, section 4 gives the positive
supermartingale `exp(s a_j Z_k - a_j^2 k/2)` with initial value one; Ville
with level `2G/(delta w_j)` yields

```text
P(exists k: s Z_k >= q_j(k)) <= delta w_j / (2G),
q_j(k) = (L_j + a_j^2 k/2) / a_j,   L_j = log(2G/(delta w_j)).
```

A union bound over `G` groups, 2 signs, and 15 components has total mass
`G * 2 * sum_j delta w_j / (2G) = delta`, so

```text
|Z_k| < q_stitch(k) = min_j q_j(k)   for all g, k
```

is independently valid with probability at least `1 - delta`. It is used only
as an audit and as the deterministic upper bracket of section 9, and is never
selected as the reported route certificate.

Bracket inequality (`q_mix(k) <= q_stitch(k)`): let `j*` attain the minimum
at `k`, so `q_stitch(k) = q_{j*}(k)`. The `j*` summand of the mixture at
`q = q_stitch(k)` satisfies

```text
w_{j*} exp(-a_{j*}^2 k/2) cosh(a_{j*} q_{j*}(k))
    >= (w_{j*}/2) exp(-a_{j*}^2 k/2 + a_{j*} q_{j*}(k))
     = (w_{j*}/2) exp(L_{j*})
     = (w_{j*}/2) * 2G/(delta w_{j*})
     = G/delta,
```

using `cosh(x) >= exp(x)/2` and `a_{j*} q_{j*}(k) = L_{j*} + a_{j*}^2 k/2`.
Therefore `M(k, q_stitch(k)) >= G/delta`, and monotonicity of `M(k, .)` gives
`q_mix(k) <= q_stitch(k)` for every integer `k >= 1`.

## 11. Radius dominance and monotonicity (numerical contract)

With `r_old(k; n) = B sqrt(2 log(2 G n / delta) / k)` exactly as each legacy
record used it, the frozen acceptance test is: for every
`n in {256, 1024, 4096, 16384}` and every integer `1 <= k <= n`,
`r_mix(k) <= r_old(k; n)` with at least one strict inequality overall, and
`r_mix` nonincreasing through 16384. Both radii are proportional to `B`, so
the check is independent of `B`; it is executed exhaustively and
deterministically by the new verifier at the frozen dimensions
`m = 6, d = 24, G = 54, delta = 0.05`, and independently recomputed by the
analyzer. The separate global `n_max = 16384` design audit
(`r_mix(k) <= r_old(k; 16384)` for all `k <= 16384`) is implied by the
per-record check because `r_old(k; n)` is increasing in `n`; both are
recorded. These checks are falsifiable numerical contracts, not proof
obligations; a certified failing count would be a valid negative result.

## 12. Deterministic route composition (inherited, radius-monotone)

On the good event the trajectory is frozen and the unchanged `[MART]`
section 4 recurrences apply, with the residual radius now supplied by
`r_mix` instead of `r_old`:

```text
Direct-Q exact:    U(L) = rho^L B + (1 - rho^L) r_X / (1 - gamma),
                   rho = 1 - alpha(1 - gamma);
Direct-Q softmax:  same with rho = 1 - 2 alpha c_Q and statistical
                   denominator 2 c_Q, c_Q = d_min - (1 + gamma)/2;
state value:       the same recurrence with state counts/diagonals;
V-first exact:     gamma U_V(L) + r_X;
V-first softmax:   gamma U_V(L) + r_X + 2B(1 - d_min).
```

Every recurrence is nondecreasing in its residual radius, and every emission
gate (support, margins, mode, divergence guard, finiteness) is independent of
the radius value. Since `r_mix(k) <= r_old(k; n)` at every frozen count,
replacing only the radius primitive leaves all emission decisions unchanged
and makes every emitted total bound nonincreasing (hypotheses 5 and 6, to be
confirmed record-by-record by the analyzer against the frozen baseline).

## Primary sources and assumption mapping

- W. Hoeffding (1963), https://doi.org/10.1080/01621459.1963.10500830:
  the interval-form conditional lemma of section 4. Assumptions: conditional
  centering and support in an interval of width at most 2 containing the
  conditional mean (not necessarily `[-1, 1]`); mapped in sections 2-3 from
  the `[MART]`-verified filtrations and width `2B`.
- S. R. Howard, A. Ramdas, J. McAuliffe, J. Sekhon, "Time-uniform Chernoff
  bounds via nonnegative supermartingales," Probability Surveys 17 (2020),
  257-317, https://doi.org/10.1214/18-PS321: nonnegative-supermartingale
  time-uniform control and Ville's inequality (sections 4-6); the
  sub-Gaussian case with variance proxy `sigma^2 = (d - c)^2 / 4 <= 1` for
  centered increments supported on an interval `[c, d]` of width at most 2;
  finite convex mixtures of supermartingales remain
  supermartingales (section 5). No assumption beyond the verified conditional
  range and centering is imported.
- J. Ville (1939), as presented in Howard et al. (2020) Theorem 1: the
  maximal inequality of section 6.
- K. Azuma (1967), https://doi.org/10.2748/tmj/1178243286: the martingale
  extension underlying the `[MART]` baseline radius that this task replaces;
  not re-imported beyond that inheritance.
- I. Waudby-Smith and A. Ramdas (2024),
  https://doi.org/10.1093/jrsssb/qkad009: consulted only for the isolated
  transition-variance feasibility assessment (separate document); its
  observed-sequence construction is not assumed to apply to the unobservable
  fixed-target residuals and nothing from it enters the mandatory certificate.

## Scope limits

Fixed policy, fixed context, synchronous updates, finite state/action spaces,
deterministic bounded edge rewards, stationary start, one trajectory. No
online control, policy improvement, nonstationary starts, layer-dependent
scores, general stochastic reward noise, learned mixture constants, or
post-hoc certificate selection. The certificate consumes only observed counts,
the declared reward bound, and public hyperparameters; `B` derives solely from
the pre-sampling declared `R_star`.
