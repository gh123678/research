# FP-TU-001 GPT route theory

Status: independent construction before first-result disclosure.

## 1. Inherited filtration facts

This route inherits only the already verified `FP-MART-001` residual and
filtration lemmas.  For state Bellman residuals, the predictable selector is
measurable in the pre-action filtration `G_t`; for pair Bellman and recovery
residuals it is measurable in the post-action filtration `H_t`.  Selecting the
first `k` visits is therefore optional skipping at predictable visit times.
Each selected fixed-target residual is conditionally centered and has
conditional range width at most `2B`, where

```text
B = R_star / (1-gamma).
```

There are `G=m+2d` families: `m` state Bellman, `d` pair Bellman, and `d`
fixed-`V^pi` recovery groups.  This construction does not condition the state
residual on `H_t`, and it does not use a kernel, occupancy, true value,
residual, route error, or true initial error as a certificate input.

## 2. Fixed-rate supermartingales and the cosh mixture

Let `S_k` be the sum of the first `k` residuals for one group and set
`Z_k=S_k/B`.  Conditional Hoeffding's lemma for a centered variable of range
width two gives

```text
E[exp(a (Z_k-Z_{k-1})) | F_{k-1}] <= exp(a^2/2).
```

Consequently `exp(a Z_k-a^2 k/2)` is a nonnegative supermartingale for every
fixed real `a`.  With the preregistered positive weights `w_j`, averaging the
processes at `+a_j` and `-a_j` gives

```text
M_k = sum_j w_j exp(-a_j^2 k/2) cosh(a_j Z_k).
```

It is a convex mixture of nonnegative supermartingales and `M_0=1`.  Ville's
inequality gives a crossing probability at most `delta/G` at threshold
`G/delta` for one group.  A union bound over the `G` groups gives total failure
probability at most `delta`, uniformly over every visit count.

The event is already uniform in `k`, so substituting a random final observed
count needs neither independence nor conditioning on that count.  The only
trajectory-level statement remains

```text
P(Emit and an emitted bound is violated) <= delta.
```

It does not assert likely support or coverage conditional on emission.

## 3. Root, stitch audit, and conservative computation

For each integer `k>=1`, `M(k,q)` is continuous and even in `q`.  On the
nonnegative half-line its derivative is strictly positive because every
weight and rate is positive and `sinh(a_j q)>0` for `q>0`.  Moreover
`M(k,0)<=sum_j w_j=1<G/delta`, while `M(k,q)` diverges with `q`.  Thus there is
one positive crossing root `q_mix(k)`.

Write

```text
L_j = log(2G/(delta w_j)),
q_j(k) = (L_j+a_j^2 k/2)/a_j.
```

For each signed line, Ville plus a union bound allocating
`delta w_j/(2G)` proves validity.  Their minimum is valid because crossing
the lower envelope means crossing at least one preregistered line.  If `j*`
attains `q_stitch=min_j q_j`, then `cosh(x)>=exp(x)/2` yields

```text
w_j* exp(-a_j*^2 k/2) cosh(a_j* q_stitch) >= G/delta.
```

Therefore `q_mix<=q_stitch`.  Stitching is only an audit and upper bracket;
it is never selected as the reported certificate.  Stable `logcosh` and
`logsumexp` avoid overflow.  Bisection maintains a strict lower endpoint and
a conservative upper endpoint, returning the latter only after both the
relative bracket-width and upper-residual tests pass within 200 iterations.

## 4. Deterministic composition

Only the residual-radius primitive changes.  The Direct-Q and state-value
recurrences remain

```text
rho = 1-2 alpha margin,
U_L = rho^L B + (1-rho^L) r_mix/(2 margin).
```

Exact matching has `margin=(1-gamma)/2`; finite softmax uses its unchanged
empirical diagonal margin.  V-first remains

```text
U_VF_exact = gamma U_V + r_mix,
U_VF_soft  = gamma U_V + r_mix + 2B(1-d_min).
```

All support, mode, divergence, and margin gates are inherited unchanged.  As
these formulas are coordinatewise nondecreasing in the residual radii,
`r_mix<=r_old` implies every previously emitted total bound is nonincreasing
without changing emission.

## 5. Transition-variance feasibility

Observed successor counts can form a confidence set for each transition row,
but a variance certificate uniform over every `V in [-B,B]^m` must also bound
unseen successor mass and optimize the nonlinear functional
`P f_V^2-(P f_V)^2` over that set.  Counts alone do not supply the predictable
conditional variance of the unobservable fixed-target residual sequence.
No complete observable, simultaneously valid construction with a separately
allocated risk budget is established here.  The exact gap is the missing
uniform transition-row confidence/optimization proof, so transition variance
is not selected, receives zero risk, and does not affect the mandatory route.

## 6. Sources and scope

The exponential-supermartingale, Ville crossing, and mixture arguments follow
Howard, Ramdas, McAuliffe, and Sekhon, *Probability Surveys* 17 (2020),
https://doi.org/10.1214/18-PS321.  The inherited bounded-increment step follows
Hoeffding (1963), https://doi.org/10.1080/01621459.1963.10500830, and the
martingale extension follows Azuma (1967),
https://doi.org/10.2748/tmj/1178243286.  The construction is restricted to the
frozen fixed-policy, stationary-start, deterministic bounded edge-reward,
fixed-context synchronous setting.
