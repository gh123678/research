# FP-MART-001 GPT route: visit-indexed martingale theory

> Status: independent GPT-route construction, preliminary until cross-verification
> Common activation commit: `c8ec7e5c3165930663e26a07f98b38cc9ec186ad`
> Construction date: 2026-09-03
> Formal outputs: not inspected when this proof and the usefulness metric below
> were frozen

## 1. Scope and primary sources

The proof covers the frozen finite-state, finite-action, fixed-policy,
fixed-context, synchronous protocol with deterministic edge rewards satisfying
`|R| <= R_star`. It does not use stationarity, occupancy, mixing, a transition
matrix, a spectral quantity, or independence across visits. Stationary start is
retained only because it belongs to the frozen experiment.

The only imported exponential-moment ingredient is Hoeffding's bounded-variable
lemma: a centered random variable whose conditional support has width `w`
satisfies a conditional moment-generating-function bound
`E[exp(lambda X) | F] <= exp(lambda^2 w^2 / 8)`. The original source is
[Hoeffding (1963), *Probability Inequalities for Sums of Bounded Random
Variables*](https://doi.org/10.1080/01621459.1963.10500830), especially its
bounded-variable exponential argument. The application to bounded martingale
differences is corroborated by the original
[Azuma (1967), *Weighted Sums of Certain Dependent Random
Variables*](https://doi.org/10.2748/tmj/1178243286).

No black-box optional-skipping theorem is imported. Section 5 constructs the
predictably selected exponential supermartingale and proves the bounded-horizon
stopping step directly. For context, the optional variance route was checked
against [Freedman (1975), *On Tail Probabilities for
Martingales*](https://doi.org/10.1214/aop/1176996452); it is not used by the
mandatory result because its predictable conditional-variance process is not
available from the allowed certificate inputs.

## 2. Sampling order and filtrations

Write the frozen sampling order as

```text
S_t -> A_t -> (R_{t+1}, S_{t+1}) -> A_{t+1}.
```

There are two observation filtrations:

```text
F^S_t = sigma(S_0, A_0, R_1, S_1, ..., A_{t-1}, R_t, S_t),
F^X_t = F^S_t join sigma(A_t).
```

Thus `F^S_t` is the history immediately before sampling `A_t`, while `F^X_t`
is the history immediately after that action and before the next reward and
state. They satisfy

```text
F^S_t subset F^X_t subset F^S_{t+1} subset F^X_{t+1}.
```

Under the fixed policy and MDP,

- the conditional law of `A_t` given `F^S_t` is `pi(. | S_t)`;
- the conditional law of `(R_{t+1}, S_{t+1})` given `F^X_t` is the fixed edge
  law at `(S_t,A_t)`;
- the conditional law of `A_{t+1}` given `F^S_{t+1}` is
  `pi(. | S_{t+1})`.

The state filtration must not condition on the sampled `A_t`: doing so would
change the Bellman target being centered.

## 3. Three fixed-target martingale differences

Let `V^pi` and `Q^pi` denote the fixed true targets, used only in this proof and
later oracle audit. Define

```text
D^V_t = R_{t+1} + gamma V^pi(S_{t+1}) - V^pi(S_t),
D^Q_t = R_{t+1} + gamma Q^pi(S_{t+1},A_{t+1}) - Q^pi(S_t,A_t),
D^R_t = R_{t+1} + gamma V^pi(S_{t+1}) - Q^pi(S_t,A_t).
```

The Bellman equations and the conditional laws above give

```text
E[D^V_t | F^S_t] = 0,
E[D^Q_t | F^X_t] = 0,
E[D^R_t | F^X_t] = 0.
```

`D^V_t` is `F^S_{t+1}`-measurable. `D^Q_t` is
`F^X_{t+1}`-measurable because it includes `A_{t+1}`. `D^R_t` is already
`F^S_{t+1}`-measurable and therefore is also `F^X_{t+1}`-measurable.

For the state residual, the conditional expectation integrates both the policy
action and next transition. For the pair Bellman residual it integrates the next
transition and next policy action. For recovery it integrates the next
transition. These are different centering statements even though their final
concentration constants coincide.

## 4. Exact conditional range

Let

```text
B = R_star / (1 - gamma).
```

The discounted-return bounds give `||V^pi||_infinity <= B` and
`||Q^pi||_infinity <= B`. Each random Bellman target before subtracting its
current fixed value lies in

```text
[-R_star - gamma B, R_star + gamma B] = [-B, B].
```

Given the appropriate pre-observation filtration, `V^pi(S_t)` or
`Q^pi(S_t,A_t)` is fixed. Subtracting it shifts the interval but does not change
its width. Consequently every residual above has conditional support width at
most `2B`, and conditional Hoeffding gives

```text
E[exp(lambda D_t) | F_pre] <= exp(lambda^2 B^2 / 2).
```

The coarser statement `|D_t| <= 2B` is true but insufficient for the desired
constant if it is naively treated as the interval `[-2B,2B]`. The proof and code
therefore record both facts and use the conditional width `2B`.

## 5. Predictable visit selection and finite-horizon stopping

For a state `s`, let

```text
I^V_{s,t} = 1{S_t=s},
```

which is `F^S_t`-measurable. For a state-action pair `x=(s,a)`, let

```text
I^Q_{x,t} = I^R_{x,t} = 1{(S_t,A_t)=x},
```

which is `F^X_t`-measurable. In every case the selection decision is known
before the selected residual is observed.

For any one residual group `g`, suppress the superscript and define through
calendar time `t`

```text
C_{g,t} = sum_{u=0}^{t-1} I_{g,u},
H_{g,t} = sum_{u=0}^{t-1} I_{g,u} D_{g,u},
M_{g,t}(lambda)
  = exp(lambda H_{g,t} - lambda^2 B^2 C_{g,t}/2).
```

Conditioning immediately before residual `D_{g,t}` is observed, predictability
of `I_{g,t}` and Section 4 imply

```text
E[M_{g,t+1}(lambda) | F_pre] <= M_{g,t}(lambda).
```

Thus `M` is a nonnegative supermartingale. Let the after-observation time of the
`k`th visit be

```text
sigma_{g,k} = min{t+1 : C_{g,t+1}=k} capped at n.
```

This is bounded by `n`. Applying the preceding one-step inequality to
`M_{t intersect sigma}` and iterating from `0` to `n` proves directly that
`E[M_{sigma}] <= 1`; no unbounded optional-stopping assertion is needed. On the
event that the `k`th visit occurs before horizon `n` and its selected-prefix
mean is at least `r`, Markov's inequality gives

```text
P(C_{g,n} >= k and H_{g,sigma}/k >= r)
  <= exp(-lambda k r + lambda^2 B^2 k/2).
```

Optimizing at `lambda=r/B^2` and repeating for the negative tail yields

```text
P(C_{g,n} >= k and |H_{g,sigma}|/k > r)
  <= 2 exp(-k r^2/(2B^2)).
```

This event formulation remains valid when the `k`th visit never occurs; it does
not condition on the occurrence of that visit.

## 6. Global event and random observed counts

There are

```text
G = m + 2d
```

families: `m` state Bellman groups, `d` pair Bellman groups, and `d` recovery
groups. The pair Bellman and recovery groups share counts but are distinct
probability events. Count indicators are not additional residual groups.

Allocate `delta/(G n)` to each two-sided `(g,k)` event, equivalently
`delta/(2 G n)` to each one-sided tail. Solving the last display for `r` gives

```text
r_H(k,delta) = B sqrt(2 log(2 G n / delta) / k).
```

A union bound over all `G n` pairs produces one event `E_n` with
`P(E_n^c) <= delta`. No independence among groups is used.

Let `N_g=C_{g,n}` be the observed final count. On `E_n`, whenever `N_g>0`, the
group's complete empirical residual mean is its first-`N_g` visit mean, so the
pathwise substitution

```text
|mean_g| <= r_H(N_g,delta)
```

is immediate. This is lookup in an event already simultaneous in every
`1 <= k <= n`; it is not conditioning on `N_g` and does not assume that the
count is independent of its residuals. If `N_g=0`, the mean is undefined and
the route does not emit.

## 7. Deterministic route composition

On `E_n`, define the support-wide radii

```text
epsilon_S = max_s r_H(N_s,delta),
epsilon_Q = max_x r_H(N_x,delta),
epsilon_R = max_x r_H(N_x,delta).
```

The last two have the same numerical values but correspond to separately
allocated residual families. For softmax matching, every empirical row is a
convex combination of group residual means, so the support-wide maximum is a
valid row bound even though its observed weights and diagonal are data
dependent. Using only the query group's own radius would be invalid because a
softmax row also weights nonmatching groups.

Let `d_min` be the observed minimum empirical matching diagonal and

```text
c = d_min - (1 + gamma)/2,
rho = 1 - 2 alpha c.
```

When full required support holds and `c>0`, the completed deterministic
recurrence gives, for any actual iteration count `L`,

```text
U_L = rho^L B + (1-rho^L) epsilon/(2c).
```

Exact matching is the special case `d_min=1`, so
`c=(1-gamma)/2`. The same all-layer inequality holds simultaneously for every
`L`; a data-dependent early-stopping iteration may therefore be substituted
without a second probability argument. A divergence guard is a rejection gate.

For V-first no-split,

```text
U_Q_exact   = gamma U_V + epsilon_R,
U_Q_softmax = gamma U_V + epsilon_R + 2B(1-d_pair_min).
```

Only the iterative state stage needs a positive contraction margin in the
softmax recovery route. The one-step recovery needs full pair support and adds
the explicit leakage term but does not need a pair contraction margin.

The legacy state evaluator clips to a symmetric interval built from the full
edge-reward bound before iteration. That interval contains `V^pi`, so projection
is nonexpansive relative to `V^pi`. The new certificate does not receive the
sampled reward tensor or that task-specific clipping value; it uses the larger
publicly declared bound described in Section 9 for its target range and initial
error. A smaller valid projection interval does not weaken the conservative
bound above.

## 8. Selective probability statement

For any route, let `Emit` mean that its observed support, empirical margin,
algorithm-mode, divergence, risk-budget, and finite-arithmetic gates pass. On
`E_n intersect Emit`, Section 7 proves the emitted bound. Therefore

```text
Emit intersect Violation subset E_n^c,
P(Emit and Violation) <= delta.
```

All four routes share the same `E_n`, so the event that any emitted route
violates its bound is also contained in `E_n^c`; no extra route-level risk split
is required. This does not imply `P(Emit)>=1-delta` or
`P(Violation | Emit)<=delta`.

The guarantee is per frozen trajectory record. The 480-record experiment is an
empirical audit and is not itself covered jointly at level `delta`.

## 9. Declared reward bound and preregistered usefulness metric

The repository generator draws its base edge rewards in `[-1,1]` and then adds
the public nonnegative `gap_bonus` only to action zero. Before inspecting the
sampled reward tensor or trajectory, the new evaluator declares

```text
R_star = 1 + gap_bonus,
B = R_star/(1-gamma).
```

This rule is part of the evaluator configuration and is the only reward-bound
input to the pure certificate. The sampled reward maximum is never passed to
the new module.

The frozen task requires numerical nontriviality to be reported separately but
does not define a threshold. Before any formal FP-MART-001 output is inspected,
the GPT route preregisters the following transparent usefulness metric:

```text
improves_over_zero_initialization := emitted and total_bound < B.
```

`B` is the same computable zero-initialization error guarantee used by the
certificate. This metric affects only descriptive usefulness rates. A finite
emitted bound at or above `B` remains a valid selective certificate and is not
relabelled as a failure.

## 10. Optional variance-adaptive route

Freedman's original martingale inequality is formulated with a predictable sum
of conditional variances and a uniform increment bound. Under the frozen
certificate-input restrictions, the exact conditional variances depend on the
unknown kernel and fixed true values, while realized squared fixed-target
increments also require the oracle targets. Counts alone do not provide an
observable variance proxy that can simply be substituted into Freedman's
theorem.

Accordingly, this route does not implement or select a variance-adaptive bound.
It records `not_implemented_no_observable_variance_proxy` in a separate optional
object. The mandatory Hoeffding event retains the full `delta`; the optional
status never enters mandatory failure reasons, and no post-hoc minimum is
taken.

## 11. Deterministic counterexamples guarding the proof

1. **Wrong state filtration.** With one state, two equiprobable actions,
   rewards `+1` and `-1`, and `gamma=0`, the state residual has mean zero given
   the state but conditional mean `+1` or `-1` after observing the action.
2. **Nonpredictable selection.** For iid Rademacher increments, selecting a time
   only when the observed increment is `+1` makes the selected mean equal one.
3. **Fixed-count-only substitution.** Choosing a count after inspecting all
   prefix deviations invalidates a single fixed-`k` tail statement; the global
   event must be simultaneous in `k`.
4. **Unadjusted post-hoc minimum.** Two delta-level bounds may fail on disjoint
   events, so their data-dependent minimum can fail with probability `2 delta`.
5. **Query-only softmax radius.** A row with nonzero mass on another residual
   group is not controlled by the query group's radius alone.

## 12. Preliminary scientific assessment

The mandatory visit-indexed Hoeffding construction is mathematically valid
under the frozen assumptions, and the exact target constant in the design is
confirmed. Implementation and empirical conclusions remain preliminary until
the GPT route passes its verifier, smoke and formal runs, both routes disclose
their sealed results, and cross-verification is complete.

## 13. Pre-run implementation and audit decisions

The following choices were recorded before inspecting any smoke or formal
FP-MART-001 output. They resolve reporting details left open by the frozen task
without changing its scientific hypotheses, risk allocation, matrix, or
acceptance criteria.

- The smoke matrix uses `tasks=2` and trajectory lengths `256` and `4096`, with
  every other formal parameter spelled out unchanged. These lengths exercise a
  sparse-support and a full-support regime; smoke output is never formal
  evidence. Its random seeds retain the frozen 30-task-per-cell spawn stride and
  select task indices zero and one in each cell, so every smoke record aligns
  with the corresponding formal baseline record.
- The declared reward rule is per public matrix cell:
  `R_star = 1 + gap_bonus`, hence `1.0` for gap zero and `1.5` for gap `0.5`.
  It is computed before sampling the MDP and never from observed rewards or the
  sampled reward tensor.
- The primary bound-usefulness field is the already defined
  `improves_over_zero_initialization`, namely emitted `total_bound < B`. Finite
  emission and usefulness remain separate.
- An oracle residual or route-bound violation is recorded only when the route
  emits and the realized value exceeds its bound by more than `1e-9`. For a
  non-emitting route, route-bound violation is `null`, not `false`.
- Baseline fields containing true occupancies, spectral quantities, true model
  summaries, or ghost residuals are grandfathered solely to satisfy exact
  legacy preservation. They remain unchanged and are never supplied to the new
  pure module. Every newly created value stays below
  `visit_indexed_certificate`, whose `certificate_inputs`, theorem objects, and
  `oracle_audit` are structurally separated.
- The mandatory failure order is `algorithm_mode_mismatch`,
  `divergence_guard_triggered`, `state_support_missing`,
  `pair_support_missing`, `state_kernel_margin_nonpositive`,
  `pair_kernel_margin_nonpositive`, and `numerical_nonfinite`. Invalid public
  arguments raise `ValueError` before a certificate exists.
- Full legacy-record, config, and summary comparison uses exact equality for
  nonnumeric leaves and `math.isclose(rel_tol=1e-12, abs_tol=1e-12)` for numeric
  leaves. Duplicate record keys are rejected before dictionary alignment.
- `config.json`, `task_results.json`, `summary.json`, `regression.json`,
  `environment.json`, `commands.log`, and `checks.log` are all hashed in the
  versioned first-result evidence after the route completes.
