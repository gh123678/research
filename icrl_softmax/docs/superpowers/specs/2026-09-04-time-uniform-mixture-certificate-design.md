# Time-uniform mixture certificate design

Date: 2026-09-04

Proposed task: `FP-TU-001`

Approved direction: finite geometric exponential mixture as the mandatory
certificate, analytic line stitching as an audit boundary, and observable
transition variance as feasibility-only work.

## 1. Context

`FP-MART-001` verified a visit-indexed Hoeffding certificate for the fixed
state Bellman, pair Bellman, and fixed-`V^pi` recovery residuals. Its radius is

```text
r_old(k) = B sqrt(2 log(2 G n_max / delta) / k),
```

where `G = m + 2d`. It is valid because the proof unions over every group and
every integer count `1 <= k <= n_max`. The resulting explicit `log n_max`
penalty is conservative: in the frozen 480-record evaluation, primary-useful
bounds (`total_bound < B`) appeared only for V-first at trajectory length
16384.

The next task will replace the count-wise union by one nonnegative
supermartingale per residual group. It will preserve all verified filtrations,
residual definitions, deterministic route recurrences, no-oracle inputs, and
selective probability semantics from `FP-MART-001`.

## 2. Research question

Can a pre-registered geometric mixture of exponential supermartingales produce
a computable, visit-indexed, time-uniform boundary that:

1. is valid simultaneously over all residual groups and visit counts;
2. uses only observed counts, the declared reward bound, and public
   hyperparameters;
3. removes the explicit count-wise `log n_max` risk penalty;
4. is no wider than the verified Hoeffding radius throughout the frozen count
   range `1 <= k <= 16384`; and
5. tightens the existing Direct-Q and V-first bounds without changing their
   emission gates or legacy outputs?

The theorem remains scientifically meaningful if item 4 is falsified: a
rigorous counterexample or exhaustive certified radius comparison will record
a verified negative result rather than silently changing the mixture.

## 3. Scope and non-goals

### In scope

- The fixed-policy, fixed-context, synchronous, finite-state/action,
  deterministic bounded edge-reward protocol verified by `FP-MART-001`.
- The same pre-action and post-action filtrations and the same three
  fixed-target residual families.
- One trajectory, random observed final counts, exact and finite-softmax
  Direct-Q, and exact and finite-softmax V-first no-split recovery.
- A finite geometric mixture optimized structurally for counts through 16384
  but valid as a nonnegative supermartingale at every time.
- A separately proved analytic line-stitching boundary used to audit the
  numerical mixture inversion.
- A proof-or-counterexample feasibility assessment for an observable
  transition-variance construction.

### Out of scope

- Changing policies, online control, policy improvement, nonstationary starts,
  layer-dependent scores, or general stochastic reward noise.
- Learning mixture weights or rates from trajectory outcomes.
- Selecting the minimum of multiple probabilistic certificates after seeing
  data.
- Making the transition-variance idea a certificate without a complete proof,
  observable input contract, and separately allocated risk.
- Modifying or reinterpreting legacy certificate status names.

## 4. Frozen probability contract

For every trajectory-based emission event `Emit`, the only required claim is

```text
P(Emit and an emitted bound is violated) <= delta.
```

The task will not claim that support or emission occurs with probability at
least `1-delta`, and it will not divide by the unknown emission probability to
claim conditional coverage.

All `G = m + 2d` residual groups share one risk budget. The mixture allocates
`delta/G` to each group. The analytic audit boundary uses the same total risk
but is not combined with the mixture certificate; it is a verification object,
not a second selectable constituent.

## 5. Mandatory mixture construction

### 5.1 Normalized martingale

For one residual group, let `S_k` be the sum of the first `k` visit-indexed
fixed-target residuals. The verified conditional range width is `2B`. Define
the normalized sum `Z_k = S_k/B`. Conditional Hoeffding's lemma gives, for
every real `a`,

```text
exp(a Z_k - a^2 k / 2)
```

as a nonnegative supermartingale in the visit-indexed filtration.

### 5.2 Frozen grid, weights, and rates

Set `J = 15` and

```text
k_j = 2^j,                         j = 0,...,14,
w_j = (j+1)^(-2) / sum_{l=0}^{14} (l+1)^(-2),
L_j = log(2 G / (delta w_j)),
a_j = sqrt(2 L_j / k_j).
```

The dimensional betting rate is `lambda_j = a_j/B`. The grid, weights, and
rates depend only on `G`, `delta`, `B`, and the declared design horizon; they
do not use states, actions, rewards, residuals, route errors, or formal output.

The exact decimal values serialized by the implementation must be derived from
these formulas, not copied from an exploratory run.

### 5.3 Two-sided mixture

For group `g`, define

```text
M_g(k,z) = sum_j w_j exp(-a_j^2 k/2) cosh(a_j z).
```

Because `cosh(x) = (exp(x)+exp(-x))/2`, substituting `z = Z_k` makes
`M_g(k,Z_k)` a convex mixture of positive supermartingales with initial value
one. Ville's inequality and a union bound over groups imply

```text
P(exists g,k: M_g(k,Z_g(k)) >= G/delta) <= delta.
```

For integer `k >= 1`, let `q_mix(k)` be the unique nonnegative solution of

```text
M_g(k,q_mix(k)) = G/delta.
```

The time-uniform residual-mean radius is

```text
r_mix(k) = B q_mix(k) / k.
```

The boundary itself does not depend on the group identity. Random final counts
are substituted directly because the crossing event is already uniform over
all visit times.

## 6. Analytic line-stitching audit

For each mixture component and sign, Ville's inequality gives a valid linear
boundary. Allocating `delta w_j/(2G)` to that signed component yields the
dimensionless upper boundary

```text
q_j(k) = (L_j + a_j^2 k/2) / a_j.
```

Define

```text
q_stitch(k) = min_j q_j(k),
r_stitch(k) = B q_stitch(k) / k.
```

This finite line stitching is independently valid by a union bound over
groups, signs, and components. It also supplies a deterministic upper bracket
for numerical inversion. If `j` attains the minimum at `q_stitch(k)`, then

```text
w_j exp(-a_j^2 k/2) cosh(a_j q_stitch(k))
    >= (w_j/2) exp(-a_j^2 k/2 + a_j q_j(k))
     = G/delta.
```

Thus at least one mixture term reaches the global threshold and
`q_mix(k) <= q_stitch(k)`.

The audit must verify that inequality for every `1 <= k <= 16384`. The stitched
boundary is never selected as the reported route certificate.

## 7. Numerical inversion contract

Compute `log M_g(k,q)` using stable `logsumexp` and a stable `logcosh` routine.
For every `k`:

1. lower bracket: `q_lo = 0`;
2. upper bracket: `q_hi = q_stitch(k)`;
3. target: `log(G/delta)`;
4. require `log M_g(k,q_lo) < target <= log M_g(k,q_hi)`, then solve by
   deterministic bisection for at most 200 iterations;
5. with `tol = 1e-12`, stop only when both conditions hold:

   ```text
   q_hi - q_lo <= tol * (1 + q_hi)
   0 <= log M_g(k,q_hi) - log(G/delta)
     <= tol * (1 + abs(log(G/delta)))
   ```

6. return the conservative upper endpoint `q_hi`.

The verifier must recompute selected roots with an independent high-precision
implementation. Failure to bracket, converge, preserve finiteness, or return a
conservative endpoint produces a deterministic non-emission reason; it cannot
fall back silently to the old radius.

## 8. Certificate module and data flow

Create four isolated files:

- `time_uniform_mixture_certificate.py` — pure grid, mixture, inversion,
  validation, and route-composition functions;
- `verify_time_uniform_mixture_certificate.py` — deterministic theorem and
  numerical contract tests;
- `evaluate_time_uniform_certificates.py` — frozen paired evaluation and
  oracle-only diagnostics;
- `analyze_time_uniform_certificates.py` — strict regression, schema, radius,
  and usefulness analysis.

The machine data flow is

```text
observed counts + declared reward bound + public hyperparameters
    -> validate grid, weights, risk, and algorithm mode
    -> invert time-uniform mixture radius
    -> compose unchanged Direct-Q / V-first recurrences
    -> serialize time_uniform_certificate
    -> compute structurally separate oracle_audit
```

The pure module must not accept a true kernel, occupancy, value, residual,
spectral quantity, route error, true reward maximum, or true initial error.
The declared reward bound determines `B = R_star/(1-gamma)` before MDP
sampling. New outputs live only below `time_uniform_certificate`; old
`visit_indexed_certificate` and legacy fields remain unchanged.

## 9. Validation and deterministic failures

The module validates:

- original-integer counts, vector dimensions, horizon sums, and state/pair
  aggregation consistency;
- `0 < delta < 1`, `0 < gamma < 1`, `B > 0`, `G >= 1`, and `k >= 1`;
- exactly 15 grid points with `k_j = 2^j`;
- positive finite weights summing to one within `1e-15`;
- positive finite rates matching the frozen formulas;
- fixed-context synchronous mode and no divergence-guard trigger;
- support and route-relevant empirical margins;
- finite log-domain arithmetic, a valid inversion bracket, convergence, and a
  conservative returned root.

Failure reasons use a fixed order. Unavailable numeric values serialize as
`null`; NaN and Infinity are forbidden. Duplicate JSON keys are rejected.

## 10. Verification design

The new verifier includes:

1. symbolic/numeric checks of weight normalization and rate formulas;
2. direct one-step checks of the exponential-supermartingale conditional MGF
   inequality on exhaustive bounded two-point fixtures;
3. a proof-level check that the cosh mixture starts at one and preserves the
   total `delta` allocation;
4. exhaustive integer-count checks for `1 <= k <= 16384` covering bracketing,
   conservative roots, `q_mix <= q_stitch`, and strict finiteness;
5. independent high-precision root comparisons at endpoints, powers of two,
   and off-grid counts;
6. comparison against the old Hoeffding radius at every frozen count;
7. positive and negative support/margin/mode/risk/numerical paths;
8. strict count-validation counterexamples inherited from `FP-MART-001`;
9. exact/softmax Direct-Q and V-first composition arithmetic;
10. strict JSON, namespace, status, and oracle-separation tests;
11. a counterexample showing why an unallocated post-hoc minimum is invalid;
12. all existing fixed-policy and visit-indexed verifiers unchanged and
    passing.

## 11. Frozen paired evaluation

Use the same 480 records and seed as `FP-MART-001`:

- tasks: 30;
- states/actions: 6/4;
- trajectory lengths: 256, 1024, 4096, 16384;
- mixing: 0.08 and 0.5;
- reward-gap bonuses: 0 and 0.5;
- minimum action probability: 0.05;
- softmax beta: 8;
- gamma/alpha/iterations: 0.70/0.65/160;
- certificate delta: 0.05;
- seed: 20260829.

Reusing the trajectories is permitted because the grid, weights, rates,
solver, thresholds, and acceptance rules in this design are frozen without
using prior per-record outcomes. No design constant may be changed after a
smoke or formal output is inspected.

Every record reports, for all four routes:

- old, stitched-audit, and mixture radii at the observed minimum counts;
- radius ratios and absolute reductions;
- old and new total bounds and their ratios/reductions;
- emission and deterministic failure reasons;
- primary usefulness `total_bound < B`;
- secondary descriptive threshold `total_bound < 2B`;
- oracle error and violation diagnostics in `oracle_audit` only.

## 12. Acceptance and falsification

The mandatory route passes only if:

1. the time-uniform theorem, group allocation, and random-count substitution
   are complete and independently verified;
2. the pure certificate uses no prohibited input;
3. every integer count through 16384 passes the inversion and stitching audits;
4. `r_mix(k) <= r_old(k)` for every frozen count, with at least one strict
   inequality;
5. every previously emitted route bound is nonincreasing and every emission
   status is identical to `FP-MART-001`;
6. the 480-record configuration, task identity, and all legacy fields have zero
   mismatch under the frozen exact/numeric tolerances;
7. exact-route emission remains 2.5%/85%/100%/100%, while all softmax rates,
   usefulness rates, radius reductions, and deviations are reported without
   tuning;
8. all empirical violations are enumerated and none is used as proof;
9. all old and new verifiers, Ruff, strict JSON, schema, provenance, and
   baseline-integrity checks pass on both routes;
10. both independent routes and both reciprocal verification reports are
    reproducible and end in `PASS`.

Failure of radius dominance is a scientific negative result, not permission to
retune weights. A valid negative completion requires a certified failing count
or proof, unchanged legacy behavior, full evidence, and reciprocal `PASS`
verification of the negative conclusion.

An increase in `<B` usefulness is an empirical secondary outcome. Its absence
does not invalidate a mathematically valid and uniformly tighter radius.

### Design-time feasibility audit

Before freezing this design, GPT evaluated the formulas themselves—not any new
trajectory or route output—at the frozen dimensions `m = 6`, `d = 24`, hence
`G = 54`, for every integer `1 <= k <= 16384`. The mixture/old radius ratio
was finite, strictly increasing with `k`, and always below one: its minimum was
approximately `0.676575` at `k = 1` and its maximum was approximately
`0.879389` at `k = 16384`. The mixture radius itself was nonincreasing over
the complete range, and every computed root satisfied `q_mix <= q_stitch`.

This calculation only establishes that the frozen acceptance test is
feasible and was not selected after inspecting new formal records. It is not
the theorem, formal evidence, or independent verification; both routes must
recompute it from the frozen formulas.

## 13. Transition-variance feasibility assessment

The optional study asks whether successor-state observations and deterministic
edge rewards can define a confidence set that upper-bounds

```text
Var_P[R(s,a,S') + gamma V(S')]
```

uniformly for every `V in [-B,B]^m`, including missing successor states, using
only observable inputs. It must explicitly account for uncertainty in the
transition distribution and cannot insert `V^pi`, `Q^pi`, a true residual, or
the empirical variance of an unobservable fixed-target sequence.

This assessment produces either a complete construction proposal or a
deterministic counterexample/identified gap. It receives no risk allocation,
does not enter the formal route certificate, and cannot delay a valid mandatory
mixture result.

## 14. Research governance and isolation

The task is long and conclusion-critical.

- Common scientific baseline:
  `28b71685ca05ae073cc847fdea230019e4bd63ea`.
- GPT branch: `codex/FP-TU-001`.
- Claude branch: `claude/FP-TU-001`.
- GPT results: `results/FP-TU-001/codex/`.
- Claude results: `results/FP-TU-001/claude/`.

GPT will create the formal task and plan. Claude performs a read-only task
pre-review before activation. After approval, both routes start from the same
frozen task-definition and activation commit, independently construct proof,
code, smoke, and formal evidence, and seal first results before disclosure.
Each then reproduces and verifies the other route. An objection blocks affected
work for user ruling; an implementation failure returns to the original author
for repair.

No merge to `main`, publication, external message, unsafe permission bypass,
or new fee category is authorized.

## 15. Evidence and stopping rules

Each route records exact commits, environment, commands, raw hashes, successful
and failed runs, anomalies, metrics, limitations, and numbered acceptance
judgments. Smoke evidence must precede the one frozen formal run. Formal outputs
contain exactly seven core artifacts: config, task records, summary,
regression, environment, commands, and checks.

Stop affected work if a primary-source assumption cannot be verified, Claude
raises an objection, the common baseline differs, an edit overlaps another
actor, a prohibited input is required, numerical inversion cannot be certified,
the frozen configuration would need to change, or external authorization/cost
would expand.

## 16. Primary sources

- Steven R. Howard, Aaditya Ramdas, Jon McAuliffe, and Jasjeet Sekhon,
  “Time-uniform Chernoff bounds via nonnegative supermartingales,” *Probability
  Surveys* 17 (2020), 257–317,
  https://doi.org/10.1214/18-PS321.
- Steven R. Howard, Aaditya Ramdas, Jon McAuliffe, and Jasjeet Sekhon,
  “Time-uniform, nonparametric, nonasymptotic confidence sequences,” *Annals of
  Statistics* 49(2) (2021), 1055–1080,
  https://doi.org/10.1214/20-AOS1991.
- Ian Waudby-Smith and Aaditya Ramdas, “Estimating means of bounded random
  variables by betting,” *JRSS Series B* 86(1) (2024), 1–27,
  https://doi.org/10.1093/jrsssb/qkad009. This source motivates the optional
  observable empirical-Bernstein question; its standard observed-sequence
  construction is not assumed to apply to unobservable fixed-target residuals.
