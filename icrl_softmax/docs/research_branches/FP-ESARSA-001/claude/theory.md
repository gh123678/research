# FP-ESARSA-001 (v1.1) — Claude route theory: fixed-policy Expected SARSA, cross-fitted residual certificate, relative-softmax step

Date: 2026-09-11
Route: `claude/FP-ESARSA-001`
Scientific/code baseline: `9d0994f03e4a659787c74df4660cdcab1d398c1c`
Common activation commit: `e2859f2` on `codex/FP-ESARSA-001`

This note derives the complete mathematical chain for the Claude main route
before implementation and before any smoke or formal output is inspected. No
formula here may be repaired after empirical output is seen. The same
development is recorded as the main-route consolidated theory in
`docs/research_branches/fixed_policy_expected_sarsa_theory.md`. Claims are
organized against hypotheses H1-H8 of the task definition.

## 1. Frozen setting and notation

Finite MDP with `|S|=6` states, `|A|=4` actions, pair space `X=S x A`,
`d=|S||A|=24`. One frozen full-support policy `pi(a|s) >= pi_min = 0.05`.
Frozen constants of the task:

```text
gamma = 0.70          alpha = 0.65         layers L = 160
R_star = 1.5          B = R_star/(1-gamma) = 5.0      2B = 10.0
delta = 0.05          mixture components = 15
zeta = xi = tau = 8   eta grid = [1.0, 0.5, 0.2, 0.1, 0.05] descending
seed = 20260829       Q_0 = 0              split m = n/2 contiguous
lengths n in {256, 1024, 4096, 16384}
```

The pair MRP has kernel and Bellman operator

```text
P_pi^X((s',b)|(s,a)) = P(s'|s,a) pi(b|s'),

(T_pi^X Q)(s,a)
  = r(s,a) + gamma sum_{s',b} P(s'|s,a) pi(b|s') Q(s',b),
```

where `r(s,a) = E[R_{t+1} | X_t=(s,a)]` is the expected one-step reward,
which may aggregate over successor-dependent realized rewards.

**Lemma 1.1 (pair-MRP representation).** `T_pi^X` is a `gamma`-contraction in
supremum norm: for any `Q,Q'`,

```text
||T_pi^X Q - T_pi^X Q'||_inf
  = gamma max_x |sum_{s',b} P_pi^X((s',b)|x) (Q-Q')(s',b)|
  <= gamma ||Q - Q'||_inf,
```

because `P_pi^X` is row-stochastic. Its unique fixed point `Q^pi` is exactly
the action-value function of the fixed policy, and bounded rewards
`|R_{t+1}| <= R_star` give `||Q^pi||_inf <= R_star/(1-gamma) = B`.

## 2. Exact grouped route is synchronous batch Expected SARSA (H1)

Canonical memory: exactly one persistent Q-memory token per logical pair.
Transition tokens carry observed fields and one signed residual; they are not
Q candidates. Equal values under different pair identifiers stay distinct;
repeated visits are repeated residual samples; one canonical token may be read
by several heads; no attention candidate set contains the same logical pair
twice.

Per layer `l`, from the unchanged `Q_l`, for every training transition `t<m`:

```text
qbar_t^l = sum_b pi(b|S_{t+1}) Q_l(S_{t+1},b),
delta_t^l = R_{t+1} + gamma qbar_t^l - Q_l(S_t,A_t),
```

and for every canonical pair `x` with training count `N_x^train > 0`,

```text
Q_{l+1}(x) = Q_l(x) + (alpha/N_x^train) sum_{t<m: X_t=x} delta_t^l,
```

leaving unvisited pairs unchanged. All updates use only `Q_l`, so the sweep is
synchronous (Jacobi), matching parallel tensor execution.

**Theorem 2.1 (exact attention equals batch Expected SARSA).** In the exact
grouped construction:

- the successor-action head for query state `s'` admits exactly the candidate
  set `{(s',b): b in A}` with scores `log pi(b|s')`; normalized weights are
  `exp(log pi(b|s')) / sum_b' exp(log pi(b'|s')) = pi(b|s')`, so its output is
  exactly `qbar`;
- the current-pair head admits only the unique token `(S_t,A_t)`, so its read
  is exactly `Q_l(S_t,A_t)`;
- the writer for query `x` admits exactly the transitions with `X_t=x` (or a
  zero-value null token when `N_x^train=0`), so its weight is `1/N_x^train` on
  matching positions and the update is the grouped residual mean.

Hence the tensor route computes the recurrence above exactly (fixture-verified
to `1e-12`; acceptance criteria 2, 4).

**Self-loops (acceptance criterion 3).** If `S_{t+1}=S_t=s` and `A_t=a`, the
same canonical token `(s,a)` appears once in the action-head candidate set and
once in the read-head candidate set. The two heads have separate softmax
denominators, so this is not duplicate probability mass, and the residual
algebra is the correct Expected SARSA branch reuse:

```text
delta = R + gamma [ pi(a|s) Q(s,a) + sum_{b!=a} pi(b|s) Q(s,b) ] - Q(s,a)
      = R - [1 - gamma pi(a|s)] Q(s,a) + gamma sum_{b!=a} pi(b|s) Q(s,b).
```

## 3. Finite-logit route: exact finite-score formulas (H2)

The finite route uses fixed one-hot pair identifiers and frozen sharpness
`zeta=xi=tau=8`. It has no input-dependent equality mask and no visited-query
gate; every candidate set is the full canonical memory or the full training
set; all scores are finite; softmax is the standard normalized softmax.

**Successor head.** Query state `s'`, candidates all `d` canonical tokens
`(u,b)`, score `zeta 1{u=s'} + log pi(b|u)`. Each policy row sums to one, so
the unnormalized mass of state group `u` is `sum_b exp(zeta 1{u=s'}) pi(b|u)
= exp(zeta 1{u=s'})`, the normalizer is `exp(zeta)+|S|-1`, and

```text
kappa_state = exp(zeta)/(exp(zeta)+|S|-1),

qbar^fin(s')
  = [ exp(zeta) sum_b pi(b|s') Q(s',b)
      + sum_{u!=s'} sum_b pi(b|u) Q(u,b) ] / (exp(zeta)+|S|-1).
```

Conditional on the requested state, action weights equal `pi(.|s')` exactly;
off-state mass `1-kappa_state` is successor-head leakage.

**Current-pair read.** Query pair `x`, score `xi 1{x=y}` over canonical tokens
`y`. Normalizer `exp(xi)+d-1`, so

```text
kappa_read = exp(xi)/(exp(xi)+d-1),

read^fin(x) = [ exp(xi) Q(x) + sum_{y!=x} Q(y) ] / (exp(xi)+d-1).
```

**Residual.** `delta_t^fin = R_{t+1} + gamma qbar^fin(S_{t+1}) -
read^fin(X_t)`; signed, carried in a value channel.

**Writeback.** Query pair `x` over the `m` training transitions, score
`tau 1{X_t=x}`:

```text
w_t(x) = exp(tau 1{X_t=x}) / (N_x^train exp(tau) + (m - N_x^train)),

Q_{l+1}(x) = Q_l(x) + alpha sum_{t<m} w_t(x) delta_t^fin.
```

Every canonical query is updated, including unvisited ones (uniform weights
`1/m`); off-group mass and unvisited-query updates are reported, not hidden.

**Theorem 3.1.** The literal tensor network's direct outputs equal these
formulas (fixture-verified to `1e-12` with finite, positive, row-normalized
weights; acceptance criterion 5).

**Honesty boundary.** At `Q^pi` the finite residual does not vanish:
`qbar^fin` and `read^fin` carry leakage, so the finite route's population
residual is not the exact Bellman residual. No asymptotic no-bias claim is
made for the full finite route; its final estimate is certified only by the
exact held-out Bellman residual of Section 6.

## 4. Exact-residual kernel population operator (H3)

Let `M_pi^X` be any row-stochastic population pair kernel induced by the fixed
writeback attention, and feed it the exact fixed-policy Bellman residual:

```text
F_pi(Q) = Q + alpha M_pi^X (T_pi^X Q - Q).
```

**Lemma 4.1 (fixed point).** `T_pi^X Q^pi = Q^pi`, so `F_pi(Q^pi) = Q^pi`.

**Lemma 4.2 (diagonal-margin contraction).** `F_pi` is affine with linear part
`L = I - alpha M + alpha gamma M P` (writing `M=M_pi^X`, `P=P_pi^X`). For
`0 < alpha <= 1` and row `x`: the diagonal entry satisfies

```text
L_xx = 1 - alpha M_xx + alpha gamma (MP)_xx >= 1 - alpha M_xx >= 0,
```

so `|L_xx| = L_xx`, while off-diagonal `|L_xz| <= alpha M_xz +
alpha gamma (MP)_xz`. Row-stochasticity of `M` and `MP` then gives

```text
||L||_inf
  <= max_x [ 1 - alpha M_xx + alpha gamma (MP)_xx
             + alpha (1-M_xx) + alpha gamma (1-(MP)_xx) ]
  = 1 + alpha (1 + gamma - 2 min_x M_xx).
```

Under the frozen premise `min_x M_xx >= (1+gamma)/2 + C` with
`0 < C < (1-gamma)/2`,

```text
||L||_inf <= 1 - 2 alpha C < 1,
  ||F_pi(Q) - F_pi(Q^pi)||_inf <= (1-2 alpha C) ||Q - Q^pi||_inf,
```

so `Q^pi` is the unique fixed point and the population iteration converges
geometrically. The theorem is conditional: the premise must be checked
empirically per instance, and an empirical failure is reported, never
oracle-repaired. (Verified on direct matrices in both the satisfied and the
violated regime; acceptance criterion 6.)

**Lemma 4.3 (no leakage cancellation).** Lemma 4.1 uses that the kernel input
is the exact Bellman residual. The finite route feeds
`R + gamma qbar^fin - read^fin`, whose population mean at `Q^pi` equals
`gamma` times successor/read leakage terms and is generally nonzero. The
finite population fixed point therefore solves a shifted equation; the
identity does not cancel successor-head or current-read leakage, and no
no-bias claim follows for the finite route.

## 5. Split measurability

The split index `m=n/2` is deterministic. `Qhat` is a function of the first
`m` transitions and frozen public constants only; validation transitions
cannot influence Q construction, stopping, or selection (structural separation
in the evaluator). Conditional on the training filtration
`F_train = sigma(transitions 0..m-1, policy, constants)`, `Qhat` is therefore
a constant vector, while each held-out transition retains its one-step Markov
conditional law given `X_t=x`. No stationarity or mixing assumption is needed
for the certificate; only the conditional one-step law enters.

## 6. Held-out visit-indexed martingale and simultaneous mixture event (H4)

Freeze `Qhat`. For held-out transitions (`m <= t <= n-1`) define the exact
Expected SARSA residual

```text
Y_t(Qhat) = R_{t+1}
  + gamma sum_b pi(b|S_{t+1}) Qhat(S_{t+1},b)
  - Qhat(S_t,A_t),
```

with exact arithmetic over the known finite action row: no sampled next
action, no environment model.

**Lemma 6.1 (conditional mean).**

```text
E[ Y_t(Qhat) | F_train, held-out history to t-1, X_t=x ]
  = (T_pi^X Qhat - Qhat)(x).
```

Proof: `Qhat` is constant given `F_train` (Section 5);
`E[R_{t+1}|X_t=x] = r(x)` by definition of the pair-MRP reward; and
`E[ sum_b pi(b|S_{t+1}) Qhat(S_{t+1},b) | X_t=x ]
= sum_{s'} P(s'|x) sum_b pi(b|s') Qhat(s',b)`. The sum is `(T_pi^X Qhat)(x)`;
subtracting `Qhat(x)` gives the claim.

**Lemma 6.2 (range).** If `||Qhat||_inf <= B` (divergence guard), then

```text
|Y_t| <= R_star + gamma B + B = B(1-gamma) + (1+gamma) B = 2B,
```

so each centered increment lies in an interval of width `4B`. Normalizing
`Z-increments = (Y - E[Y|...])/(2B)` gives width `2`, and interval-form
Hoeffding yields `E[exp(a Z_inc) | past] <= exp(a^2/2)` for every real `a`.

**Visit indexing (inherited, VERIFIED FP-MART-001 / FP-TU-001).** For pair
`x`, enumerate its held-out visit times `tau_1(x) < tau_2(x) < ...`; these are
stopping times in the held-out filtration enlarged by the visit indicators.
The visit-indexed sums

```text
S_k(x) = sum_{i<=k} [ Y_{tau_i(x)} - (T_pi^X Qhat - Qhat)(x) ]
```

are martingale partial sums with the same `exp(a^2/2)` conditional bound per
step. With the frozen 15-component geometric grid (target counts `2^j`,
weights `w_j proportional to (j+1)^{-2}`, rates
`a_j = sqrt(2 log(2 d/(delta w_j))/2^j)`), the cosh mixture

```text
M(k,q) = sum_j w_j exp(-a_j^2 k/2) cosh(a_j q)
```

is a nonnegative supermartingale in `k` with `M(0,0)=1`.

**Theorem 6.3 (simultaneous event; d=24 groups).** Ville's inequality plus a
union over exactly `d` pair groups give one event of probability at least
`1-delta` on which, for every pair `x` and every `k >= 1`,

```text
|S_k(x)| <= 2B q_mix(k; d, delta),
```

where `q_mix(k; d, delta)` is the conservative root of
`log M(k,q) = log(d/delta)` returned by the inherited `build_mixture_grid` /
`solve_mixture_boundary` (15 components, tolerance `1e-12`, upper bracket
endpoint). Because the event is uniform over all counts `k`, the random
observed count `N_x` may be substituted directly (random-count substitution;
no per-count union). On the event, for every pair with `N_x > 0`,

```text
| (1/N_x) sum_{held-out t: X_t=x} Y_t(Qhat) - (T_pi^X Qhat - Qhat)(x) |
  <= r_x := 2B q_mix(N_x; d, delta) / N_x.
```

No radius exists at `N_x = 0`.

## 7. Bellman-residual certificate (H5)

Require full held-out pair support (every pair visited in the suffix;
otherwise `heldout_pair_support_missing`), all numerical checks passing, and
the divergence guard `||Qhat||_inf <= B`. Define

```text
epsilon_res = max_x ( |Ybar_x(Qhat)| + r_x ),      E_Q = epsilon_res/(1-gamma).
```

**Theorem 7.1.** On the simultaneous event,

```text
||Qhat - Q^pi||_inf <= E_Q.
```

Proof: on the event `||T_pi^X Qhat - Qhat||_inf <= epsilon_res` by Theorem
6.3, and

```text
||Qhat - Q^pi||_inf
  <= ||Qhat - T_pi^X Qhat||_inf + ||T_pi^X Qhat - T_pi^X Q^pi||_inf
  <= epsilon_res + gamma ||Qhat - Q^pi||_inf;
```

rearrange. The certificate inputs are `Qhat`, the frozen policy, held-out
transitions, `R_star`, `gamma`, `delta`, and public mixture constants only:
model-free, with true kernels, values, errors, gaps, and returns confined to
the oracle audit (acceptance criteria 8, 14).

## 8. Exact relative-softmax improvement (H6)

For `eta >= 0` define the relative-softmax tilt

```text
pi_eta^+(a|s) = pi(a|s) exp(eta Q(s,a)) / sum_b pi(b|s) exp(eta Q(s,b)).
```

**Lemma 8.1 (KL identity).** With `Z_s = sum_b pi(b|s) exp(eta Q(s,b))`,
`log(pi_eta^+/pi) = eta Q - log Z_s`, so

```text
KL(pi_eta^+||pi) + KL(pi||pi_eta^+)
  = [ eta sum_a pi_eta^+ Q - log Z_s ] + [ log Z_s - eta sum_a pi Q ]
  = eta sum_a (pi_eta^+ - pi) Q.
```

**Theorem 8.2 (exact improvement).** For exact `Q = Q^pi`, every `eta >= 0`
and every state,

```text
I_s(eta) := sum_a (pi_eta^+ - pi) Q^pi
          = (1/eta)[ KL(pi_eta^+||pi) + KL(pi||pi_eta^+) ] >= 0
```

(for `eta=0` the candidate is the identity and `I_s(0)=0`). Since
`sum_a (pi_eta^+-pi) = 0`, equivalently `sum_a pi_eta^+ A^pi >= 0` with
`A^pi = Q^pi - V^pi`. Verified analytically here and exhaustively on finite
fixtures (acceptance criterion 9).

**Corollary 8.3 (pointwise policy improvement).**
`(T_{pi_eta^+} V^pi)(s) - V^pi(s) = I_s(eta)`, because
`Q^pi(s,a) = r(s,a) + gamma sum_{s'} P(s'|s,a) V^pi(s')`. Hence
`I_s(eta) >= 0` for all `s` implies `T_{pi_eta^+} V^pi >= V^pi`
componentwise, and monotonicity of `T_{pi_eta^+}` gives
`V^{pi_eta^+} >= V^pi` componentwise.

## 9. Approximate-Q lower bound and certified emission (H7)

On the certificate event, `||Qhat - Q^pi||_inf <= E_Q`. For the candidate
built from `Qhat`, Hoelder's inequality gives, state by state,

```text
|I_s(eta) - Ihat_s(eta)|
  <= E_Q ||pi_eta^+(.|s) - pi(.|s)||_1,

I_s(eta) >= LB_s(eta)
  := Ihat_s(eta) - E_Q ||pi_eta^+(.|s) - pi(.|s)||_1.
```

**Emission rule (deterministic, frozen).** Scan `eta` in
`[1.0, 0.5, 0.2, 0.1, 0.05]` in this descending order; emit the first
candidate whose policy is finite, strictly positive, row-normalized to
`1e-12`, changed from `pi`, and whose `LB_s(eta) >= 0` for every state with at
least one `LB_s(eta) > 0`. If no candidate passes, return the input policy
bit-for-bit and record the applicable ordered non-emission reason.

**Theorem 9.1 (emitted updates are safe on the event).** On the simultaneous
residual event, every emitted update satisfies `LB_s(eta) >= 0` for all `s`,
hence `I_s(eta) >= 0`, hence by Corollary 8.3 `T_{pi_plus} V^pi >= V^pi` and
`V^{pi_plus} >= V^pi` componentwise. Therefore

```text
P( EmitUpdate and exists s: V^{pi_plus}(s) < V^pi(s) ) <= delta.
```

**No new risk from eta selection.** The grid comparison is a deterministic
function of `(Qhat, E_Q, pi)`, all measurable with respect to the same data
that define the simultaneous event. The bound of Theorem 7.1 holds for every
candidate derived from `Qhat` at once; an emitted state-degrading update would
imply the event failed. The selection therefore spends no additional risk
budget, and there is no claim about emission probability or coverage
conditional on emission.

## 10. Ordered non-emission reasons

The frozen ordered list, mapped to the deterministic checks of the decision
pipeline:

```text
 1. algorithm_mode_mismatch        wrong route/mode flags
 2. duplicate_q_memory             duplicate logical pair candidate detected
 3. divergence_guard_triggered     ||Qhat||_inf > B (no clipping, no repair)
 4. heldout_pair_support_missing   some N_x = 0 in the held-out suffix
 5. mixture_inversion_unbracketed  inherited inversion bracket failure
 6. mixture_inversion_not_converged
 7. mixture_root_not_conservative
 8. numerical_nonfinite            NaN/Inf anywhere in the chain
 9. policy_invalid                 nonfinite, nonpositive, or misnormalized row
10. improvement_lcb_nonpositive     every eta has some LB_s < 0 (or none > 0)
11. policy_unchanged                surviving candidate equals pi
```

Reasons 10 and 11 are ordinary abstentions. No reason triggers an oracle
fallback, a different eta grid, a different Q route, or a second risk budget.

## 11. What is not claimed

- no high-probability or conditional-on-emission guarantee; the only
  probability statement is the selective one of Theorem 9.1;
- no preservation of the numerical floor `pi_min`: the emitted policy is
  strictly positive but may fall below `0.05`; repeated control is out of
  scope;
- no asymptotic no-bias claim for the finite-logit route (Lemma 4.3); its
  estimate is certified only through the held-out residual;
- the contraction theorem is conditional on the diagonal-margin premise;
  empirical premise failures are enumerated, not repaired;
- hypothesis H8 (at least one nontrivial primary emission in the frozen 480
  records) may fail; zero primary emissions is a valid verified negative
  usefulness result with no retuning of any frozen quantity;
- all reported numbers are certificates over the frozen protocol, not
  sample-complexity claims.

## 12. Claim-to-verification map

```text
H1 canonical memory, exact equivalences, self-loops, repeated visits,
   unvisited queries, synchrony       verifier sections on deterministic
                                      fixtures, tolerance 1e-12
H2 finite successor/read/write formulas, no masks or gates
                                      direct finite-score fixtures + source
                                      boundary checks
H3 fixed-point identity, contraction bound, both premise regimes
                                      direct random/explicit matrices
H4 filtration, 2B scale, mixture event at random counts, d groups
                                      high-precision independent mixture
                                      roots, random-count fixtures,
                                      train/validation leak counterexamples
H5 E_Q formula and validity           direct Bellman solves, adversarial
                                      residual corners, ordered reasons
H6 exact improvement identity         exhaustive finite fixtures
H7 lower bound, emission safety       adversarial error-box corners,
                                      oracle value comparisons
H8 usefulness without retuning        one frozen formal run, emission
                                      enumeration in the analyzer
```
