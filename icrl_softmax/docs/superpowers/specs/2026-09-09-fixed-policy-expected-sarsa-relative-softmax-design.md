# Fixed-policy Expected SARSA with certified relative-softmax improvement

Date: 2026-09-09

Proposed successor task: `FP-ESARSA-001`

Status: written design pending user review. This document records the direction
approved in conversation on 2026-09-09. It does not modify the frozen
`FP-ADV-001` scientific contract, authorize implementation, or activate a new
formal experiment.

## 1. Decision and motivation

The selected direction separates policy evaluation from policy improvement:

```text
freeze pi_k
    -> evaluate Q^{pi_k} with Expected SARSA residual iteration
    -> construct one relative-softmax candidate
    -> emit only when a simultaneous statewise certificate is nonnegative
```

This design makes the fixed-policy phase compatible with the normalized
softmax residual construction used in the fourth reference paper while
avoiding the moving-target analysis created by setting
`pi_l = softmax(beta Q_l)` inside every evaluation layer.

Three routes were considered:

1. **Fixed-policy Expected SARSA plus relative softmax.** This is the selected
   primary route. It removes next-action sampling noise, keeps the Bellman
   operator fixed across evaluation layers, and gives a direct softmax policy
   update after evaluation.
2. **Fixed-policy sampled SARSA on state-action pairs.** This is architecturally
   simpler because the observed next action supplies the bootstrap value, but
   it adds next-action sampling noise. It is retained as a control.
3. **V-first one-step action recovery.** This preserves state-value-only
   memory, but action-specific residuals do not vanish at `V^pi`; normalized
   cross-action leakage can therefore reverse an ordering and must be bounded
   directly. The existing negative usefulness result remains the comparison.

The new primary construction explicitly maintains `Q^pi`. It is therefore not
a V-only algorithm and must not be described as one.

## 2. Research question

For a finite MDP and one fixed full-support policy, can standard normalized
softmax attention implement a kernel-preconditioned Expected SARSA evaluation
whose population fixed point is `Q^pi`, obtain a finite-trajectory sup-norm
error certificate, and use that certificate to authorize one pointwise
non-degrading relative-softmax policy update?

The design separates three claims:

1. an exact algebraic implementation claim under explicit action/state
   grouping;
2. a finite-logit similarity-routing claim with all leakage included in the
   Q-error bound;
3. a policy-improvement claim conditional on the simultaneous Q-error event.

Failure to emit a nontrivial update is a valid negative result. It is not
permission to weaken the certificate after inspecting results.

## 3. Scope

### In scope

- finite state and action spaces;
- one stationary-start on-policy trajectory per outer iteration;
- one fixed full-support policy during all evaluation layers;
- one canonical Q-memory token for every logical state-action pair;
- synchronous Expected SARSA residuals computed from the same `Q_l`;
- a layer-invariant state-action similarity kernel for residual writeback;
- exact grouped attention and finite-logit standard-softmax variants;
- one relative-softmax policy candidate after evaluation;
- a simultaneous statewise lower-bound certificate and deterministic
  abstention;
- sampled SARSA and the existing V-first route as controls.

### Out of scope for the first formal task

- changing the policy inside the Q-evaluation layers;
- repeated online control under one trajectory or one certificate;
- off-policy reuse without an independently specified correction theorem;
- continuous actions, function approximation outside the declared finite
  feature construction, or learned attention parameters;
- oracle transition probabilities, oracle Q values, or realized return in a
  certificate input;
- claiming that empirical return improvement proves the theorem;
- modifying or rerunning the frozen `FP-ADV-001` matrix.

Repeated policy iteration may be studied later as a sequence of fresh outer
iterations, each with a newly frozen policy, a newly valid data protocol, and
a newly established certificate.

## 4. Inputs and memory representation

At outer iteration `k`, freeze a policy `pi_k(a|s)` with
`pi_k(a|s) >= pi_min > 0`. Let fixed logits satisfy

```text
pi_k(a|s) = softmax_a z_k(s,a).
```

Collect an on-policy trajectory

```text
D_k = {(S_t, A_t, R_{t+1}, S_{t+1}, done_t)}_{t=1}^n,
A_t ~ pi_k(.|S_t).
```

The prompt contains two logically distinct token families:

1. **Canonical Q-memory tokens.** There is exactly one token for each logical
   pair `x=(s,a)`, storing its current scalar `q_l(x)` and fixed identifiers.
2. **Transition tokens.** Each visit stores the observed transition fields and,
   after the residual block, one signed residual `delta_t^l`. A transition
   token is not an additional Q-memory candidate.

The Q table may start from zero or from a bounded warm start. The formal task
must freeze the initialization rule before any smoke output is inspected.

### Duplicate semantics

- Two numerically equal Q values belonging to different pair identifiers stay
  distinct.
- Repeated visits to the same pair produce repeated residual samples and are
  retained.
- The same canonical Q token may be read by multiple attention heads or
  computational branches.
- One attention candidate set may not contain multiple tokens representing the
  same logical pair. Otherwise softmax changes the total probability assigned
  to that pair.

If an alternate serialization necessarily contains `m_x` copies of logical
pair `x`, it is admissible only with a proved multiplicity correction
`score_copy = score_x - log m_x`, or with a prior exact aggregation that
returns one representative. The primary construction uses canonical unique
tokens and needs neither correction.

## 5. One Expected SARSA evaluation macro-layer

All residuals in macro-layer `l` are computed from `q_l`; only after every
residual is available is `q_{l+1}` written. This Jacobi-style synchronization
matches parallel Transformer execution.

### 5.1 Successor-policy expectation

For each transition, compute

```text
qbar_l(S_{t+1})
  = sum_b pi_k(b|S_{t+1}) q_l(S_{t+1},b).
```

In the exact grouped construction, the successor-action head attends once to
each canonical token in
`{(S_{t+1},b): b in A}`. Its action score is `z_k(S_{t+1},b)` and its value is
`q_l(S_{t+1},b)`, so the normalized attention weights equal the fixed policy.

In a literal finite-logit construction without an equality mask, the score
must combine a state-similarity margin and the fixed policy logit. Attention
mass on Q tokens belonging to other states is successor-head leakage. That
leakage is not silently discarded; it contributes to the finite-sample
Q-error certificate.

### 5.2 Current-pair read

A separate head reads `q_l(S_t,A_t)` from the canonical Q memory. Exact grouped
routing gives an exact scalar. Finite-logit routing contributes a second,
separately measured leakage term.

If `S_{t+1}=S_t` and the successor-policy sum includes `A_t`, the same
canonical token is intentionally read in both heads. The two heads have
separate softmax denominators, so this is not duplicate probability mass.

### 5.3 Signed residual

The transition token forms

```text
delta_t^l
  = R_{t+1}
    + gamma (1-done_t) qbar_l(S_{t+1})
    - q_l(S_t,A_t).
```

The residual is carried in a value channel and may be positive or negative.
Nonnegative normalized attention weights do not prevent a signed update.

For a self-loop with `S_{t+1}=S_t=s` and `A_t=a`, the repeated algebraic use of
the same Q entry is

```text
delta
  = r
    - [1-gamma pi_k(a|s)] q_l(s,a)
    + gamma sum_{b != a} pi_k(b|s) q_l(s,b).
```

This is the correct Expected SARSA residual, not a representation defect.

### 5.4 Pair-kernel residual writeback

For a canonical query pair `x=(s,a)`, let

```text
K_t(x)
  = softmax_t[tau sim(phi(x), phi(S_t,A_t))].
```

Use a fixed similarity feature map and fixed temperature across evaluation
layers. Update synchronously:

```text
q_{l+1}(x)
  = q_l(x) + alpha sum_t K_t(x) delta_t^l.
```

Under exact grouping, if `n_x` trajectory positions have `(S_t,A_t)=x`, then

```text
K_t(x) = 1/n_x for matching positions and 0 otherwise,

q_{l+1}(x)
  = q_l(x) + (alpha/n_x) sum_{t:X_t=x} delta_t^l.
```

Thus repeated visits are averaged rather than treated as duplicate Q-memory.
An unvisited pair cannot receive a fabricated exact update. The exact route
must abstain or retain the prior value; the finite-softmax route must expose
any null/self-token or coverage mechanism and charge its effect to the bound.

One logical Expected SARSA iteration requires sequential residual formation
and residual writeback. A literal Transformer realization therefore uses at
least two sequential blocks unless an equivalent dependency-preserving
construction is proved.

## 6. Pair-MRP operator interpretation

Lift the MDP to the pair state space `X = S x A` with transition kernel

```text
P_pi^X((s',b)|(s,a)) = P(s'|s,a) pi_k(b|s').
```

The value function of this fixed pair MRP is exactly `Q^{pi_k}`. Define

```text
(T_pi^X Q)(s,a)
  = r(s,a)
    + gamma sum_{s',b}
        P(s'|s,a) pi_k(b|s') Q(s',b).
```

The population analogue of the kernel update is

```text
F_pi(Q) = Q + alpha M_pi^X (T_pi^X Q - Q),
```

where `M_pi^X` is the row-stochastic pair-similarity operator induced by
normalized softmax attention.

Because every pair Bellman residual vanishes at the true value,

```text
T_pi^X Q^pi - Q^pi = 0,
F_pi(Q^pi) = Q^pi.
```

Consequently, off-diagonal residual mixing does not by itself move the
population fixed point. This cancellation applies to full Bellman-residual
iteration; it does not apply to one-shot V-first action ranking, where
individual action advantages are generally nonzero.

For `0 < alpha <= 1`, a sufficient contraction route is to prove a pair-space
diagonal margin

```text
min_x M_pi^X(x,x) >= (1+gamma)/2 + C,
0 < C < (1-gamma)/2.
```

Using row stochasticity gives the conservative bound

```text
||I - alpha M_pi^X + alpha gamma M_pi^X P_pi^X||_infinity
  <= 1 - 2 alpha C.
```

The formal proof must check the exact matrix definitions and may use a sharper
constant, but it may not weaken the declared diagonal assumption after seeing
empirical coverage. This condition is expected to be demanding because pair
occupancy equals state occupancy times action probability.

## 7. Finite-trajectory Q certificate

After `L` macro-layers, write

```text
Qhat_k = q_L.
```

The formal task must establish a simultaneous event of the form

```text
for every (s,a):
  |Qhat_k(s,a) - Q^{pi_k}(s,a)| <= E_k(s,a).
```

The reported bound must separately account for:

1. finite-layer contraction error;
2. empirical pair-kernel deviation from its population operator;
3. Markov-trajectory reward and successor fluctuation;
4. successor-action-head state leakage;
5. current-pair-read leakage;
6. missing or rare pair coverage;
7. numerical approximation explicitly used by the construction.

For policy improvement define the statewise radius

```text
E_k(s) = max_a E_k(s,a).
```

The certificate receives only observable inputs and predeclared constants.
True transition kernels, true values, realized errors, exact gaps, and exact
returns remain confined to an oracle audit.

## 8. Relative-softmax policy candidate

After evaluation ends, form one candidate with a predeclared or
certificate-selected `eta >= 0`:

```text
pi_{k,eta}^+(a|s)
  = pi_k(a|s) exp(eta Qhat_k(s,a))
    / sum_b pi_k(b|s) exp(eta Qhat_k(s,b)).
```

Equivalently, if `pi_k = softmax(z_k)`, then

```text
pi_{k,eta}^+ = softmax_a[z_k(s,a) + eta Qhat_k(s,a)].
```

This is a relative update: it tilts the current policy rather than replacing
it with a Boltzmann policy constructed from an implicit uniform base measure.
If the base policy is uniform, the two forms coincide.

Finite logits preserve strict positivity, but this formula does not
automatically preserve a declared numerical exploration floor. If the formal
task requires `pi_{k+1}(a|s) >= pi_min`, candidate validity additionally
requires that inequality for every pair. A candidate that violates the floor
is rejected; the frozen search rule may try a smaller eta, and otherwise must
abstain. The floor check is deterministic and precedes the improvement
certificate.

With exact `Q^{pi_k}`, exponential tilting implies for every state and every
`eta >= 0`:

```text
sum_a pi_{k,eta}^+(a|s) Q^{pi_k}(s,a)
  >= sum_a pi_k(a|s) Q^{pi_k}(s,a)
  = V^{pi_k}(s).
```

The pointwise policy-improvement theorem then yields
`V^{pi_{k,eta}^+} >= V^{pi_k}` componentwise.

## 9. Approximate-Q improvement certificate

For a candidate `pi_{k,eta}^+`, define the observable estimated improvement

```text
Ihat_s(eta)
  = sum_a [pi_{k,eta}^+(a|s)-pi_k(a|s)] Qhat_k(s,a).
```

On the simultaneous Q event,

```text
I_s(eta)
  = sum_a [pi_{k,eta}^+(a|s)-pi_k(a|s)] Q^{pi_k}(s,a)

  >= Ihat_s(eta)
     - E_k(s) ||pi_{k,eta}^+(.|s)-pi_k(.|s)||_1.
```

Define

```text
LB_s(eta)
  = Ihat_s(eta)
    - E_k(s) ||pi_{k,eta}^+(.|s)-pi_k(.|s)||_1.
```

The emission rule is deterministic:

- accept a candidate only if it satisfies every declared policy-validity
  constraint and `LB_s(eta) >= 0` for every state in the declared finite MDP;
- require at least one strictly positive lower bound and a changed policy for
  a nontrivial-emission label;
- otherwise return exactly `pi_k` and record an abstention;
- if several eta values are searched, the search rule and tie-breaking must be
  frozen before formal output. A simultaneous Q event makes the deterministic
  post-estimation comparison valid for every candidate derived from `Qhat`.

The theorem claim is selective:

```text
P(emit and exists s: V^{pi_{k+1}}(s) < V^{pi_k}(s)) <= delta.
```

It does not claim high-probability emission or improvement conditional on
emission.

## 10. End-to-end algorithm

```text
input: pi_0, gamma, alpha, L, fixed attention features, certificate level

for outer iteration k:
    freeze pi_k and its logits z_k
    collect the declared on-policy trajectory D_k
    initialize one canonical q_0(s,a) per logical pair

    for l = 0,...,L-1:
        for every transition t, in parallel:
            read q_l(S_t,A_t)
            compute qbar_l(S_{t+1})
                = sum_b pi_k(b|S_{t+1}) q_l(S_{t+1},b)
            form signed Expected SARSA residual delta_t^l

        for every canonical pair x, in parallel:
            q_{l+1}(x)
                = q_l(x) + alpha sum_t K_t(x) delta_t^l

    construct Qhat_k and simultaneous error radii E_k
    construct policy-valid relative-softmax candidates from z_k + eta Qhat_k
    emit one candidate only when every required statewise LB is nonnegative
    otherwise set pi_{k+1} = pi_k exactly
```

The first formal task implements and verifies one outer iteration only.
Repeating the displayed outer loop requires a separate protocol specifying
fresh data, dependence across iterations, risk allocation, and stopping.

## 11. Component boundaries

The later implementation plan should preserve these independent components:

1. **Memory serializer:** creates canonical pair memory and transition tokens;
2. **Fixed-policy expectation head:** computes the successor action average;
3. **Residual constructor:** combines reward, terminal flag, successor
   expectation, and current-pair read;
4. **Pair-kernel writer:** performs synchronous normalized residual updates;
5. **Q-certificate module:** returns observable simultaneous pair radii or a
   structured abstention;
6. **Relative-softmax updater:** constructs candidates without accessing
   oracle values;
7. **Improvement certifier:** computes statewise lower bounds and emits or
   abstains;
8. **Oracle audit:** measures truth-based error and return only after the
   decision and stores it in a structurally separate namespace.

Each component must have a pure mathematical reference implementation before
a literal Transformer realization is accepted.

## 12. Validation design

Deterministic tests must cover:

1. exact Expected SARSA targets against direct tabular calculations;
2. terminal transitions and self-loops where one Q entry has two algebraic
   roles;
3. repeated visits producing residual averaging without duplicate Q-memory;
4. rejection or correction of duplicate logical Q candidates;
5. successor-action attention weights equal to the frozen policy in the exact
   grouped route;
6. direct finite-logit calculations for both read-head leakage terms;
7. equality between exact grouping and batch Expected SARSA;
8. pair-MRP Bellman fixed-point identity;
9. contraction fixtures satisfying and violating the declared diagonal
   margin;
10. empirical-operator and finite-layer error decomposition;
11. exact relative-softmax statewise improvement by exhaustive finite examples;
12. the approximate-Q lower bound under adversarial errors at every corner of
    the declared error box;
13. exact policy identity on abstention, row sums, nonnegativity, finite
    logits, and any declared exploration floor;
14. strict separation of certificate inputs from oracle audit fields;
15. sampled SARSA and V-first controls under the same frozen data protocol.

The formal evaluation matrix, random seeds, eta candidate rule, attention
temperatures, initialization, and stopping conditions belong in the future
formal task. They must be frozen before Claude pre-review and before either
route observes smoke or formal outcomes.

## 13. Failure modes and stopping rules

The successor task must stop affected work rather than silently weaken the
claim if:

- the canonical Q-memory cannot be represented without unaccounted duplicate
  softmax mass;
- the successor-action head cannot isolate the intended state or bound its
  finite-logit leakage;
- pair occupancy makes the stated diagonal/contraction premise empty on the
  frozen protocol;
- an unvisited pair receives a non-observable oracle correction;
- a finite-trajectory bound omits one of the declared leakage or dependence
  terms;
- the candidate uses a policy that changed during Q evaluation;
- an emitted update has a negative statewise lower bound, violates the
  simplex, or changes the policy on abstention;
- implementation requires modifying frozen `FP-ADV-001` evidence;
- the independent pre-review returns `OBJECTION`.

The expected scientific risk is zero nontrivial emissions because the
pair-space diagonal and uniform Q requirements may be conservative. Such an
outcome falsifies usefulness under the frozen protocol but does not falsify a
correct conditional safety theorem.

## 14. Governance and next gate

`FP-ESARSA-001` is expected to be long and conclusion-critical. After the user
reviews this written design, GPT may draft a formal research task and an
implementation plan. The formal task must pass Claude's read-only pre-review
before becoming `ACTIVE`; GPT and Claude then execute independent routes from
one frozen baseline and reciprocally verify the results.

No code, experiment, task activation, merge, push, or change to
`ACTIVE_WORKSPACE.md` is authorized by this design document alone.

## 15. Reference mapping

- The fixed-policy normalized residual operator and diagonal-kernel proof
  grammar follow *Beyond Linear Attention: Softmax Transformers Implement
  In-Context Reinforcement Learning* after lifting states to state-action
  pairs.
- The Expected SARSA target supplies the exact fixed-policy pair-MRP Bellman
  residual.
- The policy step uses exponential tilting of the current policy and the
  pointwise policy-improvement theorem.
- `FP-ADV-001` remains the frozen V-first action-gap comparison and is not
  retroactively reinterpreted by this design.
