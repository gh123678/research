# Fixed-policy action-gap certificate and one safe policy update

Date: 2026-09-08

Proposed task: `FP-ADV-001`

Approved direction: use V-first action-difference certificates as the primary
construction, retain Direct-Q as a control, and make at most one certified
policy update. The user selected this direction by instructing GPT to follow
the recommendation in the linked prior task on 2026-09-08.

## 1. Context and decision

`FP-TU-001` verified a time-uniform fixed-policy certificate for complete
Direct-Q and V-first Q recovery. At trajectory length 16384, the estimators are
already accurate enough to select the true greedy action about 96% of the time,
but the complete-table certificate remains too conservative: Direct-Q has no
`total_bound < B` result in the frozen cells, while V-first exact has 101/120.
The next scientific question is therefore not whether every entry of Q can be
recovered uniformly, but whether the data can certify the action comparisons
that a policy update actually uses.

Three approaches were considered:

1. Convert the existing global Q sup-norm bound into a gap bound by subtracting
   `2 E_Q`. This is immediately valid but preserves the unrelated rare-pair
   bottleneck. It will be the frozen baseline.
2. Construct V-first pairwise bounds from the two compared recovery groups and
   propagate value error through the observed difference of their successor
   rows. This is the primary route because common value offsets cancel and
   unrelated missing state-action pairs do not block a comparison.
3. Derive a new componentwise Direct-Q interval recurrence. This could be
   valuable, but it is a separate theory problem and would combine two major
   innovations in one task. It is deferred to `FP-LOCAL-001`.

The design selects approach 2, with approach 1 as a within-record control.

## 2. Research question

Under the verified `FP-TU-001` fixed-policy assumptions, can the already
allocated time-uniform residual event yield computable lower confidence bounds
for selected action differences

```text
Delta_pi(s,a,b) = Q^pi(s,a) - Q^pi(s,b),
```

and can those bounds safely move probability from certified inferior actions
to a certified superior action while guaranteeing one pointwise non-degrading
policy update?

The task is complete as a verified negative result if the construction is
mathematically valid but never emits a useful update under the frozen protocol.
No threshold, transfer rule, route, or matrix may be changed after smoke or
formal output is inspected.

## 3. Scope and non-goals

### In scope

- One fixed policy, one stationary-start on-policy trajectory, finite state and
  action spaces, deterministic bounded edge rewards, fixed-context synchronous
  evaluation, and the exact sampling order verified by `FP-MART-001` and
  `FP-TU-001`.
- V-first no-split exact matching and finite-softmax matching as primary
  constructions.
- Existing Direct-Q and global V-first certificates as controls.
- Data-dependent selection of action pairs only after constructing one event
  simultaneous over all pre-registered residual groups.
- One deterministic probability-transfer update that preserves the declared
  exploration floor.
- Exact truth-based action gaps, Bellman improvements, and returns only inside
  a structurally separate oracle audit.

### Out of scope

- A second policy-evaluation trajectory, repeated or blockwise control, online
  learning, adaptive risk spending, or a convergence theorem.
- New Direct-Q local propagation, variance-adaptive boundaries, new mixture
  weights, or changes to the verified residual event.
- Claims of conditional coverage given emission, high-probability support, or
  strict return improvement when the update is empty or the changed states are
  unreachable.
- Changing the MDP generator, estimators, reward assumptions, seed, formal
  matrix, or any legacy output.

## 4. Reused probability event

The task spends no new risk budget. It reuses the `FP-TU-001` event over
`G = m + 2d` groups: `m` state Bellman groups, `d` pair Bellman groups, and
`d` V-first recovery groups. On that event, every observed group-specific
mixture radius and every emitted state-value bound holds simultaneously at its
random visit count.

For every emitted update, the only probability statement is

```text
P(EmitUpdate and (
    any used action ordering is false
    or V^{pi_plus}(s) < V^pi(s) for some state s
)) <= delta.
```

Candidate actions may be selected after observing the estimates because no
new statistical statement is selected: every candidate lower bound is a
deterministic consequence of the same simultaneous event. No route-level or
pair-level resplit of `delta` is permitted.

## 5. Exact V-first action-gap bound

Let `B = R_star/(1-gamma)`. For one fixed-policy V-first exact route, let
`V_hat` be the estimated state value and let the verified state certificate
emit

```text
||V_hat - V^pi||_infinity <= E_V.
```

For visited pair `g=(s,a)`, define

```text
q_hat_g = mean_{t:g_t=g} [R_{t+1} + gamma V_hat(S_{t+1})],
P_bar_g = empirical successor-state distribution for visits to g,
r_g     = FP-TU-001 recovery radius for the observed count of g.
```

Writing `e = V_hat - V^pi`, the recovery decomposition is

```text
q_hat_g - Q^pi(g) = zeta_g + gamma P_bar_g^T e,
|zeta_g| <= r_g.
```

For actions `a` and `b` in the same state,

```text
| (P_bar_sa - P_bar_sb)^T e |
    <= TV(P_bar_sa, P_bar_sb) span(e)
    <= 2 E_V TV(P_bar_sa, P_bar_sb).
```

The mandatory exact-route uncertainty and lower bound are therefore

```text
U_exact(s,a,b)
  = r_sa + r_sb
    + 2 gamma E_V TV(P_bar_sa, P_bar_sb),

LCB_exact(s,a,b)
  = q_hat(s,a) - q_hat(s,b) - U_exact(s,a,b).
```

The bound requires full state support for `E_V` and positive counts only for
the two compared actions. A missing unrelated action never blocks it. The
total-variation factor is observable, lies in `[0,1]`, and cancels common value
offsets without accessing the true transition kernel.

## 6. Finite-softmax V-first action-gap bound

For a pair query `g`, let `kappa_g` be its observed within-group softmax mass
and let `P_bar_g^beta` be the normalized effective successor distribution
induced by the verified one-hot softmax weights. Both are computed from the
trajectory and public `beta`.

The fixed-`V^pi` part splits into an on-group empirical mean and off-group
contamination. Since every recovery target lies in `[-B,B]`, define

```text
C_g = kappa_g r_g + 2 B (1-kappa_g).
```

If the verified finite-softmax state route emits `E_V^beta`, the mandatory
finite-softmax uncertainty and lower bound are

```text
U_soft(s,a,b)
  = C_sa + C_sb
    + 2 gamma E_V^beta
        TV(P_bar_sa^beta, P_bar_sb^beta),

LCB_soft(s,a,b)
  = q_hat_beta(s,a) - q_hat_beta(s,b) - U_soft(s,a,b).
```

The proof must derive the effective rows from the exact normalized attention
weights and verify that each row is finite, nonnegative, and sums to one. It
must not replace `kappa_g` by an oracle kernel diagonal or use an unobserved
fixed-target variance.

## 7. Frozen global controls and dominance

For any route with an emitted complete-Q certificate `E_Q`, define

```text
LCB_global(s,a,b)
  = q_hat(s,a) - q_hat(s,b) - 2 E_Q.
```

The controls are:

- V-first no-split exact global;
- V-first no-split softmax global;
- Direct-Q exact global;
- Direct-Q softmax global.

For identical V-first estimates and candidate pairs, the new local penalty
must be proved no larger than the corresponding global penalty. In the exact
route this follows from `TV <= 1`, `r_g <= max_h r_h`, and the existing
composition `E_Q = gamma E_V + max_h r_h`. The finite-softmax proof must also
map `kappa_g` to the verified global leakage term. Exhaustive implementation
tests must reject any record that violates this deterministic dominance when
the global control exists.

## 8. Candidate selection and policy update

Each route uses the same deterministic rule independently. For state `s`:

1. Select receiver `a_star(s) = argmax_a q_hat(s,a)`, breaking ties by the
   smallest action index.
2. A donor `b != a_star` is eligible only when its pair count is positive,
   `LCB(s,a_star,b) > 0`, and `pi(b|s) > pi_min`.
3. Freeze `theta = 1/2` before evaluation and transfer

   ```text
   eta(s,b) = theta [pi(b|s)-pi_min]
   ```

   from every eligible donor to `a_star`.
4. If no donor is eligible in any state, return the original policy and mark
   the record as an abstention, not as a failed certificate.

The update preserves row sums, nonnegativity, and `pi_plus(a|s) >= pi_min`.
On the simultaneous event,

```text
(T_{pi_plus} V^pi)(s) - V^pi(s)
  = sum_{eligible b} eta(s,b) Delta_pi(s,a_star,b)
  >= sum_{eligible b} eta(s,b) LCB(s,a_star,b)
  >= 0.
```

Monotonicity and contraction of `T_{pi_plus}` then imply
`V^{pi_plus} >= V^pi` componentwise. The theorem guarantees non-degradation;
strict improvement is an oracle-described empirical outcome, not a mandatory
claim.

This pointwise Bellman condition is deliberately stronger and simpler than an
occupancy-based conservative-policy-iteration lower bound. It is consistent
with the safe-policy-iteration observation that a nonnegative statewise
advantage yields a non-worse interpolated policy.

## 9. Module boundaries and data flow

Create four isolated Python entry points:

- `action_gap_certificate.py`: pure validation, exact and softmax gap bounds,
  deterministic candidate selection, and the one-step update;
- `verify_action_gap_certificate.py`: formula, adversarial fixture, failure,
  policy-simplex, and theorem checks;
- `evaluate_action_gap_certificates.py`: frozen trajectory reconstruction,
  route estimates, certificate composition, and separate oracle audit;
- `analyze_action_gap_certificates.py`: strict regression, schema, dominance,
  emission, usefulness, and oracle analysis.

The machine data flow is

```text
observed trajectory + unchanged fixed-policy estimators
    -> unchanged FP-TU-001 simultaneous certificate
    -> route Q estimates + observed successor histograms
    -> pairwise LCBs for one selected receiver per state
    -> deterministic probability transfers
    -> action_gap_certificate namespace
    -> structurally separate oracle_audit
```

The pure module may receive counts, observed transition histograms, the current
policy, route estimates, the existing certificate, `pi_min`, and public
hyperparameters. It must reject any true kernel, occupancy, Q value, V value,
action gap, route error, exact return, or realized Bellman improvement as an
input.

One narrowly additive, behavior-preserving helper may be added to the existing
fixed-policy evaluator if needed to expose route estimates. Legacy serialized
outputs must remain byte-compatible.

## 10. Validation and deterministic abstention

Validation covers original-integer counts; exact shape agreement; state/pair
aggregation; nonnegative transition counts; successor-row sums; finite route
estimates; policy-simplex rows; the declared exploration floor; exact beta and
attention normalization; unchanged task identity; and strict JSON without
duplicate keys, NaN, or Infinity.

Ordered pair or update reasons are:

1. `algorithm_mode_mismatch`;
2. `divergence_guard_triggered`;
3. `state_certificate_not_emitted`;
4. `candidate_pair_unvisited`;
5. `recovery_radius_unavailable`;
6. `attention_mass_invalid`;
7. `effective_transition_row_invalid`;
8. `numerical_nonfinite`;
9. `gap_lcb_nonpositive`;
10. `no_transferable_mass`.

Reasons 9 and 10 are ordinary abstentions. A failure for one state or pair may
not invalidate a valid comparison elsewhere. Numerical or input failure never
falls back silently to an oracle value or a different certificate.

## 11. Frozen paired evaluation

The formal matrix exactly reuses the deterministic `FP-TU-001` protocol:

- 30 tasks per cell;
- 6 states and 4 actions;
- trajectory lengths 256, 1024, 4096, and 16384;
- mixing 0.08 and 0.5;
- reward-gap bonuses 0 and 0.5;
- `pi_min = 0.05`, `beta = 8`, `gamma = 0.70`, `alpha = 0.65`;
- 160 evaluation iterations;
- `delta = 0.05`, seed 20260829;
- exactly 480 matched records.

The read-only GPT baseline is
`results/FP-TU-001/codex/`, with SHA-256 hashes:

- `config.json`: `43dcb96b0f6f95e76f1c0b484d6375e3727dbb5609b16a8952a69e2ac0dddf3a`;
- `task_results.json`: `0e5eab39bf49894832c5ebcdd6f7b70c453fff6b9600b234617889f8f9fa79be`;
- `summary.json`: `565fc4d261a938d13350984bb97242e79fba42f014d11f807a18942517f4444f`.

Each record reports, for all primary and control routes:

- receiver, candidate donors, estimates, uncertainties, LCBs, and reasons;
- observed total-variation factors and softmax masses;
- changed states, transferred probability mass, and updated policy;
- local-versus-global penalty ratios and dominance status;
- update emission and abstention status;
- exact pair-order, Bellman-improvement, value-improvement, and return-change
  diagnostics only in `oracle_audit`.

Primary empirical metrics are certified-update rate, certified-state rate,
certified-donor count, and transferred mass. Secondary descriptive metrics are
exact return change, false-order count, nonmonotone-update count, local/global
penalty reduction, and exact/softmax agreement. No empirical metric is theorem
evidence.

## 12. Verification design

The deterministic verifier includes:

1. algebraic fixtures for the exact recovery decomposition;
2. exhaustive finite-distribution checks of
   `|(p-q)^T e| <= TV(p,q) span(e)`;
3. common-offset fixtures where the local propagation term is exactly zero;
4. disjoint-row fixtures where the local term equals the global worst case;
5. finite-softmax weight, diagonal-mass, effective-row, and contamination
   decomposition checks against direct matrix calculations;
6. selected-pair validity under simultaneous post-data selection;
7. exact and softmax local/global dominance checks;
8. partial-support fixtures proving that unrelated missing pairs do not block
   a valid update;
9. policy row-sum, nonnegativity, exploration-floor, tie-break, and
   `theta=1/2` checks;
10. direct Bellman-operator fixtures establishing the one-step pointwise
    policy-improvement implication;
11. false positive, zero-gap, missing-radius, mode, divergence, malformed-row,
    and nonfinite counterexamples;
12. strict JSON, namespace, no-oracle, baseline-integrity, and unchanged legacy
    verifier checks.

Both independent routes must run a labelled smoke matrix before either formal
run. Formal constants cannot be revised after smoke.

## 13. Acceptance and falsification

The mandatory construction passes only if:

1. the exact and finite-softmax pairwise bounds are proved from the frozen
   `FP-TU-001` event without new risk spending or oracle inputs;
2. data-dependent receiver and donor selection is covered by the simultaneous
   event and the selective probability statement is stated exactly;
3. every emitted transfer has strictly positive LCB and the deterministic
   Bellman improvement lower bound is nonnegative in every state;
4. the policy update preserves the simplex and exploration floor exactly;
5. every local V-first penalty is no larger than its available global V-first
   control for the same estimate and candidate pair;
6. partial pair support behaves locally and an unrelated missing pair never
   suppresses a valid comparison;
7. the 480 records, task identities, estimator metrics, FP-TU certificate
   leaves, and all other legacy outputs match the frozen baseline under exact
   or `1e-12` numeric tolerance as appropriate;
8. all inherited and new verifiers, task-scoped Ruff, strict JSON, schema,
   provenance, and baseline-integrity checks pass;
9. every oracle violation is enumerated and excluded from certificate inputs
   and theorem evidence;
10. both independent routes are reproducible, reciprocal verification reports
    end `PASS`, and the final synthesis reports positive and negative outcomes
    without tuning.

The main falsifiable empirical hypothesis is that the local V-first routes emit
at least one safe probability transfer in the frozen 480 records and weakly
dominate the corresponding global V-first update decisions record by record.
Zero useful updates is a valid scientific negative outcome if the proof,
implementation, preservation checks, and reciprocal verification pass.

Any local/global dominance failure, positive-LCB oracle reversal, pointwise
value decrease, legacy mismatch, hidden risk resplit, or prohibited input is an
implementation or theory failure, not permission to change the contract.

## 14. Governance, isolation, and artifacts

The task is long and conclusion-critical.

- Scientific and Git baseline: `c579047` on `main`.
- GPT branch: `codex/FP-ADV-001`.
- Claude branch: `claude/FP-ADV-001`.
- GPT results: `results/FP-ADV-001/codex/`.
- Claude results: `results/FP-ADV-001/claude/`.

Expected shared artifacts are the four Python entry points, the formal task,
implementation plan, consolidated theory, consolidated report, and compact
`ACTIVE_WORKSPACE.md` update. Each route additionally owns `theory.md`,
`first_result.md`, `formal_result.md`, and a verification report under
`docs/research_branches/FP-ADV-001/<actor>/`.

GPT will author the formal task and plan. Claude performs the required
read-only pre-review before activation. After approval, both routes start from
one frozen activation commit, independently construct and seal their first
results before disclosure, then reproduce and verify the other route. No merge
to `main`, push, publication, external message, unsafe permission bypass, or
new fee category is authorized by this design.

## 15. Evidence and stopping rules

Each route records exact commits, environment, commands, raw-output hashes,
successful and failed runs, anomalies, metrics, limitations, and a numbered
acceptance assessment. Stop affected work if the pre-review returns
`OBJECTION`, the baseline hash or record count differs, route isolation cannot
be preserved, another edit overlaps an allowed path, a prohibited input is
needed, the softmax decomposition cannot be proved, a frozen parameter would
need changing, or external authority or cost would expand.

The untracked `docs/实验学习记录_恢复版.md` is user-owned, outside this task,
and must not be modified or included in a commit.

## 16. Primary sources

- Sham Kakade and John Langford, "Approximately Optimal Approximate
  Reinforcement Learning," ICML 2002. The paper introduces conservative policy
  iteration and policy interpolation:
  https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf
- Matteo Pirotta, Marcello Restelli, Alessio Pecorino, and Daniele
  Calandriello, "Safe Policy Iteration," ICML 2013. The paper develops lower
  bounds for safe policy improvement and records the statewise nonnegative
  advantage condition used by this design:
  https://proceedings.mlr.press/v28/pirotta13.html
