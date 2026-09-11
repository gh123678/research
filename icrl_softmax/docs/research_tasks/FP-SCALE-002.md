# FP-SCALE-002: A variance-adaptive residual certificate for one certified policy improvement

## Task metadata

- Created: 2026-09-11.
- Author: Claude, under the direct user rulings of 2026-09-11 (single-actor
  execution and verification; user selected the variance-adaptive certificate
  direction).
- Status: `ACTIVE`.
- Task version: `1.0`.
- Predecessors: `FP-SCALE-001` (`ACTIVE`, stopped at the smoke gate with the
  scale finding recorded), `FP-ESARSA-001` (`VERIFIED` by user exemption),
  `FP-ADV-001` (`VERIFIED`, negative), `FP-TU-001` (`VERIFIED`, certificate
  baseline), `FP-KERN-001` / `FP-KERN-002` (`VERIFIED`, negative).
- Scientific baseline: `467ebf591dde8421f7ed5324a18514fb5935d2f9` (`main`
  after the FP-KERN merges).
- Execution branch: `claude/FP-SCALE-002`.
- Result directory: `results/FP-SCALE-002/claude/`.
- Design:
  `docs/superpowers/specs/2026-09-11-variance-adaptive-certificate-design.md`.
- Plan:
  `docs/superpowers/plans/2026-09-11-variance-adaptive-certificate-plan.md`.
- Classification: long, conclusion-critical, single-actor under the user
  exception.

## Why this task exists

`FP-ADV-001` emitted `0/480` safe updates and `FP-ESARSA-001` emitted `0/480`.
`FP-SCALE-001` then repaired the certificate *scale* — a dedicated
certification batch rather than a split trajectory — and improved the
certificate by more than an order of magnitude (`E_Q` from a `22.47` floor to
`0.98`; worst-pair certification count from `14` to `40000`) **and still
emitted nothing**. That run established, quantitatively, that:

1. emission requires `E_Q < sigma`, the within-state action-relevant value
   spread, because the policy-improvement signal is second order in the tilt
   while the error term is first order;
2. `E_Q` and `sigma` both scale linearly with the reward bound, so widening the
   value spectrum cannot change their ratio (measured: spreads grew `6x` while
   `E_Q` grew `170x`);
3. the obstruction is therefore the **worst-case residual envelope `2B`**,
   which the verified cosh-mixture argument cannot avoid because it requires a
   *known* sub-Gaussian parameter.

This task replaces that one ingredient. It is the only lever measurement
identified as effective.

### Prototype evidence

A design probe (`results/FP-SCALE-001/claude/evidence/variance_adaptive/`) ran
the whole planned matrix twice before this task was written:

| certificate | emitted | non-degrading | strict | violations | `E_Q` |
|---|---|---|---|---|---|
| asserted-scale two-half | 31/48 | 31/31 | 31/31 | 0 | 0.064--0.188 |
| **fully rigorous two-half** | **22/48** | **22/22** | **22/22** | **0** | 0.164--0.412 |

The second row uses the construction frozen below, with every constant an
explicit named inequality and no free scaling factor. The prototype is
motivation, not evidence: it has no sealed verifier and is not independently
reconstructed.

## Research question

Can the fixed-policy Expected SARSA construction emit a certified,
componentwise non-degrading relative-softmax policy update at a reachable
certificate scale, when the residual certificate's sub-Gaussian parameter is
*estimated from held-out data under explicit concentration constants* instead
of the worst-case envelope `2B`?

## Falsifiable hypotheses

1. `H1 (construction unchanged)`: the exact grouped and finite-logit routes
   reproduce batch Expected SARSA within `1e-12`, and the certificate, decision
   rule, abstention reasons, and oracle separation are unchanged from
   `FP-ESARSA-001` except for the concentration argument.
2. `H2 (certificate validity)`: every emitted `E_Q` bounds the realized oracle
   `||Qhat - Q^pi||_infinity` in every audited record; zero certificate
   violations across the frozen matrix.
3. `H3 (emission)`: at least one primary route emits a certified non-degrading
   update in at least `25%` of the frozen route-records, i.e. at least `12` of
   `48`, matching the rigorous prototype's `22/48` within the stated
   expectation.
4. `H4 (real improvement)`: every emitted update is componentwise
   non-degrading, and at least one has strictly positive total value gain
   `sum_s (V^{pi_plus}(s) - V^pi(s)) > 0` in the oracle audit.
5. `H5 (attribution)`: the certified error at the same certification counts is
   smaller under the variance-adaptive radius than under the frozen `2B`
   envelope certificate, and the difference is attributable to the
   concentration argument alone. A `2B` control route is required in the same
   records.
6. `H6 (no free constant)`: the certificate contains no fitted, asserted, or
   tuned scaling factor; every constant traces to a named inequality and is
   reconstructible from the frozen task text.

`H1`, `H2`, and `H6` are mandatory. `H3`--`H5` are the usefulness and
attribution claims. Zero emissions is a valid negative result and forbids
retuning.

## Frozen mathematical contract

### Inherited unchanged

Everything in `FP-ESARSA-001` and `FP-SCALE-001`: the pair MRP and
`T_pi^X`/`Q^pi`; the canonical one-token-per-pair memory; the exact grouped
update `Q_{l+1}(x) = Q_l(x) + (alpha/N_x^train) sum_{t:X_t=x} delta_t^l`; the
finite-logit scores with `zeta = xi = tau = 8`; the held-out residual
`Y_t(Qhat) = R_{t+1} + gamma sum_b pi(b|S_{t+1}) Qhat(S_{t+1},b) - Qhat(S_t,A_t)`;
the relative-softmax candidate, the lower bound
`LB_s(eta) = Ihat_s(eta) - E_Q ||pi_eta^+(.|s) - pi(.|s)||_1`, its strict
`min_s LB_s > 0` passage rule, the descending eta grid, and the ordered
non-emission reasons.

The `FP-SCALE-001` protocol is inherited verbatim: `4` states, `3` actions,
`pi_min = 0.15`, gap bonus `0.5`, mixing `{0.08, 0.5}`, `12` tasks per mixing,
training trajectory `65536`, certification chains `16384 x 64` = `1048576`
items, `gamma = 0.70`, `alpha = 0.65`, `160` layers, `R_star = 1.5`,
`delta = 0.05`, `Q_0 = 0`, seed `20260911`, and the extended eta grid
`1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01`.

### The one changed ingredient: the concentration argument

Let `X = S x A` with `d = |X|`, and let `B = R_star/(1-gamma)` so every
residual lies in `[-2B, 2B]`. For each pair `x`, split its certification items
into two disjoint halves `A_x` and `B_x`, with sizes `N_A` and `N_B`, and let
`Ybar_A`, `Ybar_B` be the residual sample means.

Two explicit inequalities, with the risk split
`delta_A = delta_B = delta / (2d)` and a union bound over the `d` pairs:

**Second-moment bound on half A.** Because the residuals lie in `[-2B, 2B]`,
`Z_i = Y_i^2` lies in `[0, (2B)^2]`. Hoeffding's inequality applied to the
`Z_i` gives, with probability at least `1 - delta_A`,

```text
E[Y^2 | x] <= mean_A(Y^2) + 2 B^2 sqrt(2 log(1/delta_A) / N_A)  =: V_x,
```

using `E[Z] = E[Y^2] = Var(Y) + E[Y]^2 >= Var(Y)`. The bound is on the *second
moment*, not the variance, which is conservative and keeps it valid without
assuming the residual mean vanishes.

**Mean bound on half B.** With `s_x = sqrt(V_x) >= sqrt(Var(Y))`, Hoeffding's
lemma applied to the bounded residuals reveals the sub-Gaussian parameter
`Var(Y)/N_B <= s_x^2 / N_B`, hence with probability at least `1 - delta_B`,

```text
|Ybar_B - (T_pi^X Qhat - Qhat)(x)| <= r_x := 2 s_x sqrt(log(1/delta_B)/N_B).
```

**Simultaneous certificate.** On the intersection event, with probability at
least `1 - delta`,

```text
epsilon_res = max_x ( |Ybar_x| + r_x ),      E_Q = epsilon_res / (1 - gamma),
||Qhat - Q^pi||_infinity <= E_Q.
```

No scaling factor, no hyperparameter, and no fitted constant appears: `2 B^2`
and `2` are the literal Hoeffding constants, and `delta` is split
deterministically.

### Certification-count rule

The construction replaces `N_A` and `N_B` by their realized sizes, which is
valid because the bound holds conditionally on any fixed split sizes and the
halves are a deterministic function of the items. A record is certified only
when every pair has at least `MIN_HALF_COUNT = 5000` items in each half, i.e.
at least `10000` certification items per pair; failing records are reported
with `heldout_pair_support_missing` and excluded from `H3`.

## Routes

1. `variance_adaptive_exact`: exact grouped Expected SARSA with the
   variance-adaptive certificate, primary.
2. `variance_adaptive_finite`: finite-logit mask-free Expected SARSA with the
   variance-adaptive certificate, primary.
3. `envelope_control_exact`: the identical exact route scored with the frozen
   `2B` envelope certificate at the same counts, control for `H5`.

All three share the same tasks, training batch, certification batch, and
decision rule. The control has no claim of validity superiority; it exists only
to attribute the difference.

## Prohibited work

- No post-hoc change to any frozen formula, constant, risk split, eta rule,
  split, seed, matrix, hypothesis, or metric after activation.
- No modification of any file sealed by a closed task; the inherited programs
  are reused by import and must stay byte-identical.
- No fitted, asserted, or tuned scaling factor anywhere in the certificate, and
  no reuse of a certification item across the two halves.
- No input-dependent equality mask or visited gate in the finite route.
- No learned transition model, oracle certificate input, or leak of
  certification data into `Qhat` or hyperparameter selection.
- No repeated policy iteration, conditional-on-emission guarantee, or output
  exploration-floor claim.
- No write to `main` during execution, no push, publication, external message,
  permission bypass, or new fee category without separate authorization.
- No claim of reciprocal verification: this task has a single actor by user
  ruling, and every report must say so.

## Acceptance criteria

1. The inherited construction claims verify at `1e-12` on reduced-dimension
   fixtures, including self-loops, repeated visits, unvisited queries, and the
   absence of masks and gates in the finite route.
2. The two halves per pair are disjoint, exhaustive, and deterministic; a
   fixture proves no item is used twice and that the scale estimate never sees
   the half it certifies.
3. Every constant in the certificate is reconstructible from the task text and
   traces to a named inequality; an executable check recomputes each one.
4. Every emitted `E_Q` bounds the realized oracle error in every audited
   record; zero certificate violations.
5. `H3` is evaluated on the frozen matrix with no retuning; the emitted count
   and the selected eta values are reported exactly.
6. Every emitted policy is finite, strictly positive, row-normalized, changed,
   and componentwise non-degrading; at least one is a strict improvement.
7. Every abstention returns the input policy bit-for-bit with a frozen reason,
   and every record failing the count rule is reported and excluded.
8. Training and certification batches are separately seeded and structurally
   separated; a leak counterexample fixture is required.
9. Exact truth, realized errors, action gaps, improvements, values, and returns
   live only in `oracle_audit`.
10. The `2B` control route runs on the identical records and its certified
    errors are reported alongside, so `H5` is attributable.
11. All strict-JSON, duplicate-key, nonfinite, shape, range, identity, seed,
    configuration, and result-location checks pass, and the inherited
    `FP-ESARSA-001` programs are byte-identical to their sealed versions.
12. The formal result contains exactly `24` matched records, each with all
    three routes, reproducing the frozen generator identities and seed
    schedule.
13. The compute budget is measured in smoke and the formal run is executed
    exactly once.
14. Complete reproducibility evidence is recorded.
15. Verification is performed by the same actor under the recorded user
    exception and must be executable and derived: independent reconstruction of
    the frozen inputs, a replay of the sealed programs, and a from-scratch
    recomputation of every reported metric, with the limitation stated.
16. `ACTIVE_WORKSPACE.md` is current; `main` is unchanged without separate user
    approval.

## Failure criteria

The mandatory theory fails if the second-moment bound is not valid for the
specified range, the mean bound does not follow from Hoeffding's lemma at the
stated scale, the union over `d` pairs is not simultaneous at risk `delta`, the
Bellman residual does not imply the stated Q bound, the relative-softmax lower
bound is false, or an unstated model or oracle input is required. The
construction fails if an item is shared between halves, a scaling factor is
asserted rather than derived, the finite route uses hidden masks or gates,
certification data leaks into `Qhat`, or an emitted update can degrade any
state on the certificate event.

`H3`--`H5` may fail without invalidating the task. If they fail while `H1`,
`H2`, and `H6` pass, that is a verified negative result under a provably valid
variance-adaptive certificate, which would falsify the concentration argument
as the obstruction and move the within-state value spectrum to the leading
suspect.

## Stopping conditions

Stop affected work and notify the user if:

- the second-moment bound or the Hoeffding constant cannot be stated exactly;
- more than half the records fail the count rule, which would indicate a
  mis-specified protocol rather than ordinary occupancy noise;
- the smoke measurement projects wall time or memory beyond budget;
- implementation needs a prohibited mask, gate, oracle, leak, or fitted
  constant;
- smoke or formal output would be needed to choose a frozen parameter;
- an existing uncommitted edit overlaps an authorized path;
- execution would expand data transfer, cost, publication, merge, push,
  external communication, or permissions beyond authorization.

## Route assignments and verification

Under the direct user instruction of 2026-09-11, Claude executes and Claude
verifies this task on `claude/FP-SCALE-002`. The verification is not an
independent construction and must be labelled as such everywhere. It must
reconstruct every frozen input from source, replay the sealed programs, and
recompute every reported metric through an independent code path.

## Pre-review

- Status: `APPROVED` (2026-09-11), same-actor.
- Evidence: `docs/research_branches/FP-SCALE-002/claude/pre_review.md`.
- Checks: inheritance fidelity; that the only changed ingredient is the
  concentration argument; that every constant traces to a named inequality;
  that the prototype's `SAFETY = 1.1` assertion is gone; that the risk split is
  deterministic; and that no sealed file is modified.

## Objections and user rulings

### Objection

- Status: `NONE`.

### User ruling (single-actor execution and verification)

- Date: 2026-09-11.
- Decision: Codex is not involved and verification is assigned to Claude
  ("不管 codex 了，验证也交给你").
- Scope: this task only. It waives the reciprocal verification of `AGENTS.md`
  section 7, so this task's verification strength is strictly lower than a task
  closed with two independent actors, and that limitation must be stated in
  every report and in `ACTIVE_WORKSPACE.md`.

### User ruling (direction)

- Date: 2026-09-11.
- Decision: the user selected opening a new task for the variance-adaptive
  certificate.
- Scope: direction only. No formula, tolerance, or acceptance strength was
  specified by the user, so the contract above is frozen by this task sheet and
  is subject to objection or revision before execution.

## Definition of done

- [ ] Pre-review recorded with no unresolved objection.
- [ ] Smoke passes and the compute budget is measured.
- [ ] One frozen formal run executed exactly once.
- [ ] Every acceptance criterion has evidence.
- [ ] Same-actor derived verification recorded, with the limitation stated.
- [ ] `ACTIVE_WORKSPACE.md` is current.
- [ ] The user approves any merge to `main`.
