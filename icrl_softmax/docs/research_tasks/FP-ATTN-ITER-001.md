# FP-ATTN-ITER-001: Three certified iteration steps inside the literal attention network

## Task metadata

- Created: 2026-09-11.
- Author: Claude, under the direct user instruction of 2026-09-11 ("好的B").
- Status: `ACTIVE`.
- Task version: `1.0`.
- Scientific baseline: `eb9b03c130d9e5b03c20507bc06ce4bcb605df57`
  (`main` after FP-ITER3-001).
- Execution branch: `main` (single actor; see the user ruling below).
- Result directory: `results/FP-ATTN-ITER-001/claude/`.
- Design:
  `docs/superpowers/specs/2026-09-11-literal-attention-iteration-design.md`.
- Plan:
  `docs/superpowers/plans/2026-09-11-literal-attention-iteration-plan.md`.
- Classification: long, conclusion-critical, single-actor under the standing
  user exception.

## Why this task exists

The project's central claim is about a fixed-weight softmax **attention
network**, yet the certified iteration has only ever run in numpy:

| result | what was verified | where |
|---|---|---|
| `FP-ATTN-001` | literal network reproduces **one** certified step | literal network |
| `FP-ITER2-001` | a **second** certified step exists | numpy only |
| `FP-ITER3-001` | a **third** certified step exists | numpy only |

So the inference chain still has an unexecuted step:

```text
literal network == numpy formulas   (fixtures, then one step at scale)
literal network -> one certified step             (FP-ATTN-001)
literal network -> three certified steps          (NEVER EXECUTED)
```

This task executes it. It is the same class of gap `FP-ATTN-001` closed for a
single step, now applied to the iterated object.

## Research question

Can the literal softmax attention networks carry the three-step certified
iteration — using the network's own `Qhat` at every step to drive the frozen
certificate and decision rule — and do they reach the same decisions as the
numpy route?

## Falsifiable hypotheses

1. `H1 (step-1 agreement with FP-ATTN-001)`: at step 1 the literal network
   reproduces the `FP-ATTN-001` result, i.e. its `Qhat` gap against numpy is
   within the frozen `ATOL` and its decision, `eta` and `E_Q` match. This
   anchors the new run to the already-verified single-step result.
2. `H2 (network drives the iteration)`: at every step `k`, the `Qhat` the
   certificate is scored on is the one the **literal network** produced, not a
   numpy substitute. Verified executably by recomputing the certificate from
   the network's tensor output and by a provenance flag on every step.
3. `H3 (three steps are certifiable in the network)`: at least one route-record
   emits three certified steps driven by network `Qhat`.
4. `H4 (decisions agree with numpy)`: for every route-record and every step,
   the literal-network decision equals the `FP-ITER3-001` numpy decision, and
   where both emit the selected `eta` also agrees.
5. `H5 (flips are reported, never absorbed)`: every disagreement, whatever its
   cause, is listed individually with its step, its `E_Q`, its smallest
   per-state lower bound, the numpy and literal values, and the margin to the
   decision boundary. No disagreement may be summarised away or described as
   "numerically equivalent" without the evidence.
6. `H6 (validity under the network)`: every step the network certifies is
   componentwise non-degrading in the oracle audit, and every network `E_Q`
   bounds the realized oracle error at that step.
7. `H7 (nothing sealed changes)`: every file sealed by a closed task remains
   byte-identical.

`H1`, `H2`, `H6`, `H7` are mandatory. `H3`--`H5` are the substantive results.
`H4` failing is a valid and valuable finding — it would mean `float32` network
arithmetic changes certified decisions under iteration, which no existing result
establishes either way.

## Frozen contract

### Protocol, inherited verbatim

`4` states, `3` actions, `pi_min = 0.15`, gap bonus `0.5`, mixing
`{0.08, 0.5}`, `12` tasks per mixing, training trajectory `65536` transitions,
certification `16384 x 64 = 1048576` items, `gamma = 0.70`, `alpha = 0.65`,
`160` layers, `R_star = 1.5`, `delta = 0.05`, seed `20260911`, eta grid
`1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01`, `MAX_STEPS = 3`.

### The one structural change: the network produces Qhat

At every step, in the otherwise unchanged `FP-ITER3-001` iteration:

```text
pi_0 = the frozen target policy
for k = 1, 2, 3:
    Q_k    = LITERAL NETWORK(training batch, pi_{k-1})     <-- changed
    cert_k = variance-adaptive certificate(Q_k, pi_{k-1}, cert batch)
    decide = relative-softmax rule(pi_{k-1}, Q_k, cert_k)
    if decide abstains: stop, record the frozen reason
    pi_k   = decide.policy_plus
```

Route mapping, inherited from `FP-ATTN-001`:
`expected_exact` → `EndToEndMaskedSoftmaxExpectedSARSA`;
`expected_finite` → `EndToEndFiniteSoftmaxExpectedSARSA`.

Unchanged: the identical training and certification batches at every step, the
same frozen certificate and decision code, `pi_{k-1}` as the target at step `k`,
the oracle-audit structure with each step's realized error measured against
`Q^{pi_{k-1}}`, and every ordered abstention reason.

The literal networks run in `float32` under `torch.no_grad()`; the numpy
comparator is `FP-ITER3-001`'s sealed three-step bundle. `ATOL = 1e-4` on
`Qhat`, as frozen by `FP-ATTN-001`.

### Why agreement is not assumed

`FP-ATTN-001` measured a terminal `Qhat` gap of `1.076e-05` after one step of
`160` layers. Iteration feeds each step's output into the next, so the gap may
accumulate, and `FP-ITER3-001`'s third-step emissions include margins of order
`0.378`. A decision flip is therefore a realistic outcome, not a defect, and
`H5` exists to make it visible rather than absorbed.

## Prohibited work

- No modification of any sealed program, sealed result bundle, or closed task
  record.
- No change to any frozen formula, constant, tolerance, eta grid, matrix,
  hypothesis, or metric after the run.
- No numpy substitution anywhere the network is supposed to produce `Qhat`.
- No resampling between steps.
- No retuning to force agreement or to force an emission.
- No suppression or averaging of a disagreement.
- No claim of independent or reciprocal verification.

## Acceptance criteria

1. `H1` anchors step 1 to the `FP-ATTN-001` result with the frozen tolerance.
2. Every step records which producer generated its `Qhat`, and the verifier
   confirms it was the network.
3. `H3` is evaluated and the number of three-step route-records reported.
4. `H4` is evaluated per route-record per step, with the numpy and literal
   decisions both reported.
5. `H5` lists every disagreement individually with `E_Q`, smallest lower bound
   and boundary margin, or states that there are none.
6. Every emitted step passes componentwise non-degradation and every network
   `E_Q` bounds the realized oracle error at that step.
7. Every non-emitting step carries its frozen ordered abstention reason.
8. Exact truth confined to `oracle_audit`.
9. Batches identical across all three steps, verified by digest equality.
10. All strict-JSON, duplicate-key, finite, shape, seed and location checks pass.
11. Complete reproducibility evidence including `torch` version and dtype.
12. Same-actor derived verification recorded, with the limitation stated.
13. `ACTIVE_WORKSPACE.md` is current.

## Failure criteria

The construction fails if the network cannot run the frozen dimensions, if any
step's `Qhat` came from numpy rather than the network, if a disagreement is
suppressed, if batches differ between steps, or if the certificate code differs
from the frozen one.

`H4` failing is **not** a construction failure. It would establish that
`float32` network arithmetic changes certified decisions under iteration, which
is a substantive finding about the mechanism and would be the task's headline
result.

## Stopping conditions

Stop affected work and notify the user if:

- the network cannot complete three steps at the frozen dimensions within budget;
- a sealed file is found modified;
- the comparison would require changing a frozen tolerance or parameter;
- execution would expand cost, publication, external communication, or
  permissions beyond authorization.

## Route assignment and verification

Single actor: Claude executes and verifies, under the standing user instruction
of 2026-09-11. The verification is not independent and must be labelled as such,
recompute the comparison through a separate code path, replay the sealed
programs, and state the limitation.

## Pre-review

- Status: `APPROVED` (2026-09-11), same-actor.
- Evidence: `docs/research_branches/FP-ATTN-ITER-001/claude/pre_review.md`.

## Objections and user rulings

### Objection

- Status: `NONE`.

### User ruling (scope)

- Date: 2026-09-11.
- Decision: "好的B" — proceed with moving the three-step certified iteration
  into the literal attention network.
- Scope: this task only. Codex is out of scope by the user's instruction
  ("codex用不了不要管他"); independent verification remains outstanding and is
  not attempted here.

## Definition of done

- [x] Pre-review recorded with no unresolved objection.
- [x] Step 1 anchored to `FP-ATTN-001` (max step-1 gap `1.076e-05`).
- [x] `H1`--`H7` each evaluated with evidence; all **PASS**.
- [x] Every disagreement listed individually, or none stated (`0` exist; the
      enumeration path is exercised and empty).
- [x] Same-actor derived verification recorded, with the limitation stated.
- [ ] `ACTIVE_WORKSPACE.md` is current.

## Formal outcome (2026-09-11)

Recorded here for the task index; the route journal at
`docs/research_branches/FP-ATTN-ITER-001/claude/first_result.md` holds the full
evidence.

- `24` records, `48` route-records, `90` network step executions; the literal
  network produced the `Qhat` at **every** step (`90/90` provenance, no numpy
  substitution).
- Emissions by step: **`[22, 20, 15]`**, identical to the `FP-ITER3-001` numpy
  baseline; **`15`** network-driven three-step routes.
- **Decision agreement `90/90`; `0` decision flips; `0` selected-`eta` flips.**
- **`0` certificate violations and `0` non-degrading violations**; every emitted
  step at every level has a strictly positive total gain.
- Max `|Q_network - Q_numpy|_inf` `1.076e-05` against the frozen `ATOL` `1e-04`.
- **The `float32` difference does not accumulate**: per-step maxima are
  `1.076e-05`, `4.585e-06`, `7.176e-06`, essentially flat rather than growing.
  This is why no decision flips occur. The flatness is reported as a measurement;
  the contraction explanation offered in the journal is not claimed as a theorem.
- Same-actor derived verification: **PASS**.

The project's central claim now holds end to end: the literal attention network
produces the certified improvement, continues it for three steps, and reaches
exactly the decisions the numpy route reaches.
