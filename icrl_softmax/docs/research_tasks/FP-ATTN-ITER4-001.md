# FP-ATTN-ITER4-001: Four certified steps inside the literal attention network

## Task metadata

- Created: 2026-09-11.
- Author: Claude, under the direct user instruction of 2026-09-11 ("继续").
- Status: `ACTIVE`.
- Task version: `1.0`.
- Scientific baseline: `428c327df7e4802324e2b30fd55abdbcf318d97a`
  (`main` after FP-ITER4-001).
- Execution branch: `main` (single actor; see the user ruling below).
- Result directory: `results/FP-ATTN-ITER4-001/claude/`.
- Design:
  `docs/superpowers/specs/2026-09-11-literal-attention-four-step-design.md`.
- Plan:
  `docs/superpowers/plans/2026-09-11-literal-attention-four-step-plan.md`.
- Classification: long, conclusion-critical, single-actor under the standing
  user exception.

## Why this task exists

Two verified results currently do not align in horizon:

| result | horizon | code path |
|---|---|---|
| `FP-ITER4-001` | **four** certified steps, `12/48` | numpy |
| `FP-ATTN-ITER-001` | **three** certified steps, `15/48` | literal network |

So the project can say "the network carries the iteration" only up to three
steps, while the deepest certified result rests on numpy. This task closes that
gap by running the network at `MAX_STEPS = 4`.

It is the same class of work as `FP-ATTN-ITER-001`, extended by one step, and it
carries the same central risk with one more fold of composition: `float32`
network arithmetic versus `float64` numpy arithmetic over `160` layers per step,
now fed forward four times instead of three.

## Research question

Can the literal softmax attention networks carry a **four**-step certified
iteration — producing `Qhat` at every step — and do they reach the same
decisions as the `FP-ITER4-001` numpy route?

## Falsifiable hypotheses

1. `H1 (network step-limit monotonicity, mandatory)`: raising the network
   horizon from `3` to `4` does not change network steps 1--3. Their decisions,
   selected `eta`, certified errors and gains must equal the sealed
   `FP-ATTN-ITER-001` values exactly.
2. `H2 (three-step reproduction)`: this run reproduces the sealed
   `FP-ATTN-ITER-001` `15` three-step route-records exactly.
3. `H3 (network-produced Qhat at every step, mandatory)`: at every step of every
   route-record the `Qhat` scored by the certificate came from the literal
   network, verified by the recorded producer field.
4. `H4 (fourth step certifiable in the network)`: at least one route-record
   emits four certified steps driven by network `Qhat`.
5. `H5 (decisions agree with numpy)`: for every route-record and every step, the
   network decision equals the `FP-ITER4-001` numpy decision, and where both
   emit the selected `eta` also agrees.
6. `H6 (no accumulated drift)`: the network-versus-numpy `Qhat` gap at step 4
   stays within the frozen `ATOL = 1e-4`. This is the step where accumulation
   would first show, since `FP-ATTN-ITER-001` measured a flat gap through three
   steps.
7. `H7 (flips reported, never absorbed)`: every disagreement is listed
   individually with its step, `E_Q`, smallest per-state lower bound, both
   decisions and the `Qhat` gap.
8. `H8 (validity under the network)`: every step the network certifies is
   componentwise non-degrading in the oracle audit, and every network `E_Q`
   bounds the realized oracle error at that step.

`H1`, `H3`, `H8` and the corpus-integrity requirement are mandatory. `H4`--`H7`
are the substantive results. `H5` failing is a valid and valuable finding.

## Frozen contract

### Protocol, inherited verbatim

`4` states, `3` actions, `pi_min = 0.15`, gap bonus `0.5`, mixing
`{0.08, 0.5}`, `12` tasks per mixing, training trajectory `65536` transitions,
certification `16384 x 64 = 1048576` items, `gamma = 0.70`, `alpha = 0.65`,
`160` layers, `R_star = 1.5`, `delta = 0.05`, seed `20260911`, eta grid
`1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01`, `ATOL = 1e-4`.

### The one structural change

`MAX_STEPS = 4` instead of `3` in the otherwise unchanged `FP-ATTN-ITER-001`
evaluation, with the comparison baseline switched from the sealed
`FP-ATTN-ITER-001` three-step bundle to the sealed `FP-ITER4-001` four-step
numpy bundle, so that step 4 is compared against the matching numpy horizon.

Unchanged: the literal networks produce `Qhat` at every step; the identical
training and certification batches are reused at every step; the same frozen
certificate and decision code; `pi_{k-1}` as the target at step `k`; the oracle
audit with each step's realized error against `Q^{pi_{k-1}}`; the route→network
mapping; and every ordered abstention reason.

### Corpus-integrity requirement

The horizon knob lives in the **shared** literal evaluator
`evaluate_fp_attn_iter_001.py`, so this task necessarily changes a file that
`FP-ATTN-ITER-001` recorded by hash. Following the precedent set by
`FP-ITER4-001`, the two classes are separated explicitly:

- the **scientific corpus** — `fixed_policy_expected_sarsa.py`,
  `fixed_policy_expected_sarsa_scaled.py`,
  `fixed_policy_variance_certificate.py`, `model.py`,
  `verify_variance_adaptive_certificate.py` — must be byte-identical;
- the **task evaluators** evolve with the horizon.

The change must be paired with an **inertness proof**: re-run the network at the
frozen horizon `3` and show all `90` sealed step entries reproduce exactly. A
committed report must not be left reading `FAIL` because of a legitimate
evolution.

## Prohibited work

- No modification of any sealed program, sealed result bundle, or closed task
  record; the `FP-ITER4-001` and `FP-ATTN-ITER-001` bundles are read-only.
- No change to any frozen formula, constant, tolerance, eta grid, matrix,
  hypothesis, or metric after the run.
- No numpy substitution anywhere the network is supposed to produce `Qhat`.
- No resampling between steps.
- No retuning to force agreement or to force an emission.
- No suppression of a disagreement.
- No new claim of independent or reciprocal verification.

## Acceptance criteria

1. `H1`/`H2` confirm the network horizon change is inert and the sealed result
   reproduces.
2. Every step records its `Qhat` producer and the producer set is exactly the
   literal network.
3. `H4`--`H7` evaluated and reported, with every disagreement listed
   individually or none stated.
4. Every emitted step passes componentwise non-degradation and every network
   `E_Q` bounds the realized oracle error at that step.
5. Every non-emitting step carries its frozen ordered abstention reason.
6. Batches identical across all four steps, verified by digest equality.
7. Exact truth confined to `oracle_audit`.
8. All strict-JSON, duplicate-key, finite, shape, seed and location checks pass.
9. Complete reproducibility evidence including `torch` version and dtype.
10. The shared-evaluator change is paired with an inertness proof, and any
    affected earlier report is re-sealed rather than left failing.
11. Same-actor derived verification recorded.
12. `ACTIVE_WORKSPACE.md` is current.

## Failure criteria

The construction fails if a step's `Qhat` came from numpy, if a disagreement is
suppressed, if batches differ between steps, if the certificate code differs
from the frozen one, or if the shared-evaluator change is not paired with an
inertness proof.

`H4`--`H7` failing is **not** a construction failure. In particular `H5` or `H6`
failing would establish that `float32` network arithmetic changes certified
decisions once the iteration reaches four steps, which no existing result
establishes either way.

## Stopping conditions

Stop affected work and notify the user if:

- the network cannot complete four steps at the frozen dimensions within budget;
- a scientific-corpus file is found modified;
- the comparison would require changing a frozen tolerance or parameter;
- execution would expand cost, publication, external communication, or
  permissions beyond authorization.

## Route assignment and verification

Single actor: Claude executes and verifies. By the user's instruction of
2026-09-11 ("验证先不管"), verification effort is not the focus of this round;
the derived verification is still recorded, and no independent verification is
claimed.

## Pre-review

- Status: `APPROVED` (2026-09-11), same-actor.
- Evidence: `docs/research_branches/FP-ATTN-ITER4-001/claude/pre_review.md`.

## Objections and user rulings

### Objection

- Status: `NONE`.

### User ruling (scope and emphasis)

- Date: 2026-09-11.
- Decision: "继续，验证先不管" — continue with the next research step, and do not
  make verification the focus.
- Scope: this task. Independent verification remains outstanding and is not
  attempted; the same-actor derived verification is recorded for completeness
  only and is not emphasised.

## Definition of done

- [x] Pre-review recorded with no unresolved objection.
- [x] `H1`/`H2` confirm the network horizon change is inert (`90/90` identical).
- [x] `H1`--`H8` each evaluated with evidence; all **PASS**.
- [x] Every disagreement listed individually, or none stated (`0` exist).
- [x] Shared-evaluator change audited and paired with an inertness proof; no
      earlier report affected.
- [ ] `ACTIVE_WORKSPACE.md` is current.

## Formal outcome (2026-09-11)

Recorded here for the task index; the route journal at
`docs/research_branches/FP-ATTN-ITER4-001/claude/first_result.md` holds the full
evidence.

| step | network emissions | numpy emissions | network mean gain | network min gain | max `\|dQ\|` |
|---|---|---|---|---|---|
| 1 | `22` | `22` | `2.7177070297828965` | `0.10419714014227195` | `1.076e-05` |
| 2 | `20` | `20` | `2.2779971698013703` | `0.7799017280406759` | `4.585e-06` |
| 3 | `15` | `15` | `1.286905361647572` | `0.37819650415353206` | `7.176e-06` |
| **4** | **`12`** | **`12`** | **`0.8074994690824351`** | **`0.18490214293638285`** | **`1.044e-05`** |

- Horizon inertness: **all `90` network step-1..3 entries identical** to the
  sealed `FP-ATTN-ITER-001` run, including the recorded `Qhat` gaps; the `15`
  three-step routes reproduce exactly.
- `12` route-records emitted **all four** certified steps; `105` network step
  executions; producer set exactly `{literal_attention_network}`.
- **Decision agreement `105/105`; `0` decision flips; `0` selected-`eta` flips.**
- **`0` certificate violations and `0` non-degrading violations.**
- **No accumulated drift at four steps**: the gap is `1.076e-05`, `4.585e-06`,
  `7.176e-06`, `1.044e-05` — flat, and an order of magnitude inside `ATOL`.
  Observed across four compositions rather than three, which strengthens the
  reading that the `float32` difference acts as per-step rounding noise. The
  task claims the measurement, not the mechanism.
- Network and numpy horizons are now **aligned at four steps**, so the deepest
  certified result no longer depends on which code path is trusted.
- Shared-evaluator audit: **no sealed bundle recorded the changed file**, so no
  earlier report was affected; all prior verifiers still exit `0`.
- Same-actor derived verification recorded (**PASS**), not emphasised per the
  user's instruction of 2026-09-11.
