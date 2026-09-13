# FP-ATTN-8X-001: the literal attention network at 8x certification

## Task metadata

- Created: 2026-09-12.
- Author: Claude, under the direct user instruction of 2026-09-12 ("好的"), closing the
  last thing still on `1x`.
- Status: `ACTIVE`.
- Task version: `1.0`.
- Scientific baseline: `a3e5a2b` (`claude/FP-CENSUS-001`).
- Execution branch: `claude/FP-CENSUS-001`.
- Result directory: `results/FP-ATTN-8X-001/claude/`.
- Classification: long, conclusion-critical, single-actor under the standing user
  exception.

## Why this task exists

The project's claim is about a **fixed-weight softmax attention network**. Everything
since `FP-ATTN-ITER6-001` has been numpy.

| result | path | certification | horizon |
|---|---|---|---|
| `FP-ATTN-ITER6-001` | network | `1x` | `6` |
| `FP-TIGHT-001`, `FP-SAMPLE-001`, `FP-RANGE-001` | numpy | `8x` | step 1 |
| `FP-ITER8X-001` | numpy | `8x` | `12` |
| `FP-HORIZON-001` | numpy | `8x` | `32` |

So the network is four tasks behind. The numpy line now reports a certified iteration
reaching at least `32` steps, with a mean total value gain of `5.65`–`5.80` per
route-record and `42`–`44` of `48` records improved in every state — **none of which
has been demonstrated on the network the claim is about.**

This task brings the network to `8x` and to twelve steps, and compares it against
`FP-ITER8X-001`'s numpy bundle step by step.

## Research question

Does the literal attention network, at `8x` certification and over twelve steps,
reproduce the numpy iteration's decisions and emitting sets exactly — and does the
`float32` gap stay inside the frozen tolerance over a horizon five times longer than
any previously run on the network?

## Falsifiable hypotheses

1. `H1 (provenance, mandatory)`: every step's decision is taken on the literal
   network's `Qhat`, proven structurally at the two driving call sites, and every
   step records `qhat_producer = literal_attention_network`.
2. `H2 (validity and soundness, mandatory)`: every emitted step in both arms is
   componentwise non-degrading and strictly improving; every emitted step's certified
   error bounds the realized error against `Q^{pi_{k-1}}`. Zero violations.
3. `H3 (reach, pre-registered)`: the network reaches twelve steps on both arms, so
   its horizon matches the numpy run it is compared against.
4. `H4 (path agreement, the substantive pre-registered prediction)`: the network's
   emitting set equals numpy's at **every** step `1`–`12`, on both arms, with zero
   decision flips over `12 x 48 = 576` comparisons per arm.
5. `H5 (pre-registered)`: zero `eta` flips against the numpy comparator, at every
   step and both arms.
6. `H6 (drift, pre-registered)`: the network-versus-numpy maximum `|dQ|` stays within
   the frozen `ATOL = 1e-4` at every step, including step `12`. The five-step and
   six-step network runs held `~5e-6` flat; the prediction is that twelve steps at
   `8x` does not change that, which is a real prediction because the horizon is five
   times longer than any the network has run.
7. `H7 (agreement headroom, pre-registered)`: report `min emitted gain / max |dQ|` at
   every step. The prediction is that the headroom stays above `20x` through step
   `12`, which makes agreement a statement about the certificate rather than about
   arithmetic noise.

`H1`, `H2` are mandatory. `H3`--`H7` are pre-registered; each is reported `PASS` or
`FALSIFIED` and never reinterpreted.

### Basis of the predictions, stated explicitly

`H4`, `H5` are **consistency** predictions: the paths agreed on every decision at `1x`
through six steps, `0` flips over `105` comparisons, so agreement at `8x` through
twelve steps is the natural expectation. It is not free: the comparison count rises
roughly fivefold and, critically, the **margins thin as the iteration proceeds**.
`FP-ITER8X-001` recorded numpy minimum gains of `0.000183` at step `11` and
`0.000649` at step `12`, against an observed `float32` gap of `~5e-6`. That is
`30x`–`100x` of headroom — comfortable, but the first time in this line that the
margin has come within two orders of magnitude of the arithmetic noise. Hence `H7`,
which makes the headroom the explicit quantity rather than leaving agreement to be
read as unconditional.

`H6` is a **flatness** prediction, not an extrapolation of growth: the observed gap
sequence over six steps (`1.076e-05, 4.585e-06, 7.176e-06, 1.044e-05, 4.567e-06,
6.814e-06`) is non-monotone and shows no accumulation.

## Frozen contract

### Protocol, inherited verbatim

`4` states, `3` actions, `pi_min = 0.15`, gap bonus `0.5`, mixing `{0.08, 0.5}`,
`12` tasks per mixing, training trajectory `65536` transitions, `gamma = 0.70`,
`alpha = 0.65`, `160` layers, `R_star = 1.5`, `delta = 0.05`, seed `20260911`, eta
grid unchanged, `ATOL = 1e-4`.

### The changes

Two, both already established elsewhere and combined here for the first time on the
network:

1. **certification `8x`** (`131072 x 64 = 8,388,608` items) from the vectorised
   sampler validated in `FP-SAMPLE-001`;
2. **horizon `12`**, matching `FP-ITER8X-001` so the comparison is like for like;
3. **two arms** — `frozen` and `empirical_bernstein` — matching `FP-ITER8X-001`.

The network produces every step's `Qhat`; the numpy route is computed alongside on
the same batches purely as the comparison baseline, and the per-step provenance flag
records which producer was used.

### The comparison reference

`results/FP-ITER8X-001/claude/numpy/task_results.json`, the twelve-step `8x` numpy
run. Every per-step emitting set is compared against it.

## Prohibited work

- No modification of any sealed module, sealed bundle, or closed task record.
- No change to any frozen formula, constant, tolerance, eta grid or hypothesis after
  the run.
- No numpy substitution for any `Qhat` on the network path.
- No re-tuning of the horizon or the multiplier after the result.
- No claim that the vectorised sampler reproduces the sealed batch.
- No reinterpretation of a falsified prediction.
- No `git add -A`.

## Acceptance criteria

1. `H1` proven structurally, not asserted: the call sites whose results drive the
   iteration take the literal network's `Qhat`.
2. `H2` confirmed with per-step evidence; any violation listed individually.
3. `H3`--`H7` each evaluated with evidence.
4. Per-step emitting sets compared against the numpy bundle; every flip listed.
5. Sealed module hashes verified unchanged.
6. All strict-JSON, finite, shape, seed and location checks pass.
7. Complete reproducibility evidence including `torch` version and dtype.
8. Same-actor derived verification recorded.
9. `ACTIVE_WORKSPACE.md` updated.

## Failure criteria

The construction fails if a step's `Qhat` came from numpy on the network path, if a
validity or soundness check fails, if a sealed module changes, or if a flip is
suppressed.

`H3`--`H7` failing is **not** a construction failure. `H4` failing at a late step
would establish that the `float32` gap finally decides a marginal record — which is
the most informative outcome available here and the first such event in the line.

## Stopping conditions

Stop and report if:

- the run cannot complete within budget;
- a validity or soundness check fails;
- a sealed module's hash is found changed;
- execution would expand cost beyond the authorised scope.

## Route assignment and verification

Single actor: Claude executes and verifies. By the user's instruction of
2026-09-11 ("验证先不管"), verification is not the focus; the derived checks are
recorded for completeness and no independent verification is claimed.

## Pre-review

- Status: `APPROVED` (2026-09-12), same-actor.
- Evidence: `docs/research_branches/FP-ATTN-8X-001/claude/pre_review.md`.

## Objections and user rulings

### Objection

- Status: `NONE`.

### User ruling (scope)

- Date: 2026-09-12.
- Decision: "好的" — bring the network path to `8x`.
- Scope: this task, on `claude/FP-CENSUS-001`.

## Definition of done

- [x] `H1` proven structurally.
- [x] `H2` confirmed with per-step evidence.
- [x] `H3`--`H7` each evaluated with evidence.
- [x] Per-step set comparison against the numpy bundle reported.
- [x] Sealed module hashes verified unchanged.
- [x] Same-actor derived verification recorded.
- [x] `ACTIVE_WORKSPACE.md` updated.

## Formal outcome (2026-09-12)

Route journal: `docs/research_branches/FP-ATTN-8X-001/claude/first_result.md`.

| hypothesis | verdict |
|---|---|
| `H1` provenance | **PASS** (structural, plus `885/885` flagged entries) |
| `H2` validity and soundness | **PASS** (`837` emitted steps, `0` violations) |
| `H3` network reaches twelve steps | **PASS** |
| `H4` set agreement at every step | **PASS** (**`0`** disagreements) |
| `H5` zero `eta` flips | **PASS** (`0` over `885` entries) |
| `H6` drift within `ATOL` | **PASS** (worst `1.373e-05` vs `1e-4`) |
| `H7` headroom above `20x` | **PASS** (`40.0x`) |

- **The network reproduces numpy exactly**, not merely in counts: **zero emitting-set
  disagreements at every one of the twelve steps in both arms**, with identical
  emission sequences

  ```text
  frozen : [42, 42, 40, 40, 38, 36, 33, 29, 28, 27, 25, 22]   network == numpy
  empB   : [44, 44, 44, 42, 41, 38, 35, 34, 31, 29, 27, 26]   network == numpy
  ```

  `837` certified updates on the literal network, all componentwise non-degrading,
  strictly improving, and inside the bound that licenses them.
- **Drift does not accumulate over twelve steps** at five times the previous maximum
  network horizon: the gap is flat and non-monotone, worst `1.373e-05`.
- **The substantive finding is a trust horizon, not the reach.** The margin shrinks
  geometrically while the `float32` gap stays flat at `~1.4e-05`. Headroom falls from
  `64,221x` at step 1 to `40x` at step 11, and — post-hoc, projected against
  `FP-HORIZON-001`'s numpy margin sequence — **crosses `1x` at step `17`**. Beyond
  that the network cannot be relied on to reproduce numpy's eligibility decisions,
  because the certified improvements are smaller than the network's own arithmetic
  resolution. Steps `21`–`25` recover above `1x` because the emitting set changes;
  from step `26` they do not.
- This is a limit on the **network path**, distinct from the iteration's reach:
  `FP-HORIZON-001` found the numpy iteration still going at step `32` on margins of
  `3.8e-7`, where the network could not be trusted to make the same decision at all.
- The network path is therefore **no longer behind** on the tested configuration
  (`8x`, `12` steps); what remains ahead is numpy's `32`-step horizon, and extending
  the network there would measure arithmetic rather than the certificate.
- **Not established**: anything about the network beyond step `12` (the trust horizon
  is a projection); that the network's *values* match numpy's, only its decisions;
  and whether the certificate repairs help on the network path, since this task
  compares paths rather than arms.
