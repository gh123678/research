# FP-MEANING-001: does a certified improvement of 1e-7 mean anything?

## Task metadata

- Created: 2026-09-12.
- Author: Claude, under the direct user instruction of 2026-09-12 ("好的"), closing
  the question the horizon and network results both raised.
- Status: `ACTIVE`.
- Task version: `1.0`.
- Scientific baseline: `c1cebd7` (`claude/FP-CENSUS-001`).
- Execution branch: `claude/FP-CENSUS-001`.
- Result directory: `results/FP-MEANING-001/claude/`.
- Classification: long, conclusion-critical, single-actor under the standing user
  exception.

## Why this task exists, and why it is not merely definitional

`FP-HORIZON-001` found that the certified iteration does not stop: at `8x`
certification it is still emitting at step `32`, with a minimum realized gain of
`3.845e-07`. `FP-ATTN-8X-001` then found that the network cannot be trusted to
reproduce numpy's eligibility decisions past about step `17`, because the margins
fall below its `float32` arithmetic resolution.

Both results leave the same question on the table, and the task sheet for
`FP-HORIZON-001` recorded it as explicitly unsettled:

> a gain of `4e-7` is certified, componentwise non-degrading and strictly positive,
> and whether that counts as "policy improvement" is a definitional question the line
> has not settled.

**It is not purely definitional, because one of the candidate definitions has a
right answer.** The decision rule requires `min_s LB_s > 0`, where

```text
LB_s = I_s - E_Q * ||dpi_s||_1
```

is a **proven** lower bound on the true improvement at state `s`, given the
certificate holds. So every emitted update carries a *certified* improvement, not
merely a realized one, and `min_lb` — which every step entry already records — is its
size. Whether `LB_s = 2e-16` counts as "proving an improvement" is answerable by
measurement: it depends on the resolution of the arithmetic that computes it.

That reframes the question into one with a falsifiable answer:

> **At what step does the certified lower bound on the improvement fall to the
> resolution of the arithmetic that computes it, and does the iteration stop there?**

### What the bundle already shows

Reading `FP-HORIZON-001`'s `min_lb` column, before any new computation, the certified
bound decays geometrically by roughly a factor of `6.2` per step:

```text
step  1   min_lb 3.165e-04      step 17   min_lb 7.574e-14
step  6   min_lb 4.689e-05      step 19   min_lb 1.795e-15
step 12   min_lb 7.043e-10      step 20   min_lb 4.069e-16
step 16   min_lb 4.713e-13      step 32   min_lb 2.164e-16
```

`2.164e-16` is within a factor of `1.03` of `float64` machine epsilon
(`2.220e-16`). This task establishes that rigorously and asks what follows.

## Research question

At what step does the certified lower bound on the per-step improvement become
numerically indistinguishable from zero, and what does the iteration look like under
each defensible definition of "improvement"?

## Falsifiable hypotheses

1. `H1 (the noise floor is measured, mandatory)`: the `float64` resolution of the
   value computation is established **empirically**, per route-record, by computing
   `v^pi` two independent ways — `policy_quantities` against value iteration to
   convergence — and reporting the maximum disagreement. It is not assumed from
   machine epsilon. A floor of `~1e-15` and a floor of `~1e-10` would support
   completely different conclusions, so the number has to be earned.
2. `H2 (mandatory)`: the realized gains at the tail are **above** that floor, so they
   are real improvements rather than round-off. The prediction is that they are by
   many orders of magnitude, on the strength of the `~2e-14` agreement measured in
   `verify_policy_quantities_by_solve.py`.
3. `H3 (the headline, pre-registered)`: the **certified** bound `min_lb` falls below
   the measured floor at some step, **and the iteration keeps emitting past it**. The
   prediction is a crossing near step `20`. This is the finding that matters: the
   iteration would then be certifying improvements it cannot distinguish from zero,
   while every formal check continues to pass.
4. `H4 (pre-registered)`: `min_lb` decays geometrically, with a per-step factor
   between `3` and `10`. The bundle's apparent factor is `~6.2`, and a decaying
   rather than saturating profile is what makes the crossing inevitable.
5. `H5 (pre-registered)`: the ratio `min_lb / (realized minimum gain)` collapses by at
   least **six orders of magnitude** between step `1` and step `32`. The apparent
   ratio falls from `5e-4` to below `1e-9`, which means the certificate proves a
   vanishing fraction of what actually happens.
6. `H6 (reporting obligation, not a directional claim)`: the reach under each
   candidate definition, reported side by side —
   - `D1` realized gain `> 0`;
   - `D2` certified bound `> 0`;
   - `D3` certified bound `>` the measured `float64` floor;
   - `D4` certified bound `>` the `float32` gap the network exhibits (`~1.4e-05`,
     from `FP-ATTN-8X-001`).

`H1`, `H2` are mandatory. `H3`--`H5` are pre-registered. **`H6` is where the task
declines to choose**: it reports what each definition costs, and the choice is the
user's.

### Basis of the predictions

`H3` and `H4` are anchored on the `min_lb` column read out of the existing bundle
(both shown above), so they are not blind. `H5` is anchored on the same column.

`H2` is anchored on the `~2e-14` agreement between `policy_quantities` and a direct
linear solve measured in `verify_policy_quantities_by_solve.py`.

`H6` has no threshold because inventing one would be exactly the post-hoc choice this
task exists to expose.

## Frozen contract

### No new experiments

This task adds **no new iteration run**. It measures one new quantity — the
arithmetic floor of the value computation — and then analyses the `FP-HORIZON-001`
bundle, which is the longest certified trajectory in the line. The decision rule, the
certificates, the protocol and every sealed module are untouched.

### The measurement

For every `24` records, compute `v^pi` by two routes:

- `policy_quantities(mdp, policy)`, the routine the whole line uses;
- value iteration under `pi` to a `1e-14` tolerance, which shares no code with it.

Report `max_s |v_a(s) - v_b(s)|` per record, and take the maximum and median as the
floor. Also record `numpy.finfo(float64).eps` for reference, so the measured floor and
the assumed one can be compared rather than conflated.

### The analysis

From the `FP-HORIZON-001` bundle, per step and per arm: the minimum `min_lb`, the
minimum realized total gain, their ratio, and the emitting count. Then the reach under
each definition in `H6`.

## Prohibited work

- No modification of any sealed module, sealed bundle, or closed task record.
- No change to any frozen formula, constant, tolerance or hypothesis after the run.
- No new iteration run, and no re-tuning of any earlier task's configuration.
- **No choosing a definition and presenting it as the answer.** `H6` reports all
  four; the task sheet records that the choice belongs to the user.
- No reinterpretation of a falsified prediction.
- No `git add -A`.

## Acceptance criteria

1. `H1` the floor is measured per record, with the method stated and the numbers
   reported, not asserted.
2. `H2` confirmed with the ratio of realized tail gains to the floor.
3. `H3`--`H5` each evaluated with evidence.
4. `H6` the reach table under all four definitions, with the emitting counts.
5. The `min_lb` series reported in full, so a reader can check any crossing.
6. Sealed module hashes verified unchanged.
7. All strict-JSON, finite, shape, seed and location checks pass.
8. Same-actor derived verification recorded.
9. `ACTIVE_WORKSPACE.md` updated.

## Failure criteria

The construction fails if the floor is asserted rather than measured, if a definition
is presented as the answer rather than as one of the options, if a sealed module
changes, or if the `min_lb` series is summarised without being shown.

`H3`--`H5` failing is **not** a construction failure. `H3` failing — the certified
bound staying above the floor all the way to step `32` — would mean the tail updates
are provably meaningful and the iteration's length is a real result rather than an
arithmetic artifact.

## Stopping conditions

Stop and report if:

- the two value computations disagree by more than `1e-8`, since the floor would then
  not be a floor;
- a sealed module's hash is found changed;
- execution would expand cost beyond the authorised scope.

## Route assignment and verification

Single actor: Claude executes and verifies. By the user's instruction of
2026-09-11 ("验证先不管"), verification is not the focus; the derived checks are
recorded for completeness and no independent verification is claimed.

## Pre-review

- Status: `APPROVED` (2026-09-12), same-actor.
- Evidence: `docs/research_branches/FP-MEANING-001/claude/pre_review.md`.

## Objections and user rulings

### Objection

- Status: `NONE`.

### User ruling (scope)

- Date: 2026-09-12.
- Decision: "好的" — take up the definitional question as the next task.
- Scope: this task, on `claude/FP-CENSUS-001`.

## Definition of done

- [x] `H1` the floor measured per record.
- [x] `H2`--`H5` each evaluated with evidence.
- [x] `H6` the reach table under all four definitions.
- [x] The full `min_lb` series reported.
- [x] Sealed module hashes verified unchanged.
- [x] Same-actor derived verification recorded.
- [x] `ACTIVE_WORKSPACE.md` updated.

## Formal outcome (2026-09-12)

Route journal: `docs/research_branches/FP-MEANING-001/claude/first_result.md`.
No new iteration was run; the trajectory analysed is `FP-HORIZON-001`'s.

| hypothesis | verdict |
|---|---|
| `H1` floor measured, not assumed | **PASS** (`2.287e-14` max, `1.976e-14` median) |
| `H2` tail gains above the floor | **PASS** (`1.7e7 x`) |
| `H3` bound crosses a floor, iteration runs on | **PASS** (step `18`) |
| `H4` geometric decay | **PASS** (mean factor `6.54`) |
| `H5` at least six orders of collapse | **FALSIFIED** (`5.9` orders) |
| `H6` reach under each definition | reported, **deliberately unresolved** |

- **The floor had to be measured and the measurement decided the answer.** The two
  value routes agree to `2.287e-14`, while machine epsilon is `2.220e-16` — **two
  orders of magnitude apart**, and the gap moves the crossing by eight steps.
  Asserting `eps` would have been the natural shortcut and would have been wrong.
- **The certified bound `min_lb` decays `6.54 x` per step** from `3.165e-04` at step 1
  to `2.164e-16` at step 32, crossing the measured floor at step **`18`** and machine
  epsilon at step **`26`**.
- **The iteration runs on regardless**, still emitting `12` route-records at step 32.
  **Every formal check in this line passes throughout, because they all test `> 0`
  and `2e-16` is strictly positive.** This is a blind spot in the acceptance criteria,
  not a failure of any run: the certificate does not become *wrong*, it stops *saying
  anything*, and nothing in the current check set can see the difference.
- The realized gains are **real**: the smallest at step 32 is `1.7e7 x` the floor.
  So the realized half of the claim holds all the way to the horizon; the certified
  half evaporates.
- **`H6` reach table, and the task declines to choose**:

  | definition | first failing step | holds |
  |---|---:|---:|
  | `D1` realized gain `> 0` | never | `32/32` |
  | `D2` certified bound `> 0` | never | `32/32` |
  | `D3` certified `>` method floor (`2.3e-14`) | **`18`** | `20/32` |
  | `D4` certified `>` machine eps (`2.2e-16`) | **`26`** | `30/32` |
  | `D5` certified `>` network `float32` gap (`1.4e-05`) | **`2`** | `5/32` |

  The line has been implicitly reporting `D1`/`D2` and calling the result "reaches 32
  steps"; a reader assuming `D3` would take a very different message from the same
  data. **The choice is the user's**, and choosing one after seeing this table would be
  the post-hoc threshold selection the sheet prohibits.
- `D3`'s crossing at `18` is within two steps of the network's projected trust horizon
  of `17` — two independent routes to the same place, one from the certificate's
  resolution and one from the network's.
- **New open item, now a decision rather than an oversight**: whether the decision rule
  should require `min_s LB_s > f` for a stated floor `f` rather than `> 0`. That is a
  change to the frozen rule and therefore a new task, not a reinterpretation of this
  one.
