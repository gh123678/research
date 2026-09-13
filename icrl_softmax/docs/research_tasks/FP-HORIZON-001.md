# FP-HORIZON-001: where does the certified iteration actually stop?

## Task metadata

- Created: 2026-09-12.
- Author: Claude, under the direct user instruction of 2026-09-12 ("好"), closing the
  question `FP-ITER8X-001` raised.
- Status: `ACTIVE`.
- Task version: `1.0`.
- Scientific baseline: `a95bbee` (`claude/FP-CENSUS-001`).
- Execution branch: `claude/FP-CENSUS-001`.
- Result directory: `results/FP-HORIZON-001/claude/`.
- Classification: long, conclusion-critical, single-actor under the standing user
  exception.

## Why this task exists

`FP-ITER8X-001` ran the certified iteration at `8x` certification with a horizon of
`12`. Both arms used all twelve steps and were still emitting at step `12` — `22` and
`26` of `48`. So **twelve was the horizon, not the end**, and the iteration's true
ceiling was left unmeasured.

That is the same shape of gap as `FP-ITER6-001`'s: a plateau that might be a
horizon artifact. It was closed once by extending `6 → 12` and the answer changed
completely (`9` emitters became `36` at step `6`). This task closes it again by
extending the horizon far enough that the iteration stops **on its own**.

## Research question

At `8x` certification, on what step does the certified iteration terminate when the
horizon is no longer binding, and how much total value does the whole trajectory buy?

## Falsifiable hypotheses

1. `H1 (validity, mandatory)`: every emitted step, in both arms and on both routes,
   is componentwise non-degrading and strictly improving in total value.
2. `H2 (soundness, mandatory)`: at every emitted step the certified error bounds the
   realized oracle error against `Q^{pi_{k-1}}`. Zero violations.
3. `H3 (the headline, pre-registered)`: **the iteration terminates before the
   horizon.** The deepest trajectory is strictly less than `32`, so the stopping
   point is a property of the method rather than of the budget. If this fails — if
   some route-record is still emitting at step `32` — then the certified iteration
   has not been shown to terminate at all within `8x` certification, which is a
   finding in its own right.
4. `H4 (pre-registered)`: the emitting population is **non-increasing** at every
   step. It fell monotonically over steps `1`–`12`; a rise would mean a policy
   change made an earlier-abstaining record eligible again.
5. `H5 (pre-registered)`: the mean gain decays at every step `13` onward.
6. `H6 (pre-registered)`: every emitted gain is **strictly positive** at every step,
   so no step is emitted on the strength of a numerically null improvement.
7. `H7 (reporting obligation)`: the shape of the tail — the last step at which each
   arm emits, the population at the final emitting step, and whether the population
   tapers or falls off a cliff.
8. `H8 (pre-registered, the value question)`: the cumulative certified value gain
   from `pi_0` to the final policy of each trajectory. Reported in total and
   per-state, with the minimum per-state gain, because a large total with one
   degraded state would be a different claim.

`H1`, `H2` are mandatory. `H3`--`H6`, `H8` are pre-registered; each is reported
`PASS` or `FALSIFIED` and never reinterpreted.

### Basis of the predictions

`H3` extrapolates the observed decline: the population went `42, 42, 40, 40, 38, 36,
33, 29, 28, 27, 25, 22` over twelve steps at `8x`, a mean decline of about `1.7` per
step. Linear extrapolation from `22` reaches zero near step `25`. The prediction is
therefore that the deepest trajectory stops **before** `32` — a real prediction,
since the decline could flatten into a plateau exactly as it did at `1x`, where the
population sat at `12` for two steps before this line extended the horizon.

`H4` and `H5` are **continuations of observed trends**, not new claims, and are
registered so that a break in either is visibly reported rather than smoothed.

`H8` is a reporting obligation rather than a directional claim: it has no
pre-registered threshold, because the line has never measured cumulative value and
inventing a threshold for it would be arbitrary.

## Frozen contract

### Protocol, inherited verbatim

`4` states, `3` actions, `pi_min = 0.15`, gap bonus `0.5`, mixing `{0.08, 0.5}`,
`12` tasks per mixing, training trajectory `65536` transitions, `gamma = 0.70`,
`alpha = 0.65`, `R_star = 1.5`, `delta = 0.05`, seed `20260911`, eta grid
`1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01`.

### The change

One: **horizon `32`**, on the same evaluator, the same `8x` certification, the same
two arms (`frozen` and `empirical_bernstein`) and the same sampler. The evaluator
gains a `--task-id` override so this run's bundle carries its own identity rather
than reusing `FP-ITER8X-001`'s. No sealed module is touched.

The `frozen` arm is retained for the same reason as before: it separates the data's
contribution from the repair's, and here it also answers "does the *sealed*
certificate terminate too, or only the repair?".

## Prohibited work

- No modification of any sealed module, sealed bundle, or closed task record.
- No change to any frozen formula, constant, tolerance, eta grid or hypothesis after
  the run.
- No re-tuning of the horizon after seeing where the iteration stops.
- No claim that the vectorised sampler reproduces the sealed batch.
- No reinterpretation of a falsified prediction.
- No `git add -A`.

## Acceptance criteria

1. `H1`/`H2` confirmed with per-step evidence; any violation listed individually.
2. `H3`--`H6`, `H8` each evaluated with evidence.
3. The full population trajectory, the deepest trajectory per arm, and the final
   emitting step per arm.
4. Cumulative value gain, total and per-state, with the minimum per-state gain.
5. Sealed module hashes verified unchanged.
6. All strict-JSON, finite, shape, seed and location checks pass.
7. Complete reproducibility evidence including item counts.
8. Same-actor derived verification recorded.
9. `ACTIVE_WORKSPACE.md` updated.

## Failure criteria

The construction fails if a validity or soundness check fails, if a sealed module
changes, or if the horizon is re-tuned after the result.

`H3`--`H6` failing is **not** a construction failure. `H3` failing means the
iteration is still going at step `32` and has **not** been shown to terminate, which
would be the more interesting outcome: a certified iteration that keeps finding
tiny-but-valid improvements indefinitely, on margins near numerical dust.

## Stopping conditions

Stop and report if:

- a validity or soundness check fails;
- a sealed module's hash is found changed;
- the run cannot complete within budget;
- execution would expand cost beyond the authorised scope.

## Route assignment and verification

Single actor: Claude executes and verifies. By the user's instruction of
2026-09-11 ("验证先不管"), verification is not the focus; the derived checks are
recorded for completeness and no independent verification is claimed.

## Pre-review

- Status: `APPROVED` (2026-09-12), same-actor.
- Evidence: `docs/research_branches/FP-HORIZON-001/claude/pre_review.md`.

## Objections and user rulings

### Objection

- Status: `NONE`.

### User ruling (scope)

- Date: 2026-09-12.
- Decision: "好" — extend the horizon to find where the iteration stops.
- Scope: this task, on `claude/FP-CENSUS-001`.

## Definition of done

- [x] `H1`/`H2` confirmed with per-step evidence.
- [x] `H3`--`H6`, `H8` each evaluated with evidence.
- [x] The population trajectory and the stopping step reported per arm.
- [x] Cumulative value gain reported, total and per-state.
- [x] Sealed module hashes verified unchanged.
- [x] Same-actor derived verification recorded.
- [x] `ACTIVE_WORKSPACE.md` updated.

## Formal outcome (2026-09-12)

Route journal: `docs/research_branches/FP-HORIZON-001/claude/first_result.md`.

| hypothesis | verdict |
|---|---|
| `H1` validity | **PASS** (`1,578` emitted steps, `0` degrading) |
| `H2` soundness | **PASS** (`0` certificate violations) |
| `H3` terminates before the horizon | **FALSIFIED** — still emitting at step `32` |
| `H4` population non-increasing | **PASS** (no rises in either arm) |
| `H5` mean gain decays past twelve | **PASS** |
| `H6` every emitted gain strictly positive | **PASS** (smallest `3.845e-07`) |
| `H8` cumulative value gain | `5.649893` (`frozen`) / `5.803923` (`empB`) mean per route-record |

- **The certified iteration does not terminate within 32 steps.** Both arms are still
  emitting at step `32` — `12` and `14` of `48` — with the smallest emitted gain at
  `3.845e-07`. It keeps finding componentwise non-degrading, strictly positive,
  certified improvements indefinitely, on margins near numerical dust.
- **The population moves in a staircase, not a taper.** The `frozen` arm holds `20`
  for eight consecutive steps (`13`–`20`) and `18` for six (`21`–`26`), separated by
  sudden drops. This reframes the `1x` result: **plateaus are the normal shape of
  this process**, not an artifact of any particular horizon. The `1x` plateau at `12`
  was real *as a plateau*; what was wrong was reading it as an endpoint.
- **The value question is answered for the first time.** Mean total value gain per
  route-record `5.649893` (`frozen`) / `5.803923` (`empB`), with `42/48` and `44/48`
  records improved in **every** state. The verifier re-derives this from the recorded
  per-state value vectors, independently of the analyzer's per-step sums.
- **Post-hoc and labelled**: reach at several mean-gain floors — `0.05` last cleared
  at step `10`, `0.01` at `16`, `1e-3` at `25`, `1e-4` at `32`. The task **does not
  choose a floor**, because a gain of `3.8e-7` is certified, non-degrading and
  strictly positive, and picking a floor after seeing the table would be the post-hoc
  threshold selection the sheet prohibits.
- `1,578` certified updates, none degrading, none violating the bound that licenses
  them.
- The certified-horizon sequence `2 → 3 → 4 → 5 → 6 → 12 → 32` says more about where
  the experiments were stopped than about the method. What is now firm: **at least
  32 steps**, all valid, staircase decline, geometric gain decay, mean gain still
  above `1e-4` at step `32` but smallest gain `3.8e-7`.
- **Not established**: that it ever stops; whether the sub-`1e-3` tail counts as
  improvement; and anything about the network path at `8x`.
