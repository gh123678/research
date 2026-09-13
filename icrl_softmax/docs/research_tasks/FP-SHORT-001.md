# FP-SHORT-001: why do some records stop emitting while 45% of the gap remains?

## Task metadata

- Created: 2026-09-12.
- Author: Claude, under the direct user instruction of 2026-09-12 ("去做"), following
  the question `FP-GAP-001` relocated.
- Status: `ACTIVE`.
- Task version: `1.0`.
- Scientific baseline: `32559ac` (`claude/FP-CENSUS-001`).
- Execution branch: `claude/FP-CENSUS-001`.
- Result directory: `results/FP-SHORT-001/claude/`.
- Classification: long, conclusion-critical, single-actor under the standing user
  exception.

## Why this task exists

`FP-GAP-001` found the certified iteration converging geometrically — a constant
per-step decay factor of about `0.45`, closing `99.97%` of the initial suboptimality
on long trajectories. It also found where the iteration genuinely stops short, and it
is **not** the tail:

| emitted steps | n | mean fraction of the gap closed |
|---|---:|---:|
| `1`–`5` | `12` | **`54.89%`** |
| `6`–`11` | `26` | `94.46%` |
| `12`–`23` | `10` | `98.28%` |
| `24`–`32` | `38` | `99.97%` |

Records that stop after a handful of steps leave **`45%`** of their initial
suboptimality on the table. The gate closed on them while a large amount of value
remained uncollected.

## Research question

At the step where a short trajectory stops, **what specifically fails** — is the
certificate far too loose, is a single state blocking a conjunctive gate, or has the
`eta` grid run out of small enough steps?

## The three mechanisms, and how each is separated

The rule requires `min_s LB_s > 0` for some `eta` in the grid, with
`LB_s = I_s − E_Q·||dpi_s||_1` and `I_s = sum_a (pi_eta − pi)(s,a)·qhat(s,a)`.

1. **The certificate is far too loose.** Measured as `E_Q / h`, where the **exact gate
   margin** is `h = max_eta min_s I_s / ||dpi_s||_1`. Emission is exactly `E_Q < h`, so
   `E_Q / h` is the whole gate as a single number and its distance above `1` says how
   narrowly the record missed.
2. **One state blocks the update.** The gate is a conjunction over states, so a single
   stuck state stops the record. Measured as the number of states with `LB_s <= 0` at
   the best candidate. `1` blocking state and `4` blocking states are different
   diagnoses with different remedies.
3. **The `eta` grid bottoms out.** The smallest frozen grid value is `0.01`. A policy
   near the constrained optimum needs a *small* tilt; if `0.01` already overshoots,
   the rule has nothing smaller to try. Measured by re-running the iteration with an
   extended grid — every frozen value plus `0.005, 0.002, 0.001, 0.0005, 0.0002,
   0.0001`.

## Falsifiable hypotheses

1. `H1 (faithfulness, mandatory)`: on the frozen grid, the instrumented decision path
   reproduces `fs.improvement_for` — same status and same selected `eta` — at **every
   step of every trajectory**, and the frozen-grid arm reproduces `FP-ITER8X-001`'s
   sealed `frozen` arm step for step. Without this the diagnosis is about a
   reimplementation rather than about the frozen rule.
2. `H2 (monotonicity of the extension, mandatory)`: the extended grid changes **no
   emitting decision**, at any step, on any trajectory. This is a theorem rather than
   an observation — the frozen values are a descending prefix of the extended grid, so
   if a candidate in the prefix passed, the scan stops there regardless of what
   follows — and a single counterexample therefore falsifies the implementation, not
   the mathematics.
3. `H3 (pre-registered: the records stop narrowly)`: at the stopping step of the short
   trajectories, `E_Q / h` lies in `[1, 2]` for the majority. Registered in the
   direction that makes the problem **small** — if it holds, a modest further
   tightening or a finer grid should revive these records. If it fails, `E_Q` is far
   above what the record could ever earn, and no repair of this kind will help.
4. `H4 (pre-registered: a single state blocks)`: at the stopping step, **exactly one**
   state has `LB_s <= 0` for the majority of short trajectories. Registered as the
   conjunctive-gate diagnosis. If the majority show all states blocking, the
   obstruction is global to the record rather than a single stuck state.
5. `H5 (pre-registered: the grid is the binding constraint)`: the extended grid
   extends **at least one** short trajectory beyond its frozen-grid stopping step.
   This is the actionable form of mechanism 3.
6. `H6 (pre-registered: the extension is worth having)`: the extended grid raises the
   total emitted steps across all trajectories by at least `10%` over the frozen grid.

`H1`, `H2` are mandatory. `H3`--`H6` are pre-registered; each is reported `PASS` or
`FALSIFIED` and never reinterpreted.

### Basis of the predictions

`H5`, `H6` are predictions about a **protocol** change — the `eta` grid is frozen
protocol — and are registered because the mechanism is plausible from the arithmetic:
the rule takes the *largest* passing `eta`, and a policy close to the constrained
optimum has a small `I_s / ||dpi_s||_1`, so the passing `eta` shrinks as the
trajectory proceeds. Whether it ever shrinks below the grid floor of `0.01` is exactly
what this task measures; it is not known in advance, and a `FALSIFIED` outcome would
say the grid is not the constraint.

`H3`, `H4` are diagnostic and are registered with thresholds so that the answer is a
verdict rather than a table. `H4` in particular separates two remedies that look
similar in a summary: a single stuck state is addressable by per-state handling, while
all-states-blocking is not.

## Frozen contract

### Protocol, inherited verbatim

`4` states, `3` actions, `pi_min = 0.15`, gap bonus `0.5`, mixing `{0.08, 0.5}`,
`12` tasks per mixing, training trajectory `65536` transitions, certification
`131072 x 64 = 8,388,608` items from the sampler `FP-SAMPLE-001` validated, `gamma =
0.70`, `alpha = 0.65`, `R_star = 1.5`, `delta = 0.05`, seed `20260911`, horizon `12`.

### The two arms

Both use the **sealed variance-adaptive certificate**, so the `eta` grid is the only
thing that differs:

| arm | grid |
|---|---|
| `frozen_grid` | `1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01` |
| `extended_grid` | the frozen values **plus** `0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001` |

Keeping the frozen values as a descending **prefix** is what makes `H2` a theorem and
the comparison clean: any difference is attributable to the added candidates alone.

### Scope, and what this task does not do

The extended grid is a **diagnostic**, not a proposed protocol. The line's frozen
protocol specifies the grid, and changing it would be a protocol change requiring its
own task and its own justification. This task measures whether the grid is the binding
constraint; it does not adopt a new one.

## Prohibited work

- No modification of any sealed module, sealed bundle, or closed task record.
- No change to any frozen formula, constant, tolerance or hypothesis after the run.
- **No adoption of the extended grid as a protocol.** It is a diagnostic arm only.
- No claim that the instrumented decision path *is* the frozen rule without `H1`
  passing at every step.
- No reinterpretation of a falsified prediction.
- No `git add -A`.

## Acceptance criteria

1. `H1` verified at every step and against the sealed `FP-ITER8X-001` bundle.
2. `H2` verified at every step; any counterexample listed.
3. `H3`--`H6` each evaluated with evidence.
4. Per-short-trajectory: `E_Q`, `h`, `E_Q/h`, the blocking-state count, and the
   smallest `eta` that would have passed at the current `E_Q`.
5. Sealed module hashes verified unchanged.
6. All strict-JSON, finite, shape, seed and location checks pass.
7. Same-actor derived verification recorded.
8. `ACTIVE_WORKSPACE.md` updated.

## Failure criteria

The construction fails if `H1` fails at any step, if `H2` shows a counterexample, if a
sealed module changes, or if the extended grid is presented as an adopted protocol.

`H3`--`H6` failing is **not** a construction failure. `H5` failing would mean the grid
is **not** the constraint, which would redirect the diagnosis to the certificate or
to the conjunctive gate — and would be the more informative outcome, because the grid
is the one mechanism this line can fix without touching the certificate.

## Stopping conditions

Stop and report if:

- `H1` fails, since the diagnosis would then not be about the frozen rule;
- `H2` shows a counterexample;
- a sealed module's hash is found changed;
- execution would expand cost beyond the authorised scope.

## Route assignment and verification

Single actor: Claude executes and verifies. By the user's instruction of
2026-09-11 ("验证先不管"), verification is not the focus; the derived checks are
recorded for completeness and no independent verification is claimed.

## Pre-review

- Status: `APPROVED` (2026-09-12), same-actor.
- Evidence: `docs/research_branches/FP-SHORT-001/claude/pre_review.md`.

## Objections and user rulings

### Objection

- Status: `NONE`.

### User ruling (scope)

- Date: 2026-09-12.
- Decision: "去做" — diagnose why a subset of records stops early.
- Scope: this task, on `claude/FP-CENSUS-001`.

## Definition of done

- [ ] `H1` verified at every step and against the sealed bundle.
- [ ] `H2` verified at every step.
- [ ] `H3`--`H6` each evaluated with evidence.
- [ ] Per-trajectory diagnosis reported.
- [ ] The extended grid is not presented as an adopted protocol.
- [ ] Same-actor derived verification recorded.
- [ ] `ACTIVE_WORKSPACE.md` updated.
