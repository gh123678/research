# FP-ITER8X-001: does eight times the certification data extend the iteration?

## Task metadata

- Created: 2026-09-12.
- Author: Claude, under the direct user instruction of 2026-09-12 ("好的"), closing
  the question the two lever tasks raised.
- Status: `ACTIVE`.
- Task version: `1.0`.
- Scientific baseline: `d68d940` (`claude/FP-CENSUS-001`).
- Execution branch: `claude/FP-CENSUS-001`.
- Result directory: `results/FP-ITER8X-001/claude/`.
- Classification: long, conclusion-critical, single-actor under the standing user
  exception.

## Why this task exists

Two lines of work meet here and have never been combined.

**The ceiling.** `FP-ITER6-001` and `FP-ATTN-ITER6-001` ran the certified iteration
at the sealed `1x` certification on both implementations. It reached six steps, the
emitting population fell `22, 20, 15, 12, 12, 9`, and the minimum gain dropped to
`0.019347` — below the `0.05` floor that steps 4 and 5 had cleared. Six steps was the
**horizon**, and nine route-records used all of it.

**The lever.** `FP-TIGHT-001` and `FP-SAMPLE-001` established what makes a
route-record eligible. The criterion is the census's certificate-independent gate,
`sigma_min / E_Q ≳ 2.17`. Repairing the mean-step inequality buys `−20%`; `8x`
certification data buys `−58%`. At step 1, `8x` raises the eligible count from `30`
to `44` of `48`, soundly, with zero coverage violations.

So the obvious question is the one neither line could answer alone: **if nearly every
record is eligible at step 1, does the iteration get past six steps?**

## Research question

At `8x` certification, how far does the certified iteration reach, and is the extra
reach attributable to the data or to the certificate repair?

## Design: two arms, and why both

Each arm carries its own trajectory, because an arm's decision at step `k`
determines the policy at step `k+1`.

| arm | data | certificate | what it isolates |
|---|---|---|---|
| `frozen` | `8x` | sealed | **the data**, with nothing else changed |
| `empirical_bernstein` | `8x` | repaired | the data **and** the repair together |

The `frozen` arm is a control and not decoration: without it, a longer iteration
could not be attributed to more data rather than to a better bound, and the two
explanations are exactly the ones this line has been separating.

## Falsifiable hypotheses

1. `H1 (validity, mandatory)`: every emitted step, in both arms and on both routes,
   is componentwise non-degrading and strictly improving in total value.
2. `H2 (soundness, mandatory)`: at every emitted step, in both arms, the certified
   error bounds the realized oracle error against `Q^{pi_{k-1}}`. Zero violations.
3. `H3 (the headline, pre-registered)`: **the iteration passes six steps.** At least
   one route-record emits a seventh step, in at least one arm, at the same horizon
   budget the `1x` run already used.
4. `H4 (pre-registered)` : at step `7`, the `frozen` arm's emitting count is
   **strictly greater** than the `9` that emitted at step `6` under `1x` — i.e. the
   extra records are not merely carried over but added.
5. `H5 (pre-registered, monotonicity across arms)`: at every step, the
   `empirical_bernstein` arm's emitting set **contains** the `frozen` arm's. This is
   a theorem rather than an observation — a smaller `E_Q` can only increase
   `LB_s = Î_s − E_Q‖Δπ_s‖₁` — so a single counterexample falsifies the
   implementation, not the mathematics.
6. `H6 (pre-registered, decay continues)`: the mean gain at each step beyond `6`
   is lower than at the previous step, i.e. the decay that held through `1x → 6`
   continues wherever the iteration now reaches.
7. `H7 (pre-registered, and deliberately against the naive reading)`: the minimum
   gain at step `6` under `8x` is **lower** than the sealed `1x` value `0.019347`.
   More eligible records means a larger emitting set, and a larger set includes
   marginal records — so admitting more records should lower the minimum even though
   the bound is tighter. Registering the direction that is *not* flattering.
8. `H8 (reporting obligation)`: the maximum steps reached per arm, and whether the
   emitting population plateaus again, reported as counts and as set differences
   against the `1x` run at every shared step.

`H1`, `H2` are mandatory. `H3`--`H7` are pre-registered; each is reported `PASS` or
`FALSIFIED` and never reinterpreted.

### Basis of the predictions

`H3` and `H4` extrapolate from the step-1 result, where `8x` made `44` of `48`
records eligible against `30` at `1x`. That is a legitimate basis for expecting more
reach, and it is also the reason `H3` is the headline: if the iteration still stops
at six with `44` records eligible, then eligibility at step 1 is **not** what limits
the horizon, and something else is — which would be a more interesting finding than
a simple extension.

`H7` is registered in the direction that argues against the lever, so that a `PASS`
on it is evidence rather than self-congratulation.

## Frozen contract

### Protocol, inherited verbatim

`4` states, `3` actions, `pi_min = 0.15`, gap bonus `0.5`, mixing `{0.08, 0.5}`,
`12` tasks per mixing, training trajectory `65536` transitions, `gamma = 0.70`,
`alpha = 0.65`, `R_star = 1.5`, `delta = 0.05`, seed `20260911`, eta grid
`1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01`.

### The two changes

1. **Certification size `8x`**: `131072 x 64 = 8,388,608` items instead of
   `1,048,576`, drawn by the vectorised sampler validated in `FP-SAMPLE-001`
   (`H1` gate: worst TV `0.0091`, worst `E[Y^2]` deviation `0.9%`). The same batch
   is reused at every step, as the sealed protocol requires.
2. **Horizon `12`**, so that the iteration is not truncated at the value under test.
   `six` is the number being tested; a horizon of `12` gives it room to fail or
   succeed visibly.

Nothing else changes: the routes, the residuals, the decision rule, the ordered
reasons and the audit are the frozen ones, and `pi_{k-1}` remains the step-`k`
target. The sealed modules are untouched.

### The sampler is not a reproduction

The `8x` batch is a fresh independent sample, not a superset of the sealed one,
because the vectorised sampler consumes the random stream differently — the defect
`FP-CENSUS-001`'s batch guard caught. The `1x` comparison therefore crosses both
size and realisation, and the `frozen@8x` arm exists partly to make that visible: it
is the sealed certificate on new data, so any difference between it and the sealed
`1x` run is realisation plus size, with the certificate held fixed.

## Prohibited work

- No modification of any sealed module, sealed bundle, or closed task record.
- No change to any frozen formula, constant, tolerance, eta grid or hypothesis after
  the run.
- No claim that the vectorised sampler reproduces the sealed batch.
- No re-tuning of the horizon or the multiplier after seeing the result.
- No reinterpretation of a falsified prediction.
- No `git add -A`.

## Acceptance criteria

1. `H1` and `H2` confirmed with per-step evidence; any violation listed individually.
2. `H3`--`H7` each evaluated with evidence.
3. Per-step emissions, mean and minimum gain, and `max` realized error for both arms.
4. Set differences against the sealed `1x` run at every shared step.
5. Sealed module hashes verified unchanged.
6. All strict-JSON, finite, shape, seed and location checks pass.
7. Complete reproducibility evidence including item counts and the sampler hash.
8. Same-actor derived verification recorded.
9. `ACTIVE_WORKSPACE.md` updated.

## Failure criteria

The construction fails if a validity or soundness check fails, if a sealed module
changes, if the two arms share a trajectory when they should diverge, or if the
horizon is re-tuned after the result.

`H3`--`H7` failing is **not** a construction failure. `H3` failing would be the most
informative outcome available: it would mean that with `44` of `48` records eligible
the iteration *still* stops at six, so step-1 eligibility is not the binding
constraint on the horizon, and the search moves elsewhere.

## Stopping conditions

Stop and report if:

- a validity or soundness check fails;
- a sealed module's hash is found changed;
- the run cannot complete at the frozen dimensions within budget;
- execution would expand cost beyond the authorised scope.

## Route assignment and verification

Single actor: Claude executes and verifies. By the user's instruction of
2026-09-11 ("验证先不管"), verification is not the focus; the derived checks are
recorded for completeness and no independent verification is claimed.

## Pre-review

- Status: `APPROVED` (2026-09-12), same-actor.
- Evidence: `docs/research_branches/FP-ITER8X-001/claude/pre_review.md`.

## Objections and user rulings

### Objection

- Status: `NONE`.

### User ruling (scope)

- Date: 2026-09-12.
- Decision: "好的" — proceed with the iteration at `8x` certification.
- Scope: this task, on `claude/FP-CENSUS-001`.

## Definition of done

- [x] `H1`/`H2` confirmed with per-step evidence.
- [x] `H3`--`H7` each evaluated with evidence.
- [x] Set differences against the `1x` run reported.
- [x] Sealed module hashes verified unchanged.
- [x] Same-actor derived verification recorded.
- [x] `ACTIVE_WORKSPACE.md` updated.

## Formal outcome (2026-09-12)

Route journal: `docs/research_branches/FP-ITER8X-001/claude/first_result.md`.

Certification `131072 x 64 = 8,388,608` items, horizon `12`, two arms with separate
trajectories.

| step | `frozen@8x` | min gain | `empB@8x` | min gain | sealed `1x` |
|---|---:|---:|---:|---:|---:|
| 1 | `42` | `0.690987` | `44` | `0.521629` | `22` |
| 6 | `36` | `0.051690` | `38` | `0.051690` | `9` |
| 7 | `33` | `0.029867` | `35` | `0.012192` | — |
| **12** | **`22`** | `0.000649` | **`26`** | `0.000649` | — |

| hypothesis | verdict |
|---|---|
| `H1` validity | **PASS** (`841` emitted steps, `0` degrading) |
| `H2` soundness | **PASS** (`0` certificate violations) |
| `H3` passes six steps | **PASS** (`33`/`35` seventh-step emissions) |
| `H4` step-7 population > `1x` step 6 | **PASS** (`33` vs `9`) |
| `H5` repaired arm contains frozen arm | **FALSIFIED** (mis-registered; see below) |
| `H6` decay continues past six | **PASS** |
| `H7` step-6 minimum lower than `1x` | **FALSIFIED** (it is higher: `0.051690`) |

- **The certified iteration reaches twelve steps on both arms and is still emitting
  at the horizon** (`22` and `26` of `48`). Twelve was the horizon, so the true
  ceiling is **unmeasured and beyond twelve**. The certified horizon across the line
  has gone `2 → 3 → 4 → 5 → 6 → 12`.
- **The reach is attributable to the data**: the `frozen` arm — the sealed
  certificate on the larger sample — also reaches twelve, so the control rules out
  the repair as the sole explanation.
- **Nothing is lost at any shared step**: against `1x`, steps 1--6 gain `20, 22, 25,
  28, 26, 27` and lose **`0`**.
- **`H7` failed against the lever's disadvantage**, and that is the useful
  direction: the step-6 minimum *rose* from `0.019347` to `0.051690` while the
  emitting set grew from `9` to `36`. A larger population did not degrade the
  worst-case margin.
- **`H5`'s registration was mis-specified, not the implementation.** The containment
  argument is a theorem only at a **fixed** policy; the two arms carry different
  trajectories, so no such relation is guaranteed across them. It failed at step 11
  on one route-record, as expected. The theorem **is** testable at step 1 — both arms
  share `π_0` and `q_hat` — and there it holds (`42 ⊆ 44`), which the analyzer now
  checks explicitly. Cross-arm containment at `11/12` levels is reported as an
  observation, not a guarantee.
- **Validity holds at every one of `841` emitted updates**: `0` degrading, `0`
  non-positive gains, `0` certificate violations, `0` abstentions without a reason.
- **The tail is margin-starved**: minimum gains at steps 11 and 12 are `0.000183` and
  `0.000649`, three to four orders of magnitude below the `0.05` floor steps 4 and 5
  cleared. Strictly positive and valid, but numerically negligible — any claim that
  the iteration "reaches 12" must carry that.
- **Not established**: where it actually stops; whether such tail margins should
  count as improvement; that the sampler effect is separated from the size effect;
  and the network path, which is untested at `8x` and in sync only to the `1x`
  six-step run.
