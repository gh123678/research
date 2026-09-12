# FP-SAMPLE-001: what does it cost to revive the abstainers?

## Task metadata

- Created: 2026-09-12.
- Author: Claude, under the direct user instruction of 2026-09-12 ("都做"), following
  the two levers `FP-TIGHT-001` priced.
- Status: `ACTIVE`.
- Task version: `1.0`.
- Scientific baseline: `15a1511` (`claude/FP-CENSUS-001`).
- Execution branch: `claude/FP-CENSUS-001`.
- Result directory: `results/FP-SAMPLE-001/claude/`.
- Classification: long, conclusion-critical, single-actor under the standing user
  exception.

## Why this task exists

`FP-TIGHT-001` measured three levers on `E_Q` and found the ordering surprising:

| lever | `E_Q` reduction | sound? |
|---|---:|---|
| correct the mean-step inequality (Bernstein) | `−15%` | yes |
| correct it further (empirical Bernstein) | `−20%` | yes |
| delete the envelope | `−45%` | **no** |
| multiply certification data by `4` | **`−64%`** | yes |

The sample-size lever is the strongest, and it is a **cost** lever rather than a
mathematical one. But it was measured on a three-record subset that contained **no
abstainer**, so the only question that matters — *does more data revive records?* —
went untested. `FP-TIGHT-001` reported that honestly as `NOT EXERCISED` rather than
as a vacuous failure.

This task answers it, and turns a single point into a **cost curve**: certification
size `1x, 2x, 4x, 8x`, all `48` route-records at step 1, so the `26` abstainers are
in the population.

### Why a new sampler, and why that is sound here

The sealed certification generator loops per chain in Python. At `8x` that is `8.4M`
items per record and hours of CPU — a cost that is an artifact of the loop, not of
the science. A vectorised sampler that draws the same law reaches a `~200x` speedup.

`Generator.choice` with an explicit `p` does not consume the random stream the way
raw uniform draws do — the defect `FP-CENSUS-001`'s batch guard caught — so the
vectorised sampler **does not reproduce the sealed batch** and is never used where
the sealed batch is required.

It is legitimate here because the size question needs a *fresh independent*
certification sample, not the sealed one, and because **every rung of the ladder
uses the same sampler**, so the ladder is internally consistent. A sampler that were
fast but wrong would shift `E_Q` and invalidate the whole curve, so its unbiasedness
is a **mandatory gate** (`H1`) rather than a remark.

## Research question

How much certification data does it take to revive an abstaining route-record, and
does the reduction continue to pay as the sample grows?

## Falsifiable hypotheses

1. `H1 (sampler unbiasedness, mandatory)`: at `1x`, the vectorised sampler's
   certification reproduces the sealed generator's per-pair item distribution
   (total-variation distance `< 0.02`) and its residual second moment
   (`E[Y^2]` within `2%`). Failing this invalidates every number below.
2. `H2 (coverage, mandatory)`: both certificate arms cover the realized oracle error
   at **every** rung, `0` violations.
3. `H3 (agreement with FP-TIGHT-001)`: the `1x` rung's mean `E_Q` under empirical
   Bernstein is within `20%` of `FP-TIGHT-001`'s sealed `1x` value (`0.1940`). This
   is a cross-check on a different sample drawn by a different sampler, not a
   reproduction.
4. `H4 (monotonicity in the rung)`: mean `E_Q` decreases from `1x` to `2x` to `4x`
   to `8x`.
5. `H5 (pre-registered magnitude at 4x)`: the `4x` reduction lies in `[−70%, −50%]`,
   anchored on `FP-TIGHT-001`'s subset measurement of `−64%`.
6. `H6 (the substantive prediction)`: the abstainers revive, in three parts —
   `H6a`: at least one never-emitting route-record flips at `2x` or below;
   `H6b`: at least `16` of the `26` never-emitting route-records emit at `8x`;
   `H6c`: the number of never-emitting route-records emitting is non-decreasing in
   the rung.
7. `H7 (diminishing returns)`: the marginal mean-`E_Q` reduction from `4x` to `8x` is
   smaller than from `2x` to `4x`, i.e. the curve flattens.

`H1`, `H2` are mandatory. `H4`--`H7` are the substantive pre-registered predictions.
`H6c` can fail by noise — `E_Q` is a maximum over `12` noisy per-pair quantities, so
a single rung may regress — and a failure is reported rather than smoothed.

### Basis of the predictions, stated explicitly

`H5` is anchored on `FP-TIGHT-001`'s `4x` subset measurement, so it is not a blind
extrapolation. `H6b` is derived from the arithmetic: `FP-TIGHT-001` established that
a route-record emits when `sigma_min / E_Q` clears about `2.17`, and the abstainers'
spreads span `[0.1486, 0.4617]`. At `8x` the radius terms shrink as `1/sqrt(N)` and
`1/N` while the `|mean|` component does not, which puts `E_Q` near `0.05` and every
abstainer's ratio above `3`. `H6b` deliberately predicts `16` rather than `26`,
because the ratio is a **proxy** for the actual `min_s LB_s > 0` gate and the census
measured the proxy as imperfect.

`H7` is the shape claim that decides whether the programme should keep buying data.

## Frozen contract

### Protocol

`4` states, `3` actions, `pi_min = 0.15`, gap bonus `0.5`, mixing `{0.08, 0.5}`,
`12` tasks per mixing, training trajectory `65536` transitions, `gamma = 0.70`,
`alpha = 0.65`, `R_star = 1.5`, `delta = 0.05`, seed `20260911`, eta grid unchanged.
Certification is `16384 x 64 = 1048576` items at `1x`, multiplied by the rung.

### Scope: step 1 only

All `48` route-records at step 1 of the frozen `pi_0`, which is where the `26`
abstainers live. Whether more certification extends the **iteration** is a different
question and is not claimed here.

### The change

One new module plus one new evaluator:

- `fp_sample_vectorised_batch.py` — the vectorised sampler and its unbiasedness
  self-test.
- `evaluate_fp_sample_001.py` — the ladder.

The sealed modules are untouched, the decision rule is untouched, and the two
certificate arms come from `fixed_policy_bernstein_certificate.py` exactly as
`FP-TIGHT-001` left them.

### A confound that is stated rather than assumed away

The ladder's samples are fresh independent draws, **not** supersets of the sealed
batch, because the sampler consumes the stream differently. Each rung is therefore
a different realisation as well as a different size. The ladder is monotone in size
and internally consistent, but size and realisation are not separated. At `8x` the
sample is eight times larger, so the realisation noise is correspondingly smaller;
the confound works *against* finding a clean monotone curve, not for it.

## Prohibited work

- No modification of any sealed module, sealed bundle, or closed task record other
  than verifier updates needed to keep integrity checks correct.
- No change to any frozen formula, constant, tolerance, eta grid, hypothesis or
  metric after the run.
- No use of the vectorised sampler anywhere the sealed batch is required, and no
  claim that it reproduces the sealed batch.
- No re-tuning of the rung list after seeing results.
- No reinterpretation of a falsified prediction.
- No `git add -A`.

## Acceptance criteria

1. `H1` gate passes with its numbers reported, or the ladder is void.
2. `H2` coverage at every rung, both arms.
3. `H3`--`H7` each evaluated with evidence.
4. Per-route-record `E_Q`, decision and `sigma_min` at every rung, so a reader can
   re-derive any flip.
5. Item counts per rung, so the cost curve is explicit.
6. Sealed module hashes verified unchanged.
7. All strict-JSON, finite, shape, seed and location checks pass.
8. Complete reproducibility evidence including the sampler's hash.
9. Same-actor derived verification recorded.
10. `ACTIVE_WORKSPACE.md` updated.

## Failure criteria

The construction fails if the sampler gate fails, if a coverage violation appears at
any rung, if a sealed module's hash changes, or if a rung's numbers are reported
without its item count.

`H3`--`H7` failing is **not** a construction failure. `H6b` failing would mean the
abstainers have a hard floor that more data does not move — which would redirect the
programme away from data and back to the certificate's structure.

## Stopping conditions

Stop and report if:

- the sampler gate fails, since every number depends on it;
- a coverage violation appears at any rung;
- a sealed module's hash is found changed;
- execution would expand cost beyond the authorised scope.

## Route assignment and verification

Single actor: Claude executes and verifies. By the user's instruction of
2026-09-11 ("验证先不管"), verification is not the focus; the derived checks are
recorded for completeness and no independent verification is claimed.

## Pre-review

- Status: `APPROVED` (2026-09-12), same-actor.
- Evidence: `docs/research_branches/FP-SAMPLE-001/claude/pre_review.md`.

## Objections and user rulings

### Objection

- Status: `NONE`.

### User ruling (scope)

- Date: 2026-09-12.
- Decision: "都做" — execute both levers identified by `FP-TIGHT-001`.
- Scope: this task and `FP-RANGE-001`, on `claude/FP-CENSUS-001`.

## Definition of done

- [x] `H1` sampler gate passes.
- [x] `H2` coverage at every rung.
- [x] `H3`--`H7` each evaluated with evidence.
- [x] Cost curve reported with item counts.
- [x] Sealed module hashes verified unchanged.
- [x] Same-actor derived verification recorded.
- [x] `ACTIVE_WORKSPACE.md` updated.

## Formal outcome (2026-09-12)

Route journal: `docs/research_branches/FP-SAMPLE-001/claude/first_result.md`.

| rung | items | mean `E_Q` | vs `1x` | step-1 emitting | of the `26` frozen abstainers, revived |
|---|---:|---:|---:|---:|---:|
| `1x` | `1,048,576` | `0.1955` | — | `30` | `8` |
| `2x` | `2,097,152` | `0.1339` | `−31.5%` | `35` | `13` |
| `4x` | `4,194,304` | `0.1018` | `−47.9%` | `40` | `18` |
| **`8x`** | **`8,388,608`** | **`0.0821`** | **`−58.0%`** | **`44`** | **`22`** |

| hypothesis | verdict |
|---|---|
| `H1` sampler gate | **PASS** (worst TV `0.0091` vs gate `0.02`) |
| `H2` coverage at every rung | **PASS** (`0/192` certificate-fits, both arms) |
| `H3` agreement with the sealed `1x` | **PASS** (`0.1955` vs `0.1940`, `0.8%` apart) |
| `H4` monotone in the rung | **PASS** (strictly decreasing) |
| `H5` `4x` band | **FALSIFIED** (`−47.9%`; the band was anchored on a different baseline) |
| `H6a` a flip at `2x` or below | **PASS** |
| `H6b` `≥ 16` of `26` revived at `8x` | **PASS** (`22`) |
| `H6c` non-decreasing revival | **PASS** (`8, 13, 18, 22`) |
| `H7` diminishing returns | **PASS** (drop `−0.0321` then `−0.0197`) |

- **Certification size is the lever that works.** `2x` revives `5` more abstaining
  route-records; `8x` revives `22` of `26`. Step-1 emissions go `22/48` (frozen) →
  `30 → 35 → 40 → 44`, all with `0` coverage violations.
- A `~200x` speedup was obtained by vectorising the certification sampler, after
  which the `8x` rung becomes affordable. The sampler is validated against the
  sealed generator before the ladder is read (`H1`), because the defect it could
  hide is the same one `FP-CENSUS-001`'s batch guard caught, and it would leave the
  curve smooth and monotone with no symptom.
- **`H5` failed for a prediction-anchoring error, not a scientific one**: the band
  was anchored on `FP-TIGHT-001`'s `−64%`, which was measured against the *frozen*
  certificate and therefore contains both the repair and the size effect, while the
  ladder measures against its own repaired `1x`. Reported falsified and not
  re-scored; the sheet's rule is now "state what each number is measured against
  before writing the band".
- **The bound flattens before the outcome does**: `E_Q` gains shrink (`−0.0321` then
  `−0.0197`) while the revival count keeps climbing (`13 → 18 → 22`), because each
  further reduction moves a different part of the population across the line.
- **Four route-records never revive.** Fitting each record's own
  `eps_res = a + b/sqrt(N)`: two are priceable at `17x`–`46x` the current data; the
  other two have fitted floors (`0.0252`, `0.0272`) sitting essentially at the value
  they would need (`0.0293`), so the honest reading is **"not reachable by sample
  size"**, not a `617x` price tag. The table reports whether each fit reproduces the
  `8x` rung it was fitted on.
- **A bug caught during execution**: the first extrapolation fitted the *population
  mean* `eps_res` and applied it to individual records, which predicted a crossing
  at `6.3x` for a record already measured at `8x` without crossing. Per-record fits
  replaced it; the contradiction with a measurement already in hand was the tell.
- The size/realisation confound is stated rather than assumed away: every rung is a
  fresh independent draw, not a superset of the sealed batch.
- **Scope**: step 1 only. Nothing here claims more data extends the iteration.
