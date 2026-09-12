# FP-ITER5-001: A fifth certified step, on both the numpy and network paths

## Task metadata

- Created: 2026-09-11.
- Author: Claude, under the direct user instruction of 2026-09-11 ("好的第五步").
- Status: `ACTIVE`.
- Task version: `1.0`.
- Scientific baseline: `15311b8b87873c0109b4ff90cae8f481813135a3`
  (`main` after FP-ATTN-ITER4-001).
- Execution branch: `main` (single actor; see the user ruling below).
- Result directories: `results/FP-ITER5-001/claude/numpy/` and
  `results/FP-ITER5-001/claude/network/`.
- Design:
  `docs/superpowers/specs/2026-09-11-fifth-certified-step-design.md`.
- Plan:
  `docs/superpowers/plans/2026-09-11-fifth-certified-step-plan.md`.
- Classification: long, conclusion-critical, single-actor under the standing
  user exception.

## Why this task exists

Four certified steps are verified on both paths, and the network and numpy
horizons are now aligned. Two questions remain open that a fifth step answers.

**First, where does the iteration actually stop?** The emitting population is
`22 → 20 → 15 → 12` out of `48`. If step 5 continues the decline, the process is
approaching natural termination; if it plateaus, something sustains it.

**Second, does the `float32` gap stay flat?** The network-versus-numpy gap has
been `1.076e-05`, `4.585e-06`, `7.176e-06`, `1.044e-05` — flat across four
compositions. A fifth is one more opportunity for accumulation to appear. The
prior is that it stays flat, which makes this a low-surprise but still
worthwhile check: a surprise here would matter a great deal.

This round runs **both paths**, so the two remain aligned rather than the network
falling a step behind again.

## Research question

Does a fifth certified step exist on both paths, does the emitting population
continue to shrink, and does the network-versus-numpy gap remain flat?

## Falsifiable hypotheses

1. `H1 (numpy horizon inert, mandatory)`: raising the numpy horizon from `4` to
   `5` does not change numpy steps 1--4. Decisions, selected `eta`, certified
   errors and gains must equal the sealed `FP-ITER4-001` values exactly.
2. `H2 (network horizon inert, mandatory)`: raising the network horizon from `4`
   to `5` does not change network steps 1--4, including the recorded `Qhat`
   gaps, matching the sealed `FP-ATTN-ITER4-001` values exactly.
3. `H3 (fifth step certifiable on both paths)`: at least one route-record emits a
   fifth step on the numpy path, and at least one on the network path.
4. `H4 (fifth step valid)`: every emitted fifth step, on both paths, is
   componentwise non-degrading and strictly improving in total value.
5. `H5 (monotone value across five steps)`: no emitted step at any level
   degrades any state on either path.
6. `H6 (no certificate violations)`: across all five steps on both paths, every
   emitted certified error bounds the realized oracle error.
7. `H7 (pre-registered attrition prediction, both paths)`: `n5 < n4` (`12`) on
   the numpy path.
8. `H8 (pre-registered mean-gain prediction, both paths)`: the mean fifth-step
   gain is **lower** than the mean fourth-step gain (`0.807500` on the numpy
   path).
9. `H9 (pre-registered non-vacuity prediction)`: the minimum fifth-step gain is
   **strictly positive** and **not below `0.05`**.
10. `H10 (pre-registered drift prediction)`: the maximum network-versus-numpy
    `Qhat` gap at step 5 stays within the frozen `ATOL = 1e-4`.
11. `H11 (path agreement)`: network and numpy reach the same per-level emission
    counts and the same decisions at every step, with any disagreement listed
    individually.

`H1`, `H2`, `H4`, `H6` are mandatory. `H3`, `H7`--`H11` are the substantive
pre-registered predictions, each reported as PASS or FALSIFIED and never
reinterpreted.

### Basis of the predictions, stated explicitly

`H7`, `H8` and `H10` **extrapolate** observed sealed trends and are therefore
genuine predictions about unseen data.

`H9` is **not** an extrapolation of a monotone law: the observed minimum-gain
sequence `0.104197, 0.779902, 0.378197, 0.184902` is non-monotone in direction
(it fell, rose, then fell). `H9` is only a non-vacuity floor, the weakest
meaningful continuation claim.

`H11` is a **consistency** prediction rather than a trend: it asserts the two
paths stay aligned, which four steps of evidence support but do not guarantee.

## Frozen contract

### Protocol, inherited verbatim

`4` states, `3` actions, `pi_min = 0.15`, gap bonus `0.5`, mixing
`{0.08, 0.5}`, `12` tasks per mixing, training trajectory `65536` transitions,
certification `16384 x 64 = 1048576` items, `gamma = 0.70`, `alpha = 0.65`,
`160` layers, `R_star = 1.5`, `delta = 0.05`, seed `20260911`, eta grid
`1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01`, `ATOL = 1e-4`.

### The one structural change

`MAX_STEPS = 5` on both paths, in the otherwise unchanged iterations: identical
batches at every step, the same frozen certificate and decision code,
`pi_{k-1}` as the target at step `k`, nothing retuned. `H1`/`H2` exist to prove
the horizon changes are inert.

Reference baselines: the numpy path compares against the sealed `FP-ITER4-001`
bundle within itself; the network path compares against the sealed
`FP-ATTN-ITER4-001` for inertness and against this task's own numpy five-step
run for path agreement.

### Corpus integrity

**Audited before any change.** Extending the horizons touches two shared
evaluators:

| evaluator | recorded by |
|---|---|
| `evaluate_fp_iter2_001.py` | `FP-ATTN-ITER-001`, `FP-ATTN-ITER4-001` |
| `evaluate_fp_attn_iter_001.py` | no sealed bundle |

So this round's numpy-horizon change affects **two** sealed records, unlike
`FP-ITER4-001`'s single one. Following the established pattern:

- the **scientific corpus** — `fixed_policy_expected_sarsa.py`,
  `fixed_policy_expected_sarsa_scaled.py`,
  `fixed_policy_variance_certificate.py`, `model.py`,
  `verify_variance_adaptive_certificate.py` — must be byte-identical;
- the **task evaluators** evolve with the horizon, and each affected record's
  verifier must be updated to keep the science checks strict while reporting the
  evaluator evolution explicitly;
- every affected report must be **re-run and re-sealed as PASS**, never left
  reading `FAIL`;
- the change must be paired with an **inertness proof** at the previously frozen
  horizon.

## Prohibited work

- No modification of any sealed program, sealed result bundle, or closed task
  record other than the verifier updates needed to keep their integrity checks
  correct.
- No change to any frozen formula, constant, tolerance, eta grid, matrix,
  hypothesis, or metric after the run.
- No numpy substitution where the network must produce `Qhat`.
- No resampling between steps.
- No retuning to force a fifth emission.
- No reinterpretation of a falsified prediction.
- No new claim of independent or reciprocal verification.

## Acceptance criteria

1. `H1`/`H2` confirm both horizon changes inert.
2. Both paths run the same frozen matrix; per-level emissions compared.
3. `H3`--`H11` evaluated and reported, each prediction with its outcome.
4. Every emitted step on both paths passes componentwise non-degradation; every
   violation listed individually.
5. Every non-emitting step carries its frozen ordered abstention reason.
6. Batches identical across all five steps on both paths.
7. Exact truth confined to `oracle_audit`.
8. All strict-JSON, duplicate-key, finite, shape, seed and location checks pass.
9. Complete reproducibility evidence including `torch` version and dtype for the
   network path.
10. Both affected sealed records' verifiers updated, re-run, and re-sealed as
    PASS with the evaluator evolution documented.
11. `ACTIVE_WORKSPACE.md` is current.

## Failure criteria

The construction fails if either horizon change is not inert, if a step's
`Qhat` came from numpy on the network path, if batches differ between steps, if
the certificate code differs, or if a violation or disagreement is suppressed.

`H3`, `H7`--`H11` failing is **not** a construction failure. `H7` failing would
mean the population plateaus; `H10` or `H11` failing would establish that
`float32` network arithmetic finally diverges from numpy at five steps, which is
the single most interesting possible outcome here.

## Stopping conditions

Stop affected work and notify the user if:

- either path cannot complete five steps at the frozen dimensions within budget;
- a scientific-corpus file is found modified;
- the comparison would require changing a frozen tolerance or parameter;
- execution would expand cost, publication, external communication, or
  permissions beyond authorization.

## Route assignment and verification

Single actor: Claude executes and verifies. By the user's instruction of
2026-09-11 ("验证先不管"), verification is not the focus; the derived checks are
recorded for completeness and no independent verification is claimed.

## Pre-review

- Status: `APPROVED` (2026-09-11), same-actor.
- Evidence: `docs/research_branches/FP-ITER5-001/claude/pre_review.md`.

## Objections and user rulings

### Objection

- Status: `NONE`.

### User ruling (scope and emphasis)

- Date: 2026-09-11.
- Decision: "好的第五步" — proceed with a fifth certified step. The earlier
  instruction "验证先不管" (verification not the focus) still stands.
- Scope: this task.

## Definition of done

- [x] Pre-review recorded with no unresolved objection.
- [x] `H1`/`H2` confirm both horizon changes inert (`105/105` each, `0` mismatches).
- [x] `H1`--`H11` each evaluated with evidence.
- [x] Both affected sealed records' verifiers updated and re-sealed as PASS.
- [ ] `ACTIVE_WORKSPACE.md` is current.

## Formal outcome (2026-09-11)

Recorded here for the task index; the route journal at
`docs/research_branches/FP-ITER5-001/claude/first_result.md` holds the full
evidence.

| step | numpy emissions | network emissions | mean gain | minimum gain | network `max\|dQ\|` |
|---|---|---|---|---|---|
| 1 | `22` | `22` | `2.717707` | `0.104197` | `1.076e-05` |
| 2 | `20` | `20` | `2.277997` | `0.779902` | `4.585e-06` |
| 3 | `15` | `15` | `1.286906` | `0.378197` | `7.176e-06` |
| 4 | `12` | `12` | `0.807500` | `0.184902` | `1.044e-05` |
| **5** | **`12`** | **`12`** | **`0.361474`** | **`0.082395`** | **`4.567e-06`** |

- Both horizon changes **inert**: `105` entries compared per path, `0`
  mismatches (the network check includes the recorded `Qhat` gaps).
- `12` route-records emitted all five steps on **both** paths; **`0` certificate
  and `0` non-degrading violations** on either.
- **The substantive finding: the emitting population plateaus.** Emission deltas
  are `−2, −5, −3, **0**`. Verified directly: the step-4 and step-5 emitting
  sets are **identical** — nothing lost, nothing gained. Meanwhile the mean gain
  keeps decaying (`0.8075 → 0.3615`). The iteration is in a **regime, not a
  cliff**: a stable set of records keeps being certified, each step delivering
  less.
- **`H7` FALSIFIED** (`n5 = 12`, not fewer), reported as such and not
  reinterpreted. It was a legitimate extrapolation of a monotone decline across
  three intervals; the fourth interval broke it, and its failure is what produced
  the plateau finding.
- `H8` mean decay, `H9` non-vacuity (`min5 = 0.0824 > 0.05`), `H10` drift and
  `H11` path agreement all **PASS**. `H11`: both paths agree at **every** level
  and on **all `105`** decision comparisons, with `0` disagreements.
- **No accumulated `float32` drift**: the gap is flat across five compositions
  (`1.076e-05`, `4.585e-06`, `7.176e-06`, `1.044e-05`, `4.567e-06`).
- Corpus integrity: the horizon change affected **two** sealed records
  (`FP-ATTN-ITER-001`, `FP-ATTN-ITER4-001`). Both verifiers were updated to keep
  the scientific-corpus hash checks **strict** while reporting the evaluator
  evolution explicitly, and both re-run and re-sealed as **PASS**.
- **A vacuous check was found and fixed** in the four-step network verifier: its
  corpus-integrity section ended with
  `check(all(sha256(PROJECT / n) in {sha256(PROJECT / n)} for n in SCIENCE), ...)`,
  which compares each hash to itself and is therefore always true. Replaced by
  the strict per-file check plus the bounded evaluator-change check.
