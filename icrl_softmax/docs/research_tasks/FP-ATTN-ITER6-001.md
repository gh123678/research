# FP-ATTN-ITER6-001: A sixth certified step on the literal-network path

## Task metadata

- Created: 2026-09-12.
- Author: Claude, under the direct user instruction of 2026-09-12: asked to name
  the outstanding work, the user selected item 2, "the network path is one step
  behind".
- Status: `ACTIVE`.
- Task version: `1.0`.
- Scientific baseline: `0faffbd` (`claude/FP-CENSUS-001` after the census and the
  numpy sixth step).
- Execution branch: `claude/FP-CENSUS-001` (continues the same combined line of
  work; see the user ruling).
- Result directory: `results/FP-ATTN-ITER6-001/claude/network/`.
- Design and plan: this sheet, section 5.
- Classification: long, conclusion-critical, single-actor under the standing
  user exception.

## Why this task exists

`FP-ITER6-001` ran the sixth certified step on the **numpy** path only, and its
task sheet recorded the consequence honestly:

> The network path is **not** run in this task. [...] The consequence is recorded
> honestly: after this task the numpy path is one step ahead of the network path,
> and the network six-step run remains outstanding.

This task closes that gap. Until it is done, the headline claim of the whole line
— that a fixed-weight softmax attention network supports certified, non-degrading
policy improvement — rests on a five-step network result while the numpy path has
reached six. The two paths have agreed at every horizon so far, so the prior is
that they agree again; but "the network kept up" is exactly the kind of claim that
must be measured rather than assumed.

## Research question

Does the literal attention network carry a sixth certified step, does the
network-versus-numpy gap stay inside the frozen tolerance at six steps, and do the
two paths reach the same emitting set at step 6?

## Falsifiable hypotheses

1. `H1 (network horizon inert, mandatory)`: raising the network horizon from `5`
   to `6` does not change network steps 1--5. Decisions, selected `eta`,
   certified errors **and the recorded `Qhat` gaps** must equal the sealed
   `FP-ITER5-001` network values exactly.
2. `H2 (sixth step certifiable on the network path)`: at least one route-record
   emits a sixth step, with `Qhat` produced by the literal network.
3. `H3 (sixth step valid)`: every emitted sixth step is componentwise
   non-degrading and strictly improving in total value.
4. `H4 (no certificate violations)`: across all six steps, every emitted
   certified error bounds the realized oracle error.
5. `H5 (path agreement at step 6)`: network and numpy reach the same number of
   sixth-step emissions **and** the same set of emitting route-records.
6. `H6 (pre-registered drift prediction)`: the maximum network-versus-numpy
   `Qhat` gap at step 6 stays within the frozen `ATOL = 1e-4`. The gap sequence is
   `1.076e-05, 4.585e-06, 7.176e-06, 1.044e-05, 4.567e-06` — flat across five
   compositions and **not** monotonically growing, so this predicts no
   accumulation rather than extrapolating a trend.
7. `H7 (no decision flips)`: zero decision flips and zero `eta` flips against the
   numpy comparator at every step of the six-step run.
8. `H8 (pre-registered attrition)`: the network sixth-step emission count equals
   the numpy sixth-step count, `9`. This is a **consistency** prediction, not a
   trend: it asserts the two implementations agree on the unseen step, which five
   steps of agreement support but do not guarantee.
9. `H9 (mean-gain decay on the network path)`: the network's mean sixth-step gain
   is lower than its mean fifth-step gain.

`H1`, `H3`, `H4` are mandatory. `H2`, `H5`--`H9` are the substantive
pre-registered predictions, each reported `PASS` or `FALSIFIED` and never
reinterpreted.

### Basis of the predictions, stated explicitly

`H6` is a **flatness** prediction, not an extrapolation of growth or decay: the
five observed gaps are non-monotone, and the claim is only that the sixth stays
inside tolerance. `H8` is a cross-path consistency claim. Neither is a
continuation of a fitted law, and a failure of either would be the single most
interesting outcome available here — it would establish that `float32` network
arithmetic finally diverges from numpy at six compositions.

## Frozen contract

### Protocol, inherited verbatim

`4` states, `3` actions, `pi_min = 0.15`, gap bonus `0.5`, mixing
`{0.08, 0.5}`, `12` tasks per mixing, training trajectory `65536` transitions,
certification `16384 x 64 = 1048576` items, `gamma = 0.70`, `alpha = 0.65`,
`160` layers, `R_star = 1.5`, `delta = 0.05`, seed `20260911`, eta grid
`1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01`, `ATOL = 1e-4`.

### The one structural change

`--max-steps 6` on the otherwise unchanged literal-network iteration
(`evaluate_fp_attn_iter_001.py`): identical batches at every step, the same frozen
certificate and decision code, `pi_{k-1}` as the target at step `k`, nothing
retuned. The network produces every step's `Qhat`; the numpy route is computed
alongside purely as the comparison baseline, and the per-step provenance flag
`qhat_producer` records which producer was used.

The `--reference` bundle is set to `iter6`, this line's numpy six-step run, so
that step 6 is compared against a numpy horizon of the same length. That is the
only apples-to-apples path comparison; comparing a six-step network run against a
five-step numpy bundle would silently drop the step that matters.

### Corpus integrity

**Audited before the change.** Extending the network horizon touches one shared
evaluator:

| evaluator | recorded by |
|---|---|
| `evaluate_fp_attn_iter_001.py` | no sealed bundle |

It records no sealed bundle of its own, so no earlier record's hash is affected by
this change. `evaluate_fp_iter2_001.py` — which **is** recorded by
`FP-ATTN-ITER-001` and `FP-ATTN-ITER4-001` — is **not** touched by this task; it
was already extended once by `FP-ITER6-001`, and that evolution is documented in
those records' verifiers.

Following the established pattern:

- the **scientific corpus** — `fixed_policy_expected_sarsa.py`,
  `fixed_policy_expected_sarsa_scaled.py`,
  `fixed_policy_variance_certificate.py`, `model.py`,
  `verify_variance_adaptive_certificate.py` — must be byte-identical;
- the change must be paired with an **inertness proof** at the previously frozen
  horizon (`H1`), including the recorded `Qhat` gaps;
- any affected verifier must be **re-run and re-sealed as PASS**, never left
  reading `FAIL`.

## Prohibited work

- No modification of any sealed program, sealed result bundle, or closed task
  record other than verifier updates needed to keep their integrity checks
  correct.
- No change to any frozen formula, constant, tolerance, eta grid, matrix,
  hypothesis, or metric after the run.
- No numpy substitution for any `Qhat` on the network path. The numpy comparator
  exists only to be compared against; `H1` and the provenance flags exist to prove
  it never drives a decision.
- No resampling between steps.
- No retuning to force a sixth emission or to close the path gap.
- No reinterpretation of a falsified prediction.
- No new claim of independent or reciprocal verification.
- No `git add -A`.

## Acceptance criteria

1. `H1` confirms the network horizon change inert, including the recorded `Qhat`
   gaps on all sealed rows of levels 1--5.
2. Every step's `Qhat` on the network path carries
   `qhat_producer = literal_attention_network`.
3. `H2`--`H9` evaluated and reported, each prediction with its outcome.
4. Every emitted sixth step passes componentwise non-degradation; every violation
   listed individually.
5. Every non-emitting step carries its frozen ordered abstention reason.
6. Batches identical across all six steps.
7. Exact truth confined to `oracle_audit`.
8. Network-versus-numpy decision and `eta` flips listed individually; a suppressed
   flip is a construction failure.
9. All strict-JSON, duplicate-key, finite, shape, seed and location checks pass.
10. Complete reproducibility evidence including `torch` version and dtype.
11. Affected verifiers re-run and re-sealed as PASS.
12. `ACTIVE_WORKSPACE.md` is current.

## Failure criteria

The construction fails if the horizon change is not inert, if any step's `Qhat`
came from numpy on the network path, if batches differ between steps, if the
certificate code differs, or if a violation or flip is suppressed.

`H2`, `H5`--`H9` failing is **not** a construction failure. `H6`, `H5` or `H8`
failing would establish that `float32` network arithmetic finally diverges from
numpy at six steps, which is the most informative possible outcome of this task.

## Stopping conditions

Stop affected work and notify the user if:

- the network path cannot complete six steps at the frozen dimensions within
  budget;
- a scientific-corpus file is found modified;
- the comparison would require changing a frozen tolerance or parameter;
- any step would have to be scored on a numpy `Qhat`;
- execution would expand cost, publication, external communication, or
  permissions beyond authorization.

## Route assignment and verification

Single actor: Claude executes and verifies. By the user's instruction of
2026-09-11 ("验证先不管"), verification is not the focus; the derived checks are
recorded for completeness and no independent verification is claimed.

## Pre-review

- Status: `APPROVED` (2026-09-12), same-actor.
- Evidence: `docs/research_branches/FP-ATTN-ITER6-001/claude/pre_review.md`, with
  three mandatory conditions: the comparison reference must be the matching
  six-step numpy bundle, the recorded decision must be proven to use the network's
  `Qhat`, and the inertness proof must include the recorded `Qhat` gaps.

## Objections and user rulings

### Objection

- Status: `NONE`.

### User ruling (scope)

- Date: 2026-09-12.
- Decision: the user selected "the network path is one step behind" from the list
  of outstanding work and instructed that it be closed.
- Scope: this task, on `claude/FP-CENSUS-001`.

## Definition of done

- [x] `H1` confirms the network horizon change inert.
- [x] `H2`--`H9` each evaluated with evidence.
- [x] Path agreement at step 6 reported as counts **and** as a set comparison.
- [x] Zero decision and `eta` flips, listed individually if any.
- [x] Same-actor derived verification recorded.
- [x] Affected verifiers re-run and re-sealed as PASS.
- [x] `ACTIVE_WORKSPACE.md` updated.

## Formal outcome (2026-09-12)

Recorded here for the task index; the route journal at
`docs/research_branches/FP-ATTN-ITER6-001/claude/first_result.md` holds the full
evidence. Torch `2.11.0+cpu`; formal run `8 min 25 s` for `24` records.

| step | emissions | mean gain | minimum gain | max `\|dQ\|` vs numpy |
|---|---:|---:|---:|---:|
| 1 | `22` | `2.717707` | `0.104197` | `1.076e-05` |
| 2 | `20` | `2.277997` | `0.779902` | `4.585e-06` |
| 3 | `15` | `1.286905` | `0.378197` | `7.176e-06` |
| 4 | `12` | `0.807499` | `0.184902` | `1.044e-05` |
| 5 | `12` | `0.361474` | `0.082395` | `4.567e-06` |
| **6** | **`9`** | **`0.167057`** | **`0.019347`** | **`6.814e-06`** |

- **Every hypothesis passed.** `H1` inert on all `117` sealed route-step rows with
  `0` decision, `eta`, `E_Q` or reason mismatches **and `0` mismatches in the
  recorded `Qhat` gaps** — the gaps are float32 artifacts, so reproducing them
  exactly is a far stronger inertness statement than decision agreement alone.
- **The network is back in step with numpy.** Both paths emit
  `22, 20, 15, 12, 12, 9`, and the step-6 emitting sets are **identical**: `9`
  shared, `0` numpy-only, `0` network-only. `129` step entries, `0` decision flips,
  `0` `eta` flips.
- The plateau break reproduces on the network path: deltas `−2, −5, −3, 0, −3`,
  mean `0.361474 → 0.167057`, minimum `0.082395 → 0.019347` — below the same
  `0.05` floor.
- **No accumulating drift.** Gaps `1.076e-05, 4.585e-06, 7.176e-06, 1.044e-05,
  4.567e-06, 6.814e-06`, all within `ATOL = 1e-4`; the sixth is the second
  smallest. `H6` was registered as a flatness claim, not a trend.
- The shipped corpus bound holds: only `evaluate_fp_iter2_001.py` differs from the
  `FP-ITER5-001` record, and that change is `FP-ITER6-001`'s documented horizon
  extension. `evaluate_fp_attn_iter_001.py` is recorded by no sealed bundle.
- Two analyzer defects were found and fixed during this task: a `config["reference"]`
  check that the evaluator does not populate (vacuously true) was replaced by a
  check on the bundle's comparison depth, and the per-level `max |dQ|` aggregate was
  changed to cover **all** rows at a level rather than only emitting rows, so that
  levels 1--5 could be checked against the sealed headline figures. The corrected
  column reproduces them exactly.
- **Stated plainly**: mean gains differ between the paths in the sixth decimal at
  steps 3 and 4 (`1.286905`/`0.807499` vs `1.286906`/`0.807500`) because
  `policy_plus` is a function of `Qhat`. Path agreement must therefore be judged on
  decisions and emitting sets, not on gain figures.
- The anti-substitution guard is non-vacuous: the smallest recorded gap over all
  steps is `6.319e-08`, so no step's `Qhat` was silently replaced by numpy's.
