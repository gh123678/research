# FP-ITER6-001: A sixth certified step, scored against the census prediction

## Task metadata

- Created: 2026-09-12.
- Author: Claude, under the direct user instruction of 2026-09-12
  ("我的意思是1和2 一起做" — the eligibility census and the sixth certified step are
  one combined task).
- Status: `ACTIVE`.
- Task version: `1.0`.
- Scientific baseline: `c1e03dd4e5cd610cf5b8ee0eb57c6b1de0844d0d`
  (`main` after FP-ITER5-001).
- Execution branch: `claude/FP-CENSUS-001` (combined task; see the user ruling).
- Result directory: `results/FP-ITER6-001/claude/numpy/`.
- Design: `docs/superpowers/specs/2026-09-12-eligibility-census-design.md`,
  section 5 of which registers the prediction.
- Plan: `docs/superpowers/plans/2026-09-12-eligibility-census-plan.md`, step 5.
- Classification: long, conclusion-critical, single-actor under the standing
  user exception.

## Why this task exists

`FP-ITER5-001` found a **plateau**: emission deltas `−2, −5, −3, 0`, with the
step-4 and step-5 emitting sets identical as sets, while the mean gain keeps
decaying. Five steps tell us the process is in a regime rather than approaching a
cliff. They do not tell us whether the regime is stable, and they do not test any
explanation of it.

`FP-CENSUS-001` supplies an explanation in the form of a measured quantity: the
ratio of the within-state action-relevant value spread to the certified error,
`sigma_min(k) / E_Q(k)`, recorded at every step of every trajectory. This task
turns that explanation into a **prediction** and then runs the experiment that
scores it.

The order matters and is enforced: `FP-CENSUS-001` writes
`prediction_step6.json` with a UTC timestamp and the frozen threshold before this
task is executed. The prediction is not edited afterwards. This is the difference
between a census that describes the past and a criterion that predicts the future.

## Research question

Does a sixth certified step exist on the numpy path, and does the census's step-1
eligibility criterion predict which records reach it — better than the trivial
"the plateau carries on" extrapolation?

## Falsifiable hypotheses

1. `H1 (horizon inert, mandatory)`: raising the numpy horizon from `5` to `6` does
   not change numpy steps 1--5. Decisions, selected `eta` and certified errors must
   equal the sealed `FP-ITER5-001` values exactly.
2. `H2 (sixth step certifiable)`: at least one route-record emits a sixth step.
3. `H3 (sixth step valid)`: every emitted sixth step is componentwise
   non-degrading and strictly improving in total value.
4. `H4 (no certificate violations)`: across all six steps, every emitted certified
   error bounds the realized oracle error.
5. `H5 (pre-registered prediction P1, "the plateau carries on")`: every
   route-record that emitted at step 5 also emits at step 6, so `n6 = 12` **and**
   the emitting set is unchanged.
6. `H6 (pre-registered prediction P2, "the step-1 ratio threshold")`: with `theta`
   frozen from the census's step-1 scan and applied to the step-5 ratio, the rule
   `ratio(5) > theta` predicts step-6 emission with **fewer misclassifications
   than P1**.
7. `H7 (pre-registered attrition)` and `H7b (pre-registered plateau)`: two
   competing, mutually exclusive claims — `H7`: `n6 < 12`; `H7b`: `n6 = 12` with
   the identical emitting set. Exactly one can hold; both are reported, and one is
   `FALSIFIED`. `H7` is the same extrapolation that `FP-ITER5-001`'s `H7` already
   falsified once, re-registered rather than quietly dropped.
8. `H8 (pre-registered mean-gain decay)`: the mean sixth-step gain is **lower**
   than the mean fifth-step gain (`0.361474`).
9. `H9 (pre-registered non-vacuity)`: the minimum sixth-step gain is **strictly
   positive** and **not below `0.05`** — the same floor `FP-ITER5-001` registered,
   not a loosened one.
10. `H10 (regime stability)`: the emitting-set membership at step 6, compared with
    step 5, is reported as an explicit set difference (gained, lost, retained).
    This is a reporting obligation, not a directional claim.

`H1`, `H3`, `H4` are mandatory. `H2`, `H5`--`H9` are the substantive
pre-registered predictions. Each is reported `PASS` or `FALSIFIED` and never
reinterpreted.

### Basis of the predictions, stated explicitly

`H8` extrapolates a strictly monotone decay across four intervals
(`2.717707, 2.277997, 1.286906, 0.807500, 0.361474`). `H9` extrapolates a
**non-monotone** minimum-gain sequence (`0.104197, 0.779902, 0.378197, 0.184902,
0.082395`) and is registered as a floor, not a trend — the sequence is close to it
and this hypothesis may well fail, which would be a substantive finding about how
thin the certified margin has become.

`H5` and `H6` are the point of the combined task: they are competing predictors of
the same unseen event. `H5` is the trivial one, supported by one interval of
evidence. `H6` is the census's substantive claim. If `H6` fails while `H5` passes,
the ratio is descriptive of the past five steps but not predictive of the sixth —
a clean and useful negative result, and not a reinterpretation of anything.

## Frozen contract

### Protocol, inherited verbatim

`4` states, `3` actions, `pi_min = 0.15`, gap bonus `0.5`, mixing
`{0.08, 0.5}`, `12` tasks per mixing, training trajectory `65536` transitions,
certification `16384 x 64 = 1048576` items, `gamma = 0.70`, `alpha = 0.65`,
`160` layers, `R_star = 1.5`, `delta = 0.05`, seed `20260911`, eta grid
`1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01`.

### The one structural change

`--max-steps 6` on the otherwise unchanged numpy iteration
(`evaluate_fp_iter2_001.py`): identical batches at every step, the same frozen
certificate and decision code, `pi_{k-1}` as the target at step `k`, nothing
retuned. `H1` exists to prove the horizon change is inert.

The network path is **not** run in this task. The user's instruction scopes the
sixth step to the numpy path, and extending `evaluate_fp_attn_iter_001.py` is a
separate change with its own sealed records. The consequence is recorded honestly:
after this task the numpy path is one step ahead of the network path, and the
network six-step run remains outstanding.

### Corpus integrity

**Audited before the change.** Extending the numpy horizon touches one shared
evaluator:

| evaluator | recorded by |
|---|---|
| `evaluate_fp_iter2_001.py` | `FP-ATTN-ITER-001`, `FP-ATTN-ITER4-001` |

`evaluate_fp_attn_iter_001.py` records no sealed bundle and is untouched.

Following the established pattern:

- the **scientific corpus** — `fixed_policy_expected_sarsa.py`,
  `fixed_policy_expected_sarsa_scaled.py`,
  `fixed_policy_variance_certificate.py`, `model.py`,
  `verify_variance_adaptive_certificate.py` — must be byte-identical;
- the **task evaluator** evolves with the horizon, and the affected records'
  verifiers keep their science checks strict while bounding the evaluator change
  to exactly that one file;
- every affected verifier must be **re-run and re-sealed as PASS**, never left
  reading `FAIL`;
- the change must be paired with an **inertness proof** at the previously frozen
  horizon (`H1`).

## Prohibited work

- No modification of any sealed program, sealed result bundle, or closed task
  record other than the verifier updates needed to keep their integrity checks
  correct.
- No change to any frozen formula, constant, tolerance, eta grid, matrix,
  hypothesis, or metric after the run.
- No numpy substitution of any kind on a path that must produce `Qhat`.
- No resampling between steps.
- No retuning to force a sixth emission.
- No editing of `prediction_step6.json` after it is written.
- No reinterpretation of a falsified prediction.
- No claim that the sixth step is verified on the network path.
- No new claim of independent or reciprocal verification.
- No `git add -A`.

## Acceptance criteria

1. `H1` confirms the horizon change inert on all `105` step entries.
2. `H2`--`H9` evaluated and reported, each prediction with its outcome.
3. Every emitted sixth step passes componentwise non-degradation; every violation
   listed individually.
4. Every non-emitting step carries its frozen ordered abstention reason.
5. Batches identical across all six steps.
6. Exact truth confined to `oracle_audit`.
7. Both pre-registered rules scored with their misclassification counts over the
   population that reached step 5; `prediction_step6.json` reproduced by hash and
   shown to predate this task's bundle.
8. All strict-JSON, duplicate-key, finite, shape, seed and location checks pass.
9. Complete reproducibility evidence.
10. Both affected sealed records' verifiers re-run and re-sealed as PASS with the
    evaluator evolution documented.
11. `ACTIVE_WORKSPACE.md` updated without disturbing the concurrent cleanup.

## Failure criteria

The construction fails if the horizon change is not inert, if a step's `Qhat` came
from a substituted source, if batches differ between steps, if the certificate
code differs, if `prediction_step6.json` was edited after the fact, or if a
violation or disagreement is suppressed.

`H2`, `H5`--`H9` failing is **not** a construction failure. `H5` failing while the
census's `H2` passed would be the strongest negative result available here: the
criterion separates the first five steps but does not predict the sixth.

## Stopping conditions

Stop affected work and notify the user if:

- six steps cannot complete at the frozen dimensions within budget;
- a scientific-corpus file is found modified;
- the comparison would require changing a frozen tolerance or parameter;
- `prediction_step6.json` does not exist before this task is run;
- execution would expand cost, publication, external communication, or
  permissions beyond authorization.

## Route assignment and verification

Single actor: Claude executes and verifies. By the user's instruction of
2026-09-11 ("验证先不管"), verification is not the focus; the derived checks are
recorded for completeness and no independent verification is claimed.

## Pre-review

- Status: `APPROVED` (2026-09-12), same-actor.
- Evidence: `docs/research_branches/FP-CENSUS-001/claude/pre_review.md`.

## Objections and user rulings

### Objection

- Status: `NONE`.

### User ruling (combined scope)

- Date: 2026-09-12.
- Decision: "我的意思是1和2 一起做" — the eligibility census and the sixth
  certified step are executed as one combined task, on the new branch.
- Scope: this task and `FP-CENSUS-001`, on `claude/FP-CENSUS-001`.

## Definition of done

- [x] `prediction_step6.json` frozen before the run.
- [x] `H1` confirms the horizon change inert.
- [x] `H2`--`H9` each evaluated with evidence.
- [x] Both pre-registered rules scored against the sixth-step outcome.
- [x] Affected sealed records' verifiers re-run and re-sealed as PASS.
- [x] Same-actor derived verification recorded.
- [~] `ACTIVE_WORKSPACE.md` updated in the working tree but deliberately **not
      staged**: it carries another actor's uncommitted rewrite, and `AGENTS.md`
      forbids importing an outside actor's uncommitted changes into this task's
      commit. See `FP-CENSUS-001.md` for the same note.

## Formal outcome (2026-09-12)

Recorded here for the task index; the route journal at
`docs/research_branches/FP-CENSUS-001/claude/first_result.md` holds the full
evidence.

| step | emissions | mean gain | minimum gain |
|---|---:|---:|---:|
| 1 | `22` | `2.717707` | `0.104197` |
| 2 | `20` | `2.277997` | `0.779902` |
| 3 | `15` | `1.286906` | `0.378197` |
| 4 | `12` | `0.807500` | `0.184902` |
| 5 | `12` | `0.361474` | `0.082395` |
| **6** | **`9`** | **`0.167057`** | **`0.019347`** |

- `H1` **PASS**: all `117` sealed route-step rows of levels 1--5 reproduced
  exactly. (`105` is levels 1--4; that constant produced a spurious failure in the
  first version of the check and was replaced by a computed expected count.)
- **The plateau broke.** Emission deltas `−2, −5, −3, 0, **−3**`. Retained `9`,
  lost `3`, gained `0`. `FP-ITER5-001`'s plateau was one interval, not a fixed
  point.
- `H7` (`n6 < n5`) **PASS** — the attrition prediction that step 5 had falsified
  now holds. `H7b` **FALSIFIED**. Both were registered so that one would have to
  be reported as lost.
- `H8` mean decay **PASS** (`0.167057 < 0.361474`). `H9` non-vacuity floor
  **FALSIFIED**: `min6 = 0.019347 < 0.05`, the same floor steps 4 and 5 cleared.
  The minimum gain fell by a factor of `4.3` in one step.
- `H5` and `H6` **FALSIFIED**. Both registered rules predicted `12` emitters and
  misclassified the same `3`. This was a **degenerate test**: the scoring
  population is exactly the `12` step-5 emitters, whose step-5 ratios span
  `[2.5922, 13.1850]`, all far above `theta = 2.1687`. The degeneracy was
  detectable when the prediction was written and was not caught; it is recorded as
  a defect in this task's design, not excused.
- **Why the break was unpredictable from the criterion**: post-hoc and labelled as
  such, the best threshold on the step-5 ratio still misclassifies `1/12`, and two
  of the three drop-outs had step-5 ratios (`3.1312`, `3.2501`) **above** the
  lowest surviving record (`2.6846`). No threshold on this feature separates them.
- `H2`--`H4` **PASS**: `9` sixth-step emissions, all componentwise non-degrading
  and strictly improving, `0` certificate violations over six steps, every
  abstention carrying a frozen reason.
- The network path was **not** run; it is now one step behind the numpy path.
- Construction checks **PASS**. The analyzer reports falsified hypotheses on a
  separate line from failed construction checks.
