# FP-CENSUS-001: Why exactly these records? An eligibility census of the certified iteration

## Task metadata

- Created: 2026-09-12.
- Author: Claude, under the direct user instruction of 2026-09-12
  ("允许你开一个新的分支去做2").
- Status: `ACTIVE`.
- Task version: `1.0`.
- Scientific baseline: `c1e03dd4e5cd610cf5b8ee0eb57c6b1de0844d0d`
  (`main` after FP-ITER5-001).
- Execution branch: `claude/FP-CENSUS-001`.
- Result directory: `results/FP-CENSUS-001/claude/`.
- Design:
  `docs/superpowers/specs/2026-09-12-eligibility-census-design.md`.
- Plan:
  `docs/superpowers/plans/2026-09-12-eligibility-census-plan.md`.
- Classification: long, conclusion-critical, single-actor under the standing
  user exception.

## Why this task exists

Five certified steps are now verified on both the numpy and network paths, and
the emitting population **plateaus at `12`** from step 4 while the mean gain
keeps decaying. So we know *how far* the iteration reaches. We do **not** know
**why these particular records**.

The `48` route-records fall into three groups:

| group | size | behaviour |
|---|---|---|
| never emitted | `26` | abstained at step 1 and never moved |
| dropped out | `10` | emitted at step 1, stopped before step 5 |
| plateau | `12` | emitted all five steps |

`FP-SCALE-001` produced an *observation* from a **single** record: emission
requires `E_Q < sigma`, the within-state action-relevant value spread. That was
never tested at scale. With five steps of trajectory data it becomes testable:
**does this criterion separate the three groups?**

This matters beyond bookkeeping. If the criterion separates them, we can predict
which records are eligible without running five steps, and "is the certificate
tight enough" becomes a quantified threshold. If it does not separate them, the
binding constraint is something else — most likely the worst certified pair or
the writeback kernel's diagonal dominance — and that is a different, equally
useful conclusion.

## Research question

Does the per-record ratio `sigma_min / E_Q` — the within-state value spread over
the certified error, evaluated at each step's actual policy — separate records
that emit from records that do not?

## Falsifiable hypotheses

1. `H1 (step-1 separation)`: at step 1, the group that emitted has a strictly
   higher mean `sigma_min / E_Q` than the group that abstained.
2. `H2 (separation strength)`: the step-1 ratio distributions of emitters and
   abstainers are **strictly separated**, i.e. a threshold exists with no
   misclassification. This is the strong claim. `H2` failing while `H1` passes
   means the criterion is directionally right but not a classifier.
3. `H3 (plateau vs drop-out)`: the `12` plateau records have a strictly higher
   step-1 ratio than the `10` drop-outs. If true, the step-1 ratio predicts not
   only whether a record starts but how long it lasts.
4. `H4 (ratio decay)`: along each trajectory the ratio is **lower** at the last
   emitted step than at step 1, i.e. iteration consumes the margin.
5. `H5 (the binding term)`: for the `26` abstainers, `E_Q` is the larger term —
   their mean `E_Q` exceeds the emitters' mean `E_Q`. This distinguishes "the
   certificate is too loose" from "the policy improvement is too small".
6. `H6 (spread is a record property)`: `sigma_min` depends on the record's MDP
   and only weakly on the current policy, so the separation in `H1`, if present,
   is carried mainly by `E_Q` rather than by the spread motion.

`H1`, `H4`, `H5` are diagnostic claims. `H2` and `H3` are the strong claims and
may fail; each failure is a substantive result and is reported as `PASS` or
`FALSIFIED`, never reinterpreted.

## Frozen contract

### Protocol, inherited verbatim

`4` states, `3` actions, `pi_min = 0.15`, gap bonus `0.5`, mixing
`{0.08, 0.5}`, `12` tasks per mixing, training trajectory `65536` transitions,
certification `16384 x 64 = 1048576` items, `gamma = 0.70`, `alpha = 0.65`,
`160` layers, `R_star = 1.5`, `delta = 0.05`, seed `20260911`, eta grid
`1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01`, `MAX_STEPS = 5`.

### The measurement

For each of the `48` route-records, walk the **policy trajectory** that
FP-ITER5-001 actually produced: from `pi_0` at the frozen batch, emitting when
the frozen decision rule emits, abstaining otherwise, for up to five steps. At
**every** step, including abstaining ones, record:

```text
sigma_min(k) = min_s ptp_a Q^{pi_k}(s, a)        (from the MDP, oracle side)
E_Q(k)       = the frozen variance-adaptive certificate at pi_k
ratio(k)     = sigma_min(k) / E_Q(k)
decision(k)  = emitted or the frozen ordered abstention reason
```

Two properties make this a clean census:

- `sigma_min` is a property of the **true** MDP and the policy, computed from
  `policy_quantities` inside `oracle_audit` only. It is never a certificate
  input; it is the diagnostic yardstick.
- `E_Q` is exactly the frozen certificate the iteration used, so the ratio is
  the quantity `FP-SCALE-001` named, now measured at scale.

Abstaining steps are **included**. That is the point: the `26` abstainers have no
trajectory to walk, so their step-1 values are the only data they contribute, and
omitting them would make the census circular.

### Reference

The sealed `FP-ITER5-001` numpy bundle is the ground truth for which records
emitted at which step. This task reproduces that classification exactly
(`H0`, a mandatory check) before interpreting any ratio.

## Prohibited work

- No modification of any sealed program, sealed result bundle, or closed task
  record.
- No change to any frozen formula, constant, tolerance, eta grid, matrix,
  hypothesis, or metric after the run.
- No use of `sigma_min` or any oracle quantity as a certificate input.
- No post-hoc threshold selection presented as a pre-registered rule.
- No reinterpretation of a falsified hypothesis.
- No `git add -A`: the repository has a concurrently active documentation
  cleanup in the working tree, and only this task's own paths may be staged.
- No new claim of independent or reciprocal verification.

## Acceptance criteria

1. `H0`: the reproduced emission classification matches the sealed
   `FP-ITER5-001` numpy bundle for all `48` route-records and all five steps.
2. Every step of every trajectory carries `sigma_min`, `E_Q`, the ratio, and the
   frozen decision or reason.
3. `H1`--`H6` each evaluated with evidence and reported as `PASS` or
   `FALSIFIED`.
4. The three groups (never / dropped out / plateau) are reported with their size,
   mean and distribution of the step-1 ratio.
5. Any claimed threshold is accompanied by its misclassification count; a
   threshold with zero misclassifications is stated as such, and one with
   misclassifications is never described as a classifier.
6. Oracle quantities confined to `oracle_audit`.
7. All strict-JSON, duplicate-key, finite, shape, seed and location checks pass.
8. Complete reproducibility evidence.
9. `ACTIVE_WORKSPACE.md` is updated in a way that does not disturb the
   concurrent documentation cleanup.
10. Same-actor derived verification recorded.

## Failure criteria

The construction fails if `H0` fails, if an oracle quantity leaks into a
certificate input, if abstaining steps are omitted from the census, if a
threshold is reported without its misclassification count, or if a falsified
hypothesis is reinterpreted.

`H2`, `H3` or `H6` failing is **not** a construction failure. `H2` failing means
the criterion is directional but not a classifier; `H3` failing means the step-1
ratio does not predict persistence; `H6` failing means the spread moves with the
policy. Each is a substantive finding.

## Stopping conditions

Stop affected work and notify the user if:

- `H0` fails, since the census would then not describe the sealed iteration;
- a sealed file is found modified;
- the census would require changing a frozen parameter or tolerance;
- the concurrent documentation cleanup would be disturbed by this task's
  commits;
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

### User ruling (scope)

- Date: 2026-09-12.
- Decision: the documentation cleanup by the other session is complete; proceed
  with the eligibility census, on a new branch.
- Scope: this task, on `claude/FP-CENSUS-001`.

## Definition of done

- [x] Pre-review recorded with no unresolved objection.
- [x] `H0` reproduction of the sealed classification passes (`48/48` exact,
      `0` mismatches of decision, `E_q`, `eta` or ordered reasons).
- [x] `H1`--`H6` each evaluated with evidence.
- [x] Three-group census reported with distributions, not just means.
- [x] Same-actor derived verification recorded.
- [~] `ACTIVE_WORKSPACE.md` updated in the working tree without disturbing the
      concurrent documentation cleanup, but deliberately **not staged**: the file
      carries another actor's uncommitted rewrite, and `AGENTS.md` forbids
      importing an outside actor's uncommitted changes into this task's commit.
      Committing the index update requires the user's ruling.

## Formal outcome (2026-09-12)

Recorded here for the task index; the route journal at
`docs/research_branches/FP-CENSUS-001/claude/first_result.md` holds the full
evidence.

| hypothesis | verdict | evidence |
|---|---|---|
| `H0` sealed reproduction | **PASS** | `48/48` route-records exact; `0` mismatches |
| `H1` step-1 separation | **PASS** | emitters `4.1424` vs abstainers `1.3237` |
| `H2` separation strength | **FALSIFIED** | best of `49` thresholds misclassifies `1/48` (`2.1%`) |
| `H3` plateau vs drop-out | **PASS** | `5.0171` vs `3.0927` |
| `H4` ratio decay | **PASS** (majority) / **FALSIFIED** (literal) | `12/20` fell; the task sheet's universal phrasing fails |
| `H5` the binding term | **PASS** | `E_Q` `0.2630` vs `0.2185` |
| `H6` spread is a record property | **PASS** | mean CV `0.0177` vs `0.0264` |

- **Every abstention in the census has the same reason.** All `48` abstaining
  rows — every group, every level — report exactly one frozen reason,
  `improvement_lcb_nonpositive`. No record ever fails certification.
- **The discriminator is the spread, not the error.** `sigma_min` differs by
  `2.6×` between the groups, `E_Q` by `1.2×`; `E_Q` alone misclassifies `16/48`
  as an abstention predictor versus `1/48` for the ratio.
- `H2` is reported as a near miss, not as a classifier: the groups are
  **interleaved** (`lowest emitter 2.1721`, `highest abstainer 2.2080`).
- The step-5 level scan is reported as **vacuous** (one class only), not as a
  perfect `0/12`.
- **A batch-generator defect was caught before the census ran.** A vectorised
  rewrite of the per-chain draw produced different pair counts on all four probed
  records (`19531/19728`, `25167/25586`, `31673/31583`, `31871/32381`); the
  literal loop was restored and is now guarded by
  `verify_census_batch_frozen.py`.
- The frozen step-6 prediction was written at `2026-09-12T11:23:59Z`
  (sha256 `70b79f03…`), before `FP-ITER6-001` ran, and was not edited afterwards.
