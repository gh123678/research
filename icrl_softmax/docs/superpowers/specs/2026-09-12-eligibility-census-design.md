# Eligibility census of the certified iteration — design

- Date: 2026-09-12.
- Task: `FP-CENSUS-001`.
- Baseline: `c1e03dd4e5cd610cf5b8ee0eb57c6b1de0844d0d`.
- Branch: `claude/FP-CENSUS-001`.

## 1. The question the five-step result left open

`FP-ITER5-001` established *how far* the iteration reaches. Per-level emissions on
both paths are

```text
22, 20, 15, 12, 12
```

out of `48` route-records, and the last delta is `0`: the step-4 and step-5
emitting sets are identical as sets. So the population **plateaus at `12`** while
the mean gain keeps decaying (`0.807500 → 0.361474`).

That answers a question about the horizon. It does not answer a question about the
records. The `48` route-records partition into three groups —

| group | size | behaviour |
|---|---|---|
| never emitted | `26` | abstained at step 1 and never moved |
| dropped out | `10` | emitted at step 1, stopped before step 5 |
| plateau | `12` | emitted all five steps |

— and nothing in the sealed bundles says **why** a record lands in one group
rather than another. `FP-SCALE-001` gave a one-record *observation*: emission
requires `E_Q < sigma`, the certified error to fall below the within-state
action-relevant value spread. One record is an anecdote. Five steps of trajectory
data across `48` records is a test.

## 2. The measurement

Walk the policy trajectory `FP-ITER5-001` actually produced. For each
route-record: start at `pi_0` on the frozen batch, apply the frozen decision rule,
emit when it emits, abstain when it abstains, for up to five steps. At **every**
step, including abstaining ones, record

```text
sigma_min(k) = min_s ptp_a Q^{pi_k}(s, a)     from the MDP, oracle side
E_Q(k)       = the frozen variance-adaptive certificate at pi_k
ratio(k)     = sigma_min(k) / E_Q(k)
decision(k)  = emitted, or the frozen ordered abstention reason
```

`sigma_min` is computed by `policy_quantities`, lives only under `oracle_audit`,
and is **never** a certificate input. `E_Q` is the frozen certificate itself, so
the ratio is exactly the quantity `FP-SCALE-001` named — now measured at scale,
at every step, on every record.

### Why abstaining steps are the point, not an edge case

The `26` never-emitters have no trajectory. Their step-1 values are the only data
they can contribute, and they are also the group the question is *about*. Omitting
abstaining steps would leave a census of records that already emitted, which is
circular: it would measure the ratio only where the ratio already succeeded.

For the same reason the loop records one row per **reachable policy**. A record
that abstains at step `k` has `pi_k = pi_{k-1}`, so a step `k+1` row would be a
byte-identical duplicate; the walk stops at the first abstention. This mirrors the
sealed evaluator's own loop, which also breaks on abstention, so the number of
census rows per route equals the number of sealed rows.

### `H0` first, interpretation second

The sealed `FP-ITER5-001` numpy bundle is ground truth for which record emitted at
which step. The census must reproduce it before any ratio is read. Because a
census built on a *different* certificate would be measuring a different
quantity, `H0` is deliberately stronger than "same emitted/abstained flag": the
comparison is per step on

- the emitted/abstained decision,
- `E_Q`, by exact float equality,
- the selected `eta`,
- the frozen ordered abstention reasons,

and also on the row count per route.

### The batch generator must be the sealed one

The census has to sit on the same certification batches the sealed iteration used.
`FP-SCALE-002` recorded `cert_pair_counts` and `min_cert_count_observed` per
record, which makes that checkable rather than assumed.

A vectorised per-step rewrite of the per-chain loop was tried first, to make the
census cheaper. It **failed** the check on all four probed records:

| mixing | task | vectorised min | sealed min | pair counts equal |
|---|---|---|---|---|
| `0.5` | `0` | `19531` | `19728` | no |
| `0.5` | `1` | `25167` | `25586` | no |
| `0.08` | `0` | `31673` | `31583` | no |
| `0.08` | `3` | `31871` | `32381` | no |

`Generator.choice` with an explicit `p` does not consume the underlying stream the
way a raw uniform draw does, so "equivalent for independent chains" is false at
the level of the realised sample. The census keeps the literal per-chain loop, and
the regression guard (`tmp/census_batch_check.py`) now reports **full pair-count
vector equality** on all four records.

## 3. Groups

Assignment is by the **sealed** classification, not by the census's own:

- `never`: sealed `emitted_steps == 0`;
- `plateau`: sealed `emitted_steps == 5`;
- `dropout`: `0 < emitted_steps < 5`.

Per route, not per record: a record can have one route in `plateau` and the other
in `dropout`. The census reports the `48` route-records as the population, and
notes the record-level coincidence separately.

## 4. Hypotheses

Verbatim from the task sheet.

- `H1 (step-1 separation)` — emitters' mean step-1 ratio strictly exceeds
  abstainers'.
- `H2 (separation strength)` — the two step-1 ratio distributions are strictly
  separated: a threshold exists with zero misclassification. The strong claim.
- `H3 (plateau vs drop-out)` — the `12` plateau records' mean step-1 ratio
  strictly exceeds the `10` drop-outs'.
- `H4 (ratio decay)` — along a trajectory the ratio is lower at the **last emitted
  step** than at step 1.
- `H5 (the binding term)` — for the `26` abstainers, `E_Q` is the larger term:
  their mean `E_Q` exceeds the emitters' mean `E_Q`.
- `H6 (spread is a record property)` — `sigma_min` depends on the record's MDP and
  only weakly on the current policy, so any step-1 separation is carried mainly by
  `E_Q` rather than by motion in the spread.

`H1`, `H4`, `H5` are diagnostic. `H2`, `H3`, `H6` are strong and may fail. Each is
reported `PASS` or `FALSIFIED` and never reinterpreted.

### Prior registered before the run

The exploratory pass at step 1 over the sealed bundle gave an emitted ratio range
of about `[2.17, 12.26]` and a blocked range of about `[0.57, 2.21]`. Those
intervals **overlap**. So the registered prior is:

- `H1` likely `PASS`;
- `H2` likely `FALSIFIED`, and the interesting quantity becomes the
  **misclassification count** of the best achievable threshold, not a yes/no;
- `H6` is the hypothesis that decides how to read the rest — if the spread barely
  moves while `E_Q` moves a lot, `E_Q` is the control knob and the census has found
  the actionable quantity.

This prior is recorded here so that `H2` failing is a registered outcome and not a
post-hoc rationalisation.

## 5. Threshold reporting rule

Any threshold named in the report carries its misclassification count. A threshold
with zero misclassifications may be called a classifier; one with
misclassifications is **never** described as a classifier. The best achievable
threshold and its count are reported for the step-1 ratio and, separately, for the
ratio at each step level, so the reader can see whether separation improves or
degrades as the trajectory advances.

No threshold is chosen post hoc and presented as pre-registered.

## 6. Artifacts

- `evaluate_fp_census_001.py` — the census evaluator.
- `analyze_fp_census_001.py` — `H0`--`H6`, group statistics, threshold scans.
- `verify_fp_census_001_same_actor.py` — derived verification.
- `results/FP-CENSUS-001/claude/{smoke,formal}/`.
- `docs/research_branches/FP-CENSUS-001/claude/{pre_review,first_result,verification_same_actor}.md`.

## 7. Scope note

Single actor. By the user's instruction of 2026-09-11 ("验证先不管") verification is
not the focus of this round; derived checks are recorded and no independent
verification is claimed.
