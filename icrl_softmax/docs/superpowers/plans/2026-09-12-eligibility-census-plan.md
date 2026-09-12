# Eligibility census of the certified iteration — plan

- Date: 2026-09-12.
- Task: `FP-CENSUS-001`.
- Design: `docs/superpowers/specs/2026-09-12-eligibility-census-design.md`.

## Step 0 — pin the batch generator (done, before anything else)

The census is only a census of *this* iteration if it sits on the same
certification batches. `FP-SCALE-002` sealed `cert_pair_counts` per record, so the
generator is checkable.

1. Write `tmp/census_batch_check.py` comparing a candidate generator against the
   sealed pair-count vectors for `(0.5, 0)`, `(0.5, 1)`, `(0.08, 0)`, `(0.08, 3)`.
2. First candidate: a vectorised per-step rewrite (draw all chains' actions for a
   step at once, then all next states), kept in the loop's logical order.
   **Result: FAILS on all four records** — `19531/19728`, `25167/25586`,
   `31673/31583`, `31871/32381`. `Generator.choice` with an explicit `p` does not
   consume the stream like a raw uniform draw.
3. Revert to the literal per-chain loop, byte-identical to the generator in
   `evaluate_fp_iter2_001.py`. **Result: full pair-count vector equality on all
   four records**, ~`13.4 s` per batch.
4. Keep the check script as the regression guard.

Nothing downstream is trusted until step 0 passes.

## Step 1 — evaluator

`evaluate_fp_census_001.py`:

- imports the frozen certificate, decision rule and routes; does not reimplement
  them;
- per record: one training batch, one certification batch, both built exactly as
  the sealed iteration builds them;
- per route: walk up to five steps, recording at **every** step
  `sigma_min`, `sigma_by_state`, `status`, `e_q`, `ratio`, `min_lb`, `max_lb`,
  `update_emitted`, `eta_selected`, `ordered_reasons`, and an `oracle_audit` block;
- stop at the first abstention, matching the sealed loop, so one row per reachable
  policy;
- `H0` per route: per-step comparison of decision, `e_q` (exact equality), `eta`
  and ordered reasons against the sealed numpy bundle, plus row counts.

Writes `task_results.json`, `config.json`, `environment.json`, `commands.log`.

## Step 2 — smoke

`--tasks 2 --mixings 0.08,0.5` → `8` route-records into
`results/FP-CENSUS-001/claude/smoke/`.

Gate: `H0` exact on `8/8` with `0` mismatches of any kind. Stop if not.

## Step 3 — formal run

`--tasks 12 --mixings 0.08,0.5` → `48` route-records into
`results/FP-CENSUS-001/claude/formal/`.

Gate: `H0` exact on `48/48`.

## Step 4 — analysis

`analyze_fp_census_001.py`, reading only the formal bundle and the sealed
`FP-ITER5-001` bundle:

1. **Groups.** Assign each route-record to `never` / `dropout` / `plateau` from
   the **sealed** `emitted_steps`, so the labels do not depend on this task's own
   decisions. Report sizes; also report the record-level coincidence.
2. **`H0` restatement.** Exact match count over all `48` route-records and all
   rows; any mismatch listed individually.
3. **`H1`.** Step-1 ratio: emitter mean vs abstainer mean, with per-group
   distribution (min, quartiles, max), not just means. `PASS` iff strictly higher.
4. **`H2`.** Scan every candidate threshold between consecutive sorted step-1
   ratios; report the minimum achievable misclassification count and the interval
   of thresholds attaining it. `PASS` iff that minimum is `0`. Report the count
   regardless, since it is the informative number when `H2` fails.
5. **`H3`.** Plateau vs dropout, same test as `H1`, on step-1 ratios.
6. **`H4`.** Per trajectory, ratio at the last emitted step vs step 1; report the
   fraction that fell, and the mean change. `PASS` iff the majority fell (the
   hypothesis is stated as "lower", so it is scored as a majority claim and the
   minority is listed).
7. **`H5`.** Mean `E_Q`, abstainers vs emitters, at step 1. `PASS` iff abstainers'
   is strictly larger.
8. **`H6`.** Per record, coefficient of variation of `sigma_min` across the
   trajectory's steps, compared with the coefficient of variation of `E_Q` over the
   same steps. `PASS` iff the spread's variation is smaller on average. This is the
   hypothesis that says which quantity carries the separation.
9. **Ratio by step level.** For every level `k`, the emitter/abstainer ratio
   distributions and the best threshold's misclassification count — so we can see
   whether separation improves or degrades along the trajectory.
10. **The frozen prediction for the sixth step.** From the analysis, emit
    `prediction_step6.json`: the rule the census implies for which records should
    still emit at step 6, the rule's exact statement, its in-sample
    misclassification count at step 5, and a UTC timestamp. This file is written
    **before** `FP-ITER6-001` runs and is not edited afterwards.

## Step 5 — related sixth step

`FP-ITER6-001` (see its own task sheet), run only after step 4's prediction file
exists. It extends the numpy horizon to `6` and scores the prediction against
which `12` records actually emit.

## Step 6 — derived verification

`verify_fp_census_001_same_actor.py`:

- re-run `H0` from the bundle independently of the analyzer;
- confirm `sigma_min` appears only under `oracle_audit`-equivalent scope and never
  in a certificate input (grep the evaluator for the certificate call sites);
- confirm the batch guard still passes;
- confirm the sealed `FP-ITER5-001` bundle is unmodified by hash;
- confirm the prediction file predates the `FP-ITER6-001` bundle by mtime;
- re-run the analyzers and the affected sealed verifiers with exit `0`.

## Step 7 — records and index

- `docs/research_branches/FP-CENSUS-001/claude/pre_review.md`
- `docs/research_branches/FP-CENSUS-001/claude/first_result.md`
- `docs/research_branches/FP-CENSUS-001/claude/verification_same_actor.md`
- `ACTIVE_WORKSPACE.md`, edited only in this task's own section.

## Step 8 — commit

Explicit paths only. `git add -A` is prohibited: the working tree carries another
session's documentation cleanup. Stages exactly this task's scripts, docs and task
sheet. Result bundles under `results/` stay untracked (git-ignored).

## Abort conditions

- `H0` mismatch at smoke or formal → stop, do not interpret ratios.
- Batch guard failure → stop, do not run the census.
- A sealed scientific-corpus file found modified → stop.
- The concurrent cleanup would be disturbed → stop and report.
