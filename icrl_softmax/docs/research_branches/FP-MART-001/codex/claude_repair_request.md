# FP-MART-001 Claude author-repair request

## Authority and scope

The frozen task remains valid, has no objection, and has returned to `ACTIVE`
after GPT verification of Claude commit
`c5da2430d00e11414723d85cf42293dddae07183` ended in `FAIL`.

Read, in order:

1. the repository `AGENTS.md`;
2. `icrl_softmax/ACTIVE_WORKSPACE.md`;
3. the frozen task, design, and plan;
4. Claude's sealed files at commit `c5da2430`;
5. GPT's verification report at
   `C:\Users\Admin\Desktop\research\icrl_softmax\docs\research_branches\FP-MART-001\codex\verification_of_claude.md`.

This is an ordinary author repair, not permission to change the task. Work only
on `claude/FP-MART-001` in
`C:\tmp\research-FP-MART-001-claude` and only in Claude's already authorized
files/evidence directory plus
`C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-MART-001\claude`.
Do not edit the task, design, plan, `ACTIVE_WORKSPACE.md`, GPT files, GPT
results, old results, or any other branch. Do not merge, push, publish, send
messages, or use unsafe permission bypasses. Do not copy GPT's implementation;
repair Claude's independent construction using the verification findings.

## Mandatory repairs

1. Replace the invalid raw MGF iteration in Claude `theory.md` with the
   compensated exponential-supermartingale proof. Explicitly prove
   `E[M_n] <= 1`, apply `sum J <= k`, and recover the same two-sided constant.
2. Narrow the optional conclusion to: no valid nontrivial observable variance
   proxy was proved or implemented from the frozen allowed inputs. Do not claim
   an unproved impossibility theorem.
3. Remove the complete true reward tensor from all certificate inputs. Declare
   `R_star = 1 + gap_bonus` before sampling each MDP, pass that declared reward
   bound into the pure module, and derive `B = R_star/(1-gamma)` there. The
   actual full-table maximum may be read only after certificate construction
   inside a structurally separate `oracle_audit` that verifies the declaration.
4. Add failing-then-passing tests and pure-module validation for original
   integer counts, vector dimensions, each vector summing to the trajectory
   length, pair dimension being compatible with states, and state counts
   matching state-aggregated pair counts. The counterexample with `n=8`, state
   counts `[200,200]`, and pair counts `[100,100,100,100]` must be rejected.
5. Fix smoke seed allocation by always spawning the frozen 30 task seeds per
   cell and choosing the requested leading indices. Require all 16 smoke
   records to be baseline keys; a nonempty intersection is insufficient.
6. Preserve and regress all legacy config, task-record, and 176 summary rows.
   New values must appear only below `visit_indexed_certificate`. Reject
   duplicate keys; perform strict-JSON and complete schema/oracle-provenance
   checks; audit every residual group against its own visit radius; retain the
   read-only baseline inventory check.
7. Report both deterministic descriptive thresholds without selecting between
   them after seeing results: primary
   `improves_over_zero_initialization := total_bound < B`, and secondary
   `below_two_B_range := total_bound < 2B`. Keep validity/emission separate.
8. Correct all statements in the new repair evidence that were falsified by
   GPT verification. Preserve the historical first-result commit; write new
   repair evidence rather than rewriting history.

## Required order and commits

Follow this order exactly:

1. Add the new failing tests and record the expected failures.
2. Repair proof, module, evaluator, and analyzer.
3. Run Ruff, all five verifiers, and a corrected 16-record smoke matrix.
4. Analyze smoke and require 16/16 alignment, zero legacy mismatch, strict
   namespace/schema/oracle separation, and no baseline mutation.
5. Write `docs/research_branches/FP-MART-001/claude/repair_smoke.md` with exact
   commands, failures repaired, environment, metrics, limitations, paths, and
   hashes.
6. Commit code, tests, proof, and smoke evidence with subject beginning
   `[claude]`; verify the branch is clean. This is the repaired passing
   implementation/smoke seal.
7. Only from that exact clean commit, run the one frozen 480-record formal
   matrix with every argument explicitly spelled out and the canonical absolute
   Claude output path. Run the strict analyzer. Do not tune or rerun with
   changed scientific settings after inspection.
8. Write `docs/research_branches/FP-MART-001/claude/repair_result.md` with the
   exact implementation seal, environment, commands, seven raw-output hashes,
   all emission and both usefulness rates, every audit violation, anomalies,
   limitations, and a numbered acceptance assessment.
9. Commit only the new formal evidence with a `[claude]` subject and leave the
   branch clean. This is the repaired formal-evidence seal.

If any mandatory repair cannot be satisfied without changing the frozen task,
return `OBJECTION` and stop rather than weakening a gate. Otherwise, finish by
printing only:

```text
REPAIRED
IMPLEMENTATION_SEAL=<full sha>
FORMAL_EVIDENCE_SEAL=<full sha>
BRANCH_CLEAN=true
FORMAL_ANALYZER=PASS
```

Do not verify the GPT route in this repair run; reciprocal Claude verification
will be a separate post-repair instruction.
