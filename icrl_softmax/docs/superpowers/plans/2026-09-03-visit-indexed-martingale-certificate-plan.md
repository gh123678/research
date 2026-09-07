# Visit-indexed martingale certificate implementation plan

> Date: 2026-09-03
> Intended task: `docs/research_tasks/FP-MART-001.md`
> Design: `docs/superpowers/specs/2026-09-03-visit-indexed-martingale-certificate-design.md`
> Scientific code baseline: `b4b2769`
> Constraint: no research implementation or experiment begins before Claude's
> read-only pre-review returns `APPROVED` and the task becomes `ACTIVE`.

## Task 1: Freeze the research contract

Create the complete `FP-MART-001` task in `DRAFT`, including the falsifiable
hypotheses, fixed matrix, exact paths, route isolation, acceptance criteria,
failure criteria, and stopping conditions. Commit the DRAFT task, approved
design, and this plan together.

Move task metadata to `REVIEW` in a second commit, record the DRAFT commit as
the task-definition baseline, and add a compact `ACTIVE_WORKSPACE.md` pointer
to the task under review. Launch local Claude Code with read-only tools against
`AGENTS.md`, `CLAUDE.md`, `ACTIVE_WORKSPACE.md`, the design, plan, task, and
relevant completed theory. Require exactly one outcome:

- `APPROVED`, with a criterion-by-criterion check; or
- `OBJECTION`, with the disputed clause, evidence, validity impact, and options
  for user ruling.

On `OBJECTION`, record `BLOCKED_BY_OBJECTION` in both the task and workspace
pointer and stop. On approval, record the review, update the workspace status,
set the task to `ACTIVE`, and commit that activation. That activation commit is
the common execution start for both routes.

## Task 2: Create isolated execution worktrees

Keep the root worktree on `codex/FP-MART-001`. Create a separate worktree and
branch `claude/FP-MART-001` from the activation commit. Confirm both worktrees
resolve the same baseline, task version, design, plan, seed, and protocol.

Create only the ignored result roots in the canonical root worktree:

- `results/FP-MART-001/codex/`;
- `results/FP-MART-001/claude/`.

Because ignored files are not populated into a secondary Git worktree, the
Claude commands use the absolute canonical baseline and Claude-result paths
under `C:\Users\Admin\Desktop\research\icrl_softmax\results\`. They do not
write ignored outputs inside `C:\tmp\research-FP-MART-001-claude`.

Claude may write only its branch, worktree, and result subdirectory. GPT may
write only its branch, root worktree, and result subdirectory. Do not expose or
inspect the other route's first-result files or conclusions before both
first-result commits are recorded.

## Task 3: Independently establish the probability argument

Each route independently writes its own proof record under its assigned
evidence directory. It must derive, rather than assume:

1. the pre-action filtration for state Bellman residuals;
2. the post-action, pre-transition filtration for pair Bellman and fixed-value
   recovery residuals;
3. the measurability of each visit indicator and the associated visit stopping
   times;
4. the martingale-difference property after optional skipping;
5. the conditional interval width from `|R| <= R_star` and
   `B = R_star / (1 - gamma)`;
6. a two-sided union allocation over `G = m + 2d` groups and every
   `1 <= k <= n`;
7. substitution of the random final observed count;
8. the selective statement
   `P(Emit and bound violation) <= delta`;
9. the deterministic composition with Direct-Q and V-first no-split.

Use primary sources for any imported martingale or optional-skipping theorem
and record the exact assumptions and mapping. The mandatory result is the
finite-horizon Hoeffding route. A Freedman or empirical-Bernstein route is
optional and must have a separately valid observable variance construction and
preallocated risk.

Stop before implementation if any mandatory residual fails to form the stated
martingale difference or the proposed random-count substitution is invalid.
That is a scientific negative result, not an invitation to change the protocol.
Each route still seals its proof or counterexample and evidence independently;
the code-specific tasks are then marked not applicable rather than fabricated.

## Task 4: Add failing contract tests first

Independently create `verify_visit_indexed_martingale_certificate.py`. Before
creating the certificate module, run it and record that failure is caused by
the missing module rather than an unrelated import or environment problem.

The verifier must cover at least:

- exact risk allocation over groups and counts;
- monotone `1/sqrt(k)` radius behavior and the proved constants;
- state and pair visit-count selection;
- full-support and missing-support fixtures;
- random-count lookup without data-dependent risk reallocation;
- zero-initialization bound `B` without true values;
- Direct-Q exact/softmax composition;
- V-first no-split exact/softmax composition;
- relevant empirical-margin rejection;
- invalid delta, mode mismatch, and nonfinite arithmetic;
- deterministic failure-reason ordering;
- `selective_high_probability_certified` status semantics;
- strict JSON and separation of certificate versus oracle-audit fields;
- proof-level counterexample fixtures for the incorrect state filtration and
  unadjusted post-hoc minimum of two bounds.

## Task 5: Implement the pure certificate module

Independently create `visit_indexed_martingale_certificate.py` with no sampling,
plotting, file-writing, or true-model dependency. Keep focused public functions
for:

- input and risk-budget validation;
- the simultaneous Hoeffding radius;
- optional, separately budgeted variance-adaptive radii;
- state/pair support summaries;
- empirical matching-kernel gates;
- Direct-Q route composition;
- state-value and V-first no-split composition;
- deterministic status/failure handling;
- strict-JSON conversion.

The module must not accept stationary occupancy, transition matrices, spectral
quantities, `Q^pi`, `V^pi`, true residuals, or true initial errors as
certificate inputs. Run the new verifier until it passes.

## Task 6: Integrate through an additive evaluator

Create `evaluate_visit_indexed_certificates.py`. Reuse the existing MDP,
policy, rollout, estimator, and route functions. Add new data only below a
`visit_indexed_certificate` namespace. Do not overwrite the completed
`finite_sample_certificate`, route metrics, or legacy coverage flags.

If a minimal public extraction helper is genuinely required, an additive,
behavior-preserving change is allowed in:

- `evaluate_fixed_policy_q_routes.py`;
- `fixed_policy_finite_sample_certificate.py`.

Any such change must be covered by all old verifiers and zero-mismatch
regression. Broader refactoring is prohibited.

The evaluator must keep certificate inputs structurally separate from an
`oracle_audit` namespace. True values and realized residuals may enter only the
audit path and must never be passed to the certificate module.

## Task 7: Build strict regression and analysis

Create `analyze_visit_indexed_certificates.py` to:

- load strict JSON and reject `NaN` or `Infinity`;
- align all 480 records by the existing task key;
- require exact equality for nonnumeric legacy leaves;
- use `math.isclose(rel_tol=1e-12, abs_tol=1e-12)` for numeric legacy leaves;
- report mismatch count and maximum common numeric difference;
- summarize missing support, emission, and bound-nontriviality separately;
- summarize every rejection reason by trajectory length and route;
- report oracle-audit violations without treating empirical coverage as proof;
- compare Hoeffding and any valid variance-adaptive radius without unadjusted
  post-hoc selection.

Save `config.json`, `task_results.json`, `summary.json`, `regression.json`, and
the exact command/environment log in each assigned result directory. Plots are
derived results in that directory and are not required versioned figures.

## Task 8: Run deterministic regression and smoke tests

Before the formal matrix, each route runs:

```text
python -B verify_finite_sample_theorems.py
python -B verify_fixed_policy_q_routes.py
python -B verify_crossfit_markov_certificate.py
python -B verify_end_to_end_sarsa.py
python -B verify_visit_indexed_martingale_certificate.py
```

Then run a two-task, two-length smoke matrix into a route-local `smoke/`
subdirectory. Confirm:

- strict JSON parses;
- no nonfinite JSON value occurs;
- no old result directory changes;
- all legacy common fields match;
- missing-support and margin failures are stable;
- certificate input serialization contains no oracle field.

Do not proceed to the formal matrix until these checks pass. Smoke results are
not formal evidence and must be clearly labelled.

## Task 9: Produce sealed first results independently

Each route commits its code, verifier, proof record, and smoke evidence with the
required `[codex]` or `[claude]` subject. Record the first-result commit in its
own evidence file. Do not read the other route's branch, result directory,
report, or conclusion until both first-result commits exist.

The principal researcher records only the existence and commit identity of the
sealed Claude result before disclosure; it does not summarize or inspect its
content.

## Task 10: Run the frozen 480-item matrix

After each route has a passing sealed implementation, run its own formal
matrix. The GPT command is:

```text
python -B evaluate_visit_indexed_certificates.py
  --tasks 30
  --trajectory-lengths 256 1024 4096 16384
  --n-states 6
  --n-actions 4
  --pi-mins 0.05
  --betas 8
  --mixing 0.08 0.5
  --gap-bonuses 0 0.5
  --gamma 0.70
  --alpha 0.65
  --iterations 160
  --certificate-delta 0.05
  --seed 20260829
  --output-dir results/FP-MART-001/codex
```

Claude uses the identical arguments and its `claude` output directory. The
line-broken representation is documentation; the reproducibility log must also
record a directly executable Windows command. From the Claude worktree, both
`--output-dir` and the analyzer's baseline directory are passed as absolute
paths in the canonical root worktree.

Each route then runs its analyzer against the read-only baseline
`results/fixed_policy_finite_sample_certificates/` and its own new directory.
No parameter, threshold, risk split, or stopping rule may change after formal
results are viewed. Each route commits its formal evidence record before
disclosure; the ignored raw results remain identified by hashes and paths.

## Task 11: Disclose and cross-verify

Only after both formal result commits and evidence records exist may GPT and
Claude inspect each other's artifacts.

GPT verifies Claude by:

- inspecting the Claude proof and implementation without modifying its branch;
- rerunning Claude's verifier and smallest reproduction in a clean validation
  worktree;
- checking the Claude result schema, frozen config, legacy regression, and raw
  metrics;
- mapping evidence to every acceptance criterion;
- writing `PASS`, `FAIL`, or `OBJECTION` with exact evidence.

Claude performs the symmetric read-only verification of GPT. A task-definition
defect becomes `OBJECTION`; an implementation or inference defect becomes
`FAIL` and returns the affected route to `ACTIVE` for author repair.

## Task 12: Synthesize without erasing disagreement

After disclosure, GPT writes the final shared theory and report:

- `docs/research_branches/visit_indexed_martingale_certificate_theory.md`;
- `docs/research_branches/visit_indexed_martingale_certificate_report.md`.

They must distinguish common conclusions, route-specific differences,
negative results, empirical audit observations, and unresolved limitations.
Disagreement is explained with evidence or sent to the user for ruling; one
route's result is never silently substituted for the other's.

Update `FP-MART-001.md` and `ACTIVE_WORKSPACE.md`, move through `VERIFYING`, and
perform a final scope, placeholder, strict-JSON, finite-value, verifier,
regression, result-location, and worktree-cleanliness audit. The task becomes
`VERIFIED` only after both cross-verification reports pass or the user records
an explicit exception. No merge to `main` occurs without separate user
approval.
