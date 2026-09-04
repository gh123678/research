# Time-uniform mixture certificate implementation plan

> Date: 2026-09-04
> Intended task: `docs/research_tasks/FP-TU-001.md`
> Design: `docs/superpowers/specs/2026-09-04-time-uniform-mixture-certificate-design.md`
> Scientific baseline: `28b71685ca05ae073cc847fdea230019e4bd63ea`
> Constraint: no research implementation or experiment begins before Claude's
> read-only pre-review returns `APPROVED` and the task becomes `ACTIVE`.

## Task 1: Freeze and pre-review the research contract

Create the complete task in `DRAFT`, including frozen formulas, baseline
hashes, paths, protocol, role isolation, acceptance and negative-result paths,
failure criteria, and stopping conditions. Commit the DRAFT task and this plan
together.

In a second commit, set the task to `REVIEW`, record the DRAFT commit as the
task-definition baseline, and update `ACTIVE_WORKSPACE.md` with a compact
pointer. Launch local Claude Code in read-only mode against the governance,
task, design, plan, prior verified theory, relevant code, and frozen hashes.
Require exactly one outcome:

- `APPROVED`, with a numbered contract and feasibility audit; or
- `OBJECTION`, with disputed text, evidence, validity impact, and options for
  user ruling.

On objection, record `BLOCKED_BY_OBJECTION` and stop affected work. On
approval, record the full pre-review in the task, set status to `ACTIVE`, and
commit. That activation commit becomes the common execution start.

## Task 2: Create isolated execution worktrees

Keep the root worktree on `codex/FP-TU-001`. Create
`C:\tmp\research-FP-TU-001-claude` and branch `claude/FP-TU-001` from the
activation commit. Confirm both worktrees resolve the same scientific
baseline, task version, design, plan, formulas, seed, and protocol.

Create only the ignored canonical result roots:

- `results/FP-TU-001/codex/`;
- `results/FP-TU-001/claude/`.

Claude uses the absolute canonical baseline and Claude-result paths because
ignored files do not populate into a secondary worktree. Neither route reads
the other's first-result evidence, code conclusion, or output before both
first-result seals exist.

## Task 3: Independently establish the time-uniform proof

Each route independently writes `theory.md` in its evidence directory. The
proof must derive or explicitly inherit with exact references:

1. the three residual families and their verified pre/post-action filtrations;
2. predictable visit selection, visit stopping times, and optional skipping;
3. conditional centering and range width `2B`;
4. the fixed-`a` exponential supermartingale;
5. convex mixing of both signs through `cosh`;
6. initial value one and group allocation `delta/G`;
7. Ville crossing control uniform over all counts;
8. random-final-count substitution and selective emission;
9. uniqueness of `q_mix(k)`;
10. independent validity of every line boundary and their minimum;
11. the algebra proving `q_mix(k) <= q_stitch(k)`;
12. unchanged deterministic route composition.

Use primary sources for imported results and map each assumption. Stop before
implementation if a mandatory proof step fails; seal a negative proof or
counterexample instead.

## Task 4: Write failing contract tests first

Independently create `verify_time_uniform_mixture_certificate.py`. Run it
before the module exists and record the expected missing-module failure.

Tests cover at least:

- exact grid, normalized weights, rates, and group/sign risk allocation;
- stable `logcosh`, `logsumexp`, and direct small-number comparisons;
- exhaustive conditional MGF fixtures and mixture initial value;
- bracketing, root conservativeness, and iteration/tolerance behavior;
- all `1 <= k <= 16384` for finiteness, monotone radius,
  `q_mix <= q_stitch`, and dominance over `r_old`;
- independent high-precision roots at endpoints, powers of two, and off-grid
  counts;
- strict integer/count aggregation and horizon validation;
- exact/softmax Direct-Q and V-first composition;
- support, margin, mode, risk, numeric, and ordered failure paths;
- strict JSON, namespace/status preservation, and oracle separation;
- a counterexample to an unallocated post-hoc minimum.

## Task 5: Implement the pure certificate module

Independently create `time_uniform_mixture_certificate.py` with no sampling,
plotting, file writing, or true-model dependency. Keep focused public
functions for:

- grid/weight/rate construction and validation;
- stable log-mixture evaluation;
- analytic stitch calculation;
- conservative bisection inversion;
- state/pair radius summaries from observed counts;
- Direct-Q and V-first no-split composition;
- ordered status/failure handling;
- strict-JSON conversion.

Do not accept any prohibited oracle quantity. Run the new verifier until it
passes, then run every existing fixed-policy and visit-indexed verifier.

## Task 6: Integrate through an additive evaluator

Create `evaluate_time_uniform_certificates.py`. Reuse the existing MDP,
policy, rollout, estimator, route, and visit-indexed functions. Add data only
below `time_uniform_certificate`; preserve `visit_indexed_certificate` and all
legacy leaves exactly.

The evaluator reports old, stitch-audit, and mixture radii at route-relevant
minimum observed counts; radius and total-bound ratios/reductions; unchanged
emission decisions and ordered reasons; `<B` and descriptive `<2B`
usefulness; and oracle errors/violations only inside `oracle_audit`.

If a reusable helper is essential, make only the task-authorized additive,
behavior-preserving change and cover it with old verifiers and regression.

## Task 7: Build strict regression and analysis

Create `analyze_time_uniform_certificates.py` to:

- parse strict JSON and reject duplicate keys, NaN, and Infinity;
- validate all frozen hashes, config values, task keys, and 480 records;
- require exact equality for nonnumeric legacy leaves;
- require zero numeric mismatch under
  `math.isclose(rel_tol=1e-12, abs_tol=1e-12)`;
- exhaustively recompute all frozen count-radius checks;
- confirm identical emission and nonincreasing emitted bounds;
- summarize radius reductions, route-bound reductions, support, emission,
  `<B`/`<2B` usefulness, and every deterministic rejection reason;
- enumerate oracle violations separately without treating them as proof;
- write strict `regression.json`, `summary.json`, and `checks.log` evidence.

## Task 8: Assess observable transition variance separately

Each route asks whether successor observations and deterministic edge rewards
can yield an observable confidence set that uniformly upper-bounds
`Var_P[R(s,a,S') + gamma V(S')]` for all `V in [-B,B]^m`, including unseen
successors.

Record one of:

- a complete proposal with observable inputs, proof obligations, and future
  risk allocation; or
- a deterministic counterexample or exact unresolved gap.

Do not implement it as a certificate, allocate it risk, combine it with the
mixture, or let it delay the mandatory route.

## Task 9: Run and seal smoke evidence

Before any formal run, execute Ruff, every old verifier, the new verifier, and
a small matrix covering all four routes, both mixing regimes, both gap
bonuses, short/long counts, support/non-support, and exact/softmax gates.

Require strict JSON, zero legacy mismatch, no namespace/oracle leakage, and all
radius/inversion audits. Record exact commands, versions, hashes, successful
and failed runs, anomalies, metrics, limitations, and numbered acceptance
judgments in `first_result.md`.

Each route commits implementation and smoke evidence as a blind first-result
seal before reading the other route.

## Task 10: Run the one frozen formal matrix

After smoke passes and seals, each route runs exactly once with all parameters
spelled out:

```text
C:\Users\Admin\anaconda3\python.exe -B evaluate_time_uniform_certificates.py
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
  --output-dir results/FP-TU-001/codex
```

Claude uses identical arguments and its `claude` result directory. Each route
runs its analyzer against the frozen absolute `FP-MART-001/codex` baseline.
The line-broken command is documentation; `commands.log` also records a
directly executable Windows command.

Do not change or rerun the formal matrix to improve results. A mechanical
interruption before a complete readable output may be recorded and resumed or
restarted only under the frozen command; a scientifically complete output is
the one formal result. Commit `formal_result.md` with hashes and exact metrics
before disclosure.

## Task 11: Disclose and cross-verify

Only after both blind first-result and formal-evidence seals exist may routes
inspect one another.

GPT verifies Claude by checking its proof, code, history, schema, frozen
configuration, raw hashes, regression, and every acceptance item, then reruns
all verifiers and a fresh reproduction in a clean validation worktree. Claude
performs the symmetric read-only verification of GPT.

A task-definition defect is `OBJECTION`; an implementation, evidence, or
inference defect is `FAIL` and returns that route to `ACTIVE` for author repair.
Each report ends with exactly `PASS`, `FAIL`, or `OBJECTION` and points to
reproducible evidence.

## Task 12: Synthesize without erasing differences

After reciprocal verification, GPT writes:

- `docs/research_branches/time_uniform_mixture_certificate_theory.md`;
- `docs/research_branches/time_uniform_mixture_certificate_report.md`.

The synthesis distinguishes the mandatory mathematical result, empirical
secondary outcomes, transition-variance feasibility result, route-specific
differences, anomalies, and limitations. Disagreement is explained with
evidence or sent to the user for ruling.

Move the task through `VERIFYING` to `VERIFIED` only after both final reports
pass or the user records an explicit exception. Update `ACTIVE_WORKSPACE.md`
and run the final placeholder, scope, strict-JSON, finite-value, verifier,
regression, result-location, baseline, branch, and clean-worktree audits. No
merge to `main` occurs without separate user approval.
