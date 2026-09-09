# FP-KERN-002 Claude route: git-seal environment blocker

Date: 2026-09-09.

Route: Claude independent executor on `claude/FP-KERN-002`, worktree
`results/FP-KERN-002/claude_worktree/`, common execution-start commit
`ffdf26b029efde08ea794454a7b5da890108c355`.

Status: `BLOCKED` at the blind implementation/smoke seal step. The earlier
session-environment blocker is resolved (see `execution_blocker.md`,
resolution section): Bash and Python are fully usable, and all executable
gates up to the seal have passed. Git mutation is now the only failing step.

## Completed before this blocker

- Frozen common input verified (hashes, manifest, 480-record matrix).
- Verifier-first: `verify_kernel_reuse_diagnostics.py` written before the
  analyzer; required initial failing run recorded
  (`ModuleNotFoundError: No module named 'analyze_kernel_reuse_diagnostics'`);
  after implementation, all 14 fixtures pass.
- `analyze_kernel_reuse_diagnostics.py` implemented independently (blind: no
  GPT FP-KERN-002 branch, implementation, result, or conclusion was read).
- Stage 0: exact (bit-for-bit) replay of all four predecessor routes on all
  480 records; old metrics/intervals match the sealed predecessor
  `summary.json` to absolute tolerance 1e-12; `NOT_SUPPORTED` reproduced.
- Task-scoped Ruff: all checks passed.
- Fixed 16-record smoke: full pipeline PASS; strict JSON bundle inspected;
  temporary smoke directory deleted; formal result directory never created.
- Inherited verifiers PASS: `verify_kernel_state_generalization.py`,
  `verify_fixed_policy_q_routes.py`, `verify_finite_sample_theorems.py`,
  `verify_visit_indexed_martingale_certificate.py`,
  `verify_time_uniform_mixture_certificate.py`, and the read-only legacy
  strict-analysis regression (`analyze_kernel_state_generalization.py` on the
  FP-KERN-001 canonical directory, classification `NOT_SUPPORTED`).
- Full evidence: `implementation_seal.md` in this directory.

All of the above is preserved uncommitted in this worktree:

```text
?? icrl_softmax/analyze_kernel_reuse_diagnostics.py
?? icrl_softmax/verify_kernel_reuse_diagnostics.py
?? icrl_softmax/docs/research_branches/FP-KERN-002/claude/
```

## Blocker

Every git write fails at OS level before any repository data is touched:

```text
$ git add <the four staged paths above>
fatal: Unable to create
'C:/Users/Admin/Desktop/research/.git/worktrees/claude_worktree1/index.lock':
Permission denied
```

Diagnosis:

1. The failure is identical with the harness sandbox enabled and disabled,
   so it is not a command-level or harness permission denial.
2. `icacls` shows explicit `DENY (W,D,Rc,DC)` access-control entries for two
   sandbox SIDs (`S-1-5-21-1085831768-3489467200-433082987-2850326224` and
   `S-1-5-21-2983289630-2183854612-99023474-2914504594`) set directly on
   `C:\Users\Admin\Desktop\research\.git` and inherited by every child,
   including `worktrees\claude_worktree1` and `objects`. DENY entries
   override the inherited Modify grants, so this session account
   (`CodexSandboxOffline`) can read the repository but cannot create the
   index lock, write objects, or update refs. No stale lock file exists.
3. Reads work: `git rev-parse`, `git status`, `git log` all succeed. Writes
   to the worktree working directory and to route output paths work; only
   `.git` internals are denied.
4. No bypass was attempted or will be attempted: changing ACLs, elevating,
   or committing through another identity from inside this session would be
   an unsafe permission bypass prohibited by the task and `AGENTS.md`.

## Consequence

The frozen plan requires sealing the blind implementation with a `[claude]`
commit *before* the sole full 480-record diagnostic, and sealing the formal
result with a second `[claude]` commit. Because the once-only formal output
rule makes the full run effectively irreversible, the formal diagnostic was
**not** run; `results/FP-KERN-002/claude/` remains absent. Execution is
halted exactly at the seal boundary with all pre-seal gates green.

This is an environment blocker, not a task-level defect; no `OBJECTION`
against the frozen task definition is raised.

## Required user action (one of)

1. Remove the two inherited DENY write ACEs on
   `C:\Users\Admin\Desktop\research\.git` (and let inheritance clear the
   children) from an unrestricted shell, e.g. inspect with
   `icacls C:\Users\Admin\Desktop\research\.git` and remove the deny entries
   for the two sandbox SIDs; then restart or resume this session — the route
   will immediately create the `[claude]` implementation/smoke seal commit,
   run the sole formal diagnostic, verify it, and create the `[claude]`
   formal-result seal commit; or
2. Perform only the two seal commits manually from an unrestricted account
   using the exact `git add`/`git commit` steps listed in the next progress
   report (the working tree content is already final for the first seal), in
   which case the route continues with the formal run after the first seal
   exists.

After repair, no scientific step needs repetition: the verifier, Stage 0,
Ruff, smoke, and inherited-verifier evidence is recorded and remains valid
because none of it depends on git state.
