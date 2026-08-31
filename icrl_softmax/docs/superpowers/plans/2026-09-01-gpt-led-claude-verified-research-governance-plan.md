# GPT-led, Claude-verified research governance implementation plan

> Date: 2026-09-01
> Task: `docs/research_tasks/GOV-001.md`
> Design: `docs/superpowers/specs/2026-08-31-gpt-led-claude-verified-research-governance-design.md`
> Constraint: this is a governance-only change; preserve all scientific code, evidence, and conclusions.

## Task 1: Freeze and pre-review the implementation contract

Create `docs/research_tasks/GOV-001.md` with the approved authority model, bounded file scope, acceptance criteria, and stopping conditions. Commit the task and this plan before changing the active rules.

Launch the locally installed Claude Code in read-only plan mode. Ask it to inspect the design, task record, current `AGENTS.md`, and this plan. Require one structured outcome:

- `APPROVED`, with an itemized coverage check; or
- `OBJECTION`, with the exact disputed clause, evidence, validity impact, and options for user ruling.

Acceptance:

- the task is frozen at a named commit;
- Claude has no edit-capable tools during pre-review;
- implementation does not start after `OBJECTION`.

## Task 2: Replace the canonical collaboration rules

Rewrite the root `AGENTS.md` in concise Chinese. Preserve repository structure and environment facts that remain true, and replace the equal peer-maintainer model with the approved hierarchy.

The canonical rules must cover:

- user, GPT, and Claude authority;
- GPT-only formal task initiation;
- Claude read-only pre-review and bounded execution rights;
- short-task and long-task routing;
- automatic local Claude orchestration;
- task states and objection pause;
- quota handoff;
- isolated branches, worktrees, and result paths;
- reproducible evidence and two-way verification;
- user-controlled merge into `main`;
- preservation of unrelated and historical work.

Acceptance:

- no old statement implies equal task-initiation authority;
- no old serial-handoff rule prevents required dual execution of long tasks;
- no rule permits either assistant to merge into `main` without user approval.

## Task 3: Add the Claude entry point

Create a short root `CLAUDE.md`. It must tell Claude to read, in order:

1. root `AGENTS.md`;
2. `icrl_softmax/ACTIVE_WORKSPACE.md`;
3. the active task record named there.

It must also require Claude to remain read-only unless the task explicitly grants an execution scope, and to return `OBJECTION` rather than changing a disputed task.

Do not copy the complete governance policy into `CLAUDE.md`; `AGENTS.md` remains canonical.

## Task 4: Add reusable research-task templates

Create `icrl_softmax/docs/research_tasks/README.md` with copyable sections for:

- task definition and metadata;
- falsifiable hypotheses;
- permitted and prohibited work;
- acceptance, failure, and stopping criteria;
- route-specific branches and output paths;
- execution evidence;
- objection and user-ruling records;
- quota handoff;
- GPT and Claude verification outcomes;
- definition of done.

The template must distinguish `FAIL` from `OBJECTION` and identify the only legal state transitions.

## Task 5: Add a compact active-governance pointer

Update `icrl_softmax/ACTIVE_WORKSPACE.md` without rewriting its scientific objective or active evidence. Add a compact section containing:

- governance mode;
- active task identifier and status;
- GPT and Claude branches or roles;
- current blocker and next action;
- links to the canonical rules, task, and design.

## Task 6: GPT verification

Inspect the complete diff and verify every GOV-001 acceptance criterion. Run read-only checks equivalent to:

- whitespace and patch-integrity checks;
- placeholder scans;
- scans for obsolete peer-authority language;
- required-term and required-section checks;
- changed-path comparison against the allowed file list;
- confirmation that research code, results, manuscripts, and archive content are untouched.

Record an itemized `PASS` or `FAIL` in GOV-001. A failure returns implementation to `ACTIVE`; a task-definition defect becomes `OBJECTION`.

## Task 7: Claude final verification

Launch Claude again in read-only mode against the frozen design, GOV-001, and the complete implementation diff. Require it to check:

- authority consistency;
- no unauthorized Claude self-modification power;
- long-task dual execution and final cross-verification;
- objection pause and user ruling;
- quota handoff behavior;
- non-duplication between `AGENTS.md` and `CLAUDE.md`;
- preservation of scientific work;
- every acceptance criterion.

Record Claude's exact status and evidence in GOV-001. Do not ask Claude to repair the GPT branch.

## Task 8: Commit and hand off for merge approval

Commit the governance implementation with a `[codex]` subject explaining why the authority model changed. Push only `codex/gpt-led-research-governance`.

Report:

- commits and branch;
- files changed;
- GPT verification result;
- Claude verification result;
- any limitations or user rulings;
- the fact that `main` remains unchanged.

Request the user's explicit approval before any merge into `main`.
