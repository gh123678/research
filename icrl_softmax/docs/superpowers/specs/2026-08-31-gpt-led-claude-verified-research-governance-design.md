# GPT-led, Claude-verified research governance design

> Date: 2026-08-31
> Status: user-approved design
> Scope: repository-wide research governance, with `icrl_softmax/` as the active project

## 1. Purpose

Replace the current peer-maintainer model with a principal-investigator model:

- the user remains the project owner and final arbiter;
- GPT is the principal researcher and the sole author of formal research tasks;
- Claude Code is an auxiliary researcher and independent verifier;
- long or conclusion-critical work is executed independently by both systems;
- neither system may call a result verified until the required independent review is complete.

The user supplies objectives and priorities. GPT translates them into versioned research tasks with explicit hypotheses, methods, boundaries, and acceptance criteria.

## 2. Authority and responsibilities

### 2.1 User: project owner

The user has final authority over:

- research direction and priority;
- disputed assumptions, methods, and acceptance criteria;
- exceptions to the verification policy;
- approval to merge verified work into `main`.

### 2.2 GPT: principal researcher

GPT must:

- originate every formal research task;
- define the question, falsifiable hypotheses, permitted methods, prohibited actions, outputs, and acceptance criteria;
- classify and decompose the work;
- assign Claude's bounded execution and verification scope;
- execute GPT's route on `codex/*` branches;
- collect both evidence sets and synthesize the final conclusion;
- issue a written handoff before transferring unfinished work because of quota or availability limits.

Only GPT may revise a task definition after recording the user's ruling. GPT may not describe an unreviewed result as verified.

### 2.3 Claude Code: auxiliary researcher and independent verifier

Claude must:

- review each assigned task for clarity, falsifiability, reproducibility, and technical defects before execution;
- return `APPROVED` or `OBJECTION` with evidence;
- execute only the explicitly assigned scope on `claude/*` branches;
- independently verify GPT's artifacts and conclusions;
- preserve conflicting evidence and report it to the user.

Claude may repair ordinary implementation errors within its own assigned scope. Claude may not independently change a research objective, key assumption, method boundary, acceptance criterion, or task allocation. A task-level objection pauses affected work until the user rules.

## 3. Research task record

Every task is stored at:

`icrl_softmax/docs/research_tasks/<task-id>.md`

Each record must contain:

1. task identifier, creation date, baseline commit, and status;
2. research question and falsifiable hypotheses;
3. input data and fixed evaluation protocol;
4. permitted methods and prohibited actions;
5. expected artifacts and exact output locations;
6. acceptance criteria, failure criteria, and stopping conditions;
7. estimated runtime and resource needs;
8. GPT execution scope and Claude execution scope;
9. independent verification required from each system;
10. objections, user rulings, handoffs, and final evidence links.

`ACTIVE_WORKSPACE.md` remains a compact pointer to the current task, branches, state, blockers, output locations, and next action. It does not replace the full task record.

## 4. Task lifecycle

The normal state sequence is:

`DRAFT -> REVIEW -> ACTIVE -> VERIFYING -> VERIFIED`

An objection changes the affected task to `BLOCKED_BY_OBJECTION`. Work resumes only after the user's ruling is recorded. A failed verification returns the task to `ACTIVE` for a bounded correction under the unchanged task definition, or to `BLOCKED_BY_OBJECTION` if the definition itself is disputed.

The lifecycle is:

1. GPT writes and freezes the task record.
2. Claude performs a read-only pre-review.
3. Claude returns `APPROVED` or a documented `OBJECTION`.
4. GPT starts the approved execution routes.
5. Each executor submits a reproducible evidence package.
6. GPT and Claude independently verify the other route.
7. GPT reconciles consistent evidence and documents discrepancies.
8. Unresolved discrepancies go to the user.
9. The user approves or rejects the merge into `main`.

## 5. Short and long tasks

A task is long if any of the following applies:

- expected wall-clock runtime exceeds 30 minutes;
- it contains a large experimental matrix;
- it requires multiple research or implementation stages;
- its conclusion is important enough to require an independent construction;
- GPT may be unable to complete it within the current model quota.

For a short task, GPT executes and Claude independently verifies.

For a long task, GPT automatically assigns the frozen task to both systems. They receive the same baseline, inputs, random seeds, and evaluation protocol, but use separate branches and output directories. Each system forms its initial result before reading the other's conclusion.

If GPT reaches a quota or availability limit, Claude may complete the handed-off scope. That result remains provisional until GPT later verifies it, unless the user explicitly waives GPT verification.

## 6. Automatic Claude orchestration

The user authorizes GPT to start the locally installed Claude Code automatically for qualifying long tasks without requesting permission each time.

Orchestration has two phases:

1. **Read-only review.** Claude inspects the task and returns `APPROVED` or `OBJECTION` without editing the repository.
2. **Isolated execution.** After approval, GPT launches Claude in its assigned worktree and branch with bounded repository access.

GPT monitors the run, collects the report, and stops or surfaces any request that exceeds the approved scope. Automatic authorization does not include:

- bypassing permission or safety controls;
- direct commits or merges to `main`;
- deleting or overwriting another route's code or results;
- changing the frozen task definition;
- continuing work affected by an unresolved objection;
- publishing externally, creating pull requests, or sending external messages;
- incurring an unapproved new cost category.

If Claude is unavailable because of authentication, quota, permissions, or environment failure, GPT records the blocker and informs the user. A Codex subagent may not be presented as Claude verification.

## 7. Isolation and Git policy

Both routes start from the same recorded baseline commit:

- GPT branch: `codex/<task-id>`;
- Claude branch: `claude/<task-id>`;
- GPT output: `results/<task-id>/codex/`;
- Claude output: `results/<task-id>/claude/`.

Neither system may edit the other's branch or overwrite its results. The verifier reports findings through the task record or a review report; the original author fixes its own implementation unless GPT assigns a separate repair task.

Experimental changes must not be made directly on `main`. Commit subjects start with `[codex]` or `[claude]` and explain why the change is needed. Merging into `main` requires completed verification and user approval.

Unrelated dirty working-tree content is preserved, excluded from task commits, and reported in the handoff. Existing experimental code and results must not be removed merely because they are obsolete or disputed.

## 8. Evidence and cross-verification

Each execution route submits:

- exact commit, environment, and reproducible commands;
- code and output locations;
- successful, failed, and anomalous run records;
- raw metrics and their calculation path;
- a conclusion, limitations, and applicability boundary;
- an itemized acceptance-criteria assessment.

GPT verifies Claude's route, and Claude verifies GPT's route. Each verification ends with one status:

- `PASS`: the evidence reproduces and supports the claimed conclusion;
- `FAIL`: an implementation, experiment, or inference does not satisfy the frozen task;
- `OBJECTION`: the task definition or acceptance basis is materially defective.

A task is `VERIFIED` only when both verifiers return `PASS`, or when the user explicitly resolves an exception. Cross-verification must include artifacts and evidence, not only prose agreement.

## 9. Objection protocol

An objection must identify:

- the exact disputed task clause;
- theoretical, code, or experimental evidence;
- the effect on validity or reproducibility;
- concrete options for the user's ruling.

The objecting party preserves its evidence and pauses affected work. It may continue unrelated work only when doing so cannot prejudice the ruling. Claude must not silently convert an objection into a changed implementation. The user's ruling is recorded verbatim or faithfully summarized in the task record; GPT then issues the revised task version when revision is required.

## 10. Quota and continuity handoff

Before GPT transfers unfinished work to Claude, GPT records:

- current branch, commit, and task version;
- completed, running, and pending work;
- commands already executed and output locations;
- current findings and uncertainty;
- immutable task boundaries;
- exact next actions for Claude;
- items GPT must verify after returning.

Claude may continue within this handoff. Any necessary scope expansion triggers the objection protocol. On return, GPT audits the handoff result before initiating another research task.

## 11. Rule files and enforcement

Implementation will establish:

- `AGENTS.md` as the complete, canonical governance document;
- `CLAUDE.md` as a concise mandatory entry point directing Claude to read `AGENTS.md`, `ACTIVE_WORKSPACE.md`, and the active task record;
- `icrl_softmax/docs/research_tasks/README.md` as the task, objection, handoff, and verification template;
- `ACTIVE_WORKSPACE.md` as the current-state index.

Duplicated rule text is avoided so the two assistants cannot silently diverge. If entry-point text conflicts with `AGENTS.md`, the canonical document controls, except that a direct user instruction always has higher authority.

## 12. Definition of done

A research task is complete only when:

- no unresolved objection remains;
- both routes' artifacts are preserved and reproducible;
- both verification reports are recorded;
- every acceptance criterion has explicit evidence;
- discrepancies are reconciled or decided by the user;
- `ACTIVE_WORKSPACE.md` records the final state and next action;
- the user has approved any merge into `main`.
