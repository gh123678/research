# Research task records

Every formal research task is authored by GPT and stored in this directory as `<task-id>.md`. The task record is the frozen contract for execution, objections, handoffs, evidence, and verification.

## Legal states

Normal flow:

`DRAFT -> REVIEW -> ACTIVE -> VERIFYING -> VERIFIED`

Exceptional transitions:

- `REVIEW`, `ACTIVE`, or `VERIFYING` -> `BLOCKED_BY_OBJECTION` when the task definition or acceptance basis is materially defective;
- `VERIFYING` -> `ACTIVE` when an implementation or inference fails but the frozen task remains valid;
- `BLOCKED_BY_OBJECTION` -> `REVIEW` after the user's ruling and a GPT-authored task revision.

No assistant may skip `REVIEW`, self-resolve a task-level objection, or mark a task `VERIFIED` without the required evidence.

## Task template

Copy the following structure into `<task-id>.md`.

```markdown
# <task-id>: <title>

## Task metadata

- Created:
- Author: GPT
- Status: `DRAFT`
- Task version:
- Baseline commit:
- Design:
- Plan:
- GPT branch: `codex/<task-id>`
- Claude branch or role: `claude/<task-id>` or read-only verifier
- Estimated runtime and resources:

## Research question

State one answerable question.

## Falsifiable hypotheses

List predictions that evidence can reject.

## Inputs and fixed protocol

Record data, seeds, metrics, evaluation rules, and shared assumptions.

## Allowed work

List exact files, directories, tools, and external systems in scope.

## Prohibited work

List actions and artifacts outside scope.

## Expected artifacts

Give exact code, document, and result paths.

## Acceptance criteria

Number every checkable success condition.

## Failure criteria

Define outcomes that reject the hypothesis or implementation.

## Stopping conditions

List objections, conflicts, safety issues, resource limits, or missing authority that require a pause.

## Execution evidence

### GPT route

- Commit and environment:
- Commands:
- Outputs:
- Metrics and anomalies:
- Conclusion and limitations:
- Acceptance assessment:

### Claude route

- Commit and environment:
- Commands:
- Outputs:
- Metrics and anomalies:
- Conclusion and limitations:
- Acceptance assessment:

## Objections and user rulings

### Objection

- Status: `NONE` or `OBJECTION`
- Disputed clause:
- Evidence:
- Validity impact:
- Options for user ruling:

### User ruling

- Date:
- Decision:
- Required GPT task revision:

## Quota or continuity handoff

- Current branch, commit, and task version:
- Completed, running, and pending work:
- Commands and outputs:
- Current findings and uncertainty:
- Immutable boundaries:
- Claude's exact next actions:
- GPT's required return verification:

## Verification reports

### GPT verifies Claude

- Status: `PASS`, `FAIL`, or `OBJECTION`
- Reproduction or inspection performed:
- Evidence:
- Acceptance-criteria mapping:
- Required next state:

### Claude verifies GPT

- Status: `PASS`, `FAIL`, or `OBJECTION`
- Reproduction or inspection performed:
- Evidence:
- Acceptance-criteria mapping:
- Required next state:

## Definition of done

- [ ] No unresolved objection remains.
- [ ] Both required routes are reproducible.
- [ ] Both verification reports are recorded.
- [ ] Every acceptance criterion has evidence.
- [ ] Discrepancies are reconciled or ruled on by the user.
- [ ] `ACTIVE_WORKSPACE.md` is current.
- [ ] The user approved any merge into `main`.
```

## Status semantics

- `APPROVED` is used only for the pre-execution task review.
- `PASS` means an execution or result satisfies the frozen task.
- `FAIL` means the frozen task is valid but the implementation, experiment, or inference is not.
- `OBJECTION` means the task definition or acceptance basis is materially defective and requires the user's ruling.
