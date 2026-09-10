# FP-ITER-001 Codex verification of Claude route

Verifier: GPT/Codex, completed after both first-result seals. The prior PASS
below is historical and is superseded by the follow-up FAIL at the end.
Numerical reproduction remains successful; the original author must repair
the identified claim/evidence defects before this direction can pass.

## Identity and scope

- Frozen scientific baseline: 6d2376968aeb9379c978bd1a7af7929b70fdeb09.
- Claude blind seal: 40344989bc88ea422d5535a7a4b199b98a669413,
  subject `[claude] seal FP-ITER-001 independent blind first result`.
  Its worktree is clean at that commit and contains only the four authorized
  source/report/theory files.
- Claude's execution-start manifest records the common baseline before coding.
  Its task hash is 88a07ff9...62b8. The current Codex task hash is
  301c8804...18f24 because the user-authorization ruling and historical
  execution-blocker record were appended after the common activation commit.
  `git diff 6d23769 -- FP-ITER-001.md` shows those are administrative
  authorization/history lines only; frozen inputs, formulas, tolerances,
  lifecycle and artifact scope are unchanged.
- Claude reported and preserved four ordinary failed runs (two verifier
  defects, one syntax defect, one strict-JSON NumPy-bool defect). The final
  run was performed after each repair without changing the protocol.
- No Codex source, report or result was read by Claude before its blind seal.
  The user authorized the scoped transfer before this reciprocal review.

## Executable reproduction

From the Claude worktree's `icrl_softmax` directory, GPT ran:

```
C:/Users/Admin/anaconda3/python.exe -B docs/research_branches/FP-ITER-001/claude/verify.py
C:/Users/Admin/anaconda3/python.exe -B docs/research_branches/FP-ITER-001/claude/witness.py
C:/Users/Admin/anaconda3/python.exe -m ruff check docs/research_branches/FP-ITER-001/claude/witness.py docs/research_branches/FP-ITER-001/claude/verify.py
```

The replayed verifier exited 0 with 1,633 checks and 0 failures; its output is
`results/FP-ITER-001/codex/claude_replay_verification.json`. Ruff exited 0.
The replayed witness exited 0 and produced strict JSON of the same 680,882-byte
size with the same PASS H1--H5 summaries; replay output is
`results/FP-ITER-001/codex/claude_replay_results.json`. The verification JSON
digest is exactly the sealed verification digest
D9341B74276AC11F98DF18FCF509AB587E5E8DB17A759A08B0DE37FAA1F88E2A. The witness
digest differs only because the embedded runtime commit/environment metadata
changes between the original and replay; numerical and H-check fields match.

## Numerical cross-reconstruction

GPT loaded both raw JSON artifacts and compared every saved Q trace:

- 16 aligned exact/finite sequences (Claude stores direct and attention
  separately; Codex stores the corresponding four exact and eight finite
  routes) had maximum absolute difference 1.7763568394002505e-15.
- Codex and Claude finite operators C/S/W differed by at most
  3.3306690738754696e-16 across C and M, flat and sharp configurations.
- Exact C contraction was 0.85 and exact M norm was 1.0 in both routes.
  C/Z/sharp and C/A/sharp final finite-versus-exact gaps, bounds and fixed
  point match the independent Codex values: q_f_inf approximately
  [2.4599552801374, 0.95794182893235, 1.70894855453487, 2.20961970493656],
  attention bias 0.001621946804065, steady bound 0.003921311230829.
- Both routes report q_hat=q_pi=(59,23,41,53)/24 on C, and explicitly call
  this fixture's population/data discrepancy zero rather than presenting a
  sampling claim.
- Both routes report exact preservation of absent pair 11 for M, uniform
  finite writer weights 1/5 for that absent query, no strict full-table
  contraction, and no unique M finite fixed point when the contraction test
  fails. Their flat and M-sharp noncontractive traces and loose E_64 bounds
  agree numerically.

## Source and proof audit

I inspected the committed Claude witness and theory. In finite mode,
attention eligibility is only the static role/position mask; pair equality
appears in dot-product scores, not in a content-dependent mask or visited
gate. The exact route separately uses declared equality/absent-row null
probabilities. Dynamic Q enters only as memory values, rewards are immutable
context fields, and pair writes pass through the writer value/output matrices.
The fixed feed-forward map forms the signed residual and the final projection
updates Q then clears scratch. The scalar finite implementation is written
separately from the literal network. The population transition/Q calculation
is labeled audit-only and does not enter prompt construction or network
weights. The theory correctly distinguishes vector telescoping from a norm
bound, treats c_f<1 as sufficient, and does not infer divergence from c_f>=1.

The source-level checks, replayed verifier, raw sequence comparison and proof
audit support every H1--H5 claim made in Claude's report. No undeclared oracle,
protocol retuning, missing artifact, or task-definition defect was found.

## Remaining governance state

The required reciprocal direction is blocked operationally: after the user
authorized the scoped transfer, the second Claude invocation was rejected by
automatic approval with the message that the account usage limit was reached.
No workaround or substitute agent was used. The first direction is therefore
PASS, while Claude's verification_of_other.md for the Codex route is still
pending. The task cannot become VERIFIED until that report is PASS or the user
explicitly rules an exception.

## Follow-up finding: stronger claims exceed numerical evidence

Claude theory section 10, report M-sharp paragraph and witness supplementary
interpretation infer exact consistency, fixed-point existence and convergence
from floating-point eigenvalues and ell^T bf approximately zero. Those
measurements support a numerical indication, not the exact implication.
This violates the frozen requirement to justify stronger spectral claims.
Narrow the claims and raw JSON labels to observed finite-horizon behavior, or
provide rigorous independent justification. This is an ordinary report/proof
defect; it does not change the task definition or require a task objection.

The literal witness's residual_field_after_feedforward snapshot is also taken
after scratch reset, making its label incorrect. Actual residuals are still
available in writer value projections; correct the snapshot label or capture
location and verify the evidence. Only the original Claude author may repair
its files. Previously reported numerical matches do not resolve these issues.

The prior automatic-review usage failure belonged to Codex's approval path;
no evidence established a Kimi quota failure. Current approval availability
is being rechecked under the existing explicit user authorization.

FAIL
