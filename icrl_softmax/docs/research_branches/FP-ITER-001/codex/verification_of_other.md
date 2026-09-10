# FP-ITER-001 Codex verification of Claude route

Current verdict: PASS on Claude repair commit
f6feef6221a5d623f385094650d8a033b605c420. GPT independently replayed the
repaired verifier (1,665 checks, 0 failures), witness and Ruff, audited the
source/proof delta and completed 6,040 raw cross-checks with 0 failures.
The final section contains current evidence and limitations. All earlier
PASS, FAIL and operational-blocker passages below are historical audit trail,
not current status. Claude's reciprocal verdict is still pending.

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

## Original-author repair and final GPT verification

Claude repaired its own four author files and sealed them at
f6feef6221a5d623f385094650d8a033b605c420. I inspected the delta from 4034498:
the witness now captures scratch before reset and labels its columns, the
verifier adds independent residual/writeback/reset checks, and the theory,
report and emitted JSON qualify M-sharp spectral and convergence statements
as numerical indications. The exact flat-case rank-one proof remains valid.
The frozen formulas, fixtures, masks, Q updates and reset behavior are unchanged.

GPT repeated the three reproduction commands listed above against f6feef6
in the Claude worktree's icrl_softmax directory. GPT's new outputs, all under
the root project's results/FP-ITER-001/codex directory, are:

- claude_repaired_replay_results.json (witness exit 0, H1-H5 and capture PASS);
- claude_repaired_replay_verification.json (exit 0, 1,665 checks, 0 failures);
- corresponding .stderr.txt files (empty); scoped Ruff exit 0;
- repair_invariance.json: exact equality before/after repair of all 24 Q
  traces, frozen inputs, operators, fixed-point values, error decompositions,
  flat analysis and structural numerical measurements. The 16 captured
  updates now have residual-versus-writer-value error exactly 0.

Repaired witness SHA-256:
7e27d1cbbd37f73011fa7bdd133aa755bed8997a70b06597cb2eb83965c6cd9f;
verifier SHA-256:
c0b2d88fdb3659ce8f6288d3d29e24b9156242fc08aa8ae9e7e6aa9d3c11281e.
The witness replay SHA-256 is
9404a353946465fa171b3e52d8d22d167deee250fb28307bc100483fc1ba57a0;
verifier replay SHA-256 is
302288adf16c888be1d28430b6142cd9793b8a6c8f7e174e9bc8e4ace1273b7e.

An independent audit script imports neither actor's source. Reproduce from
the repository root with:

    C:/Users/Admin/anaconda3/python.exe -B icrl_softmax/results/FP-ITER-001/codex/cross_check.py --claude icrl_softmax/results/FP-ITER-001/codex/claude_repaired_replay_results.json --output icrl_softmax/results/FP-ITER-001/codex/cross_check_repaired.json

It independently reconstructs the frozen operators, checks all 24 aligned
literal/direct/scalar traces, signed stage vectors, all E_k recursions and
inequalities, and applicable fixed-point/steady/transient quantities.
6,040 comparisons pass. Maximum trace difference is 3.109e-15 (the earlier
16-trace comparison omitted the eight scalar traces); matrix difference is
3.331e-16. The largest absolute bound-rounding difference is 9.346e-5 on an
E_k of order 1e10, within the frozen scaled tolerance. Transient-bound
differences are at most 2.887e-15.

The first raw cross-check attempt used the residual upper bound where both
actors saved the tighter actual-bias transient expression, producing 260
audit-script failures. cross_check_before_repair.json preserves that attempt;
cross_check_corrected.json and cross_check_repaired.json use the correct
expression without changing either actor's scientific output. Failed inline
JavaScript patch invocations made no file change. Exact recursive comparison
of the original Claude witness and initial replay established that only
/code/baseline_commit_at_start differed, replacing the earlier unverified
metadata-only assertion with a checked result.

Execution-scope deviations remain disclosed: the blind CLI run created two
global memory files outside the authorized actor directory, and used a
Python capture wrapper after shell-redirection denial. The repair invocation
also used Python for backup copies after cp/mv denial, but honored the later
output-redirection denial and left its sealed JSON files unchanged. Its
repaired self-run evidence is in the CLI log; GPT's separately authorized
independent replay supplies the full repaired raw outputs above. This report
does not relabel the old Claude JSON files as repaired outputs. Creation
evidence and read-only memory backups are in codex/scope_cleanup; automatic
review rejected deletion and originals remain pending explicit user approval.
These operational deviations are not presented as scope-compliant execution.

The two scientific/evidence defects have been repaired by their original
author and independently verified. This verdict covers the repaired route's
reproducibility and frozen scientific acceptance criteria; it does not grant
a cleanup permission, an operational exception, a main merge or a claim of
policy improvement.

PASS
