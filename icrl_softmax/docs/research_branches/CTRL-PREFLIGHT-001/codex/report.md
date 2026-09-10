# Control-route comparison: source review

Date: 2026-09-10. Baseline: db9d63043489f4ec5660e71c53e7e842a280d04f.
Status: diagnostic evidence VERIFIED after author repairs and reciprocal PASS.
Research recommendation remains provisional. Earlier failure and handoff sections
are preserved as history; the final section records the accepted revision.

## Comparison of actual computation interfaces

| Interface | Existing two-stage control | Fixed-policy Expected SARSA draft |
|---|---|---|
| Persistent value memory | Q table; reusable canonical pair representation | One canonical token per pair, explicitly specified |
| Current Q retrieval | Direct indexing in model.py, TwoStageSoftmaxQControl.forward | A finite-score current-pair head is specified, not implemented in the draft |
| Successor action set | q_values[next_states] externally supplies the exact action row | All canonical tokens with state similarity plus log policy probability |
| Action weighting | GroupedSoftmaxMax implements beta*Q scores and Q values | Fixed policy log probabilities supply action weights |
| Residual formation | Arithmetic after exact indexing | Fixed linear combination after two attention reads |
| Writeback | KernelizedSoftmaxQTD updates all passed queries, including unvisited pairs | Finite writer explicitly updates all canonical queries |
| Policy change during evaluation | Current Boltzmann continuation changes with Q | Policy stays fixed through evaluation |
| Final policy | Existing performance analysis/experiments use external policy extraction; actual output head needs an explicit witness | Relative-softmax output is specified; literal realization remains to be constructed |
| Full literal witness | Existing sampled-SARSA witness is reusable, but does not establish this entire finite two-stage route | No Expected-SARSA literal witness in the inspected source |
| Environment execution | External sampling and simulator | External sampling and simulator |
| Guarantee scope | Ideal approximate-greedy analysis plus separately stated perturbations | Fixed-policy evaluation and one selective safety test; current certificate preflight is adverse |

Sources: model.py classes GroupedSoftmaxMax, KernelizedSoftmaxQTD,
TwoStageSoftmaxQControl; verify_end_to_end_sarsa.py;
docs/research_tasks/FP-ESARSA-001.md;
docs/superpowers/specs/2026-09-09-fixed-policy-expected-sarsa-relative-softmax-design.md;
manuscript method_experiments.md sections 3.1--3.4.

The existing sampled finite-logit SARSA class still uses a visited-query gate.
It must not be conflated with the two-stage writer, which passes all Q entries
as queries. Reusing a module requires checking its actual support semantics.

## Candidate-specific obstacles

For two-stage control, replacing pre-grouped action rows by all-memory
attention is not just exchanging indexing for a fixed state score. The
proposed finite score combines state similarity with beta*Q. Values in other
states can compete with the requested-state margin. Thus action sharpness and
state-routing sharpness must be analyzed together. Sharpening action choice
alone is not a valid remedy for state leakage.

For fixed-policy Expected SARSA, action scores are log pi(b|u). Each policy
row sums to one, so the draft's requested-state mass has a simpler expression.
This is a real architectural advantage to test. It does not remove current-
read leakage, writer leakage, or the need for a literal policy-output head.

The previously disclosed certificate obstruction remains a separate issue.
The preflight task will reproduce its arithmetic, but will not change the
FP-ESARSA certificate. A working finite head would not cure an always-rejecting
safety test, and a rejecting test would not establish that the head is useless.

## Provisional choice and exact next gate

Existing code maturity favors two-stage control as the initial candidate.
The fixed-policy candidate has cleaner successor-state normalization. There
is not yet enough evidence to decide the winner of the finite routing test.

The approved task must test both successor heads on the same fixed small
fixtures, including the separated-state-value case. It must separately report
agreement with the finite formula and disagreement with the grouped formula.
The selected next implementation should minimize remaining external content
operations while giving an explicit error condition that is nonvacuous on a
declared case. If both fail that gate, report no selection.

Potential publication value is not validated by this audit. A later task must
specify the full permitted architecture, policy output, unvisited-pair behavior,
and a matched literal-network/control reference. It must not call compact
operator tests end-to-end control evidence or infer per-update safety from
an approximate-optimality bound.

## Execution record

- Read-only source inspection completed; user-owned untracked learning record
  preserved.
- New isolated GPT branch and task created. DRAFT seal: ebe6bd3.
- Claude pre-review launch was rejected by automatic approval review before
  process creation. No private files were sent by that attempted launch.
- Reason: authorization did not explicitly cover transfer of AGENTS.md and
  this new task to the configured external Claude Code service.
- No diagnostic implementation or experiment proceeded past the review gate.
- Next action requires explicit scoped external-data authorization; after
  approval, resume the same task rather than create another plan or matrix.

## Delegated execution and independent verification, 2026-09-10

User subsequently authorized scoped delegation. Task v1.1 received Claude
pre-review APPROVED; shared activation commit is
1d53bb0ff34c2553feb6ddfe5931fedf466500f6. GPT remains on
codex/CTRL-PREFLIGHT-001. Claude's branch is claude/CTRL-PREFLIGHT-001 in
C:/Users/Admin/Desktop/research/icrl_softmax/results/CTRL-PREFLIGHT-001/claude_worktree.

Claude created an uncommitted diagnostic at
icrl_softmax/docs/research_branches/CTRL-PREFLIGHT-001/claude/diagnostic.py
inside that worktree; SHA256:
7FD7911590BF568778F4B4A6644CD9F52D165FEE4274B144E29821D90A1DD931.
No report.md or Claude result commit exists. Tracked files remain identical
to the activation commit. GPT did not edit or commit Claude's artifact.

GPT inspected the entire diagnostic and independently executed it with:

```powershell
& 'C:/Users/Admin/anaconda3/python.exe' -B icrl_softmax/docs/research_branches/CTRL-PREFLIGHT-001/claude/diagnostic.py
```

Working directory: the Claude worktree above. Exit 0, 36 rows. GPT compared
all rows against independently calculated state partition factors rather
than trusting the embedded scalar reference. Maximum discrepancy across
finite value, requested-state mass and grouped error: 5.551115123125783e-16.
Maximum normalization error: 2.220446049250313e-16. Claude's internal
finite/scalar weight discrepancy: 8.604228440844963e-16. These pass 1e-12.
The code uses unscaled dot products; an implementation using scaled dot-product
attention would need its fixed scale absorbed into queries/keys. This is not
a full Transformer projection/residual/FFN witness.

For Q=[[-1,-1],[1,1]], requested state 0, zeta=8, action beta=8:

| Read rule | Requested-state mass | Finite/grouped value error |
|---|---:|---:|
| Two-stage all-memory candidate | 0.0003353501304664781 | 1.999329299739067 |
| Fixed-policy all-memory candidate | 0.9996646498695335 | 0.000670700260932966 |

This supports the local normalization obstacle, not empirical superiority or
failure of the existing grouped TwoStageSoftmaxQControl class. Current-read
mass is 0.9989946239146034; the absent-query write fixture records 1/3 per
transition. The latter is formula-level arithmetic, not an end-to-end writer
implementation. The fixed-policy head is a cleaner next *local construction*
candidate, but no control-route winner or publication value is established.

GPT also independently ran the two inherited verifiers with Python -B from
C:/Users/Admin/Desktop/research/icrl_softmax. Both exited 0. Literal SARSA
reference error: 5.551115123125783e-17. Two-stage verifier: all four checks PASS.
Environment: Python 3.13.9, torch 2.11.0+cpu, Windows PowerShell, CPU.
Evidence is in results/CTRL-PREFLIGHT-001/codex/:
verify_end_to_end_sarsa_rerun.json, verify_two_stage_q_control_rerun.json,
independent36.json, verification_audit.json (ignored local evidence).

### Blocking implementation defect

Claude diagnostic.py line 150 computes the mixture contribution as
2B*q_mix(k)/(1-gamma), omitting division by k. The frozen contract requires
2B*q_mix(k)/((1-gamma)*k). GPT independently called the baseline mixture
functions to verify the correction:

| n | k | Analytic floor | Correct numerical contribution | Claude reported |
|---|---:|---:|---:|---:|
| 256 | 5 | 52.3822960080 | 63.6165329709 | 318.0826648543 |
| 1024 | 21 | 25.5599432477 | 32.8189808809 | 689.1985984989 |
| 4096 | 85 | 12.7045729031 | 16.8423640178 | 1431.6009415108 |
| 16384 | 341 | 6.3429654099 | 8.6010214449 | 2932.9483127118 |

The analytic floor still exceeds B=5 even at the largest n. For any positive
minimum pair count K<=floor(n/(2d)), the analytic floor is at least the listed
value. This follows from M(K,q)<=exp(q^2/(2K)), giving
q_mix(K)>=sqrt(2K log(d/delta)). Hence the old certificate has E_Q>B;
Ihat<=B*||pi_plus-pi||_1 implies a negative lower bound for any changed state.
Missing held-out support instead causes rejection. Numerical contributions
at k are not asserted to bound all smaller counts without a monotonicity proof.
This is a diagnosis of the frozen certificate, not all possible certificates.

### Acceptance and exact handoff

- Frozen 36-row head calculation and independent comparison: PASS.
- Existing verifiers and no inherited tracked-source modification: PASS.
- Numerical mixture contribution: FAIL; author must repair line 150.
- Claude executable source-comparison report, provenance, recommendation,
  acceptance checklist and result commit: MISSING.
- Reciprocal verification and overall task acceptance: NOT COMPLETE.

Claude stopped with error_max_budget_usd after its configured USD 0.50 call
limit. The CLI reported USD 0.533722 (post-response overshoot), session
62b4189c-f1ce-4f93-9dfe-ce75a6c4f96b. Together with pre-review USD 0.141851,
reported cost is USD 0.675573; both allowed calls are consumed. No further
call is authorized by the current task. A future call needs a user ruling
on the call-count limit; do not silently reallocate the remaining total.

Accurate next step for Claude if authorized: retain existing diagnostic and
failure outputs; fix only the missing /k; rerun the same frozen checks; finish
the scoped report verifying GPT's source comparison; commit only the two
allowed artifacts on its own branch. No new protocol, fixtures, trajectories,
algorithm, certificate, provider change, push or merge. GPT must then inspect
the fix, rerun the diagnostic, verify raw outputs and report, and record the
independent verdict. The task stays ACTIVE after this failed VERIFYING attempt.

The next scientific gate remains one literal fixed-policy successor/current
read/residual/writeback witness with an explicitly declared finite-error
condition. This task does not authorize implementing that gate, and the old
always-rejecting safety rule must not be mistaken for a useful control result.

FAIL (historical first verification; superseded by the repaired verdict below)

## Final targeted verification and decision

Date: 2026-09-10. User clarified that GPT quota, not Claude quota, should be
conserved. Execution and report repairs were delegated; GPT reused unchanged
evidence and independently verified the changed computation and report diff.

Accepted Claude commits: 651fc6e (code repair and initial report), then
da6a72758537446bcd872f6dd7dc68fedd13a2a1 (report-only repair). The complete
Claude artifacts remain on claude/CTRL-PREFLIGHT-001 in the worktree above;
no merge or push occurred. Its worktree is clean; exactly the two permitted
files differ from the shared activation commit. GPT's unrelated untracked
learning record was preserved. All inherited scientific sources are unchanged.

GPT compared the failed-source snapshot against the repaired diagnostic:
exactly the missing /k was added. An independent rerun exited 0; all four
mixture contributions agree with GPT's earlier independent reference within
1.4210854715202004e-14, below 1e-12. All 36 head rows still pass, maximum
finite/reference error 8.604228440844963e-16. The prior independent closed-form
36-row comparison and inherited-verifier checks are retained because their
code and inputs are unchanged. New evidence:
results/CTRL-PREFLIGHT-001/codex/verification_repair_audit.json.

GPT initially rejected the report at 651fc6e: it called the universal mixture
inequality unproved while claiming its unconditional consequence. The author
fixed the report, preserved the failed revision, and independently checked:

```text
exp(-lambda^2 K/2) cosh(lambda q)
 <= exp(-lambda^2 K/2 + lambda q)
 = exp(q^2/(2K) - (sqrt(K) lambda - q/sqrt(K))^2/2)
 <= exp(q^2/(2K)),   K>0, q>=0, lambda>=0.
```

Normalized nonnegative mixture weights preserve this bound. A conservative
root therefore has q_mix(K)>=sqrt(2K log(d/delta)) for every positive K.
The minimum positive held-out count is at most floor(n/(2d)); the analytic
floor is decreasing in K and exceeds B=5 at all four frozen lengths. Thus
E_Q>B and every changed state has LB<0. Missing support instead rejects.
This proves the stated old-certificate obstruction without assuming that
the four numerical mixture ratios are universal lower bounds. GPT checked
this proof and the entire repaired diff; the earlier writer/current-read
overstatement and exact-equality wording were also corrected.

Acceptance: source audit, all rows/adverse cases, finite-score tolerance,
inherited checks, source preservation/no-oracle inputs, explicit limitations,
one next gate, Claude executable source-review PASS and GPT independent
implementation/report PASS are all satisfied. No outstanding scientific
objection or implementation defect remains within this diagnostic's scope.

Decision: retain fixed-policy Expected SARSA as the preferred *next local
construction gate*, because its successor normalization decouples state
selection from Q scale. Retain two-stage as the code-reuse/control comparison;
do not declare an overall winner. Next gate is one literal fixed-policy
read/residual/writeback and relative-softmax output witness with a declared
nonvacuous finite-error condition. It requires its own reviewed task; it has
not been implemented here. Do not run the old FP-ESARSA matrix merely to
rediscover an always-rejecting certificate. No proof of full control
improvement, practical superiority, or publication novelty is claimed.

PASS
