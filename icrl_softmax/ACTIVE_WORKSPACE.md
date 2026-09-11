# Active research workspace

## Current state

No task is active. The most recent task,
`docs/research_tasks/FP-ESARSA-001.md` (v1.1), closed `VERIFIED` on
2026-09-11 by an explicit user exemption of the independent GPT acceptance
(see the acceptance-exemption ruling in the task sheet); no GPT acceptance
artifact exists and this route was never independently reconstructed, so the
task's verification rests on single-route evidence only.

Task summary: fixed-policy Expected SARSA with a cross-fitted Bellman-residual
certificate and one certified relative-softmax improvement step on the frozen
480-record protocol inherited from FP-ADV-001. Activation prerequisite
satisfied (FP-ADV-001 VERIFIED). By the direct user ruling of 2026-09-11
(Codex quota exhausted), this task used the FP-EXPL-001 v1.1 responsibility
pattern: Claude main execution on `claude/FP-ESARSA-001` with author seal.
Claude's itemized pre-review returned APPROVED
(`docs/research_branches/FP-ESARSA-001/codex/claude_pre_review.md`).
Scientific baseline `9d0994f03e4a659787c74df4660cdcab1d398c1c`; all
inherited science files are byte-identical to the v1.0 baseline. Design:
`docs/superpowers/specs/2026-09-09-fixed-policy-expected-sarsa-relative-softmax-design.md`.

Claude's main route is sealed at `1cb59f0` on `claude/FP-ESARSA-001`; it was
pushed to `origin` and fast-forwarded into `main` by explicit user
authorization on 2026-09-11 (`main` and `origin/main` both at `0ac8eb6`).
One frozen 480-record formal run into `results/FP-ESARSA-001/claude/`:
283/480 certificates per route, 0 oracle certificate/residual-event/value
violations, generator identity 0/480 mismatches against the frozen FP-TU-001
baseline, and no safe update emitted, so hypothesis 8 is a verified negative
usefulness result with no retuning. Route evidence:
`docs/research_branches/FP-ESARSA-001/claude/` (`theory.md`, `first_result.md`,
`formal_result.md`, `report.md`, `failure_history.md`); main-route report
`docs/research_branches/fixed_policy_expected_sarsa_report.md`. Verdict for
acceptance criteria 1-19: PASS; criterion 20 closed by the
acceptance-exemption user ruling.

### Measured scale of the current bottleneck (2026-09-11)

The two policy-improvement tasks both emitted zero updates. Reading their
numbers together localizes the obstruction to certificate scale rather than to
the construction:

- `results/FP-ESARSA-001/claude/summary.json`: among emitted certificates
  `E_Q` has minimum `22.474` and mean `57.686`; the minimum oracle bound slack
  is `22.302`; the largest realized oracle Q sup-error is `3.771`;
- non-emission reasons are exactly `heldout_pair_support_missing` (0.410) and
  `improvement_lcb_nonpositive` (0.590); no other reason fires;
- `FP-ADV-001` independently emitted `0/480` on all six routes.

So the certificate is roughly one order of magnitude looser than the realized
error, and the relative-softmax one-step improvement is far smaller than that
slack. This is the same obstruction `CTRL-PREFLIGHT-001` predicted
analytically (error lower bound `6.343 > B = 5`). Any next task must choose a
reachable guarantee or a tighter certificate scale deliberately, and record the
choice before execution; no frozen parameter of a closed task may be retuned.

## Verified kernel tasks: FP-KERN-001 and FP-KERN-002

Both tasks are `VERIFIED` with both reciprocal verification directions `PASS`.
They were executed on branches forked from `c579047` and, through a scheduling
gap, were never merged to `main`; they were merged under explicit user
approval on 2026-09-11. Their results are independent of, and convergent with,
the action-gap and Expected SARSA negative results above: they close the
cross-state-generalization direction rather than the certificate direction.

### FP-KERN-001: data-derived cross-state kernel feasibility

Status: `VERIFIED`. Both independent 480-record routes and the
user-authorized corrective GPT rerun classify the frozen kernel route as
`NOT_SUPPORTED`: both families pass zero-count coverage and false-improvement
control but fail the zero-count RMSE, `1-4`-count RMSE, and sparse-state
top-action requirements.

The frozen design compares `oracle_q_nearest2`,
`oracle_generator_cluster`, and `observable_balanced_cluster` against the
reconstructed predecessor controls. It reuses the predecessor's five-item
sparse-estimation screen and assigns one ordered scoped conclusion. Only a
passing observable route may be described as promising; oracle results remain
diagnostic.

`FP-KERN-001` is the verified predecessor at
`403884ae6bde46c7c3578ae01d77422ed03faf05`. Its corrected Gaussian-kernel
classification remains `NOT_SUPPORTED` and is not modified by this task.

### FP-KERN-002: reused-record oracle and learnability diagnostic

Status: `VERIFIED`, classification `NO_BORROWING_EVIDENCE`. Holding all 480
verified `FP-KERN-001` trajectories and results fixed, three diagnostic
constructions were compared against the reconstructed predecessor controls:
`oracle_q_nearest2`, `oracle_generator_cluster`, and
`observable_balanced_cluster`. All four ordered decision inputs are false, so
no route passed the unchanged five-item screen.

Both routes sealed independently and reached the same classification: GPT
implementation/smoke `f37b9730aea4ba692b854dcfb89f8f3d17faa34e`, formal
`0815d0dbef3a8f7784438ac89e2df195d3cab00b`; Claude implementation/smoke
`a39323c8011acebfb651c8431d8598e1d1aee244`, formal
`718d77053801c6f9e3dd958513b7a06918a5e274`. GPT verification of Claude and
Claude verification of GPT both `PASS` (248,566 cross-route checks, zero
failures); synthesis at
`docs/research_branches/FP-KERN-002/codex/final_synthesis.md`.

The scientifically important part is the diagnosis, not the null: coverage was
high everywhere (true-Q nearest-two `100%`; observable balanced cluster
`99.45%` current / `96.60%` hidden), so simple action inaccessibility is not
the failure mode. Even the favorable true-Q nearest-two route worsened the
`1-4`-count RMSE by `33.42%` (hidden) and `49.45%` (current) while improving
zero-count RMSE, and the observable partition recovered little latent
structure (mean adjusted Rand index `0.1516`, peer precision `0.4910`).
Unconditional count-weighted same-action borrowing across states therefore
fails on this corpus under the frozen peer and partition rules.

Boundary recorded by the task: this does not reject every form of cross-state
generalization. Count-aware gating or shrinkage, borrowing only for zero-count
actions, learned state representations, cross-fitting, and estimators that
retain local evidence are untested and require a new frozen hypothesis.

### FP-KERN evidence location correction

Both task sheets advertise `results/FP-KERN-001/codex/` and
`results/FP-KERN-001/claude/` as the canonical result directories. Those
host-level paths do not exist: because `results/` is Git-ignored and route
worktrees were created under it, the sealed 480-record corpora live inside the
registered worktrees at

```text
results/FP-KERN-001/codex_worktree/icrl_softmax/results/FP-KERN-001/codex/
results/FP-KERN-001/claude_worktree/icrl_softmax/results/FP-KERN-001/claude/
results/FP-KERN-002/input/            (common immutable 480-record corpus)
results/FP-KERN-002/codex_worktree/icrl_softmax/results/FP-KERN-002/codex/
results/FP-KERN-002/claude_worktree/icrl_softmax/results/FP-KERN-002/claude/
```

The corpora were verified present on 2026-09-11 (`task_results.json`
33,805,703 bytes, plus `summary.json`, `checks.log`, `commands.log`,
`environment.json`, `analysis.json`, `config.json`, and the preserved
`original_formal/`, `correction_smoke/`, `original_smoke/` archives under the
GPT route). No artifact was moved or regenerated by this index correction; the
recorded task-sheet paths were left as authored and this section is the
authoritative locator.

## Verified fixed-exploration task: FP-EXPL-001

`docs/research_tasks/FP-EXPL-001.md` (v1.1) is `VERIFIED` (2026-09-11).
Claude executed the frozen protocol as main author (user ruling), GPT
performed independent post-seal acceptance, and Claude performed the renewed
executable reciprocal review.

- Frozen protocol: one 64-transition Markov batch (seed 20260911), fixed
  behavior and target policies, grouped-mean Q iteration, 64 updates from
  Q0=0, sharpness xi=zeta=tau=8, reward of pair (1,1) changed to 0.25 before
  sampling. No resampling or scans.
- Coverage passed with counts [19,16,12,17]. Direct reference, exact grouped
  attention, literal finite network and independent scalar finite formula
  agree (max cross-route deviation <= 1.34e-15). Exact and finite operators
  both contract; a rational-arithmetic certificate proves Gf strictly
  positive with exact row sums 17/20, hence c_f = 0.85 exactly for this
  batch. Final error decomposition at k=64: iteration 4.894e-05, finite
  softmax 1.79092e-03, data bias 0.03806150093295335; the two true state
  values differ. Bounds are reported, not sample-complexity claims.
- Author seal: `6512252934614807916953a65eb5e6ba7ee4a5a5` on
  `claude/FP-EXPL-001`. GPT post-seal replay (verify/witness/Ruff all exit
  0) and independent acceptance: 2589 checks, 0 failures, PASS at
  `results/FP-EXPL-001/codex/verification.json`; replay manifest under
  `results/FP-EXPL-001/codex/postseal_replay/6512252934614807916953a65eb5e6ba7ee4a5a5/`.
- Claude reciprocal review: executable rerun of the GPT acceptance (exit 0,
  identical 2589-check record), all snapshot SHA-256 values match the replay
  manifest, mutation guards 13/13, no q_pi network input, no hidden Q lookup
  or visitation-frequency multiplier. Report ends PASS at
  `docs/research_branches/FP-EXPL-001/claude/verification_of_other.md`,
  commit `5025cdf535b8f1c9d1460930ccc5ffdd92d3bd87`.
- The earlier 20-check GPT PASS and the text-only reciprocal PASS were
  superseded and archived; the resumed-audit repairs (probe boundary,
  data-bias proof) were sealed by the author and are covered by the final
  acceptance. The GPT/Claude execution-provenance deviation (session-env
  EPERM) remains disclosed in both reports.
- Codex exhausted its execution quota during closure; the user explicitly
  authorized Claude on 2026-09-11 to take over all closure work, including
  this update, the synthesis/codex report final texts, the merge to main and
  the remote push. GPT's acceptance verdict stands on its own executable
  artifact cited above.
- Synthesis: `docs/research_branches/FP-EXPL-001/codex/synthesis.md`.
  This task's closure led directly to the activation of FP-ESARSA-001
  (see Current state); it does not itself authorize policy improvement
  or online control claims beyond that task's frozen scope.
  The question it leaves open is the one the next task should answer: under a
  fixed policy, how do the coverage event of a real trajectory and the
  sampling error enter a provable bound, and how do those differ from the
  audit-only bound that depends on the true `q_pi`?

## Verified predecessor: FP-ITER-001

Current construction gate: `docs/research_tasks/FP-ITER-001.md` (v1.1, `VERIFIED`). It narrows the
next step to a two-state/two-action fixed-policy Expected SARSA iteration
witness: direct reference, exact grouped attention, finite-logit attention,
and a per-iteration error/stability decomposition. The v1.0 review missed
unfrozen details. V1.1 freezes complete/missing batches, initial Q, 64 updates,
two sharpness triples and dual independent routes. The dedicated branch now
exists; v1.1 pre-review returned APPROVED. Both actors start independently from
the common activation commit, with reciprocal verification after blind seals.
Codex preliminary construction is now implemented at cb26a339, with 6,332
self-checks passing. Complete coverage at sharpness 8 contracts; missing-pair
write leakage remains. The complete fixture has zero population/data bias,
which limits that diagnostic. Claude's blind route is sealed at 4034498.
GPT's follow-up audit found unsupported M-sharp convergence claims and a
mislabeled scratch snapshot. Claude repaired both at f6feef6; GPT's replay
of the repair passes 1,665 checks plus 6,040 raw cross-checks and the scientific
review now returns PASS. All 24 aligned traces differ by at most 3.109e-15;
the repair changes none of their values. Claude's reciprocal review is sealed
at 5c7aea1 and ends PASS. It independently replayed the Codex verifier and
witness, matched 84,697 witness leaves and 17,779 cross-route comparisons,
and found no scientific mismatch. Both reciprocal reports now PASS; the task
is VERIFIED. Operational deviations and the retained global-memory cleanup
blocker remain disclosed in the handoff and reports.
See docs/research_branches/FP-ITER-001/codex/synthesis.md, verification_of_other.md
and handoff.md.
It does not activate the broader FP-ESARSA certificate
matrix, policy improvement, or online control.

## Prior verified diagnostic (CTRL-PREFLIGHT-001)

Prior diagnostic: `docs/research_tasks/CTRL-PREFLIGHT-001.md` (`VERIFIED`), an
economical source comparison and minimal finite-head diagnostic authorized on
2026-09-10. GPT branch: `codex/CTRL-PREFLIGHT-001`. Source comparison is recorded
in `docs/research_branches/CTRL-PREFLIGHT-001/codex/report.md`. User authorized
scoped external transfer and delegated execution: GPT defines requirements,
Claude pre-reviews and executes the diagnostic, GPT independently verifies.
Task pre-review returned APPROVED. After user-authorized continuation, Claude
repaired the missing /k and its report's analytic proof. Both reciprocal
verdicts are PASS; accepted Claude commit da6a727. The 36 head cases and
inherited checks pass, and the old certificate's no-emission obstruction is
confirmed analytically. See codex/report.md for preserved failures and final
acceptance. No current blocker. User requests conserving GPT quota: delegate
execution/report work to Claude, retain targeted independent GPT verification.
Provisional next gate: literal fixed-policy Expected SARSA construction with
explicit finite-error conditions; no overall control-route winner is claimed.
FP-ESARSA remains DRAFT; no new formal matrix or control algorithm was run.

## Verified action-gap task (FP-ADV-001)

Status: `FP-ADV-001` is `VERIFIED` as a negative usefulness result.  Claude's
task-scoped pre-review returned `APPROVED`; Codex and Claude then independently
sealed matching 480-record formal results, and both reciprocal verification
directions returned `PASS`.  All six routes emitted zero updates, so the safe
one-step theorem is retained while empirical usefulness hypothesis 8 is
falsified without retuning.

That task asked whether the verified `FP-TU-001` event can certify only
the action differences used by one policy update. V-first local exact and
finite-softmax bounds are primary; complete-Q V-first and Direct-Q bounds are
controls. The update moves half of each certified donor's mass above `pi_min`
and must guarantee `V^{pi_plus} >= V^pi` componentwise.

That task reused the frozen 480-record protocol and spends no new risk budget.
It excludes repeated control, new Direct-Q local theory, variance adaptation,
oracle inputs, outcome tuning, and conditional-on-emission claims.

`FP-TU-001` remains the verified certificate baseline. Its local merge and
remote synchronization are complete at `c579047950dfabb2600020cd2e53dd24b3e39c84`.

## Research governance

- Mode: GPT principal researcher; Claude Code auxiliary executor and independent verifier; user final arbiter.
- Canonical rules: `../AGENTS.md`.
- Active governance task: `docs/research_tasks/GOV-001.md` (`VERIFIED`).
- GPT branch: `codex/gpt-led-research-governance`.
- Claude role for GOV-001: read-only pre-review and final verification.
- Current blocker: none; GPT verification and Claude content/Git verification returned `PASS`.
- Next action: apply the verified governance rules to the next GPT-authored research task.
- Approved design: `docs/superpowers/specs/2026-08-31-gpt-led-claude-verified-research-governance-design.md`.

## Verified action-gap execution evidence

- Task: `docs/research_tasks/FP-ADV-001.md` (`VERIFIED`).
- Git and scientific baseline:
  `c579047950dfabb2600020cd2e53dd24b3e39c84`.
- Frozen DRAFT task-definition baseline:
  `919c26f8f2dc90b093cc1f40c9b36d2be2b45b03`.
- Common execution-start commit:
  `10a9a94e24ec92a59e7c756f9af6ce07b2f30e59`.
- Common route execution-start commit:
  `4078f6911cbfb4654205772685f49896e4e8cad2`.
- GPT branch: `codex/FP-ADV-001`.
- Claude role: read-only pre-review completed with `APPROVED`; isolated
  independent execution is explicitly authorized.
- GPT blind first-result seal:
  `996641d9ce2088e95c0bf2a0661e3b24b6e0d6fe`; all proof, verifier, Ruff,
  eight-record smoke, reconstruction, preservation, policy, separation, and
  dominance checks passed before the seal.
- GPT formal-evidence seal:
  `5f190ebb78697acd3cd877c1d64898929bb88024`; the sole 480-record formal
  evaluator run passed, and all six routes emitted zero updates. A bounded
  no-donor serialization defect was fixed at
  `440710c8c9c77ee8128d4abfd96919ffafe7991c` and mechanically repaired from
  saved observable inputs without rerunning the matrix.
- Claude isolated route: formal result sealed at
  `191821b26b16e13de323fb31651343ffe1eb9656` after blind seal
  `191e13ba5472ce4c183643008169236979c000ff`; GPT independently replayed all
  480 records and 17,280 route-state entries with zero failures.
- Claude executable reciprocal verification: `PASS` on 2026-09-10.  The
  verifier, strict read-only 480-record analyzer, and task-scoped Ruff all
  passed; no formal evaluator rerun or repository mutation occurred.  Claude
  recorded its own report on `claude/FP-ADV-001` at commit `42fb0cc`.
- Current blocker: none.
- Shared conclusion:
  `docs/research_branches/action_gap_certificate_report.md`.
- Next task: `docs/research_tasks/FP-ESARSA-001.md` v1.1 became `ACTIVE` after
  this verification and later closed `VERIFIED` on 2026-09-11 by an explicit
  user exemption of the independent GPT acceptance (see Current state). Its
  activation prerequisite (this task being VERIFIED) was satisfied.
- Design:
  `docs/superpowers/specs/2026-09-08-action-gap-safe-update-design.md`.
- Plan:
  `docs/superpowers/plans/2026-09-08-action-gap-safe-update-plan.md`.
- Prior verified task: `docs/research_tasks/FP-TU-001.md` (`VERIFIED`).

## FP-KERN-001 execution evidence

- Task: `docs/research_tasks/FP-KERN-002.md` (`ACTIVE`).
- Predecessor baseline:
  `403884ae6bde46c7c3578ae01d77422ed03faf05`.
- Approved design commit:
  `be6b7213eeba08d3b9750a850ac2de99ff17be89`.
- DRAFT task-definition baseline:
  `b7ef163f11eb5ee41499344c296587efd3516651`.
- Activation commit:
  `2308c372eb47ce7c181f98caee952121f4e47644`.
- Common route execution-start commit:
  `28c4ae0f68ca51c7c9a0fd981159e85b7742dd4c`.
- GPT branch and isolated worktree: `codex/FP-KERN-001` in
  `results/FP-KERN-001/codex_worktree/`.
- Claude route: formal seal `1a820467683d137e6527edbd99bb486000b2fc58`;
  classification `NOT_SUPPORTED`; GPT verification `PASS`.
- GPT route: original formal seal
  `640a3f8fb2d0f41b96eef8d9bb76fc5ef2e9b93b`; corrective implementation and
  smoke seal `5af3dc6134a13779908e87558941fa82ed0829eb`; corrected formal seal
  `1001d23273bdf29b92d9b84a3f3956e83819da4a`; classification
  `NOT_SUPPORTED`.
- Reciprocal verification: GPT verification of Claude is `PASS` at
  `5e538ad06e1167c8c644db9db7f7f25573b470cb`; Claude verification of the
  corrected GPT route is `PASS` at
  `c17561620d02a08210b1702f4ff576e7bde17366`.
- Current blocker: none; all acceptance evidence is recorded. The merge to
  `main` was authorized by the user on 2026-09-11 and performed with the
  FP-KERN-002 closure.
- Design:
  `docs/superpowers/specs/2026-09-09-kernel-state-generalization-feasibility-design.md`.
- Plan:
  `docs/superpowers/plans/2026-09-09-kernel-state-generalization-feasibility-plan.md`.

## FP-KERN-002 execution evidence

- Task: `docs/research_tasks/FP-KERN-002.md` (`VERIFIED`); classification
  `NO_BORROWING_EVIDENCE`.
- Predecessor and branch baseline:
  `403884ae6bde46c7c3578ae01d77422ed03faf05` (`FP-KERN-001` verified).
- Frozen DRAFT task-definition baseline:
  `e04db4c17c7648bc751bef1620ab0d01fe1cb3a3`.
- REVIEW clarification closure:
  `04584e44e788b7cb50289c6dd447c356bd4081b6`.
- Activation commit:
  `698ebdc62859ec26ae6b623099ab499fc6b410fa`.
- Common execution-start commit:
  `ffdf26b029efde08ea794454a7b5da890108c355`.
- GPT branch and isolated worktree: `codex/FP-KERN-002` in
  `results/FP-KERN-002/codex_worktree/`.
- User decision: reuse the exact verified 480-record predecessor corpus; no
  new trajectories.
- Claude read-only pre-review: `APPROVED` on
  `f72d4fec81bf43f2efe39a770afad68f63d559d9`, with 12 passed checks and seven
  nonblocking wording/secondary-metric clarifications now closed.
- Common input: created once and frozen at host path
  `results/FP-KERN-002/input/`; both source hashes match the task and manifest
  SHA-256 is
  `670648f7a2761d919f58b25881db45dc6d9d49d4fec25e307c6fe72bf8b31966`.
- Claude branch and isolated worktree: `claude/FP-KERN-002` in
  `results/FP-KERN-002/claude_worktree/`, created from the common start before
  implementation.
- Current blocker: none; task verification is complete.
- GPT blind route: verifier-first implementation/smoke sealed at
  `f37b9730aea4ba692b854dcfb89f8f3d17faa34e`; the sole 480-record formal run
  and all post-run checks passed with initial classification
  `NO_BORROWING_EVIDENCE`.
- GPT blind formal seal: `0815d0dbef3a8f7784438ac89e2df195d3cab00b`.
- Claude blind route: implementation/smoke seal
  `a39323c8011acebfb651c8431d8598e1d1aee244`, formal seal
  `718d77053801c6f9e3dd958513b7a06918a5e274`, independently reaching the same
  `NO_BORROWING_EVIDENCE` classification.
- GPT verification of Claude: `PASS`; every gate and metric agrees, 506
  independent reconstruction groups and all inherited checks passed.
- Claude verification of GPT: `PASS`, sealed at
  `1567603d5d8f8fd99c266c17a3741ae8fa30c7d3`; 248,566 cross-route checks
  passed with zero failures.
- Final task status: `VERIFIED`; final classification:
  `NO_BORROWING_EVIDENCE`. The merge to `main` was authorized by the user on
  2026-09-11.
- Next action: user decides whether to open a new task for zero-only borrowing,
  count-aware gating/shrinkage, or learned state representations. The
  certificate-scale direction recorded under "Measured scale of the current
  bottleneck" is the separate, currently unopened alternative.
- Design:
  `docs/superpowers/specs/2026-09-09-kernel-reuse-oracle-learnability-design.md`.
- Plan:
  `docs/superpowers/plans/2026-09-09-kernel-reuse-oracle-learnability-plan.md`.
- Prior verified task: `docs/research_tasks/FP-KERN-001.md` (`VERIFIED`).

## Active implementation

- `action_gap_certificate.py`
- `verify_action_gap_certificate.py`
- `evaluate_action_gap_certificates.py`
- `analyze_action_gap_certificates.py`
- `evaluate_fixed_policy_q_routes.py`
- `fixed_policy_finite_sample_certificate.py`
- `verify_finite_sample_theorems.py`
- `analyze_fixed_policy_finite_sample_certificates.py`
- `verify_fixed_policy_q_routes.py`
- `crossfit_vfirst.py`
- `markov_coverage_certificate.py`
- `verify_crossfit_markov_certificate.py`
- `evaluate_blockwise_q_routes.py`
- `mdps.py`
- `visit_indexed_martingale_certificate.py`
- `evaluate_visit_indexed_certificates.py`
- `analyze_visit_indexed_certificates.py`
- `verify_visit_indexed_martingale_certificate.py`
- `time_uniform_mixture_certificate.py`
- `verify_time_uniform_mixture_certificate.py`
- `evaluate_time_uniform_certificates.py`
- `analyze_time_uniform_certificates.py`
- `fixed_policy_expected_sarsa.py`
- `verify_fixed_policy_expected_sarsa.py`
- `evaluate_fixed_policy_expected_sarsa.py`
- `analyze_fixed_policy_expected_sarsa.py`
- FP-KERN-001 implementation: `kernel_state_generalization.py`,
  `kernel_generalization_mdps.py`, `verify_kernel_state_generalization.py`,
  `evaluate_kernel_state_generalization.py`, and
  `analyze_kernel_state_generalization.py` (Codex corrected route at these
  canonical paths; the Claude route's byte-identical copies live under
  `docs/research_branches/FP-KERN-001/claude/route/`).
- FP-KERN-002 implementation: `analyze_kernel_reuse_diagnostics.py` and
  `verify_kernel_reuse_diagnostics.py`.
- Planned FP-KERN-002 implementation: `analyze_kernel_reuse_diagnostics.py`
  and `verify_kernel_reuse_diagnostics.py` (not yet created at activation).

## Active evidence

- `docs/research_branches/`
- `docs/superpowers/specs/2026-08-29-direct-q-v-first-balanced-exploration-design.md`
- `docs/superpowers/specs/2026-08-29-crossfit-markov-certificate-design.md`
- `docs/superpowers/specs/2026-08-31-shared-fixed-policy-finite-sample-theorem-design.md`
- `docs/superpowers/plans/2026-08-31-shared-fixed-policy-finite-sample-theorem-plan.md`
- `docs/research_branches/shared_fixed_policy_finite_sample_theory.md`
- `docs/superpowers/plans/2026-08-29-direct-q-v-first-balanced-exploration-plan.md`
- `docs/superpowers/plans/2026-08-29-crossfit-markov-certificate-plan.md`
- `results/fixed_policy_q_routes/`
- `results/fixed_policy_q_routes_crossfit/`
- `results/fixed_policy_finite_sample_certificates/`
- `results/blockwise_q_routes/`
- `docs/research_branches/visit_indexed_martingale_certificate_theory.md`
- `docs/research_branches/visit_indexed_martingale_certificate_report.md`
- `results/FP-MART-001/codex/`
- `results/FP-MART-001/claude/`
- `docs/superpowers/specs/2026-09-04-time-uniform-mixture-certificate-design.md`
- `docs/superpowers/plans/2026-09-04-time-uniform-mixture-certificate-plan.md`
- `docs/research_tasks/FP-TU-001.md`
- `docs/research_branches/time_uniform_mixture_certificate_theory.md`
- `docs/research_branches/time_uniform_mixture_certificate_report.md`
- `results/FP-TU-001/codex/`
- `results/FP-TU-001/claude/`
- `docs/research_tasks/FP-ADV-001.md`
- `docs/research_branches/action_gap_certificate_theory.md`
- `docs/research_branches/action_gap_certificate_report.md`
- `results/FP-ADV-001/codex/`
- `results/FP-ADV-001/claude/`
- `docs/research_tasks/FP-ESARSA-001.md`
- `docs/research_branches/fixed_policy_expected_sarsa_theory.md`
- `docs/research_branches/fixed_policy_expected_sarsa_report.md`
- `results/FP-ESARSA-001/claude/`
- `docs/superpowers/specs/2026-09-09-kernel-state-generalization-feasibility-design.md`
- `docs/superpowers/plans/2026-09-09-kernel-state-generalization-feasibility-plan.md`
- `docs/research_tasks/FP-KERN-001.md`
- `docs/research_branches/FP-KERN-001/codex/formal_result.md`
- `docs/research_branches/FP-KERN-001/codex/corrected_formal_result.md`
- `docs/research_branches/FP-KERN-001/codex/correction_first_result.md`
- `docs/research_branches/FP-KERN-001/codex/verify_claude.md`
- `docs/research_branches/FP-KERN-001/claude/formal_result.md`
- `docs/research_branches/FP-KERN-001/claude/verify_codex.md`
- `docs/research_branches/FP-KERN-001/claude/verify_codex_corrected.md`
- Kernel-route result corpora: see "FP-KERN evidence location correction"
  above; the task-sheet `results/FP-KERN-001/...` paths resolve to the
  registered route worktrees, not to host-level directories.
- `docs/research_branches/FP-KERN-001/claude/route/` (relocated Claude route
  code, byte-identical blobs, plus the path-remapping README)
- `docs/superpowers/specs/2026-09-09-kernel-reuse-oracle-learnability-design.md`
- `docs/superpowers/plans/2026-09-09-kernel-reuse-oracle-learnability-plan.md`
- `docs/research_tasks/FP-KERN-002.md`
- `docs/research_branches/FP-KERN-002/codex/claude_pre_review.md`
- `docs/research_branches/FP-KERN-002/codex/first_result.md`
- `docs/research_branches/FP-KERN-002/codex/formal_result.md`
- `docs/research_branches/FP-KERN-002/codex/claude_verification.md`
- `docs/research_branches/FP-KERN-002/codex/final_synthesis.md`
- `docs/research_branches/FP-KERN-002/claude/route/` (relocated Claude route
  code, byte-identical blobs, plus the path-remapping README)

Each independent formal route contains 480 same-seed comparisons, strict-JSON
route certificates, zero-mismatch legacy regression, certificate/failure
summaries, exact/softmax analysis, and complete execution evidence. Exact
emission is 2.5%/85%/100%/100%; primary usefulness (`total_bound < B`) appears
only at length 16384 for V-first exact (40%) and V-first softmax (3.33%).

## Historical archive

Earlier diagnostics, generated outputs, smoke runs, superseded specs/plans, and temporary build trees were moved intact to:

`C:\Users\Admin\Desktop\research\_archive\icrl_softmax_history_20260831`

The archive is recoverable and remains outside the active project directory. Python and Ruff caches were deleted because they are reproducible.
