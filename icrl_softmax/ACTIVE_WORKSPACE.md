# Active research workspace

## Current objective

Current task: `docs/research_tasks/FP-ESARSA-001.md` (v1.1, `ACTIVE`).
Fixed-policy Expected SARSA with a cross-fitted Bellman-residual certificate
and one certified relative-softmax improvement step on the frozen 480-record
protocol inherited from FP-ADV-001. Activation prerequisite satisfied
(FP-ADV-001 VERIFIED). By the direct user ruling of 2026-09-11 (Codex quota
exhausted), this task uses the FP-EXPL-001 v1.1 responsibility pattern:
Claude main execution on `claude/FP-ESARSA-001`, GPT independent post-seal
acceptance on `codex/FP-ESARSA-001` when its quota is restored, then Claude
reciprocal review. Claude's itemized pre-review returned APPROVED
(`docs/research_branches/FP-ESARSA-001/codex/claude_pre_review.md`).
Scientific baseline `9d0994f03e4a659787c74df4660cdcab1d398c1c`; all
inherited science files are byte-identical to the v1.0 baseline. Design:
`docs/superpowers/specs/2026-09-09-fixed-policy-expected-sarsa-relative-softmax-design.md`.

Claude's main route is sealed at `1cb59f0` on `claude/FP-ESARSA-001`; the
workspace-pointer commit `cc49599` was pushed to `origin` and fast-forwarded
into `main` by explicit user authorization on 2026-09-11 (`main` and
`origin/main` both at `cc49599`). One frozen 480-record formal run into
`results/FP-ESARSA-001/claude/`: 283/480 certificates per route, 0 oracle
certificate/residual-event/value violations, generator identity 0/480
mismatches against the frozen FP-TU-001 baseline, and no safe update emitted,
so hypothesis 8 is a verified negative usefulness result with no retuning.
Route evidence: `docs/research_branches/FP-ESARSA-001/claude/`
(`theory.md`, `first_result.md`, `formal_result.md`, `report.md`,
`failure_history.md`); main-route report
`docs/research_branches/fixed_policy_expected_sarsa_report.md`. Verdict for
acceptance criteria 1-19: PASS. Results remain **preliminary** until GPT
post-seal acceptance (criterion 20), which is quota-blocked, or an explicit
user exemption.

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
  (see Current objective); it does not itself authorize policy improvement
  or online control claims beyond that task's frozen scope.

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
- Next task: `docs/research_tasks/FP-ESARSA-001.md` is now v1.1 `ACTIVE`
  (see Current objective). Its activation prerequisite (this task being
  VERIFIED) is satisfied.
- Design:
  `docs/superpowers/specs/2026-09-08-action-gap-safe-update-design.md`.
- Plan:
  `docs/superpowers/plans/2026-09-08-action-gap-safe-update-plan.md`.
- Prior verified task: `docs/research_tasks/FP-TU-001.md` (`VERIFIED`).

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

Each independent formal route contains 480 same-seed comparisons, strict-JSON
route certificates, zero-mismatch legacy regression, certificate/failure
summaries, exact/softmax analysis, and complete execution evidence. Exact
emission is 2.5%/85%/100%/100%; primary usefulness (`total_bound < B`) appears
only at length 16384 for V-first exact (40%) and V-first softmax (3.33%).

## Historical archive

Earlier diagnostics, generated outputs, smoke runs, superseded specs/plans, and temporary build trees were moved intact to:

`C:\Users\Admin\Desktop\research\_archive\icrl_softmax_history_20260831`

The archive is recoverable and remains outside the active project directory. Python and Ruff caches were deleted because they are reproducible.
