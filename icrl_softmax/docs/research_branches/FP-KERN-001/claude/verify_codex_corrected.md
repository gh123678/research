# FP-KERN-001 Claude final verification of the corrected Codex route

Date: 2026-09-09. Verifier branch: `claude/FP-KERN-001` at Claude formal seal
`1a820467683d137e6527edbd99bb486000b2fc58` and prior reciprocal report
`verify_codex.md` (commit `f9666e9d54091e8d563f7c3f097d294ef3ee9ddd`).
Verified object: corrected Codex route on `codex/FP-KERN-001`, worktree
`results/FP-KERN-001/codex_worktree/icrl_softmax`, at corrective
implementation/smoke seal `5af3dc6134a13779908e87558941fa82ed0829eb` and
corrected formal seal `1001d23273bdf29b92d9b84a3f3956e83819da4a` (both full
hashes confirmed via `git rev-parse`; worktree `git status` clean).

This is the final reciprocal verification of the single user-authorized
corrective GPT rerun ("授权纠正重跑", 2026-09-09, recorded in task section
"Corrective-rerun ruling"). Both prior blind seals and both first reciprocal
reports exist, so disclosure is authorized. Verification was read-only with
respect to all Codex files: the evaluator was not rerun, the strict analyzer
ran without `--write-results`, and no Codex file, result, task, design, or
workspace file was modified.

## 1. Authorized scope vs actual change set

The user ruling authorizes exactly three definition repairs and nothing else:

1. self weight exactly one whenever the target state has target observations,
   without requiring two common non-target signature actions for the self
   contribution;
2. false improvement only as estimated positive action difference with true
   difference nonpositive;
3. secondary Spearman diagnostic computed within each record/action before
   averaging finite correlations.

`git diff --stat 640a3f8..1001d23` covers exactly: `ACTIVE_WORKSPACE.md`
(pointer updates), `docs/research_tasks/FP-KERN-001.md` (discrepancy record
and user ruling), three docs under `docs/research_branches/FP-KERN-001/codex/`
(`verify_claude.md`, `correction_first_result.md`,
`corrected_formal_result.md`), and three source files
(`kernel_state_generalization.py` 22 lines,
`analyze_kernel_state_generalization.py` 114 lines,
`verify_kernel_state_generalization.py` 78 added lines).
`kernel_generalization_mdps.py` and `evaluate_kernel_state_generalization.py`
are byte-identical to the original seal — no environment generator, random
stream, matrix, or CLI change.

All five corrected source files were read in full. The code changes implement
exactly the authorized repairs:

- **Repair 1** (`kernel_state_generalization.py`): the self distance in
  `_leave_one_action_out_distances` is now set to `0.0` unconditionally
  (self weight `exp(0)=1`), while cross-state distances still require two
  common observed non-target actions (`common_count < 2` stays NaN). In
  `_kernel_route`, a positive-count primary pair no longer needs a finite
  neighbor; a zero-count primary pair still does. Zero-count estimates are
  therefore unchanged by construction (self weight 1 times count 0
  contributes nothing), matching the sealed claim.
- **Repair 2** (`analyze_kernel_state_generalization.py`,
  `_false_improvement_counts`): the old two-sided sign-disagreement event
  `(est>0,true<=0) or (est<0,true>=0)` is replaced by the literal one-sided
  event `est>0 and true<=0`, on the same common-finite comparison set, for
  both compared routes symmetrically.
- **Repair 3** (`_record_action_correlations`): one Spearman correlation per
  record/action over eligible cross-state pairs (minimum 3 pairs, finite
  statistic required), then averaged over finite correlations with a
  Student-t interval; summary key renamed `record_action_summary`.
- **Disclosed label alignment**: a nonpositive/nonfinite kernel denominator
  is now labelled `kernel_denominator_invalid` (frozen ordered contract
  reason 7) instead of `target_source_unavailable`, and the
  `target_source_unavailable` check now sits after the bandwidth check,
  matching the frozen reason order 4-5-6-7. This was disclosed in both Codex
  correction documents; it changes no estimate, coverage, or screen numeric
  (the denominator-failure path does not occur in the sealed data; primary
  reasons remain `ok`/`insufficient_common_actions`, and the anchor still
  abstains on all zero-count pairs: bin-0 n=0 in both families in the
  corrected `summary.json`).
- **Test-first regression evidence**: `verify_kernel_state_generalization.py`
  adds exactly three tests, one per repair
  (`verify_sparse_positive_self_weight_without_signature_support`,
  `verify_one_sided_false_improvement`,
  `verify_record_action_spearman_granularity`);
  `correction_first_result.md` records the red-first run
  (`ImportError: cannot import name '_false_improvement_counts'`) before the
  implementation change.

No kernel form, bandwidth rule, cross-state support rule, environment
family, seed, stream construction, formal matrix, threshold, or decision
rule was changed. No scientific tuning beyond the three repairs was found.

## 2. Hash verification (recomputed by Claude)

Corrected canonical `results/FP-KERN-001/codex/` — all seven recomputed
SHA-256 hashes match `corrected_formal_result.md` exactly:

- `analysis.json` `5fef6fd0f615b6301d274544af9c18e54fe04634cd6d5cee16ef929bf79d32a7`
- `checks.log` `a8666ad610f4fcc969dd589b0b2a8dbef94276e14268fea58afc824dd53780ed`
- `commands.log` `2c53a81f617d244a0965b48d4fdabdc7eafea28e7a973766750a1343558f37ee`
- `config.json` `78aa1bcb5bd2529ab7346412a818ec95e2058deb07dff6436777424e074fb33a`
- `environment.json` `53602b573c9ae80948e5d0403d4d314a6f381e9ddda01beb32992e1a17f830d4`
- `summary.json` `0c5d1b039720acb3b3710846f1e490589a07f5eb7fd6f2ed859ba3520515b640`
- `task_results.json` `9f3e777e819fb64625bc2c119bdc5ad462a62277b2ff337b04d2f360253c5da1`

Preserved first-run artifacts `results/FP-KERN-001/codex/original_formal/` —
all seven recomputed hashes are byte-identical to the original seal record in
`formal_result.md` (`776656e2…`, `a8666ad6…`, `0a0887a8…`, `78aa1bcb…`,
`a2000e7d…`, `381b230e…`, `d5392da0…`). `config.json` is identical across
runs (frozen matrix unchanged) and `checks.log` is identical across runs
(both runs logged the same two PASS lines: 480 records, NOT_SUPPORTED);
`environment.json` differs only by capture timestamp (original vs corrected
run time, consistent with the commit timeline 14:06 seal → 14:08 run → 14:23
seal). Corrective smoke `correction_smoke/` — all seven hashes match
`correction_first_result.md` and it contains exactly 16 records. After my
read-only analyzer run below, all seven canonical hashes were recomputed
again and remained identical.

## 3. Executable verification (run in the Codex worktree, `python -B`)

- `verify_kernel_state_generalization.py`: PASS ("kernel state-generalization
  checks passed"), now including the three corrective regression tests.
- `verify_fixed_policy_q_routes.py`: PASS.
- `verify_finite_sample_theorems.py`: PASS.
- `verify_visit_indexed_martingale_certificate.py`: PASS.
- `verify_time_uniform_mixture_certificate.py`: PASS.
- Task-scoped `ruff check` on the five task files: PASS ("All checks
  passed!").
- `analyze_kernel_state_generalization.py --result-dir
  results/FP-KERN-001/codex` (no `--write-results`): PASS — "strict analysis
  with 480 records; classification=NOT_SUPPORTED". The analyzer independently
  regenerated every record from the serialized seed components and asserted
  nested equality at 1e-12; zero reconstruction mismatches; its internal
  before/after hash guard confirms frozen inputs unmodified.

No evaluator or formal sampling was run by this verification.

## 4. Corrected frozen screen and cross-route agreement

Recomputed from the corrected canonical `summary.json`/`analysis.json` (480
records, `reconstruction_mismatches: 0`, classification `NOT_SUPPORTED`):

| Item | Codex corrected current | Claude current | Codex corrected hidden | Claude hidden |
|---|---|---|---|---|
| 1 coverage >= 50% | 100% (183/183) pass | 100% (227/227) pass | 100% (235/235) pass | 99.56% (224/225) pass |
| 2 zero-RMSE improvement | +7.25% fail | +4.50% fail | +1.47% fail | +7.58% fail |
| 3 1-4 RMSE improvement | -142.90% fail | -112.51% fail | -118.39% fail | -119.78% fail |
| 4 top-action | +2.64pp fail | +4.41pp fail | +3.71pp fail | +4.56pp fail |
| 5 false improvement | 20.51% vs 21.61% pass | 18.75% vs 19.79% pass | 17.51% vs 18.80% pass | 17.99% vs 19.48% pass |
| Screen / classification | FAIL | FAIL | FAIL | FAIL |

Every screen item agrees with the independent Claude route: both families
fail items 2-4 and pass items 1 and 5. Both routes classify `NOT_SUPPORTED`
under the frozen decision rule (hidden-cluster screen fails). The corrected
secondary record/action Spearman diagnostic is positive with intervals
excluding zero in both families (current 0.2809 [0.2568, 0.3050]; hidden
0.3163 [0.2927, 0.3399]) and remains non-screen evidence, consistent with
the Claude route (0.283/0.307). Per-record numeric levels differ because the
two routes use different authorized deterministic substream derivations
(documented in `verify_codex.md` section 5); pass/fail outcomes are
identical on all ten item-family pairs.

## 5. Corrected-vs-original metric changes

Comparing the corrected screen with the original GPT screen
(`formal_result.md`):

- **Zero-count coverage and item 2 unchanged** (183/183, 235/235; +7.25%,
  +1.47%): repair 1 only frees the self contribution, which has weight times
  zero count at zero-count pairs, and cross-state support is untouched.
- **Item 3 gap narrows** (current -154.41% → -142.90%; hidden -121.51% →
  -118.39%): sparse-positive pairs now always carry self weight one, and
  positive-count pairs without an eligible neighbor now emit the
  self-only (local-equivalent) estimate instead of abstaining, diluting the
  paired RMSE difference toward zero. Still catastrophically negative; item 3
  still fails.
- **Item 4 rises** (current +2.12pp → +2.64pp; hidden +2.65pp → +3.71pp):
  more primary emissions create more complete sparse-state rows. Still below
  the 5pp threshold (current CI includes zero); item 4 still fails.
- **Item 5 rates roughly halve symmetrically** (current 38.13/39.58 →
  20.51/21.61; hidden 34.39/35.99 → 17.51/18.80): the one-sided event is a
  strict subset of the old two-sided sign disagreement, applied to both
  compared routes. Primary remains within 1pp of the pool; item 5 still
  passes.
- **Spearman means rise** (0.2170 → 0.2809; 0.2355 → 0.3163) under the
  record/action granularity; still positive and significant; non-screen.
- **No screen item changed pass/fail in either family**; classification
  remains `NOT_SUPPORTED`. The ruling's prediction ("none changes any
  screen-item pass/fail result or the common NOT_SUPPORTED classification")
  is confirmed empirically.

## 6. Single authorized rerun

- Canonical `commands.log` records exactly one formal `EVALUATION`, writing
  to the isolated `results/FP-KERN-001/codex/corrected_formal` directory with
  the frozen matrix and seed; `original_formal/commands.log` records the one
  original formal run; `correction_smoke/commands.log` records the one
  16-record smoke. No other formal command exists anywhere in the evidence.
- `_prepare_output` refuses a nonempty directory, so the corrective formal
  run could not have overwritten the original artifacts; the originals were
  moved byte-for-byte to `original_formal/` (hashes re-verified above).
- Git history shows one corrective implementation/smoke seal (`5af3dc6`)
  before the single corrected formal seal (`1001d23`), preceded by the
  user-ruling record (`7fc2d53`) and the discrepancy record (`ad8c3d1`).
- `checks.log` in each directory shows exactly one evaluator PASS line and
  one analyzer PASS line.

No new formal run occurred beyond the one user-authorized corrective rerun,
and no Claude formal rerun was authorized or needed (the Claude route already
followed the literal frozen definitions).

## 7. Conclusion

The corrected Codex route is mechanically sound and reproducible: 480 records
reconstructed independently with zero mismatches, all seven canonical hashes
match the corrected seal, all preserved original and smoke artifacts remain
byte-identical to their recorded seals, all five verifiers and task-scoped
Ruff pass, and the change set is exactly the three user-authorized definition
repairs plus the disclosed frozen-contract label alignment. The corrected
frozen screen agrees with the independent Claude route on every item in both
families, and the classification remains `NOT_SUPPORTED` under the frozen
decision rule.

PASS
