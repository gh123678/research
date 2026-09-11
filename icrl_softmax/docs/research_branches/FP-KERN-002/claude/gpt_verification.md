# FP-KERN-002 Claude verification of the GPT route

Date: 2026-09-09.

Verifier: Claude Code reciprocal verifier on `claude/FP-KERN-002`, worktree
`results/FP-KERN-002/claude_worktree/`. Frozen task:
`docs/research_tasks/FP-KERN-002.md` (version 1.0). This report is the
symmetric executable verification of the GPT route required after both blind
formal-result seals existed and disclosure was authorized by the user.

## Scope and seals under verification

- GPT implementation/smoke seal: `f37b9730aea4ba692b854dcfb89f8f3d17faa34e`
  (parent `9a60a9861059e24897b2402ee9dc4602f8604417`).
- GPT formal seal: `0815d0dbef3a8f7784438ac89e2df195d3cab00b` (parent
  `f37b9730aea4ba692b854dcfb89f8f3d17faa34e`).
- GPT sealed worktree (read-only during this verification):
  `results/FP-KERN-002/codex_worktree/`.
- GPT formal bundle:
  `results/FP-KERN-002/codex_worktree/icrl_softmax/results/FP-KERN-002/codex/formal/`.
- Claude route seals used as the independent reference:
  `a39323c8011acebfb651c8431d8598e1d1aee244` (implementation/smoke) and
  `718d77053801c6f9e3dd958513b7a06918a5e274` (formal), bundle at
  `results/FP-KERN-002/claude/`.
- No GPT file or result was modified; all GPT-side commands were read-only
  verifier/verify-mode executions.

## Blindness and commit sequence

Checked with `git merge-base --is-ancestor` in both worktrees:

- neither Claude seal (`a39323c`, `718d770`) is an ancestor of the GPT formal
  seal `0815d0d`;
- neither GPT seal (`f37b973`, `0815d0d`) is an ancestor of the Claude formal
  seal `718d770`;
- `git ls-tree` of `0815d0d` and `f37b973` contains no
  `docs/research_branches/FP-KERN-002/claude/` content — only the mandated
  shared read-only task set and GPT's own `codex/` evidence;
- both routes descend from the common execution-start identity
  `ffdf26b029efde08ea794454a7b5da890108c355` (GPT branch via `9a60a98`),
  which both routes record in their bundles (`source_manifest.json`
  `common_execution_start_commit`, GPT `formal_result.md`, Claude
  `formal_result_seal.md`);
- seal order per route is implementation/smoke before formal
  (GPT `f37b973` -> `0815d0d`; Claude `a39323c` -> `718d770`).

Blindness and sequencing conform to the frozen task.

## Single-formal-run evidence

- GPT `formal/commands.log` contains exactly one command:
  `analyze_kernel_reuse_diagnostics.py --mode formal --input-dir
  C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-KERN-002\input
  --output-dir results\FP-KERN-002\codex\formal`.
- All nine formal bundle files carry one write timestamp
  (2026-09-09 16:46:56 +08:00), matching `environment.json`
  `created_at=2026-09-09T16:46:56.952900+08:00`; the smoke bundle predates it
  (16:41), matching the frozen smoke-then-full protocol.
- The GPT analyzer enforces the absent-or-empty output guard
  (`_prepare_empty_output`, analyzer line 1158-1163), so a rerun over an
  existing bundle would have aborted.
- `environment.json` `script_sha256`
  (`a065d295066c747f872799c6b20b5c34b189399700c5e6d4013c341c8cdb9c09`)
  independently re-hashed equal to the sealed analyzer file.

## Independent hash checks

- Frozen common input (recomputed SHA-256, all three match the frozen task
  values exactly): `config.json`
  `78aa1bcb5bd2529ab7346412a818ec95e2058deb07dff6436777424e074fb33a`,
  `task_results.json`
  `9f3e777e819fb64625bc2c119bdc5ad462a62277b2ff337b04d2f360253c5da1`,
  `source_manifest.json`
  `670648f7a2761d919f58b25881db45dc6d9d49d4fec25e307c6fe72bf8b31966`.
- GPT `formal/artifact_hashes.json`: all eight entries re-hashed equal to the
  files on disk; the bundled `source_manifest.json` hash equals the frozen
  input manifest hash.
- The four artifact hashes sealed in GPT `formal_result.md`
  (`diagnostic_records.json` `90ae246b...`, `summary.json` `7b52e13f...`,
  `analysis.json` `dbb66d55...`, `artifact_hashes.json` `88f33238...`) all
  match the files on disk.
- Smoke bundle `artifact_hashes.json` likewise re-verified (also covered by
  the strict bundle verifier run below).
- Predecessor canonical corpus unchanged: the FP-KERN-001 canonical
  `config.json` / `task_results.json` hash byte-identical to the frozen
  values, and the canonical `summary.json` file hash equals GPT's recorded
  frozen file identity
  `0c5d1b039720acb3b3710846f1e490589a07f5eb7fd6f2ed859ba3520515b640`.

## Executable checks run in this session (GPT worktree, read-only)

1. GPT task verifier:
   `python -B verify_kernel_reuse_diagnostics.py --input-dir <common input>`
   -> `PASS FP-KERN-002 verifier and frozen common-input checks`
   (fixtures: ten balanced partitions, target-action exclusion, prohibited
   oracle input rejection, missing-distance 1.0, tie abstention, both oracle
   constructions, ARI, one-sided false improvement, full decision table,
   unique-optimum state-permutation equivariance).
2. Strict saved-bundle reconstruction (GPT analyzer `--mode verify`, which
   re-derives baseline, all records, summary, and analysis from the frozen
   input and asserts nested-exact equality plus artifact hashes):
   - formal: `PASS FP-KERN-002 strict bundle verification; records=480;
     classification=NO_BORROWING_EVIDENCE`;
   - smoke: `PASS FP-KERN-002 strict bundle verification; records=16;
     classification=NO_BORROWING_EVIDENCE`.
3. Task-scoped Ruff:
   `ruff check analyze_kernel_reuse_diagnostics.py verify_kernel_reuse_diagnostics.py`
   -> `All checks passed!`.
4. Inherited predecessor verifier (byte-identical in both worktrees; `diff`
   confirmed `verify_kernel_state_generalization.py`,
   `kernel_state_generalization.py`, and
   `analyze_kernel_state_generalization.py` identical across routes):
   `python -B verify_kernel_state_generalization.py`
   -> `kernel state-generalization checks passed`.
5. Read-only legacy strict-analysis regression on the unchanged canonical
   predecessor result:
   `python -B analyze_kernel_state_generalization.py --result-dir
   <FP-KERN-001 canonical>` ->
   `PASS FP-KERN-001 strict analysis with 480 records; classification=NOT_SUPPORTED`.
6. Test-first record: both entry points first appear in seal `f37b973`; GPT
   `first_result.md` records the verifier failing first with
   `ModuleNotFoundError` before the analyzer existed, consistent with the
   sealed commit content.

## Independent cross-route reconciliation

Fresh reconciliation code written for this verification (imports neither
route's analyzer; Ruff-clean), preserved as
`docs/research_branches/FP-KERN-002/claude/gpt_bundle_reconciliation.py`:

```text
python -B docs/research_branches/FP-KERN-002/claude/gpt_bundle_reconciliation.py
checks=248566 failures=0
PASS GPT formal bundle reconciles exactly with the Claude sealed route and frozen input
```

Coverage of the 248,566 checks, all passing:

- strict-JSON parse (NaN/Infinity and duplicate keys rejected) of every GPT
  bundle file;
- per-record identity fields against the frozen input for all 480 records;
- per record/action, for all three routes: abstention reasons, denominators,
  and estimates reconciled; every emitted GPT estimate reconstructs exactly
  from GPT's declared source list, sums, and counts via the route formula
  (NumPy summation in declared order);
- oracle-Q route: at most two positive-count peers, frozen ranking
  (|true-Q gap|, then state index) verified from GPT's ranked serialization,
  self-inclusion exactly when the target count is positive — on all 480
  records;
- generator-cluster route: serialized `groups` equal the frozen input
  `cluster_by_state` labels on every hidden record; all sources share the
  target's label; `oracle_use` labels confirm no true-Q input;
- observable route: `observable_input_keys` equal the frozen allowed input
  set on every record; all ten candidate partitions with state zero first;
  all ten candidate scores equal the Claude route's serialized scores exactly
  (same enumeration order); unique-minimum selection and zero
  `partition_tie` cases on the full matrix; missing-distance replacement 1.0;
  every serialized leave-one-action-out distance re-derived from the frozen
  input (NumPy formula) exactly, including symmetry and the at-least-two-
  common-actions availability rule;
- zero-count targets never self-pool; abstained cells always serialize null
  estimates;
- abstention tallies recomputed from GPT's bundle equal the Claude route's:
  `oracle_q_nearest2` 11520 ok; `oracle_generator_cluster` 5745 ok + 15
  `target_source_unavailable` + 240 `not_applicable_family`;
  `observable_balanced_cluster` 11499 ok + 21 `target_source_unavailable`;
- every screen quantity for every route/family — eligible/covered coverage
  (183 current, 235 hidden denominators kept intact), route/baseline RMSE
  means, relative improvements, all paired 95% Student-t intervals
  (n/mean/lower/upper), top-action means and intervals, false-improvement
  counts/rates, all five criterion booleans, and `screen_pass` — exactly
  equal between routes;
- secondary diagnostics exactly equal: ARI n=960, mean
  0.1516203703703704, CI [0.1217019663212322, 0.1815387744195086]; peer
  precision 5656/11520 = 0.4909722222222222; oracle-benefit recovery
  unavailable in both bins under the frozen rule (generator means
  -0.02207533508899333 zero bin, -0.40363349849569985 1-4 bin), ratio null
  and unclipped;
- gates all false in both routes (`peer_headroom`,
  `generator_structure_useful`, `observable_structure_useful`,
  `observable_current_family_pass`) and the ordered classification is
  `NO_BORROWING_EVIDENCE` in both;
- Stage 0: GPT `baseline_reproduction.json` reports zero route
  reconstruction mismatches, classification `NOT_SUPPORTED`, and family
  screens exactly equal to the Claude route's replayed screens; GPT's
  canonical payload hash
  `e6327594ccca29e02eeb718849be09d7445a56f1ccbc2d53ee8129d79d2da9d0`
  independently reproduced here by CRLF->LF normalization of the sealed
  predecessor `summary.json` (whose on-disk file hash matches GPT's recorded
  `0c5d1b03...` identity).

## Harmless serialization/schema differences (explicit statement)

All differences below were quantified by the reconciliation script; none
affects any gate, metric, interval, diagnostic, abstention, or the
classification:

1. Source-list order (9,366 record/action cells, `oracle_q_nearest2` only):
   GPT serializes peers in frozen ranking order after the target state
   (`[target, nearest, second]`); Claude serializes the same set in ascending
   state order. The selected sets are identical everywhere; the frozen rule
   constrains selection, not list order.
2. Zero-count group members in declared source lists (1,482 record/action
   cells, generator and observable routes): GPT declares only positive-count
   members of the selected group; Claude declares all selected group members.
   Every divergent member has `target_counts == 0`, contributing exactly zero
   to numerator and denominator; denominators and estimates are identical.
3. `oracle_q_nearest2` estimate floating-point evaluation order (1,564 of
   11,520 cells): GPT sums in ranked order, Claude in ascending order; under
   cancellation (large sums, small quotient) the serialized doubles differ by
   at most 260 ULP, i.e. maximum relative difference 2.98e-14. Each route's
   estimates reconstruct exactly from its own declared source order, and all
   downstream screen metrics and intervals are bit-identical between routes.
4. Schema naming and shape: GPT `summary.json` uses `gates`,
   `route_mean/route_count/route_rate`, a bare record list, per-partition
   `candidate_scores` with `tie_count`/`missing_pair_count`, extra route
   fields (`oracle_use`, generator `groups`), an object form of
   `not_applicable_family` in screens, and `artifact_hashes.json`; the Claude
   bundle uses `hidden_family_gates` + `observable_current_family_pass`,
   `primary_mean/primary_count/primary_rate`, a metadata wrapper around
   `records`, a flat `scores` list, and `output_hashes.json`. All shared
   quantities reconcile exactly under this mapping.
5. Stage 0 identity recording: GPT records the frozen CRLF file hash and the
   canonical LF payload hash separately; the Claude route recorded an
   `atol_1e-12_match` comparison. Both identities were independently
   re-derived in this session and agree.

## Acceptance-relevant judgments for the GPT route

1-4 (source identity, matrix completeness, Stage 0, old-result immutability):
verified above. 5-9 (oracle peer order and cap, generator labels-only
construction, observable boundary, exact partition machinery, equivariance):
fixture-verified by GPT's verifier run here and record-level re-derivation in
the reconciliation. 10-14 (estimate reconstruction, zero-count other-state
rule, coverage denominators, full metric reconstruction, ordered
classification): verified exactly. 15 (test-first, smoke, strict bundle,
hashes, inherited verifier, strict JSON, Ruff): all executed and passing.
16 (commands, environment, anomalies): recorded; no failures, repairs, or
unrecorded abstentions observed; the sole documented repair is the Stage 0
CRLF provenance-label correction recorded in `first_result.md`, which changes
no science. 17 (common start, common input, blind seals, reciprocal
reproduction): satisfied.

## Verdict

The GPT route is reproducible, byte-identical at the sealed hashes, internally
and externally consistent with the frozen task, and reconciles exactly with
the independently sealed Claude route on every gate, screen metric,
confidence interval, secondary diagnostic, abstention, and the ordered
classification `NO_BORROWING_EVIDENCE`, with only the harmless
serialization/schema differences enumerated above.

PASS
