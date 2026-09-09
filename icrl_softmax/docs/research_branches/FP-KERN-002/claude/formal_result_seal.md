# FP-KERN-002 Claude route: formal 480-record result and seal

Date: 2026-09-09.

Route: Claude independent executor on `claude/FP-KERN-002`, worktree
`results/FP-KERN-002/claude_worktree/`, common execution-start commit
`ffdf26b029efde08ea794454a7b5da890108c355`.

Blindness statement: at the time of this seal, no GPT FP-KERN-002 branch
content, GPT implementation, GPT smoke/full result, or GPT conclusion was
read. The only GPT-authored FP-KERN-002 material consumed is the mandated
read-only task set (task, design, plan, activation metadata, recorded Claude
pre-review) plus the frozen common input and the verified FP-KERN-001
task/code/evidence as shared read-only inputs.

## Implementation seal provenance (externally created)

The blind implementation/smoke seal commit
`a39323c8011acebfb651c8431d8598e1d1aee244`
(`[claude] seal FP-KERN-002 implementation and smoke`) was created by the
principal account outside this session because OS-level DENY ACLs on
`C:\Users\Admin\Desktop\research\.git` block all git writes from this session
(see `git_seal_blocker.md`). This session prepared exactly the five committed
files; the principal account committed them without content changes. Verified
in this session before the formal run:

- `git status` clean; `git diff a39323c HEAD` empty; the worktree content of
  `analyze_kernel_reuse_diagnostics.py` (81542 bytes),
  `verify_kernel_reuse_diagnostics.py` (19988 bytes), and the three evidence
  documents is exactly what this session produced.
- The commit contains only those five files (3003 insertions, no deletions).

## Pre-formal gate rerun (this session, after the seal)

- Frozen input hashes (match frozen task values exactly):
  - `input/config.json` SHA-256
    `78aa1bcb5bd2529ab7346412a818ec95e2058deb07dff6436777424e074fb33a`;
  - `input/task_results.json` SHA-256
    `9f3e777e819fb64625bc2c119bdc5ad462a62277b2ff337b04d2f360253c5da1`;
  - `input/source_manifest.json` SHA-256
    `670648f7a2761d919f58b25881db45dc6d9d49d4fec25e307c6fe72bf8b31966`.
- `python verify_kernel_reuse_diagnostics.py` ->
  `kernel reuse diagnostic checks passed (14 fixtures)`.
- `ruff check analyze_kernel_reuse_diagnostics.py verify_kernel_reuse_diagnostics.py`
  -> `All checks passed!`.
- Inherited verifiers, all PASS:
  `verify_kernel_state_generalization.py`,
  `verify_fixed_policy_q_routes.py`,
  `verify_finite_sample_theorems.py`,
  `verify_visit_indexed_martingale_certificate.py`,
  `verify_time_uniform_mixture_certificate.py`.
- Read-only legacy strict-analysis regression:
  `python analyze_kernel_state_generalization.py --result-dir <FP-KERN-001 canonical>`
  -> `PASS FP-KERN-001 strict analysis with 480 records; classification=NOT_SUPPORTED`.
- Old-input immutability: the sealed FP-KERN-001 canonical `config.json` and
  `task_results.json` hash byte-identical to the frozen values above, and the
  common input was re-hashed unchanged after the formal run.

## Sole formal run (exactly once)

`results/FP-KERN-002/claude/` was absent before generation (confirmed by
`ls` failure and by the analyzer's own absent-or-empty guard, which aborts
otherwise). Command:

```text
python analyze_kernel_reuse_diagnostics.py --input-dir C:/Users/Admin/Desktop/research/icrl_softmax/results/FP-KERN-002/input --result-dir C:/Users/Admin/Desktop/research/icrl_softmax/results/FP-KERN-002/claude --mode full --write-results
```

Output:

```text
PASS FP-KERN-002 full diagnostic on 480 records; classification=NO_BORROWING_EVIDENCE
```

Environment (`environment.json`): Python 3.13.9 (Anaconda,
`C:\Users\Admin\anaconda3\python.exe`), NumPy 2.4.6, SciPy 1.16.3,
Windows-11-10.0.22631-SP0.

## Formal classification and gates

Ordered frozen decision rule applied exactly; outcome:
**`NO_BORROWING_EVIDENCE`**.

Hidden-family gates (`summary.json`):

- `peer_headroom` (`oracle_q_nearest2`): **false** — fails criterion 3
  (`sparse_rmse`, relative improvement -0.3342, paired CI
  [-0.1772, -0.0416], i.e. the oracle-Q route is significantly worse than
  unpooled local estimates on 1-4-count pairs). Criteria 1, 2, 4, 5 pass:
  zero coverage 235/235 = 1.0; zero RMSE -39.37% relative with CI
  [0.2280, 0.4114] excluding zero; top-action +23.82pp with CI
  [0.1800, 0.2965]; false improvement 9.21% vs 18.88% baseline.
- `generator_structure_useful` (`oracle_generator_cluster`): **false** —
  fails criteria 2 (zero RMSE relative -0.46%, CI [-0.0989, 0.0914] includes
  zero) and 3 (sparse RMSE relative -1.2336, CI [-0.5030, -0.3043]).
  Criteria 1 (coverage 223/235 = 0.9489), 4 (+11.99pp, CI
  [0.0592, 0.1806]), and 5 (14.71% vs 18.89%) pass.
- `observable_structure_useful` (`observable_balanced_cluster`): **false** —
  fails criteria 2 (zero RMSE relative +5.32% but CI [-0.0575, 0.1423]
  includes zero) and 3 (sparse RMSE relative -0.9754, CI
  [-0.3924, -0.2459]). Criteria 1 (coverage 227/235 = 0.9660), 4
  (+6.34pp, CI [0.0068, 0.1201]), and 5 (16.39% vs 18.89%) pass.

Current-family observable screen: **false** (criteria 2, 3, 4 fail),
consistent with the ordered rule; with all three hidden gates false the
classification is `NO_BORROWING_EVIDENCE` regardless.

Secondary hidden-family diagnostics (`analysis.json`):

- Emitted record/actions: 960; excluded `partition_tie`: 0.
- Adjusted Rand index vs generator labels: n=960, mean 0.1516203703703704,
  95% Student-t [0.1217019663212322, 0.1815387744195086].
- Micro peer precision: 5656 / 11520 = 0.4909722222222222 (chance 0.5 for
  balanced clusters: the observable partition carries essentially no
  generator-label information).
- Oracle-benefit recovery: unavailable in both bins — the generator-cluster
  mean improvement is nonpositive (zero bin -0.0221, 1-4 bin -0.4036), so
  the frozen rule reports `null`; not clipped.

Abstentions (all ordinary, recorded, none silent): `oracle_q_nearest2`
11520 ok; `oracle_generator_cluster` 5745 ok + 15 `target_source_unavailable`
+ 240 `not_applicable_family` records on the current family;
`observable_balanced_cluster` 11499 ok + 21 `target_source_unavailable`.

## Stage 0 inside the formal run

`baseline_reproduction.json`: exact (bit-for-bit) replay of all four
predecessor routes on all 480 records from saved observable fields; sealed
predecessor `summary.json` matched at absolute tolerance 1e-12
(`sealed_summary_comparison: atol_1e-12_match`); replayed classification
`NOT_SUPPORTED`; both predecessor family screens reproduced.

## Post-run independent verification (this session)

1. Strict JSON: every bundle file (`analysis.json`,
   `baseline_reproduction.json`, `diagnostic_records.json`,
   `environment.json`, `output_hashes.json`, `source_manifest.json`,
   `summary.json`) parses with `NaN`/`Infinity` rejected.
2. Output hashes: independent `sha256sum` of all eight files matches
   `output_hashes.json` exactly:
   - `analysis.json`
     `08c5f2b0859d8feac8ac770ddce730b67b78dd38d02904aed03c584747d8c015`;
   - `baseline_reproduction.json`
     `b6eee0ef9dd657b9ddabde52b7c152dfcdfabc24203801ac062d1992f5b81d2a`;
   - `checks.log`
     `efa3fa60442982a6ff155e3e011329338172a07cbcb8972ffc3edd4e273cbc7e`;
   - `commands.log`
     `cd816bf7bb57b9db53f35d8876b175da2dd249b15e6b48c8ffce71bebeaa55af`;
   - `diagnostic_records.json`
     `4964276d392e020c02f3185e7e0b2cf868530ce5211f5c4944d9cb2da121fecc`;
   - `environment.json`
     `4e3534d08c41dee1d48ed5966d6e1fcbf7c4762b3ad9567ae06b6f91334b59fd`;
   - `source_manifest.json`
     `58ba090e4fd5e2b2b2b5ae77fba77e3e1d2c57c6564ba7f85b7c8b85bf5a97c5`;
   - `summary.json`
     `1e6f5452d49ebb6b03e6aad482989cca1548c133a0acde5cdd45ed7dc9d5d018`.
3. Independent reconstruction script `formal_output_check.py` (this
   directory; written fresh, does not import the analyzer; Ruff-clean):
   `PASS all independent reconstruction, screen, secondary, abstention, and
   classification checks` over 506 check groups, including:
   - matrix identity: 480 records, 240/240 families, all 240 frozen matrix
     cells present once, per-record identity fields equal to the frozen input;
   - bit-for-bit re-derivation of every estimate, denominator, peer list,
     source set, abstention reason, and all ten serialized partition scores
     per action from the frozen input (zero mismatches across 480 records);
   - pool invariants: zero-count targets never self-pool, positive-count
     targets always self-include, abstentions always null estimates;
   - predecessor signature-eligibility and zero-count coverage denominators
     re-derived independently (183 current / 235 hidden eligible pairs);
   - all five screen items, all metrics, and all 95% Student-t intervals for
     every route/family recomputed and matched at absolute tolerance 1e-12;
   - ARI and peer precision recomputed from serialized partitions with an
     independent ARI implementation (mean matched at 1e-12);
   - abstention tallies and the ordered classification recomputed and
     matched.
4. Anomaly review: no failures, repairs, retries, or unrecorded abstentions
   occurred during the formal run; no `partition_tie` occurred on the full
   matrix (tie path is fixture-covered and smoke-exercised);
   oracle-benefit recovery unavailability follows the frozen rule and is
   reported, not an anomaly.

## Acceptance criteria assessment (task order)

1. PASS — source files hash byte-identical to frozen values (above).
2. PASS — 480 records, every frozen matrix cell once (independent check).
3. PASS — Stage 0 exact replay + 1e-12 metrics + `NOT_SUPPORTED`.
4. PASS — old input/result directories unchanged (canonical hashes equal the
   frozen values; input re-hashed unchanged after the run).
5. PASS — oracle-Q peer order and two-peer cap re-derived bit-for-bit.
6. PASS — generator-cluster sources use labels only; construction reads no
   true Q (verifier fixtures + independent source-set re-derivation).
7. PASS — observable boundary rejects oracle fields (verifier fixtures) and
   consumes only allowed observable inputs by construction.
8. PASS — ten partitions, missing-distance 1.0, sorted scoring, unique
   minimum, tie abstention all re-derived exactly (fixtures cover ties; the
   full matrix produced none).
9. PASS — state-relabelling equivariance fixture-verified; unique-optimum
   selection serialized.
10. PASS — every estimate reconstructs from declared source sets, sums, and
    counts (bit-for-bit, all 480 records).
11. PASS — every zero-count estimate uses only other-state observations.
12. PASS — coverage denominators keep every frozen eligible pair (183/235).
13. PASS — common sets, errors, actions, false improvements, intervals,
    cluster diagnostics, and screens independently reconstructed (506 groups).
14. PASS — ordered classification applied exactly: `NO_BORROWING_EVIDENCE`.
15. PASS — test-first record, smoke, full reconstruction, source/output
    hashes, inherited verifiers, strict JSON, task-scoped Ruff all pass.
16. PASS — commands, environment, failures (none), repairs (none this run),
    anomalies (none), hashes, limitations, and this assessment recorded.
17. PARTIAL / IN PROGRESS — both routes share the activation commit and
    common input; Claude blind seals are recorded; reciprocal reproduction
    awaits disclosure and the GPT route's completion.
18. OPEN — reciprocal reports not yet due in the blind phase.
19. PASS — `main` untouched; no merge or push performed.

## Limitations

- This is empirical feasibility evidence on the sealed FP-KERN-001 corpus
  only; it certifies no policy nondegradation and no general
  identifiability.
- `NO_BORROWING_EVIDENCE` is conditioned on the frozen five-item screen:
  the oracle-Q route shows real zero-count headroom (criteria 1, 2, 4, 5
  pass) but materially degrades 1-4-count pairs, so even favorable fixed
  peer selection fails the unchanged screen; the injected generator clusters
  are not useful for target-Q pooling even with known membership, and the
  observable partitions recover almost no generator-label structure
  (ARI 0.152, peer precision 0.491 vs 0.5 chance).
- Oracle-benefit recovery ratios are unavailable by the frozen rule
  (nonpositive generator-cluster improvement in both bins).
- No `partition_tie` occurred on the full matrix, so the tie-abstention
  path is evidenced by deterministic fixtures and smoke, not by formal data.
- The implementation/smoke seal commit was created by the principal account
  from this session's prepared files because this session cannot write git
  objects (DENY ACLs on `.git`); content identity was verified before the
  formal run as recorded above.

## Seal request

The formal result bundle at `results/FP-KERN-002/claude/` (nine files,
hashes above) and the evidence files in
`docs/research_branches/FP-KERN-002/claude/` are final. The `[claude]`
formal-result seal commit will be attempted from this session; if the git
write blocker persists, the exact staging list is reported in the progress
reply and in `git_seal_blocker.md` remains the standing diagnosis. No rerun
of the formal diagnostic is permitted without user authorization.
