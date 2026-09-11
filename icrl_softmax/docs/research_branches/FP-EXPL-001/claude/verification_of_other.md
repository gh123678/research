# FP-EXPL-001 (v1.1) — Claude executable reciprocal review of GPT post-seal acceptance

Date: 2026-09-11. Reviewer: Claude (claude/FP-EXPL-001). Scope: task
FP-EXPL-001 v1.1 section 7 step 4 — Claude's renewed, executable reciprocal
review of GPT's post-seal acceptance of author seal
`6512252934614807916953a65eb5e6ba7ee4a5a5`.

This document **replaces and supersedes** the earlier text-only reciprocal
PASS previously stored at this path; that review referred to GPT's pre-seal
preflight and is void per task section 10. No scientific input, seed, policy,
sample, formula, or tolerance was changed by this review. Nothing was pushed
or merged.

## 1. Reviewed objects and hash identity

All SHA-256 values below were recomputed independently by Claude in this
review and compared with GPT's records.

| object | SHA-256 | matches |
|---|---|---|
| GPT acceptance source `verify_claude.py` (codex checkout, `docs/research_branches/FP-EXPL-001/codex/`) | `8a37c790b2fc9a5dcca6dbe7400c135a2761e681ea49d9177ae1b00d5c3435c7` | GPT `verification.json` digests.acceptance_source_sha256 |
| snapshot `results.json` (post-seal replay output) | `8eea2216c6c7df21c5a25d46bd2adcf4067a8a9189db977d1909c38550302990` | replay_manifest replay_raw_hashes; GPT digests.results_sha256 |
| snapshot `verification.json` (author self-verify) | `e4750b34e0b979fb7bf94f46d906bbd0eb0326320c80edc8f73f11c440277db5` | replay_manifest sealed+replay hashes; GPT digests.author_verification_sha256 |
| snapshot `witness.py` | `c48a134b67712aa5361ab6a9e9459a6a5afcda4f0153d36cf97f900d7fa84834` | replay_manifest source_hashes; GPT digests.witness_sha256 |
| snapshot `verify.py` | `f56ce6df3690b4c55a855dae4eafa50f6b1fce6db7ee66561008717730afc4cb` | replay_manifest source_hashes; GPT digests.verify_sha256 |
| snapshot `failure_history.md` / `report.md` / `theory.md` / old `verification_of_other.md` | `a51fbc8c…`, `7a0970d6…`, `f25bcb5b…`, `e0f34187…` | replay_manifest source_hashes (all four, exact) |
| author-sealed raw `results.json` (Claude worktree) | `8f84b6aaa62562b4e97d7794e43cb1e5a081f785fe867d523e6dcc6b67200c89` | replay_manifest sealed_raw_hashes; seal commit message |

The snapshot directory
`results/FP-EXPL-001/codex/postseal_replay/6512252934614807916953a65eb5e6ba7ee4a5a5/`
and its `replay_manifest.json` were read in full. Every hash recorded in the
manifest was reproduced byte-for-byte from the snapshot files. The replay
`results.json` differs from the author-sealed raw `results.json` only in the
`code.baseline_commit_at_start` run-checkout metadata field (039a53a vs
6512252, both documented in the manifest); after normalizing that single
field the two JSON payloads are exactly equal, confirming the manifest's
`scientific_payload_exact_match: true` and
`non_code_top_level_differences: []`.

## 2. Commands executed by Claude in this review

Note on the review request path: the literal script path
`…\icrl_softmax\icrl_softmax\docs\…\verify_claude.py` does not exist (the
project root `icrl_softmax/` appears once inside the repository root
`research/`). The command was executed with the real path
`C:\Users\Admin\Desktop\research\icrl_softmax\docs\research_branches\FP-EXPL-001\codex\verify_claude.py`;
its SHA-256 equals GPT's recorded acceptance source hash, so the executed
program is byte-identical to the one GPT used.

1. GPT acceptance program against the sealed snapshot evidence
   (stdout/stderr captured; `--output` redirects the acceptance record into
   Claude's own evidence directory so GPT's
   `results/FP-EXPL-001/codex/verification.json` is not overwritten):

   ```
   C:\Users\Admin\anaconda3\python.exe -B C:\Users\Admin\Desktop\research\icrl_softmax\docs\research_branches\FP-EXPL-001\codex\verify_claude.py --results "C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-EXPL-001\codex\postseal_replay\6512252934614807916953a65eb5e6ba7ee4a5a5\icrl_softmax\results\FP-EXPL-001\claude\results.json" --output <claude evidence dir>\verify_claude_rerun.json
   ```

   - exit code: **0**; stderr empty (SHA-256 `e3b0c442…b855`, the empty hash)
   - stdout (SHA-256 `deb9371d4adcd70a38c12d298aa57923a4e692ebcdf6ce02d3d3d28ef01e90c8`):
     `{"total_checks": 2589, "total_failures": 0, "verdict": "PASS", "coverage_counts": [19, 16, 12, 17], "c_f": 0.8500000000000001}`
   - acceptance record `verify_claude_rerun.json` SHA-256
     `add35b42fb55c25c39ca8fe0376e46eb9a6c34e7e96ff71d59bd6599234dd676`

2. Acceptance-program mutation self-test (guard checks only; no new research
   samples): same program with
   `--self-test --output …\verify_claude_mutations.json`.
   - exit code: **0**; stdout: `{"total_checks": 13, "total_failures": 0, "verdict": "PASS"}`
   - record SHA-256 `017640877488bb07545e07969bee23156fdcf9dab4a27dcc1e508aac227da35d`

3. Author self-verifier re-run inside the snapshot (`cwd` = snapshot
   `icrl_softmax`):

   ```
   C:\Users\Admin\anaconda3\python.exe -B docs/research_branches/FP-EXPL-001/claude/verify.py
   ```

   - exit code: **0**; regenerated `verification.json` is byte-identical
     (SHA-256 `e4750b34…`) to both the sealed and GPT-replayed copies;
     summary: 1279 checks, 0 failures, verdict PASS, max group errors
     ≤ 1.29e-06 (group C; all others ≤ 6.3e-16 or exactly 0).
   - `witness.py` was deliberately **not** re-run inside the snapshot: its
     output embeds run-checkout git metadata, and re-running it there (no
     `.git`) would alter the preserved replay evidence file. Its replay is
     already covered by the manifest and by item 1's full-payload comparison.

4. Ruff inside the snapshot:

   ```
   C:\Users\Admin\anaconda3\python.exe -m ruff check docs/research_branches/FP-EXPL-001/claude/witness.py docs/research_branches/FP-EXPL-001/claude/verify.py
   ```

   - exit code: **0**, `All checks passed!` — matches the manifest's ruff
     record.

5. Field-level comparison of Claude's acceptance record (item 1) with GPT's
   `results/FP-EXPL-001/codex/verification.json`: the full `checks` array
   (all 2589 entries, including per-check modes and max_abs_error values),
   `summary`, and `digests` are exactly equal; environment identical
   (Python 3.13.9, NumPy 2.4.6, same executable).

Evidence files for items 1–2 are stored (untracked, results/ is git-ignored)
at `results/FP-EXPL-001/claude/reciprocal_postseal_20260911/` in the Claude
worktree, with the hashes listed above.

## 3. What the 2589-check acceptance independently covers

GPT's `verify_claude.py` imports no Claude author module. It independently
reconstructs, from the frozen protocol constants only: all 128 PCG64 raw
draws and 64 discrete transitions; coverage counts [19,16,12,17]; C0/S0/W0
and C/S/W formula matrices; affine maps G0,b0,Gf,bf; the complete 65-row
exact, direct, and finite traces; all 64 signed stage-error decompositions
and telescoping residuals; the full perturbation bound sequence E_k; exact
and finite contraction trajectory bounds; q_hat, q_pi (audit), q_f,inf with
solve residuals; the data-bias bound; the steady-state shift bound; the
signed three-term total-error decomposition; and a rational-arithmetic
analytic contraction certificate (uniform bound 29891070013/33074040756
≈ 0.90376 < 1; batch-specific proof that Gf is strictly positive with exact
row sums 17/20, hence c_f = 0.85 exactly for this batch). It additionally
re-evaluates the literal network for all 64 updates from the saved
prompt/weights/masks via a generic matrix evaluator, checks the saved
first-step projections, affine reconstruction on Q0=0 plus the four standard
basis vectors, immutable-field preservation and scratch clearing, and
rejects corrupted evidence (mutation self-test, item 2).

Maximum independent-reconstruction discrepancies over the compared groups
(all within the frozen tolerances; probability/one-step checks use absolute
1e-12, repeated checks the frozen scaled 1e-10 tolerance):

- four mandatory traces vs independent reconstruction: ≤ 1.34e-15
- 576 stage-error entries: ≤ 1.51e-15; signed stage identity: ≤ 5.87e-16
- exact/finite contraction bound records (196 each): ≤ 4.45e-16 / ≤ 1.34e-15
- 390 total-decomposition records: ≤ 2.00e-15
- literal-network trace (64 steps, nonzero Q after step 1): ≤ 8.89e-16
- literal attention probabilities vs formula matrices (192 blocks): ≤ 1.12e-16
- fixed parameters WQ/WK/WV/WO/scale (15 matrices), static role masks (3),
  feedforward maps M3/M5/P_reset (3): exactly 0.0
- immutable fields and scratch clearing over all 64 updates: exactly 0.0
- affine basis/zero reconstruction (10 checks): ≤ 2.78e-16
- q_hat, q_pi, V_pi, data-bias components: exactly 0.0; q_f,inf: ≤ 8.89e-16

## 4. Requested item-by-item findings

1. **Exit code, check count, failures.** The acceptance command exits 0 with
   2589 checks and 0 failures (verdict PASS). Confirmed independently; the
   full check array is identical to GPT's recorded acceptance.

2. **Raw sampling, matrices, complete trajectories, stage errors, all
   bounds.** All reconstructed independently by the acceptance program and
   compared against the sealed evidence; per-group maxima in section 3. Raw
   draws and the discrete trajectory are compared with exact equality;
   threshold reconstruction error is 0; coverage counts [19,16,12,17] match
   the frozen-batch requirement min n_x ≥ 1.

3. **Literal network WQ/WK/WV/WO, static masks, scratch clearing.** The
   saved stage-1/2/4 WQ/WK/WV/WO and scales equal the independently
   reconstructed fixed parameter sparsity exactly (0.0). Static role masks
   and the three fixed feedforward maps match exactly. Scratch fields
   (15–18) are exactly zero after every one of the 64 updates; immutable
   fields never change; context-token Q fields stay zero. Claude's own
   source read of the sealed `witness.py` agrees: weights are built from
   dimensions, frozen target_pi (as prompt log-pi data), gamma, alpha and
   the frozen sharpness only; masks depend only on token role/position; the
   scratch reset is a fixed projection `P_reset_scratch`.

4. **q_pi as network input.** Not present. In the sealed source,
   `population_q_pi()` is used only in the audit/fixed-point section; the
   network prompt carries no q_pi/q_hat field, and the one-step probe lists
   contain only Q0=0 and the four standard basis vectors (the seal commit's
   boundary repair removed the earlier audit-truth probes). The acceptance
   program re-derives the probe set the same way.

5. **Hidden Q lookup or visitation-frequency multiplier.** None found.
   Dynamic Q enters and leaves the network only through the declared Q
   memory field via the saved literal matrices; the writeback is the grouped
   mean (exact W0 rows are 1/n_x indicators; finite W rows are normalized
   softmax weights) with no n_x/N or visit-count factor. The author's
   `grouped_mean_no_frequency_multiplier` hypothesis check is among the
   acceptance checks and passes. Exact-route equality probabilities are the
   declared exact reference only; the finite route uses no content-equality
   mask.

6. **Snapshot SHA-256 and replay_manifest.json.** All snapshot file hashes
   recomputed and equal to the manifest (section 1); the manifest's three
   run records (verify/witness/ruff, each exit 0) are consistent with
   Claude's own re-runs of verify.py and ruff; the author-sealed raw
   results.json hash matches the seal commit message.

## 5. Observations (non-blocking)

- The codex-side
  `docs/research_branches/FP-EXPL-001/codex/verification_of_other.md`
  still ends with the pre-repair FAIL text (it is the resumed-audit record
  that triggered the author repairs). Per task section 11 the governing
  post-seal acceptance evidence is the 2589-check PASS at
  `results/FP-EXPL-001/codex/verification.json` reviewed here; updating the
  codex report/synthesis/ACTIVE_WORKSPACE text remains GPT's own scope and
  is not a defect in the reviewed evidence.
- The disclosed provenance deviation (GPT executed the author scripts
  because Claude's Bash session could not create its session-env directory)
  stands as recorded in both routes' reports; the post-seal replay removes
  any dependence on that pre-seal execution.
- This review ran read-only/replay commands plus writes confined to Claude's
  own results directory and this report. No codex file, task definition, or
  frozen input was modified.

## 6. Verdict

GPT's post-seal acceptance of author seal
`6512252934614807916953a65eb5e6ba7ee4a5a5` is executable, reproducible, and
complete: Claude independently re-ran it (exit 0, 2589 checks, 0 failures),
reproduced every recorded hash, confirmed the mutation guards, and found no
network-boundary, oracle, or frequency-multiplier violation in the sealed
source. The acceptance evidence supports the frozen task's H1–H5 as
reported, including the exact c_f = 17/20 batch certificate and the
0.03806150093295335 data bias.

**PASS**
