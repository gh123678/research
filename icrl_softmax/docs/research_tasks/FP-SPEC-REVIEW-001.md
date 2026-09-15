# FP-SPEC-REVIEW-001 — Paper, local specification, and implemented network correspondence

- Author: GPT; date: 2026-09-15; version: 1.0.
- State: VERIFIED (final reciprocal PASS; only this bounded correspondence audit, not earlier experiments).
- User authorization: “你去审查”, following discussion prioritizing paper/specification correspondence.
- Common code baseline: 16ee0f652cde56b4f2ef6e3f1a43b58818bb0108.
- Classification: long / conclusion-critical independent construction, no experiments.
- Branches: codex/FP-SPEC-REVIEW-001 and claude/FP-SPEC-REVIEW-001, both from the common baseline.
- Root: C:/Users/Admin/Desktop/research/icrl_softmax/results/FP-SPEC-REVIEW-001; each branch has its corresponding codex_worktree or claude_worktree directory.

## Question and falsifiable claims

Determine whether the two Expected-SARSA classes used in FP-COMPOSE-001/002 correspond to an external paper, a declared local extension, or neither. The purpose is to test correspondence, NOT to establish agreement regardless of evidence. H1: an explicit, checkable mathematical specification defines each implemented operator. H2: each claimed paper attribution and theorem reference resolves and supports the attributed claim. H3: any differences in memory, routing, sampling, update timing, logits, and control are explicitly accounted for. A mismatch, justified adaptation, or unresolved source is an acceptable scientific result if evidenced accurately.

## Frozen inputs and protocol

Primary external sources are the locally archived PDFs Xie_2026_beyond_linear_attention.pdf (arXiv 2605.07333) and Liang_Lai_2026_linear_attention_policy_improvement.pdf (arXiv 2605.05755), in the host icrl_softmax/papers/. Freeze their SHA256 and extract page-indexed text in a common input manifest before activation; preserve PDF printed version metadata and use official source pages to check provenance. Do not silently replace these PDFs with a newer version. A version ambiguity is reported as unresolved, not automatically an objection.

Local sources, all at the common baseline: model.py (primary classes EndToEndMaskedSoftmaxExpectedSARSA, EndToEndFiniteSoftmaxExpectedSARSA, FixedPolicyActionExpectation and called helpers); fixed_policy_expected_sarsa.py and fixed_policy_expected_sarsa_scaled.py; docs/research_tasks/FP-ESARSA-001.md and FP-ATTN-001.md; 论文_草稿/端到端_softmax_SARSA_构造性证明.md and its references. Trace referenced prior artifacts/history read-only when needed, and identify the exact version. Historical code classes and external theorems are examined only as necessary to disambiguate lineage, not as a full audit of all model.py classes or all proofs in either paper.

Each actor independently derives source operators from the original papers and local construction text, then maps code to those operators. Prior baseline reports may identify provenance but are not accepted as correctness evidence. Before submitting its first result, neither actor reads the other's task results or conclusions. The shared task, source corpus and input manifest are allowed. Sources already seen in earlier work cannot be unseen; record this limitation without claiming pristine blind discovery.

For every primary operator compare: token contents and candidate multiplicity; score, mask, softmax axis and normalization; value projections and signed residual; simultaneous versus sequential writes and memory transport; null/unvisited handling; sampled versus expected successor; policy input versus learned parameters; finite-logit leakage versus exact equality; dependence on dimension/support; and outer certificate/policy update versus network-internal operation. Algebra, source inspection and bounded hand examples are allowed. No benchmark reruns, training, parameter changes or edits to model.py/NumPy reference/sealed outputs. No new simulation matrix or independent implementation requested.

## Deliverables and acceptance

In each worktree: results/FP-SPEC-REVIEW-001/{codex|claude}/ for raw output, extracted evidence and execution metadata; docs/research_branches/FP-SPEC-REVIEW-001/{codex|claude}/first_result.md for the independent operator derivation, comparison table, exact code-line and paper page/equation citations, findings with severity, permitted final terminology and limitations. Record code/source hashes, commands and any failed retrieval. Each row is exact correspondence, explicit adaptation, mismatch, or unresolved; it must not infer correspondence merely from numerical agreement with the in-repo reference.

Acceptance requires both classes covered across all listed dimensions, distinguishing original external paper claims from local propositions, and a supported answer about what network name and paper attribution are justified. No requirement to prove all external paper theorems, strengthen empirical guarantees, or upgrade previous tasks. Missing evidence must stay unresolved. An actual scientific mismatch is not a task-definition objection.

After BOTH first results are saved and hashed, switch to VERIFYING. Each actor reads the other's report, checks cited primary evidence and derivations, and writes verification.md ending in PASS, FAIL, or OBJECTION. Original authors repair their own reports for ordinary evidence/derivation failures, preserving prior reports. GPT writes final_synthesis.md and updates its ACTIVE_WORKSPACE index. VERIFIED only after reciprocal PASS and every acceptance item accounted for; source fidelity can fail scientifically while an accurate negative audit passes. No merge to main.

## Responsibility, resources, stopping and governance

GPT drafts and freezes task, obtains actual Claude Code read-only pre-review, independently executes its report, verifies Claude, and synthesizes. Claude pre-reviews, then independently executes only its branch/report directories and verifies GPT. GPT may capture verbatim CLI responses in designated result logs; these remain Claude-authored evidence and are not substituted with Codex agents. No actor changes task definition independently.

Budget: approximately 1–3 hours wall time, local reading/algebra and existing Claude Code subscription; no GPU, training, paid new service or external messages. Stop affected work immediately for task-level OBJECTION and seek user ruling; do not stop merely because a source or attribution is unresolved. If Claude is unavailable, record the failure and report the verification block. Preserve unrelated untracked files. No pushing or merging required by this audit.

## Evidence and state history

- Repository governance and current index read; code baseline matches remote Claude branch; remote Codex prior review 6d8690e and main c1e03dd confirmed read-only.
- Existing untracked files in host preserved; execution uses isolated worktrees.
- DRAFT → REVIEW: task version 1.0 submitted for pre-review; pending.
- Final state history: initial VERIFYING produced Claude→GPT PASS with documentary corrections and GPT→Claude FAIL for repairable report errors. Returned ACTIVE for author repairs; Claude v2 introduced two normalization statements and one counterexample overstatement, producing a second report-level FAIL. Final report corrected them; final VERIFYING produced GPT verification_final.md PASS and Claude verification_v2.md PASS. No task-definition objection and no user exception required. All drafts, failures, revisions and actual CLI responses retained. Canonical synthesis: docs/research_branches/FP-SPEC-REVIEW-001/codex/final_synthesis.md. Claude evidence is mirrored verbatim there under codex/claude_evidence/ with archive_manifest.json; original actor files remain in the Claude worktree. Both code trees match the common baseline. No experiments, model changes, pushes or main merge.
- REVIEW → ACTIVE: actual Claude Code returned APPROVED in input/pre_review_retry.txt; first attempt was unable to read outside its worktree and performed no review (input/pre_review.txt). No task-level objection. The page-indexed extracted text already existed; manifest now includes its paths and SHA256, printed v2/v1 labels, and repeated PDF hash checks (both true). This discharges C1/C2. Official arXiv abstract pages confirm version metadata; Jina retrieval failed, built-in web access succeeded. All report/result paths are relative to each worktree's icrl_softmax directory. The action-expectation helper is included in both-operator coverage. If a handoff becomes necessary, AGENTS §8 applies verbatim. These are execution records/clarifications, not scientific scope changes.
