# FP-SPEC-REVIEW-001 — Claude evidence and command record

Route: `claude/FP-SPEC-REVIEW-001`, worktree `claude_worktree`.
Date: 2026-09-15. Author: Claude Code (independent route).
Companion report: `docs/research_branches/FP-SPEC-REVIEW-001/claude/first_result.md`.
Sealed before reading any Codex report for this task.

All paths below are relative to this worktree's `icrl_softmax` directory, per task record line 44,
unless written as an absolute path.

## 1. Identity pins

### 1.1 Code baseline

| item | value |
|---|---|
| commit | `16ee0f652cde56b4f2ef6e3f1a43b58818bb0108` |
| source of value | FP-SPEC-REVIEW-001 task record, "Common code baseline" (line 6) |
| verification status | **not re-verified in this session**; no shell was available to run `git rev-parse` (see §4) |

Every `file.py:line` citation in the report is a line number in the working-tree file as read in this
session. Working-tree content is assumed to equal the baseline commit; that assumption is stated, not
proven, in report §9.

### 1.2 External sources (values copied from the frozen manifest)

From `input/manifest.json` as read in this session. These are pre-reviewed, frozen values; they are
quoted here for citation, not recomputed (§4).

| file | sha256 | note |
|---|---|---|
| `icrl_softmax/papers/Xie_2026_beyond_linear_attention.pdf` | `04aef63b86239100c0ec836e6105486db125a4b4ec1dbfdbc0138bfc2f9647d3` | 28 pages; printed `arXiv:2605.07333v2  [cs.LG]  17 May 2026`; `pdf_hash_rechecked: true` |
| `input/Xie_2026_beyond_linear_attention.txt` | `9026158eef03b6d5c8bd83f0742a99f30aa882b63c0d7e2cbbc220e7941134f7` | page-indexed extracted text |
| `icrl_softmax/papers/Liang_Lai_2026_linear_attention_policy_improvement.pdf` | `6f579b4fdc818d493350878ba042bba6ae324d8631c8ed9a8341298c4eed7f9b` | 25 pages; printed `arXiv:2605.05755v1  [stat.ML]  7 May 2026`; `pdf_hash_rechecked: true` |
| `input/Liang_Lai_2026_linear_attention_policy_improvement.txt` | `bf254437a04af611d531b4f706a4d4093674684c6d15351880f5d67cb0ccbe52` | page-indexed extracted text |

Absolute paths of the extracted texts, per manifest:

- `C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-SPEC-REVIEW-001\input\Xie_2026_beyond_linear_attention.txt`
- `C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-SPEC-REVIEW-001\input\Liang_Lai_2026_linear_attention_policy_improvement.txt`

### 1.3 Deliverable hashes

The SHA256 of `first_result.md` and of this file **could not be computed in this session**
(no shell, no Python, no hashing utility available — §4). They are left unrecorded rather than guessed.
A route with a working shell should record them before the VERIFYING phase.

## 2. Files read, with ranges

Read-only. No file in this worktree was modified, and no file outside
`docs/research_branches/FP-SPEC-REVIEW-001/claude/` and
`results/FP-SPEC-REVIEW-001/claude/` was written.

| file | range read | purpose |
|---|---|---|
| `AGENTS.md` | full | governance; §5 long-task dual execution, §6 branch/result isolation, §7 PASS/FAIL/OBJECTION |
| `ACTIVE_WORKSPACE.md` | 1–120, 120–240, 240–360 (in chunks; file exceeds the single-read token limit) | FP-MODEL-REVIEW-001 residual limits; the user ruling not to change `model.py` |
| `CLAUDE.md` | full | repository entry point |
| `model.py` | 1–1101 (full) | audited classes and helpers |
| `fixed_policy_expected_sarsa.py` | 1–595 (full) | NumPy contract / declared operator |
| `fixed_policy_expected_sarsa_scaled.py` | 1–140 | frozen constants; confirms it inherits the contract module |
| `verify_fixed_policy_expected_sarsa.py` | 1–70 | which classes are actually exercised, and on what fixtures |
| `docs/research_tasks/FP-ESARSA-001.md` | full | the frozen local specification (H1 evidence) |
| `docs/research_tasks/FP-ATTN-001.md` | full | class usage and tolerance provenance |
| `论文_草稿/端到端_softmax_SARSA_构造性证明.md` | full | local construction text |
| `input/manifest.json` | full | source freeze |
| `input/Xie_2026_beyond_linear_attention.txt` | full, 28 pages | external operator derivation |
| `input/Liang_Lai_2026_linear_attention_policy_improvement.txt` | full, 25 pages | external operator derivation |
| `<codex_worktree>/icrl_softmax/docs/research_tasks/FP-SPEC-REVIEW-001.md` | full | the shared task (allowed input) |

**Not read.** No file under `<codex_worktree>/results/FP-SPEC-REVIEW-001/codex/` and no
`docs/research_branches/FP-SPEC-REVIEW-001/codex/` report was opened. The shared task record and the
shared `input/` corpus are the only Codex-route-adjacent artifacts read, which the task explicitly permits.

## 3. Commands executed

**None.** No shell command completed in this session. Consequently:

- no `git rev-parse` / `git status` / `git log`;
- no `sha256sum` / `certutil` / `Get-FileHash`;
- no `python -c` arithmetic or hashing;
- no directory listing, so the set of files in the deliverable directories was not enumerated;
- no `grep`-style absence proof.

Two consequences for the report's evidence quality, recorded rather than hidden:

1. **Absence claims rest on full sequential reads, not on search.** Statements of the form "the paper
   contains no Theorem 3.1 / no equality mask / no Expected SARSA" were established by reading the whole
   extracted text, not by a query that could have missed a match. This is a stronger method per page read
   but offers no mechanical guarantee against a rendering artifact in the extraction step.
2. **Line numbers are working-tree line numbers.** They could not be cross-checked against the baseline
   commit blob, since no git command ran.

All algebra in report §5 was done by hand from transcribed formulas. No numeric execution occurred, so
none of the report's conclusions depend on a computed quantity.

## 4. Failed retrieval and tooling record

| tool | outcome | impact |
|---|---|---|
| Bash | every invocation failed with `EPERM: operation not permitted, mkdir 'C:\Users\Admin\.claude\session-env\1b89a67d-c022-4d73-a35d-eb3823c0b994'`; identical with and without `dangerouslyDisableSandbox: true`; the failure occurs while the harness sets up the session environment, before any command runs | no local hashes, no git, no Python, no file listing |
| Grep | tool not enabled in this session | content search replaced by full-file sequential reading |
| Glob | tool not enabled in this session | file discovery replaced by explicit paths from the task and manifest |
| `mcp__agentflow__namespace_list` | denied by permission policy in don't-ask mode | no MCP-based git/HEAD/hash confirmation; denial was not worked around |
| `ACTIVE_WORKSPACE.md` single Read | rejected — file exceeds the 25 000-token read limit (34 735 tokens) | resolved by reading in three offset/limit chunks |

No external network retrieval was attempted. Both PDFs were available locally and the page-indexed
extracted text was used, so no provenance question depends on a live fetch.

## 5. Independence statement

Derivation order was: external paper text (Xie, Liang-Lai) → local construction text → the frozen
FP-ESARSA-001 specification → code. Code was mapped onto operators only after those operators were
written down. Numerical agreement with the in-repo NumPy reference was never used as provenance evidence,
and the report states this explicitly in §1.

Limitations that cannot be removed:

- Sources were already seen in earlier work on this repository (FP-ESARSA-001, FP-ATTN-001,
  FP-MODEL-REVIEW-001), so this is not pristine blind discovery. The task anticipates this and asks for
  it to be recorded rather than claimed away.
- The `input/` corpus is shared between routes by design, so any extraction artifact in the page-indexed
  text would affect both routes identically.
- One extracted-text path in the report's early notes was corrected against the working tree
  (`论文_草稿/端到端_softmax_SARSA_构造性证明.md`); the report cites the corrected path.

## 6. Reproducing this audit

With a working shell, in `claude_worktree/icrl_softmax`:

1. `git rev-parse HEAD` — confirm `16ee0f652cde56b4f2ef6e3f1a43b58818bb0108`.
2. `sha256sum ../papers/*.pdf` — compare against §1.2; also re-verify the two extracted-text hashes.
3. Read the four code files at the cited line ranges; read the two extracted texts in full.
4. Re-do the four hand checks in report §5 and the ten-dimension tables in report §6.
5. Record the SHA256 of both deliverable files at that point.
