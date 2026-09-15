# Execution and evidence — FP-SPEC-REVIEW-001

## Isolation

Both worktrees were created by git worktree add from 16ee0f652cde56b4f2ef6e3f1a43b58818bb0108, named codex/FP-SPEC-REVIEW-001 and claude/FP-SPEC-REVIEW-001. Host untracked files preserved (LITERATURE_ALERTS.md, audit helper files, backups, paper extracts, temporary folders); no tracked host edits. Read-only git ls-remote confirmed Claude composition 16ee0f6, Codex previous review 6d8690e and main c1e03dd. Initial SSH known_hosts access failed in restricted shell; explicit read-only escalation succeeded.

## Source preparation and checks

Python C:/Users/Admin/anaconda3/python.exe; PyMuPDF fitz.open then page.get_text for every source page. Each text page prefixed with its 1-based PDF page. PDF and text SHA256 values in the shared input/manifest.json. PDF render check used page.get_pixmap(matrix=fitz.Matrix(1.25,1.25)); Xie pp.5–6 and Liang p.4 visually inspected. These are read-only source inspections, not experiments. Initial terminal print failed on GBK Unicode output; setting PYTHONIOENCODING=utf-8 resolved it. Some rg wildcard invocations failed under Windows; repeated with actual directories and -g patterns. No scientific inference rests on the failed searches.

The agent-reach Jina web read (curl.exe -L --max-time 30 https://r.jina.ai/https://arxiv.org/abs/2605.07333) failed. Built-in web opened the two official arXiv abstract pages successfully; metadata agrees with the local printed versions. No new PDF replaced the frozen corpus. The metadata check does not claim fresh-download byte identity.

## Actual Claude Code dispatch

Executable: C:/Program Files/claude-code/claude.exe. All invocations use -p --permission-mode dontAsk --no-session-persistence --output-format text. This is actual Claude Code, not a Codex subagent.

1. Read-only pre-review (--tools Read) could not read sibling/shared paths. Raw response input/pre_review.txt; no review verdict or task-level objection issued.
2. Explicit allowed Read and --add-dir pointing to this task root enabled access. Raw response input/pre_review_retry.txt: APPROVED. Manifest was augmented with the already extracted page-text paths/hashes and printed versions; PDF hashes rechecked before ACTIVE. Task scope unchanged.
3. Independent Claude execution used --tools Read,Write,Bash --allowedTools Read Write Bash and --add-dir for the task root and host papers. Prompt prohibited reading Codex conclusions, touching models/sealed outputs, task edits, experiments and pushes. Raw completion input/claude_independent_execution.txt. The actor wrote its own first_result.md and evidence_and_commands.md. Its Bash environment failed to create a session directory; it could read/write reports but not independently hash them. GPT hashed the first report before cross-review and separately verified baseline/content identity for both worktrees. This is a GPT integrity supplement, not a claim that Claude executed hashing. No further permission workaround was used for its shell.
4. Only after both first reports were saved and hashed did GPT start reciprocal review. Claude's independent report SHA256: 72e3d57046097f3507cccd36bf8b39cca51396d3788288e86b285ce575d159a9; GPT's: 1a3dd3e1b0d5b62de5f440f5c1e2ef6b0dd7bf245b6fe5545a60cfbd8760fe49. Cross-review responses are in input/claude_cross_review.txt and subsequent revision records.

## Reproduction without experiments

Read the task and frozen manifest; verify all listed SHA256 values. Read Xie equations (1)–(23), Liang equations (1)–(5) and their theorem statements, then the local sampled-SARSA construction and FP-ESARSA mathematical contract. Independently derive the attention weights and update maps. Compare to baseline model.py:888–1100 and the outer FP-COMPOSE-002 evaluator. Recompute the short symbolic counterexamples by substitution. No existing experimental evaluator or verification script was rerun.

Both worktrees' relevant tracked source diffs against the common baseline are empty; integrity_crosscheck.json records heads and hashes. Source and result provenance are separate from scientific agreement. Earlier tasks retain their states and evidence boundaries.
