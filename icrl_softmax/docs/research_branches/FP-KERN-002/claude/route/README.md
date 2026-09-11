# FP-KERN-002 Claude independent route — relocated code

These two files are the **Claude independent route's own implementation** of
the FP-KERN-002 frozen diagnostic, sealed on `claude/FP-KERN-002`. They were
extracted byte-for-byte from that branch when the verified FP-KERN-001 and
FP-KERN-002 tasks were merged into `main` on 2026-09-11 under explicit user
approval.

## Why they are not at the canonical top-level paths

Both independent routes were required to provide their own versions of the
same two entry points, so both used the identical top-level filenames under
`icrl_softmax/`. Git cannot hold two different files at one path, and the two
implementations are genuinely different code — for example this route's
analyzer imports `scipy.stats` for the Spearman and Student-t diagnostics,
while the Codex route derives them from the standard library.

The canonical top-level paths in `main` therefore hold the **Codex route**
(GPT formal seal `0815d0dbef3a8f7784438ac89e2df195d3cab00b`), because
`docs/research_tasks/FP-KERN-002.md` names the corrected Codex output as the
canonical 480-record source corpus. This route's code is preserved here,
byte-identical, with a documented path remapping.

## Byte-identity verification

Each file was extracted with `git show claude/FP-KERN-002:<path>` and compared
against its committed blob with `git hash-object`. Both match exactly:

| file | git blob hash (local = committed) |
|---|---|
| `analyze_kernel_reuse_diagnostics.py` | `ef085cdcb6a3db1034a4e770eb5274d5bf98fead` |
| `verify_kernel_reuse_diagnostics.py` | `5ca3f9abaa4d7bfd44c6f29d1e0797d10746497a` |

## Reproducing this route

This route's sealed `commands.log` recorded its programs at the canonical
top-level names, executed from that route's own worktree
`results/FP-KERN-002/claude_worktree/icrl_softmax` on branch
`claude/FP-KERN-002` (implementation/smoke seal
`a39323c8011acebfb651c8431d8598e1d1aee244`, formal seal
`718d77053801c6f9e3dd958513b7a06918a5e274`).

That worktree and branch are the authoritative reproduction environment. To
rerun from `main` instead, copy these two files back to the canonical
top-level names **in a scratch copy of the tree** — never over the Codex route
files at those paths — because both consume the frozen common input at
`results/FP-KERN-002/input/` and are invoked by their canonical module names:

```text
analyze_kernel_reuse_diagnostics.py  -> icrl_softmax/analyze_kernel_reuse_diagnostics.py
verify_kernel_reuse_diagnostics.py   -> icrl_softmax/verify_kernel_reuse_diagnostics.py
```

## Provenance

- Task: `docs/research_tasks/FP-KERN-002.md` (`VERIFIED`).
- Branch: `claude/FP-KERN-002`.
- Classification reached independently by this route:
  `NO_BORROWING_EVIDENCE` (identical to the Codex route).
- Route evidence: `docs/research_branches/FP-KERN-002/claude/`
  (`implementation_seal.md`, `formal_result_seal.md`, `gpt_verification.md`,
  `gpt_bundle_reconciliation.py`, `formal_output_check.py`,
  `execution_blocker.md`, `git_seal_blocker.md`).
- Sealed 480-record corpus: inside the registered worktree at
  `results/FP-KERN-002/claude_worktree/icrl_softmax/results/FP-KERN-002/claude/`.

No scientific content, input, metric, threshold, or decision rule was changed
by this relocation; only the file location and this note were added.
