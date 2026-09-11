# FP-KERN-001 Claude independent route — relocated code

These five files are the **Claude independent route's own implementation** of
the FP-KERN-001 frozen contract, sealed on `claude/FP-KERN-001`. They were
extracted byte-for-byte from that branch when the verified FP-KERN-001 and
FP-KERN-002 tasks were merged into `main` on 2026-09-11 under explicit user
approval.

## Why they are not at the canonical top-level paths

Both independent routes were required to provide their own versions of the
same five entry points, so both used the identical top-level filenames under
`icrl_softmax/`. Git cannot hold two different files at one path, and the two
implementations are genuinely different code (for example the Codex route
builds its hidden-cluster family through `make_environment`, while this route
provides `make_hidden_cluster_mdp`). Neither implementation is a superset of
the other, and both are backed by independent sealed 480-record results.

The canonical top-level paths in `main` therefore hold the **Codex corrected
route** (seal `1001d23273bdf29b92d9b84a3f3956e83819da4a`), because that is the
route whose final, authoritative 480-record corpus is the one recorded in the
task sheet. This route's code is preserved here instead, with a documented
path remapping.

## Byte-identity verification

Each file was extracted with `git show claude/FP-KERN-001:<path>` and compared
against its committed blob with `git hash-object`. All five match exactly:

| file | git blob hash (local = committed) |
|---|---|
| `kernel_state_generalization.py` | `1192b4950bd51cf7af099de56639407f48a7560d` |
| `kernel_generalization_mdps.py` | `974dc9c72694351432fef9162634d2f21c20332c` |
| `verify_kernel_state_generalization.py` | `8f064c14be861721a2324c1226b1974716f27cd8` |
| `evaluate_kernel_state_generalization.py` | `13949e868c83b9dfc3e6e62eab6df1f1f5c42af6` |
| `analyze_kernel_state_generalization.py` | `ae4e48209e866e0ef77dbc52ebab6f790043ba54` |

## Reproducing this route

This route's sealed `commands.log` recorded its programs at the canonical
top-level names, executed from that route's own worktree
`results/FP-KERN-001/claude_worktree/icrl_softmax` on branch
`claude/FP-KERN-001` (formal seal `1a820467683d137e6527edbd99bb486000b2fc58`).

That worktree and branch are the authoritative reproduction environment. To
rerun from `main` instead, copy these five files back to the canonical
top-level names **in a scratch copy of the tree** — never over the Codex
route files at those paths — because all five import each other by their
canonical module names, exactly as in the sealed command:

```text
kernel_generalization_mdps.py          -> icrl_softmax/kernel_generalization_mdps.py
kernel_state_generalization.py         -> icrl_softmax/kernel_state_generalization.py
evaluate_kernel_state_generalization.py-> icrl_softmax/evaluate_kernel_state_generalization.py
analyze_kernel_state_generalization.py -> icrl_softmax/analyze_kernel_state_generalization.py
verify_kernel_state_generalization.py  -> icrl_softmax/verify_kernel_state_generalization.py
```

The shared, route-independent dependencies this route imports
(`evaluate_fixed_policy_q_routes.py`, `fixed_policy_finite_sample_certificate.py`,
`mdps.py`, `verify_fixed_policy_q_routes.py`) are byte-identical on both
branches and are unchanged in `main`.

## Provenance

- Task: `docs/research_tasks/FP-KERN-001.md` (`VERIFIED`).
- Branch: `claude/FP-KERN-001`, formal seal
  `1a820467683d137e6527edbd99bb486000b2fc58`.
- Classification reached independently by this route: `NOT_SUPPORTED`.
- Route evidence: `docs/research_branches/FP-KERN-001/claude/`
  (`theory_note.md`, `first_result.md`, `formal_result.md`, `verify_codex.md`,
  `verify_codex_corrected.md`).
- Sealed 480-record corpus: inside the registered worktree at
  `results/FP-KERN-001/claude_worktree/icrl_softmax/results/FP-KERN-001/claude/`.

No scientific content, matrix, seed, threshold, or decision rule was changed
by this relocation; only the file location and this note were added.
