# GPT verification of the Claude FP-TU-001 route

Reviewed Claude formal seal:
`0fa824fb4e565bf0fff8c589ecf4b0cf32c21f4d`.

## Reproduced checks

- Claude branch history begins at the common activation commit and modifies
  only its authorized code/evidence paths.
- The seven canonical formal artifact hashes exactly match
  `claude/formal_result.md`.
- All five inherited verifiers and Claude's new verifier pass from a detached
  validation worktree at the formal seal.
- Ruff initially failed only because it could not create a cache in the
  detached worktree; `ruff check --no-cache` passes all four new files.
- Claude's formal analyzer records 480 records, zero legacy mismatches,
  unchanged exact emissions, exhaustive radius dominance, nonincreasing
  emitted bounds, and zero oracle-audit violations.  These scientific metrics
  agree with the independent GPT route.

## Repair-required findings

1. Proof interval error.  `claude/theory.md` section 3 says a conditionally
   centered increment with conditional range width `2B` lies in `[-B,B]`.
   This does not follow: subtracting the conditional mean shifts the support,
   so a normalized centered interval can be asymmetric, for example
   `[-0.2,1.8]`.  Hoeffding's lemma still gives the required `exp(a^2/2)`
   bound for any interval of length two, so the theorem and code constant are
   repairable without changing the frozen construction.  The verifier should
   add asymmetric centered two-point fixtures extending outside `[-1,1]`.

2. Mutable frozen grid and iteration-cap contract.  `mixture_grid` returns a
   cached mutable dictionary.  A caller can set `grid["weights"][0]=0` and a
   subsequent `mixture_grid(54,0.05)` call returns the corrupted value.
   `mixture_root(..., max_iterations=201)` is also accepted although the
   frozen numerical contract says at most 200.  The author must prevent cache
   mutation from affecting later certificates, validate any supplied grid
   against the frozen formulas, enforce the 200-iteration ceiling, and add
   regression tests.  No fallback or parameter retuning is permitted.

3. Transition-variance overclaim.  The corner example proves the uniform
   worst-case variance over unrestricted `V in [-B,B]^m` can attain `B^2`; it
   does not prove that a data-dependent confidence set coupled to the Bellman
   equations can never tighten on some datasets.  The unseen-mass uncertainty
   can also shrink with sample size.  The feasibility document must state the
   exact universal-worst-case obstruction and the unresolved confidence-set
   construction gap, not a global impossibility theorem.  This optional study
   remains excluded from the mandatory certificate.

4. Formal-run procedure.  Claude completed a first 480-record evaluation,
   then its analyzer found a derived-summary compatibility defect; after a
   one-line repair it reran the identical frozen evaluation.  The report is
   transparent and shows no metric tuning, but the history is not literally
   the preregistered single-run procedure.  This cannot be repaired by another
   run.  It was retained as a documented governance exception and later
   accepted by the user's 2026-09-07 final ruling.

The Claude author must repair findings 1--3 on its own branch, rerun only
deterministic verifiers/Ruff (not the formal matrix), and seal the repair.
GPT will then re-review the code and reproduce the sealed formal artifacts in
an isolated result directory.  Finding 4 was reserved for, and later accepted
by, the user's final adjudication.

## Final re-verification after Claude repairs

Claude repaired findings 1--3 without rerunning its formal matrix.  From a
detached validation worktree at Claude repair commit
`02e74cf6d238cfeed03495b4807eccef98c4fa78`, GPT ran the deterministic verifier
suite and Ruff (`--no-cache`); all passed.  The repaired API now isolates the
cached frozen grid, enforces the 200-iteration ceiling, validates every frozen
grid field, and rejects invalid inversion brackets with ordered deterministic
failure reasons.  The asymmetric-support tests and transition-variance
qualification are present in Claude's repair evidence.

GPT then reproduced Claude's frozen formal command in the separate result
directory
`icrl_softmax/results/FP-TU-001/codex/claude_reproduction/`.  The evaluator
and formal analyzer both passed.  Parsed `config.json`, `task_results.json`,
and `summary.json` are byte-for-byte identical to Claude's sealed formal
artifacts; all 480 records match, the frozen baseline hashes remain intact,
and the reproduced route metrics are unchanged (emissions 2.5%/85%/100%/100%,
maximum mixture/old ratios 0.9573574/0.9263356/0.8975201/0.8793887, zero oracle
violations).

The duplicate Claude formal evaluation after its summary-only repair remains a
documented procedural exception.  It does not invalidate the implementation or
the reproduced scientific data, and the user explicitly accepted the exception
on 2026-09-07 before final task closure.

PASS
