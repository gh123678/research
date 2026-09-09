# FP-KERN-001 Claude route: blind first-result seal

Date: 2026-09-09. Branch: `claude/FP-KERN-001`. Common route baseline:
`28c4ae0f68ca51c7c9a0fd981159e85b7742dd4c`. Task version 1.0.

This seal was recorded before any formal run and before inspecting any GPT
route artifact. Blindness was maintained: no `codex/FP-KERN-001` branch,
`codex_worktree`, `results/FP-KERN-001/codex*`, or codex evidence file other
than the authorized `claude_pre_review.md` was read.

## Test-first evidence

`verify_kernel_state_generalization.py` was created before the pure module
and failed first:

```text
ModuleNotFoundError: No module named 'kernel_state_generalization'
```

After implementing `kernel_state_generalization.py` and
`kernel_generalization_mdps.py`, all 17 contract tests pass.

## Implementation files (Claude independent versions)

- `icrl_softmax/kernel_state_generalization.py`
- `icrl_softmax/kernel_generalization_mdps.py`
- `icrl_softmax/verify_kernel_state_generalization.py`
- `icrl_softmax/evaluate_kernel_state_generalization.py`
- `icrl_softmax/analyze_kernel_state_generalization.py`

Method statement: `docs/research_branches/FP-KERN-001/claude/theory_note.md`.

## Smoke defect and repair (implementation-level, disclosed)

First smoke attempt failed in `make_current_mdp` validation: the unchanged
generator's float32 Dirichlet `p0` sums to 1 within ~1e-7, exceeding the
wrapper's 1e-9 simplex tolerance. Repair: renormalize `p0` to the simplex in
float64 inside the wrapper. Transition and reward construction are unchanged
(verifier K16 checks bit-identity of P and R against the unchanged
`make_mdp`). No scientific constant changed. The partial first-attempt smoke
directory (commands.log, checks.log, config.json only) was removed and the
smoke rerun.

## Commands (smoke)

```text
python -B verify_fixed_policy_q_routes.py
python -B verify_finite_sample_theorems.py
python -B verify_visit_indexed_martingale_certificate.py
python -B verify_time_uniform_mixture_certificate.py
python -B verify_kernel_state_generalization.py
python -m ruff check kernel_state_generalization.py kernel_generalization_mdps.py verify_kernel_state_generalization.py evaluate_kernel_state_generalization.py analyze_kernel_state_generalization.py
python -B evaluate_kernel_state_generalization.py --tasks 1 --families current_unstructured hidden_cluster --trajectory-lengths 256 1024 --mixing 0.08 0.50 --gap-bonuses 0.0 0.50 --n-states 6 --n-actions 4 --pi-min 0.05 --gamma 0.70 --alpha 0.65 --iterations 160 --seed 20260909 --output-dir results/FP-KERN-001/claude_smoke
python -B analyze_kernel_state_generalization.py --result-dir results/FP-KERN-001/claude_smoke
```

All four inherited verifiers, the new verifier (17 checks), and task-scoped
Ruff passed. The evaluator's internal preflight re-ran all five verifiers
into `checks.log`.

## Smoke result (16 records, no tuning)

- 16/16 records written; 0 record-level failures; strict JSON written.
- Analyzer: full seed-regeneration of all 16 records, V-first bitwise
  reconstruction, route re-derivation, summary verification, and oracle
  separation all passed (`PASS FP-KERN-001 analysis: records=16`).
- Smoke classification field is `NOT_SUPPORTED`, which is expected and
  meaningless at one task per cell: paired intervals with fewer than two
  contributing records are unavailable and nonpassing by the frozen rule.
  The smoke validates mechanics and schema only; no scientific quantity was
  read as a tuning signal.
- Mechanical sanity: primary zero-count coverage 15/20 (current family) and
  7/10 (hidden cluster); all four routes emitted; abstention paths exercised
  (`target_source_unavailable`, `bandwidth_unavailable`,
  `insufficient_common_actions` all observed in smoke output).

## Smoke artifact SHA-256

Directory: `results/FP-KERN-001/claude_smoke/`

- `analysis.json`: `5148029b87e602d97be98dc4035c65b230303359e72507681dde862458f24fd8`
- `checks.log`: `b656252430c93eb8cb4c0f4899c3dcde3b269622152daf1689a65114bf66c81f`
- `commands.log`: `3d6d1aad10d4d5058ce30fc20684b783ef213dba892e1d3f5cd93abf35581ae2`
- `config.json`: `3d168ab1d4f4a2a77e59f5acbe2e24997c36790f1811cb760a2920729f182a98`
- `environment.json`: `194b75c622b10eb13c2d33dee131d36bec7e93923d95ec3dece84c3b99e70ad9`
- `summary.json`: `ce38c74484d7557f8bd957d8a7ca1e58610fb15cae9e544c5122d5c0a17bbdec`
- `task_results.json`: `a0db086c937931880bf7fe330b4a0f4166d3e2da2178b7334b1ae32569900137`

## Environment

- Python 3.13.9 (Anaconda), numpy 2.4.6, scipy 1.16.3, ruff 0.12.0,
  Windows 11, git HEAD at smoke: `28c4ae0f68ca51c7c9a0fd981159e85b7742dd4c`
  (see `environment.json` for the authoritative record).

## Statement

The implementation, smoke evidence, and this note are sealed in this commit
before the sole formal run. The canonical formal output directory
`results/FP-KERN-001/claude/` does not exist yet and is therefore empty; the
smoke output lives unchanged in `results/FP-KERN-001/claude_smoke/`.
