# FP-MART-001 GPT verification of Claude repair

Date: 2026-09-03. Verifier: GPT. Claude implementation/smoke seal:
`e0343871fd313c80f4f49ffa773db10eb75667d6`. Claude formal-evidence seal:
`e3c37197905e3ced080197dba989cdab36b4d4e8`.

## Outcome

The repaired proof and implementation defects are fixed. All five required
verifiers and Ruff pass under independent GPT execution, the prohibited
out-of-horizon count fixture is rejected, and an independent 480-record run
reproduces the repaired Claude deterministic outputs exactly. The repair is
not yet accepted because its evidence record remains incomplete in two
specific, non-scientific respects described below.

## Independent reproduction

GPT ran, from the Claude worktree and repaired code, the following checks:

```text
C:\Users\Admin\anaconda3\python.exe -B verify_finite_sample_theorems.py
C:\Users\Admin\anaconda3\python.exe -B verify_fixed_policy_q_routes.py
C:\Users\Admin\anaconda3\python.exe -B verify_crossfit_markov_certificate.py
C:\Users\Admin\anaconda3\python.exe -B verify_end_to_end_sarsa.py
C:\Users\Admin\anaconda3\python.exe -B verify_visit_indexed_martingale_certificate.py
C:\Users\Admin\anaconda3\python.exe -m ruff check --no-cache visit_indexed_martingale_certificate.py evaluate_visit_indexed_certificates.py analyze_visit_indexed_certificates.py verify_visit_indexed_martingale_certificate.py
```

All passed; the new verifier passed 27/27 tests. GPT then ran the frozen
480-record evaluator and strict analyzer into the isolated directory
`C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-MART-001\codex\verification_claude_repair`.
The analyzer reported 480/480 aligned records, no duplicate, missing, or new
keys, zero legacy mismatch, maximum numeric difference 0, and passing config,
task-record, 176-row summary, schema, oracle-provenance, and baseline-hash
checks.

The independently reproduced hashes equal Claude's repaired formal hashes for
all deterministic core files:

- `config.json`: `a2276eae06ba8689864cac2ad3d3d0e92b069014b2046d0180ee172bb8296ea0`;
- `task_results.json`: `365a0c979835890165c3ed1acc34938afcccaebef187af51780b165663d9afe5`;
- `summary.json`: `bb052450e498e82fcfef594718d4552cd6d3d1161b96a100bb04456c8a0d614d`;
- `regression.json`: `9a88e6944d13cfc44d1c6e7d95f8f42a4b1799b1e01ad769bed73a3e7896ca7a`.

Environment and command-log hashes differ as expected because the reproduction
has a different timestamp and output directory. The reproduced route metrics,
emission rates, both preregistered usefulness thresholds, and zero oracle and
per-group residual violations agree with `repair_result.md`.

## Repair findings

The compensated exponential-supermartingale proof is valid and replaces the
invalid raw MGF iteration. The optional variance statement is correctly
narrowed to a construction gap. The certificate derives `B` from the public
pre-sampling reward declaration; the true reward maximum appears only in the
post-construction `oracle_audit`. Count types, dimensions, horizon totals, and
state/pair aggregation are enforced. Smoke seed allocation uses the frozen
30-seed stride, and the analyzer performs complete legacy and provenance
checks. No remaining scientific or implementation defect was found.

## Remaining evidence defects

1. `repair_smoke.md` says that five required verifiers ran, but its command
   list substitutes `verify_theory.py` for the required
   `verify_end_to_end_sarsa.py`. GPT's later independent run shows that the
   omitted verifier passes, but the Claude route must correct its own record
   and must not claim the required pre-formal sequence occurred when it did
   not.
2. The repaired smoke directory has six files and lacks `checks.log` despite
   the task's per-result-directory artifact contract. `repair_smoke.md` records
   only four hashes and omits its existing `summary.json` and
   `regression.json`. The repaired formal directory itself has all seven files
   and its seven recorded hashes are exact.
3. The repaired formal report numbers only the eight repair-request items.
   The historical `first_result.md` contains the task's 1--18 assessment, but
   several of those statements describe the superseded failing route. A
   current post-repair 1--18 assessment is required for an unambiguous audit.

These are ordinary evidence-record defects. They do not require a task
revision or a new formal matrix, but the original author must repair them and
explicitly record the temporal anomaly that the missing required verifier was
run only after the formal seal.

## Acceptance mapping

Criteria 1--10 and 12--15 pass on direct inspection and reproduction.
Criterion 11 is scientifically supported, but the required-verifier execution
record has the sequencing anomaly above. Criterion 16 fails because the
repaired smoke artifact set and current numbered assessment are incomplete.
Criterion 17 remains pending until this evidence repair passes and Claude
verifies GPT. Criterion 18 remains pending final synthesis.

FAIL
