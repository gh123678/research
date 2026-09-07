# FP-TU-001 consolidated report

## Result

The mandatory mathematical and numerical route is positive.  GPT's formal
matrix and an independent reproduction of Claude's repaired route each contain
480 matched records under the frozen protocol.  The reproduced
`config.json`, `task_results.json`, and `summary.json` are byte-for-byte equal
to Claude's sealed artifacts; the canonical FP-MART-001 baseline remains
unchanged.

All counts `1..16384` pass stable inversion, conservative-root, stitch-bracket,
and radius-monotonicity checks.  Maximum mixture/legacy radius ratios for
trajectory lengths 256, 1024, 4096, and 16384 are respectively
`0.9573574`, `0.9263356`, `0.8975201`, and `0.8793887`; 21,760 horizon/count
comparisons are strict.  The largest `q_mix/q_stitch` is `0.9959406`, with at
most 42 bisection steps.  Every emitted total bound is nonincreasing and the
oracle audit reports zero violations.

Exact-route emission is unchanged at `2.5%`, `85%`, `100%`, and `100%` across
the four lengths.  `<B` usefulness is a secondary empirical outcome: V-first
exact has 101/120 at length 16384 and V-first softmax 58/120; it is not used as
the probability proof.  Direct-Q and V-first softmax/direct comparisons and
all legacy leaves remain unchanged.

## Route and verification record

- GPT implementation: `4656fd3`; blind seal `00f7d89b`; formal seal
  `6f73def5`; defensive numeric-failure repair `f01a49d`.
- Claude blind seal: `450881d`; formal seal `0fa824f`; repair seal
  `02e74cf`.  Repairs corrected asymmetric centered-interval wording,
  protected the frozen grid and iteration cap, and qualified the
  transition-variance claim.
- GPT independently re-reviewed the repaired Claude route and reproduced its
  formal data in `results/FP-TU-001/codex/claude_reproduction/`; this report is
  recorded as `PASS` in the GPT verification file.
- Claude independently reproduced GPT's six verifier checks, task-scoped Ruff,
  frozen hashes, 480-record metrics, and analyzer result; it also reviewed the
  theory and numeric-failure paths.  Its 20-item report ends `PASS` at commit
  `ae82bb8bd7a269c878697b19280a00c4ad2c2f23`.

## Governance exception and status

Claude's analyzer found a summary-composition defect after its first complete
formal evaluation, repaired that derived-summary code, and reran the identical
frozen command.  The rerun was not metric-tuned and matches the first data, but
it is not literally the preregistered single-run procedure.  On 2026-09-07 the
user explicitly accepted this transparent procedural exception and the
one-file environment exception under which GPT committed Claude's unchanged,
already-written reciprocal report after Claude could not write shared Git
metadata.

Current task state: `VERIFIED`.  Both reciprocal reports end `PASS`; all 20
acceptance criteria are satisfied or explicitly ruled on.  Main is unchanged
and still requires separate user approval before merge.
