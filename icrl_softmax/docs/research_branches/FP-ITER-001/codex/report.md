# FP-ITER-001 Codex first independent result

Status: **preliminary**, self-verification PASS; independent Claude execution
and reciprocal verification are not complete. Task remains ACTIVE.

## Versions, independence and artifacts

- Task: FP-ITER-001 v1.1; common activation baseline
  6d2376968aeb9379c978bd1a7af7929b70fdeb09.
- Branch: codex/FP-ITER-001. Scientific source/theory commit:
  cb26a339b3a9160cc508f0558c507786e79b8304.
- The witness, verifier and theory were constructed independently. No Claude
  implementation, theory or experiment result was read. Claude execution was
  rejected before process creation; its isolated branch remains at baseline.
- Source: witness.py, verify.py and theory.md in this report's directory.
- Raw evidence: results/FP-ITER-001/codex/results.json and verification.json,
  relative to icrl_softmax. Full projected first-step tensors, every weight,
  all 65 time points, score/probability matrices, stage errors and bounds are
  present. Subsequent tensors can be reconstructed from the fixed weights and
  saved per-step Q; attention scores do not depend on Q.
- results.json SHA-256:
  2F38DE6588D69C6C0B7F8CECBC0D1C707178D4206B3CAC5B32669631DC342434.
- verification.json SHA-256:
  3C731ECF4BD002AFF92AD2B0D123CE424FB323DF76C49F91638BE6CB1A98B13A.
- Environment: Windows 11, C:/Users/Admin/anaconda3/python.exe, Python 3.13.9,
  NumPy 2.4.6, CPU float64; Ruff 0.12.0. No random draws, GPU or training.
  Raw JSON records full environment and source-file SHA-256 digests.

## Exact commands and execution history

From C:/Users/Admin/Desktop/research/icrl_softmax in PowerShell:

```powershell
& 'C:\Users\Admin\anaconda3\python.exe' -B docs/research_branches/FP-ITER-001/codex/witness.py > results/FP-ITER-001/codex/results.json
& 'C:\Users\Admin\anaconda3\python.exe' -B docs/research_branches/FP-ITER-001/codex/verify.py > results/FP-ITER-001/codex/verification.json
& 'C:\Users\Admin\anaconda3\python.exe' -m ruff check docs/research_branches/FP-ITER-001/codex/witness.py docs/research_branches/FP-ITER-001/codex/verify.py
```

The first pre-commit verifier run passed all 6,332 checks; its output and
empty stderr are retained as verification_attempt1.json/.stderr.txt. Ruff
passed. The identical code was committed, then the canonical witness and
verifier were run at cb26a339; both exited zero with empty stderr. No
implementation/experiment failure, fixture tuning or numerical correction
occurred. The verifier internally reproduces the same bounded protocol; this
is not an additional sample or a new scientific experiment matrix.

## Numerical findings

Infinity norms, 64 synchronous updates; finite sharpness means all three
parameters are equal to the displayed number. E64 is the proved audit bound.

| Batch / initial / sharpness | c_f | rho(Gf), numerical | finite vs exact at k=64 | E64 |
|---|---:|---:|---:|---:|
| C / Z / 0 | 1.175 | 1 | 0.930539 | 26913.6517 |
| C / A / 0 | 1.175 | 1 | 0.652769 | 15666.0560 |
| C / Z / 8 | 0.850770 | 0.85 | 0.001621893 | 0.003921135 |
| C / A / 8 | 0.850770 | 0.85 | 0.001621903 | 0.003921250 |
| M / Z / 0 | 1.175 | 1 | 1.722141 | 45139.3668 |
| M / A / 0 | 1.175 | 1 | 1.666616 | 31647.7934 |
| M / Z / 8 | 1.464630 | about 1 | 1.493323 | 1.1130042e10 |
| M / A / 8 | 1.464630 | about 1 | 1.198093 | 9.2992586e9 |

For complete coverage, exact c0=0.85. Finite sharpness 8 has c_f<1 and thus a
unique limit. Its limiting bias from the empirical point is 0.001621946804065,
below the steady bound 0.003921311230829. Its fixed point is approximately
[2.45995528013740, 0.957941828932350, 1.70894855453487, 2.20961970493656].
The two initializations approach this same finite-attention limit.

At zero sharpness the sufficient contraction check fails. This does NOT show
divergence: theory.md proves eigenvalues .85,1,1,1, preservation of all
pairwise Q differences, and convergence to an initialization-dependent
member of a fixed-point family. The common-limit evaluation property fails.

Missing coverage changes the semantics. Exact iteration preserves Q11 at
0 for Z and 0.5 for A. Finite sharpness 8 changes those values at k=64 to
1.49332250566767 and 1.69809294958123. Its absent-pair writer is uniform over
all five observed transitions for either sharpness, so increasing sharpness
does not supply the missing pair's data. No strict full-table contraction or
unique-limit claim is made for M. Its E64 bounds are valid but far too loose
to be practically informative. A spectral radius approximately one is not
presented as a proof of divergence or a unique limit.

## Diagnostic limitation: population and empirical points coincide

The frozen rewards have pi-weighted immediate reward 5/8 in BOTH states.
Hence V_pi=25/12 in both states, for any row-stochastic transition matrix,
and q_pi=[59,23,41,53]/24. The complete empirical matrix therefore has the same
q_hat. The saved batch-discrepancy norm is 4.44e-16, numerical roundoff around
zero. This fixture does not demonstrate a nonzero sampling/data error.

Both complete-batch initializations also have equal pi-weighted state values;
exact iteration preserves that equality. Accordingly the successor-routing
term along the C exact trajectories is at most 7.53e-17. The M trajectories
do exercise nonzero successor terms (up to 0.0274003 at sharpness 0 and
9.18e-5 at sharpness 8). Basis-vector verifier checks additionally test the
literal operator off the two initial trajectories. None of these facts
justifies claiming a nonzero complete-batch population-error demonstration.
The signed decomposition is verified, but its C data term is degenerate.
The frozen inputs were retained rather than changed after this observation.

## Acceptance evidence and limitations

| Check | Evidence and decision |
|---|---|
| H1 exact witness | Four 65-point traces; max direct/exact accumulated difference 8.88e-16. PASS. |
| H2 finite witness | Explicit prompt and projections; all per-trajectory single steps within 8.88e-16 of scalar; accumulated finite/scalar max 2.44e-15. PASS. |
| H3 error split | Max signed telescope residual 7.82e-16; conditional policy error 1.12e-16; triangle inequality checked. Data degeneracy disclosed above. PASS. |
| H4 exact iteration | C c0=.85 with checked contraction envelopes; M identity coordinate preserved exactly. PASS with the frozen coverage distinction. |
| H5 finite iteration | All E_k and applicable steady/transient inequalities checked; worst positive horizon-bound excess 1.12e-16, within tolerance. Loose bounds and noncontractive cases explicitly retained. PASS. |

All 6,332 self-checks pass, including scratch reset, immutable fields, null
token, normalization, positional finite masks, signed residuals, repeated
visits, self-loops, absent writes, basis-vector affine equality and malformed
input rejection. No shared project module was changed; legacy experiment
regression was not needed for these isolated scripts.

This establishes only this small fixed-policy evaluation construction and
the proved conditional statements. Static role masks and positional memory
layout are explicit interfaces. No learned routing, policy improvement,
online-control behavior, useful statistical certificate or general benchmark
performance has been established.

## Remaining work and approval blocker

Claude's v1.1 pre-review is APPROVED. Its independent execution launch was
rejected by automatic approval review before process creation: the private
task/source transmission destination and broad read/write/Bash scope were
not explicitly authorized to the reviewer's satisfaction. A subsequent
read-only inspection of the selected non-secret backend field identifies
HTTPS api.kimi.com:443. The existing CLI's model metadata labels were
kimi-k2.6/k3; backend model identity is not independently authenticated.
No workaround launch was attempted. The earlier CTRL-PREFLIGHT transfer
authorization was task-scoped and is not asserted as explicit authorization
for all FP-ITER payloads.

Required remaining step: explicit approval for transmitting this task's
governance, frozen task/design/plan, actor-scoped source and results to the
existing Kimi backend through Claude Code, for isolated execution and
subsequent reciprocal verification. Independent execution must first see
only the common baseline, not this report or Codex code. Cross-reading is
allowed only after both first-result seals. Main merge/push is not included.
Until then Claude is not replaced by another Codex agent and the task is not
VERIFIED. Exact continuation is recorded in handoff.md.

Self-verification verdict (not reciprocal validation):

PASS
