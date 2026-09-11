# FP-EXPL-001 (v1.1) — Claude route report (preliminary)

> **Current correction (2026-09-11, author repair after GPT source review).**
> GPT's source review of the sealed Claude route found one ordinary boundary
> error: both `witness.py` and `verify.py` included the audit-only `q_pi`
> (and additionally `q_hat` and an arbitrary non-basis vector) in the
> one-step **network** probe lists, although frozen task section 4 declares
> `q_pi` audit-only and never a network input. The repair below removes those
> three probes from both scripts; the four standard basis vectors plus
> `Q0 = 0` remain (they already fix the affine maps, per task section 5).
> Population audit, residual/bias/decomposition mathematics and the direct
> formula application at `q_pi`/`q_hat` are unchanged. Consequently the
> verifier's one-step unit checks drop from 8 to 5 per equivalence group
> (groups C and D: 9 -> 6 checks each; total 1285 -> 1279, still 0 failures,
> verdict PASS). All scientific outputs are bit-identical to the pre-repair
> sealed run: see `results/FP-EXPL-001/claude/rerun_probe_boundary_repair_20260911T154700/scientific_invariance_evidence.json`.
> Prior sealed results are preserved under
> `results/FP-EXPL-001/claude/preserved_pre_repair_20260911T154017/` and
> `results/FP-EXPL-001/claude/archived_pre_probe_boundary_repair_20260911T154700/`;
> the full history is in `failure_history.md`. Counts and hashes below now
> describe the repaired route.

Status of this document: **preliminary Claude main-route report**. Under the
user ruling recorded in task FP-EXPL-001 v1.1, Claude performs the main
execution and GPT performs the final acceptance. This report summarizes the
Claude route's own evidence only. It is **not** the acceptance verdict: final
acceptance requires GPT's independent reconstruction and replay
(`docs/research_branches/FP-EXPL-001/codex/verify_claude.py` run by GPT
against the Claude `results.json`, plus GPT's read-only source review and
`verification_of_other.md`), the Claude cross-review of that acceptance, and
the VERIFIED lifecycle transition. No VERIFIED claim is made here.

Provenance note: this documentation step ran no commands and produced no new
numbers. Every number below is transcribed from the recorded evidence files
`results/FP-EXPL-001/claude/verification.json` (schema
`FP-EXPL-001/claude-verification/v1`) and the frozen protocol in
`docs/research_tasks/FP-EXPL-001.md` v1.1; the full witness evidence (all raw
draws, transitions, matrices, traces, stage errors, bounds and fixed points)
is stored in `results/FP-EXPL-001/claude/results.json` (schema
`FP-EXPL-001/claude-witness/v1`).

## 1. The actual frozen batch

- Sampler: numpy.random.Generator(numpy.random.PCG64(20260911)), initial
  state 0, 64 sequential transitions, exactly two scalar rng.random() draws
  per transition in the task order (action, then transition) — 128 draws
  total, no extra draws, no resampling, no shuffling, no seed choice.
- The independent verifier regenerated the batch twice and required exact
  equality of raw draws and the discrete trajectory, exact threshold
  reconstruction of every action/next-state/reward, trajectory continuity,
  and the initial state (group A, 323 checks, 0 failures, max error 0.0).
- Coverage (H1): visit counts by pair (00, 01, 10, 11) = [19, 16, 12, 17];
  min n_x = 12 >= 1, so the batch passed the coverage gate and the frozen
  batch Q iteration proceeded. Coverage is a falsifiable diagnostic of this
  batch, not a guarantee of the behavior policy.

## 2. Checks and PASS evidence (independent verifier)

`verify.py` is a self-contained independent re-implementation (no import of
`witness.py`): it re-derives the frozen batch, the direct grouped-mean
reference, the exact grouped matrices, the finite operator as independently
written scalar loops, and a separately coded pure-Python literal fixed-weight
softmax attention network, then checks the protocol end to end.

Recorded summary (`verification.json`):

- total checks: 1279; total failures: 0; verdict: PASS.
- Per-group checks / failures / max error:

  | group | checks | failures | max error |
  |---|---|---|---|
  | A_sampling_and_coverage | 323 | 0 | 0.0 |
  | B_probability_structure | 520 | 0 | 1.11e-16 |
  | C_H2a_exact_equals_direct | 6 | 0 | 1.29e-06 (scaled trace ratio; raw tolerance 1e-10*(1+max_abs)) |
  | D_H2b_literal_equals_scalar | 6 | 0 | 0.0 |
  | E_H3_telescoping | 64 | 0 | 6.25e-16 |
  | F_affine_basis | 11 | 0 | 1.67e-16 |
  | G_H4_contraction | 66 | 0 | 0.0 |
  | H_perturbation_bounds | 64 | 0 | 0.0 |
  | I_fixed_points_decomposition | 201 | 0 | 4.44e-16 |
  | J_policy_preservation | 2 | 0 | 1.11e-16 |
  | K_network_integrity | 7 | 0 | 7.82e-17 |
  | L_input_rejection | 9 | 0 | 0.0 |

- What the groups establish: both equivalences of H2 (exact grouped attention
  equals the direct grouped-mean reference; the literal finite network equals
  the independent scalar finite formula, one-step probes on the four standard
  basis vectors plus Q0 = 0, and full 64-update traces); the H3 three-term
  signed stage-error identity
  Ff(q) - F0(q) = e_current + e_successor + e_write holds at every step
  (telescoping residuals at round-off level) and the perturbation recursion
  E_(k+1) = c_f E_k + ||Ff(q_exact,k) - F0(q_exact,k)||_inf upper-bounds the
  actual finite-vs-exact error at every k = 1..64; affine reconstruction from
  the four standard basis vectors matches the directly derived G0, b0, Gf, bf;
  the successor conditional equals target_pi and is not the behavior policy
  (0.5, 0.5); the grouped mean carries no visitation-frequency multiplier
  (W0[x, t] * n_x = 1 on matches); immutable prompt fields are bit-identical
  and scratch fields exactly zero after each of the 64 updates; malformed and
  nonfinite inputs are rejected.
- Tolerances (frozen): 1e-12 absolute for probabilities and one-step
  equivalences; 1e-10*(1+max_abs(left,right)) for repeated traces, solves,
  decompositions and bounds (added on the right side of inequalities); exact
  equality for raw draws and the discrete trajectory.

## 3. Contraction (H4) evidence

- Exact operator: ||G0||_inf equals the proved bound
  c0 = 1 - alpha(1 - gamma) = 0.85 on this covered batch, and the trajectory
  bound ||q_exact,k - q_hat||_inf <= 0.85^k ||Q0 - q_hat||_inf held at every
  k = 0..64 (group G, 0 failures).
- Finite operator: the computed sufficient statistic is c_f = ||Gf||_inf =
  0.85 < 1, so the contraction sufficient condition holds on this batch; the
  finite fixed point q_f,inf is applicable and was computed, its residual and
  the steady-state bound ||q_f,inf - q_hat|| <= ||Ff(q_hat)-q_hat||/(1-c_f),
  the trajectory bound ||q_finite,k - q_f,inf|| <= c_f^k ||Q0 - q_f,inf||, and
  the exact three-term signed decomposition
  q_finite,k - q_pi = (q_finite,k - q_f,inf) + (q_f,inf - q_hat) +
  (q_hat - q_pi) were all checked (group I, 201 checks, 0 failures).
- rho(Gf) was recorded only as a floating-point numerical diagnostic
  (0.8499999999999994); it is not used to prove spectral properties, a unique
  fixed point, or convergence, and c_f >= 1 would not have implied divergence.

## 4. Data bias and H5 metrics (reported as-is)

From `verification.json` (group I details):

- V_pi by state = [1.7689620758483031, 1.4096806387225547]; the two true state
  values differ by 0.35928143712574845. This is consistent with the proved
  nondegeneracy argument (target-policy mean immediate rewards by state are
  0.625 and 0.25, which cannot correspond to a constant state value).
- Data bias ||q_hat - q_pi||_inf = 0.03806150093295335 — nonzero on this
  batch. Per the task, the data-bias magnitude is a reported metric, not a
  success condition; the deterministic audit upper bound
  ||q_hat - q_pi|| <= ||F0(q_pi) - q_pi||/(1 - c0) was checked as a one-sided
  inequality (0 failures).

## 5. Limitations

- One frozen batch of 64 transitions from one fixed seed; no seed, sharpness,
  batch-size, or initial-value sweeps were run or are claimed. Nothing
  statistical (confidence, sample complexity) is asserted.
- The data-bias upper bound uses audit ground truth (P and q_pi); it is not a
  learner-computable statistical certificate.
- c_f = 0.85 < 1 is a property of this batch's actual finite matrices; it is
  not guaranteed in advance for other batches, and c_f >= 1 elsewhere would
  not imply divergence. rho(Gf) is a floating-point diagnostic only.
- The construction implements fixed-policy fixed-batch evaluation only: no
  training, no policy improvement, no online control, no per-round new data,
  and no paper-style time-weighted batching; convergence/sample-complexity
  guarantees of any related paper are not inherited.
- The finite network uses static role/position masks and one-hot identity
  dot-product logits with frozen log-pi bias; the exact route's declared
  equality masks are a reference device, not a finite-network capability.
- The group C H2a trace maximum (1.29e-06) is the recorded scaled trace
  ratio's raw group maximum as stored in `verification.json`; it is within the
  frozen repeated-trace tolerance 1e-10*(1+max_abs(left,right)) per check
  (0 failures), and all one-step equivalences met 1e-12.
- All results remain preliminary until GPT acceptance completes (Section 7).

## 6. Reproducibility

Commands recorded by the task (section 6) and the evidence files, to be run
from the Claude worktree's `icrl_softmax` directory (stdout and exit codes
saved per protocol):

    C:\Users\Admin\anaconda3\python.exe -B docs/research_branches/FP-EXPL-001/claude/witness.py
    C:\Users\Admin\anaconda3\python.exe -B docs/research_branches/FP-EXPL-001/claude/verify.py
    C:\Users\Admin\anaconda3\python.exe -m ruff check docs/research_branches/FP-EXPL-001/claude/witness.py docs/research_branches/FP-EXPL-001/claude/verify.py

Recorded environment for the verifier run (`verification.json`): Python
3.13.9 (`C:\Users\Admin\anaconda3\python.exe`), NumPy 2.4.6, Windows
11 (10.0.22631), CPU float64. The witness additionally records its own SHA-256
and the verifier's SHA-256 plus the git HEAD at start inside `results.json`.
That `code.baseline_commit_at_start` field records the run checkout's HEAD at
witness start (after a seal this denotes the sealed run checkout, not the
frozen scientific baseline); it is kept under its original name and
transparently documented alongside the explicit
`code.frozen_scientific_baseline_commit`
(`c710e32b24d77085adea134c3f470773037ffbb1`, task section 1) and
`code.common_active_publication_commit`
(`8c915c4bf2bb9533e2374f5d2c91cf34e5c13c77`, the common ACTIVE publication
commit). No relabeling of either baseline is made or implied.

Final raw artifact SHA-256 of this repaired route (pinned at seal):

- `results/FP-EXPL-001/claude/results.json`:
  `8f84b6aaa62562b4e97d7794e43cb1e5a081f785fe867d523e6dcc6b67200c89`
- `results/FP-EXPL-001/claude/verification.json`:
  `e4750b34e0b979fb7bf94f46d906bbd0eb0326320c80edc8f73f11c440277db5`
- `docs/research_branches/FP-EXPL-001/claude/witness.py`:
  `c48a134b67712aa5361ab6a9e9459a6a5afcda4f0153d36cf97f900d7fa84834`
- `docs/research_branches/FP-EXPL-001/claude/verify.py`:
  `f56ce6df3690b4c55a855dae4eafa50f6b1fce6db7ee66561008717730afc4cb`

GPT acceptance replays the three commands above first, then runs from its own
worktree:

    C:\Users\Admin\anaconda3\python.exe -B docs/research_branches/FP-EXPL-001/codex/verify_claude.py --results <Claude原始results.json绝对路径>

This report does not claim that any acceptance-side command has been run.

## 7. Preliminary Claude route vs. final GPT acceptance

- This report, `theory.md`, `witness.py`, `verify.py`,
  `results/FP-EXPL-001/claude/results.json` and
  `results/FP-EXPL-001/claude/verification.json` constitute the **Claude main
  route**: the witness evidence and a self-contained independent Claude-side
  re-implementation check (PASS, 1279/1279 checks). The task explicitly does
  not claim two complete blind-route network implementations.
- **Final acceptance is GPT's**: independent reconstruction of the frozen
  seed, all traces, all matrices, both operators, all Q iterations, error
  terms, bounds and applicable fixed points via `verify_claude.py`; replay of
  the three commands above; read-only source review for external dynamic-Q
  lookups or hidden oracles; and a `verification_of_other.md` ending in PASS,
  FAIL or OBJECTION — followed by Claude's cross-review of the acceptance
  evidence. Only after both reviews pass and the lifecycle conditions of task
  section 7 are met may the task be called VERIFIED, and any merge to main
  requires separate user approval.
- Until then, every number in this report is a preliminary, reproducible
  Claude-route result, not an accepted result.
