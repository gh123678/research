# FP-SCALE-001 same-actor pre-review

Date: 2026-09-11.
Reviewer: Claude, acting as both executor and verifier under the user
instruction of 2026-09-11 ("不管 codex 了，验证也交给你").

**This is not an independent review.** There is no second actor. `AGENTS.md`
section 4 presumes a reviewer distinct from the author; the user waived that for
this task. Everything below is a same-actor check, and confidence in it is
strictly lower than an independent pre-review. It is recorded so the check is
auditable, not to claim independence.

Task under review: `docs/research_tasks/FP-SCALE-001.md` v1.0, plus
`docs/superpowers/specs/2026-09-11-reachable-certificate-scale-design.md` and
`docs/superpowers/plans/2026-09-11-reachable-certificate-scale-plan.md`.

Outcome: `OBJECTION` against v1.0 on two blocking protocol defects, both
repaired in v1.1; **v1.1 is `APPROVED`**.

## Section 1: consistency of task, design, and plan

| check | result |
|---|---|
| Task sheet contains every element `AGENTS.md` section 3 requires | PASS |
| Design and plan agree with the task's research question and hypotheses | PASS |
| Acceptance criteria are checkable and match the design's success/failure semantics | PASS |
| `H4`/`H5` failure is declared non-invalidating and forbids retuning | PASS |
| Prohibited-work list matches the design's out-of-scope list | PASS |

## Section 2: inheritance fidelity against the sealed FP-ESARSA-001 contract

The task claims the mathematical contract is inherited unchanged. Checked
against `icrl_softmax/fixed_policy_expected_sarsa.py` on `main`:

| claimed inheritance | verified |
|---|---|
| exact grouped update `Q + (alpha/N_x^train) sum delta` | PASS (`run_expected_exact`) |
| finite-logit scores `zeta = xi = tau = 8` | PASS (`finite_successor_all`, `finite_read_all`, `finite_writeback_all`) |
| canonical one-token-per-pair memory | PASS (`validate_canonical_memory`) |
| held-out residual definition | PASS (`build_residual_certificate`) |
| mixture event with `d` groups, risk `delta` | PASS (delegates to `time_uniform_mixture_certificate`) |
| certificate `E_Q = max_x(|Ybar_x| + r_x)/(1-gamma)` | PASS |
| lower bound `LB_s(eta) = Ihat_s(eta) - E_Q TV_s` | PASS (`decide_policy_update`) |
| ordered abstention reasons | PASS |

Additional finding, favourable: the inherited route and certificate functions
are already dimension-general and take `n_states`, `n_actions`, `pi_min`,
`layers`, `alpha`, `gamma`, and `value_bound` as arguments. The reduced `4x3`
protocol therefore needs **no modification** of any sealed file; the new code
can import them. This satisfies acceptance criterion 10 (byte-identity of the
inherited programs) by construction rather than by after-the-fact checking.

## Section 3: oracle separation and leakage

| check | result |
|---|---|
| Certificate inputs exclude true kernel, true value, realized error, action gap, return | PASS by inherited API; to be re-verified executably (criterion 8) |
| Certification batch cannot influence `Qhat` | PASS in v1.1: the two batches are separate rollouts and `Qhat` is built before the certification batch is read |
| A leak counterexample fixture is required | PASS (criterion 7) |
| Exact truth confined to `oracle_audit` | PASS (inherited record schema) |

## Section 4: `H1` arithmetic re-derived from the inherited verified code

Re-derived with `time_uniform_mixture_certificate.build_mixture_grid` and
`solve_mixture_boundary` at `n_groups = d = 12`, `delta = 0.05`,
`components = 15`, `B = R_star/(1-gamma) = 5`:

| `N_x` | `r_x` | `r_x/(1-gamma)` |
|---|---|---|
| 1000 | 1.4805 | 4.9349 |
| 5000 | 0.6739 | 2.2463 |
| 10000 | 0.4809 | 1.6032 |
| 20000 | 0.3484 | 1.1613 |
| 25000 | 0.3174 | 1.0579 |
| 40000 | 0.2703 | 0.9009 |

So the `H1` target `r_x/(1-gamma) <= 1.2` corresponds to `N_x >= 20000`, which
is exactly the frozen count rule. The arithmetic in the task sheet is correct.

## Section 5: blocker 1 — v1.0's trajectory lengths cannot reach the `H1` target

**Objection.** v1.0 froze trajectory lengths `32768` and `131072` with a
contiguous half split, expecting `N_x >= 20000`. This is arithmetically
unreachable, for a reason v1.0 did not consider: the certificate's radius is
governed by the **rarest** pair, and at `4x3` the sticky chain's stationary
occupancy caps the average pair count at `heldout/d`. A `131072`-length
trajectory holds out `65536` transitions across `12` pairs, so the average pair
count is about `5460` — already far below `20000` — and the rarest pair is
lower still.

Measured, three tasks per cell:

| mixing | length | held out | worst-pair count |
|---|---|---|---|
| 0.08 | 32768 | 16384 | 248 |
| 0.5 | 32768 | 16384 | 338 |
| 0.08 | 131072 | 65536 | 1271 |
| 0.5 | 131072 | 65536 | 1504 |
| 0.5 | 262144 | 131072 | 2980 |

No affordable trajectory length reaches `20000`; the deficit is a factor of
about `13` at the frozen maximum, and reaching the target by length alone would
need roughly `1.7` million transitions in a single trajectory while the
training half grew uselessly.

**Validity impact if unrepaired.** Every record would be excluded by
`heldout_pair_support_missing`, `H4` would be reported as unavailable, and the
task would spend its budget to learn nothing about the question it asked.

**Repair in v1.1.** Separate the two roles the single trajectory was serving:

- a **training trajectory** of `65536` transitions, whose only job is to build
  `Qhat`;
- an independent **certification batch** of `1048576` transitions, whose only
  job is to give every pair a large held-out count.

This is the statistically natural design — the certificate is a held-out
quantity, so its precision should be controlled directly rather than inherited
from the estimator's trajectory. Measured worst-pair counts at
`1048576` certification steps over five probes: `23498, 27013, 28154, 31815,
34051`, all above `20000`, giving `r_x` in `0.2840`--`0.3254` and
`r_x/(1-gamma)` in `0.947`--`1.085`, inside the `H1` target.

**No criterion weakened.** `H1` keeps its numeric target; the guarantee type,
the certificate, and every acceptance criterion are unchanged. Only the
sampling design that was supposed to achieve the target is corrected.

## Section 6: blocker 2 — the inherited eta grid cannot emit at a non-negligible certified error

**Objection.** v1.0 inherited the eta grid `{1.0, 0.5, 0.2, 0.1, 0.05}`. The
emission condition is

```text
LB_s(eta) = Ihat_s(eta) - E_Q * ||pi_eta^+(.|s) - pi(.|s)||_1 >= 0  for all s.
```

Write `s` for the within-state Q spread and suppose, generically, that the
relative-softmax tilt produces `Ihat ~ eta * s^2 / 2` and
`TV ~ eta * s / 2`. Then `LB > 0` requires `E_Q < s`, essentially independent
of `eta`. At `N_x = 20000` the certified error is `E_Q ~ 1.1` plus the residual
term, so a state whose Q spread is around `1` sits exactly on the boundary, and
states with a smaller spread cannot emit at any `eta` in the inherited grid.

**Validity impact if unrepaired.** `H4` would likely fail for a reason that is
about the grid's resolution rather than about the method or the scale,
confounding the very attribution the task exists to make.

**Repair in v1.1.** Extend the candidate grid downward by two entries to
`{1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01}`, ascending in selectivity as before.
This is a protocol parameter, fixed before any execution. It does not change
the decision rule, the certificate, or the acceptance criteria, and it makes
`H4` a sharper test: the hypothesis now also predicts that any emission selects
`eta <= 0.1`, which is falsifiable in either direction.

**Disclosure.** This repair was motivated by analysis, not by observed output;
no smoke or formal result existed. It is recorded here so that a reader can
judge whether it is legitimate design or tuning. The judgement rests on the
fact that the condition `E_Q < s` is eta-independent, so extending the grid
cannot manufacture an emission that the mechanism does not support: it can only
reveal whether a small tilt suffices.

## Section 7: residual risks

| risk | assessment |
|---|---|
| Certified error `E_Q ~ 1.1 + residual` may still exceed the achievable margin | Real. `H4` may fail. The task is designed so that this failure is informative (`H6` isolates attribution) rather than wasted. |
| Certification batch `1048576` steps x `24` records x `3` routes | Sampling-dominated; smoke must measure it. Compute cost is negligible (about `0.43` s per record-route). |
| Same-actor verification | Not mitigated; disclosed. |
| Reduced matrix is not comparable cell-for-cell with the `480`-record protocol | Acknowledged; no cross-protocol numeric comparison is claimed. |

## Section 8: verdict

v1.0: `OBJECTION` on Section 5 (blocking) and Section 6 (blocking).

v1.1: **`APPROVED`**. Both blockers are repaired by pre-execution measurement
and analysis; no scientific contract, hypothesis strength, tolerance, or
acceptance criterion was weakened; no closed task's file is modified; oracle
separation and the leak requirement remain in force.

This approval is a same-actor approval and carries correspondingly less
assurance than an independent pre-review would.
