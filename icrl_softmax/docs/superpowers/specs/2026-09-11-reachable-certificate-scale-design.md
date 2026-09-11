# Design: a certified relative-softmax improvement step at a reachable certificate scale

Date: 2026-09-11.
Task: `docs/research_tasks/FP-SCALE-001.md`.
Status: approved direction (user ruling 2026-09-11); task in `DRAFT`.

## 1. Problem statement

The project's core question is whether a softmax attention network can convert
in-context experience into a genuinely better policy. Step one (value
estimation) is constructed and verified. Step three, proving that a policy
change cannot make things worse, is the open gate:

- `FP-ADV-001` emitted `0/480` safe updates on all six routes;
- `FP-ESARSA-001` emitted `0/480` on all three routes;
- `CTRL-PREFLIGHT-001` had already predicted analytically that the certificate
  in use must reject everything in the frozen matrix.

## 2. Diagnosis (read-only, on sealed records)

The full arithmetic is recorded in the task sheet's "Why this task exists".
The essential decomposition, for records where a certificate was emitted on
the `expected_exact` route of `FP-ESARSA-001`:

```text
E_Q = max_x ( |Ybar_x| + r_x ) / (1 - gamma)
```

- worst-pair `|Ybar_x|` (empirical held-out Bellman residual at `Qhat`):
  at most `2.4125`;
- worst-pair `r_x` (frozen time-uniform mixture radius): `37.7612`;
- worst-pair held-out count in that record: `1`.

The certified error is therefore the concentration radius, not the residual.
The radius scales as `N_x^{-1/2}` through the frozen inversion:

| `N_x` | `r_x` | `r_x/(1-gamma)` |
|---|---|---|
| 1 | 37.7612 | 125.871 |
| 128 | 4.1505 | 13.835 |
| 1024 | 1.5110 | 5.037 |
| 4096 | 0.7668 | 2.556 |
| 16384 | 0.5463 | 1.821 |

Separately, the frozen inversion is not wasteful: it is about `1.32` to
`1.41` times a calibrated two-sided sub-Gaussian radius at the same count and
risk. Replacing the inversion cannot recover the factor of roughly `150`
between `E_Q` and the realized error.

### 2.1 Why weakening the guarantee does not help by itself

The natural fallback, "certify average value instead of componentwise value",
does not remove the obstruction. For a `mu`-weighted value guarantee the
radius is still governed by the worst pair, because `mu` has full support over
the states. The obstruction is not the type of guarantee; it is how many
observations the rarest pair receives.

### 2.2 Why this is about scale, not about the method

Two independent lines of evidence say the construction is sound:

- `FP-ESARSA-001` met acceptance criteria 1--19, including the `1e-12`
  agreement between the literal attention construction and batch Expected
  SARSA, and reported zero certificate, residual-event, and value violations on
  every emitted certificate;
- `FP-KERN-002` showed separately that cross-state statistical borrowing is not
  the missing ingredient, so pooling cannot be used to rescue occupancy.

What is missing is per-pair data. That is a protocol parameter, and choosing it
before execution on the basis of a stated calculation is legitimate
experimental design, not outcome tuning.

## 3. Design decision

Keep the mathematical contract exactly as verified, and change only the
protocol scale, fixing it by the arithmetic above.

- Reduce the state-action space to `4` states and `3` actions (`d = 12` pairs)
  so a long trajectory can concentrate observations.
- Raise the behavior-policy floor to `pi_min = 0.15` so occupancy is more even.
- **Separate the two roles a single trajectory was serving** (v1.1 correction):
  a *training trajectory* of `65536` transitions builds `Qhat`, and an
  independent *certification batch* of `1048576` transitions from the same
  frozen behavior policy supplies the held-out residual certificate. The v1.0
  design used one contiguously split trajectory of length `32768`/`131072`,
  which measurement showed cannot give the rarest pair more than about `1500`
  observations, because the sticky chain's stationary occupancy caps the
  average pair count at `heldout/d`.
- Require every pair to reach `N_min >= 20000` certification observations; a
  record that fails is excluded with the frozen
  `heldout_pair_support_missing` reason.
- Fix the reward-gap bonus at `0.5`, the nontrivial setting, because the
  zero-bonus cells were the support-limited ones.
- Use `2` mixing settings (`0.08`, `0.5`) and `12` tasks per cell, for `24`
  matched records.
- **Extend the eta candidate grid downward** (v1.1 correction) to
  `1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01`. The emission condition
  `E_Q < within-state Q spread` is essentially eta-independent, so the
  inherited grid could not emit at a certified error of order `1`; the
  extension makes `H4` sharper rather than easier, and it also predicts the
  selected eta.

The target is `N_x >= 20000` per certification pair, at which the radius
contribution satisfies `r_x/(1-gamma)` in `0.947`--`1.085`, inside the `H1`
target of `1.2`. Measured worst-pair counts at `1048576` certification steps
over five probes were `23498, 27013, 28154, 31815, 34051`.

Compute is not the binding constraint: the exact grouped `65536`-train route
costs `0.34` s and a `262144`-point certificate `0.02` s, so the formal
iteration is about two minutes in total and the rollout sampler dominates.

## 4. What success and failure mean

- Success (`H4` and `H5`): at least `3` of the frozen records emit a certified
  update, every emission is componentwise non-degrading in the oracle audit,
  and at least one is a strict improvement. This is the "small complete policy
  improvement example" the project has been trying to reach.
- Failure: zero emissions at a scale where the certificate is non-vacuous.
  Then the scale explanation is falsified as the sole obstruction, the
  certificate form itself becomes the leading suspect, and the negative result
  is reported without retuning.

Both outcomes are publishable-quality evidence. The design deliberately makes
the protocol scale the only variable so the outcome is interpretable either
way.

## 5. Risks and mitigations

- **Compute risk.** The `131072`-length, `160`-layer iteration is the dominant
  cost. Mitigation: the smoke run must measure wall time before the formal run,
  and the task stops rather than silently shrinking the matrix.
- **Occupancy risk.** A pair may still be starved. Mitigation: `H1` is checked
  per record, starved records are excluded with the frozen reason, and `H4` is
  reported as unavailable if more than half the records are excluded.
- **Single-actor verification.** The user assigned verification to the executor.
  Mitigation: mandatory derived reconstruction (independent code path,
  from-source generator identity, metric recomputation, `H1` re-derivation),
  and a limitation statement in every report. This does not restore
  independence and must not be described as if it did.
- **Comparability risk.** The reduced matrix is not comparable cell-for-cell
  with the inherited `480`-record protocol. Mitigation: the task claims no
  cross-protocol numeric comparison; `H6` only compares emission rates and
  attributes the difference to occupancy, and the inherited protocol remains
  untouched.

## 6. Out of scope

- Any change to a closed task's frozen parameters, results, or reports.
- Variance-adaptive or Bernstein-type residuals: the diagnosis shows the mean
  is not the binding term, so this is deferred until a scale-adequate run shows
  a residual-dominated certificate.
- Cross-state generalization, pooling, learned representations.
- Repeated policy iteration, online control, or conditional-on-emission claims.
- Any claim of independent verification.
