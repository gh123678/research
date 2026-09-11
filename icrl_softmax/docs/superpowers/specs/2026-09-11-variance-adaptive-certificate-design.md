# Design: a variance-adaptive residual certificate for certified policy improvement

Date: 2026-09-11.
Task: `docs/research_tasks/FP-SCALE-002.md`.
Status: `ACTIVE`.

## 1. The finding this design acts on

The project emitted `0/480` safe updates in `FP-ADV-001` and `0/480` in
`FP-ESARSA-001`. `FP-SCALE-001` then separated the training trajectory from a
dedicated certification batch and tightened the certificate by more than an
order of magnitude:

| quantity | FP-ESARSA-001 | FP-SCALE-001 |
|---|---|---|
| worst-pair certification count | 14 | 40000 |
| `max_x r_x` | 37.7612 | 0.2703 |
| `max_x |Ybar_x|` | 2.4125 | 0.0236 |
| `E_Q` | 22.47 (min) | 0.98 |

It still emitted nothing. Two reachable-scale diagnostics then isolated the
cause:

- **Emission needs `E_Q < sigma`.** The policy-improvement signal `Ihat` is
  second order in the tilt (`~ eta^2 Var_pi(q)/2`) while the error term
  `E_Q * TV` is first order (`TV ~ eta sqrt(Var_pi(q))`), so `LB > 0` requires
  `E_Q` below the within-state, action-relevant value spread `sigma`,
  essentially independent of `eta`.
- **Reward scaling is invariant.** At `gamma = 0.95`, gap `6.0`,
  `R_star = 7.5`, the spreads grew about `6x` but `E_Q` grew `170x`, because a
  Bellman-residual certificate with envelope `2B` and `B = R_star/(1-gamma)` is
  itself proportional to `R_star`. `E_Q` and `sigma` scale together, so their
  ratio cannot be improved by enlarging rewards.

The obstruction is the **worst-case residual envelope `2B`**: it assigns the
same sub-Gaussian parameter to every pair, while the measured residual spread
is of order `0.5`. Replacing the envelope by an estimated, provably bounded
scale was measured to change the radius at `N_x = 40000` from `0.2703` to
`0.0166`, and the count needed for `E_Q <= 0.15` from above `2**26` to about
`6,000`.

## 2. Why the envelope is unavoidable for the inherited certificate

The verified cosh-mixture argument requires a *known* sub-Gaussian parameter.
`|Y_t| <= 2B` supplies one immediately through Hoeffding's lemma, and that is
why the inherited certificate is valid but loose. Removing the looseness means
estimating the parameter, and estimating it from the same data that is being
certified would be circular. The design's whole problem is therefore to obtain
a data-driven scale without circularity and without an asserted constant.

## 3. The construction

Per pair, split the certification items into disjoint halves `A` and `B`.

1. **Second-moment bound from A.** With `Z_i = Y_i^2` in `[0, (2B)^2]`,
   Hoeffding gives `E[Y^2] <= mean_A(Y^2) + 2 B^2 sqrt(2 log(1/delta_A)/N_A)
   =: V_x` at confidence `1 - delta_A`. Bounding the *second moment* rather
   than the variance keeps the statement valid without assuming the residual
   mean vanishes, which matters because at `Qhat != Q^pi` it does not.
2. **Mean bound from B.** With `s_x = sqrt(V_x) >= sqrt(Var(Y))`, Hoeffding's
   lemma bounds the half-B mean deviation by
   `r_x = 2 s_x sqrt(log(1/delta_B)/N_B)` at confidence `1 - delta_B`.
3. **Simultaneity.** `delta_A = delta_B = delta/(2d)` and a union bound over the
   `d` pairs give the whole event at risk `delta`, with
   `E_Q = max_x(|Ybar_x| + r_x)/(1-gamma)`.

Every constant is a literal Hoeffding constant; there is no fitted or asserted
scaling factor. Because the halves are a deterministic function of the items,
using the realized `N_A` and `N_B` is valid.

### Why this is not circular

The scale that certifies half `B` is computed only from half `A`, and the
second-moment bound is a one-sided high-probability statement about the true
second moment, not about the observed value. Conditioning on it and applying
Hoeffding to the independent half `B` closes the argument.

### Why the count rule changed

The inherited certificate needed a minimum count because the frozen mixture
inversion only brackets inside a narrow, non-monotone band. The
variance-adaptive certificate does not use that inversion at all, so the
`FROZEN_CERT_COUNT = 40000` subsample is unnecessary. In its place the task
freezes a simple floor: at least `5000` items per half, i.e. `10000` per pair.

## 4. Prototype validation

Ran before the task was frozen, on the full planned matrix shape, two primary
routes, `16384 x 64 = 1048576` certification items per record:

| certificate | emitted | non-degrading | strict | violations | `E_Q` |
|---|---|---|---|---|---|
| asserted scale (`SAFETY = 1.1`) | 31/48 | 31/31 | 31/31 | 0 | 0.064--0.188 |
| **fully rigorous, no free constant** | **22/48** | **22/22** | **22/22** | **0** | 0.164--0.412 |

The rigorous row is the construction frozen above. It demonstrates that
emission survives the removal of every asserted constant, at an emission rate
of about `46%`.

## 5. What success and failure mean

- Success (`H3`, `H4`): at least `12` of `48` route-records emit, every
  emission is componentwise non-degrading, and at least one is a strict
  improvement. This is the project's first certified policy improvement under a
  provably valid, non-vacuous certificate.
- Failure: fewer than `12` emissions while `H1`, `H2`, `H6` pass. Then the
  concentration argument is exonerated, the within-state value spectrum becomes
  the leading suspect, and the negative result is reported without retuning.

## 6. Risks and mitigations

- **Conservatism risk.** The second-moment bound is looser than a variance
  bound because it does not subtract `E[Y]^2`. Mitigation: the prototype
  measures the effect (`31/48` to `22/48`) and `H3`'s threshold is set from the
  rigorous prototype, not the optimistic one.
- **Count risk.** Some records had per-pair counts as low as `12849`, near the
  `10000` floor. Mitigation: failing records are reported and excluded, and a
  majority failure trips a stopping condition.
- **Single-actor verification.** Not mitigated; disclosed.
- **Reduced matrix.** Not comparable cell-for-cell with the inherited
  `480`-record protocol; no cross-protocol numeric comparison is claimed.

## 7. Out of scope

- Any change to a closed task's frozen parameters, results, or reports.
- Cross-state generalization, pooling, or learned representations.
- Repeated policy iteration, online control, conditional-on-emission claims.
- Tightening the second-moment bound to a true variance bound, empirically
  Bernstein constants, or a self-normalized martingale construction: these are
  further refinements, and the frozen construction already emits.
- Any claim of independent verification.
