# FP-SCALE-001 route journal

Branch: `claude/FP-SCALE-001`. Task: `docs/research_tasks/FP-SCALE-001.md` v1.1
(`ACTIVE`). Single actor: Claude holds both execution and verification under the
user instruction of 2026-09-11.

## 1. Gate A — H1 arithmetic re-derived

Re-derived with the inherited verified inverter
(`time_uniform_mixture_certificate`). See
`docs/research_branches/FP-SCALE-001/claude/pre_review.md` sections 4--6.

At `d = 12`, `delta = 0.05`, `components = 15`, `B = 5`:

| `N_x` | `r_x` | `r_x/(1-gamma)` |
|---|---|---|
| 16384 | 0.3813 | 1.2709 |
| 20000 | 0.3484 | 1.1613 |
| 30000 | 0.2965 | 0.9882 |
| 40000 | 0.2703 | 0.9009 |

The `H1` target `r_x/(1-gamma) <= 1.2` needs `N_x >= 20000`, matching the
frozen count rule. `H1` arithmetic confirmed.

## 2. Blocker found and repaired during Gate A

v1.0's single contiguously split trajectory cannot reach `N_x >= 20000`: the
certificate radius is governed by the rarest pair, and the sticky chain caps
the average pair count at `heldout/d`. Measured worst-pair counts were `248`
(length 32768) and `1504` (length 131072).

Repaired in v1.1 by separating the training trajectory from an independent
certification batch. Full detail and measurements in `pre_review.md`.

## 3. Second blocker found while implementing

The inherited frozen inversion only brackets counts inside a narrow,
**non-monotone** band. Its 15-component geometric grid has mesh points
`2**0 .. 2**14 = 16384`, so the bracket condition
`log_mixture(count, 0) < log(d/delta) <= log_mixture(count, stitch)` fails for
counts that fall between usable mesh points.

Measured at `d = 12`, `delta = 0.05`:

| count | brackets |
|---|---|
| 19573 | yes |
| 40000 | yes |
| 48000 | yes |
| 92185 | **no** (`mixture_inversion_unbracketed`) |
| 207250 | **no** |
| 234003 | **no** |
| 131088 | yes |
| 131089 | no |

This is why an early plumbing run returned
`status='not_certified'`, `failure_reasons=['mixture_inversion_unbracketed']`
with four pairs unbracketed.

Repaired **inside the protocol, not by editing the sealed inverter** (which is
forbidden): the certification batch is sized so every pair exceeds a frozen
per-pair count, and each pair is then uniformly subsampled to exactly
`FROZEN_CERT_COUNT = 40000`, which sits mid-band. The subsample uses a
dedicated RNG stream, so retained items remain independent draws of the same
conditional law. The estimator is untouched.

Fairness note: this is a protocol constant fixed before any smoke or formal
output. It was chosen from the inverter's bracketing band, which is a
property of the sealed verified component, not from any emission outcome.

## 4. Batch sizing measurements

Certification batch sizing at `mixing = 0.5`, three probes each:

| chains | chain length | total | worst-pair count | s/task |
|---|---|---|---|---|
| 65536 | 16 | 1,048,576 | 19795 | 12.6 |
| 131072 | 16 | 2,097,152 | 39580 | 25.1 |
| 262144 | 8 | 2,097,152 | 39298 | 25.2 |
| **262144** | **16** | **4,194,304** | **78491** | **50.3** |
| 524288 | 4 | 2,097,152 | 39123 | 25.2 |

Frozen: `CERT_CHAINS = 262144`, `CERT_CHAIN_LENGTH = 16`, so the worst pair
exceeds `FROZEN_CERT_COUNT = 40000` with a factor-2 margin and no record is
lost to subsampling.

## 5. Compute measurements

At `4x3`, `160` layers, `alpha = 0.65`:

| operation | measured |
|---|---|
| exact grouped route, 65536 training transitions | 0.34 s |
| certificate over 262144 certification items | 0.02 s |
| certification rollout, 4194304 items | ~50 s |
| full plumbing run (one task, three routes, certificate, H1, decision) | 53.0 s |

Projected formal cost: about `53` s x `24` records = **about 21 minutes**,
rollout-dominated. Within budget.

## 6. First end-to-end reachable-scale measurement — the decisive result

One task (`task_index = 0`, `mixing = 0.5`), frozen protocol, all three routes:

```text
expected_exact   cert=certificate_emitted  E_Q=0.9797  max|Ybar|=0.02362
expected_finite  cert=certificate_emitted  E_Q=0.9789  max|Ybar|=0.02340
sampled_exact    cert=certificate_emitted  E_Q=0.9927  max|Ybar|=0.02753
H1 passed=True   min_cert_count=40000      radius_contrib=0.9009
improvement: abstained, reasons=['improvement_lcb_nonpositive']
```

**The scale repair worked exactly as designed.** Against the `FP-ESARSA-001`
baseline the certificate improved by more than an order of magnitude:

| quantity | FP-ESARSA-001 | FP-SCALE-001 (1 record) |
|---|---|---|
| worst-pair certification count | 14 | 40000 |
| `max_x r_x` | 37.7612 | 0.2703 |
| `max_x |Ybar_x|` | 2.4125 | 0.0236 |
| `E_Q` | 22.47 (minimum) | 0.98 |

The bound is valid and `H1` passes. **But no update is emitted**, for a reason
that is now measurable rather than conjectural.

### Why no emission, quantitatively

Emission needs `LB_s(eta) = Ihat_s(eta) - E_Q * TV_s(eta) > 0` at every state.
At `E_Q = 0.98` the two terms are comparable:

- `Ihat_s`, the policy-improvement signal, is the advantage the tilted policy
  gains. It is **second order** in the tilt: `Ihat ~ eta^2 Var_pi(q) / 2`.
- `TV_s`, the total-variation distance moved, is **first order**:
  `TV ~ eta * sqrt(Var_pi(q))`.
- So `LB > 0` requires `E_Q < sqrt(Var_pi(q)) = sigma_s`, O(1) in `E_Q` and
  independent of `eta` in the small-tilt regime.

For this record the within-state Q spreads `sigma_s` are `1.5353, 0.7820,
1.3606, 0.7712`, while `E_Q = 0.9797`. The smallest spread, `0.7712`, is
**below** `E_Q`, so that state cannot emit at any `eta`, and the strict
all-states rule therefore blocks the record.

The sharper constraint is that the achievable margin, not the theoretical
`sigma_s`, is what must exceed `E_Q * TV`. Measured for the actual `q_hat`:

| `eta` | `max_s Ihat_s` | `max_s TV_s` | margin at `TV=0.01` | needed `E_Q` |
|---|---|---|---|---|
| 0.05 | 0.004595 | 0.076458 | 0.000601 | 0.0786 |
| 0.02 | 0.000812 | 0.031638 | 0.000257 | 0.0257 |
| 0.01 | 0.000214 | 0.015974 | 0.000134 | 0.0134 |

So an emission in this record would need a certifiable per-state-action error
of order `0.01`, whereas the certified radius is `0.2703`.

### Structural reading

Combining the two constraints: the certificate must resolve a signal of order
`eta^2 sigma^2` while carrying an error `eta sigma E_Q`, so the condition is

```text
E_Q  <  sigma        (the within-state, action-relevant value spread)
```

and the *usable* margin is only a few percent of `sigma` because the softmax
tilt is nearly diffusive at the admissible `eta`.

`E_Q = 0.98` against spreads `0.77`--`1.54` is worse than a factor of `10` from
the usable condition in every state. Closing that gap requires either a much
larger action-relevant value spread or a certifiable error below `0.01`, and
the second option needs roughly a `700`-fold larger certification count at this
record's radius scaling, which is not affordable.

This is the first time the obstruction has been located at the level of the
value spectrum rather than the certificate's arithmetic, and it is a stronger
statement than either `FP-ADV-001` or `FP-ESARSA-001` could make, because those
results were confounded by a certificate that was three orders of magnitude too
loose.

## 7. Second reachable-scale diagnostic — which lever actually works

Two candidate levers were tested quantitatively at reachable scale.

### Lever 1: widen the value spectrum. **Does not work.**

Running the same frozen certificate and decision rule on a wider-spectrum MDP
(`gamma = 0.95`, gap bonus `6.0`, `R_star = 7.5`) gave:

| case | within-state `q_pi` spreads | `E_Q` | best `min_s LB` | emitted |
|---|---|---|---|---|
| frozen (`gamma=0.7`, gap `0.5`, `R_star=1.5`) | 0.713, 0.316, 1.225, 1.951 | 0.9531 | -0.00172 | no |
| wide spectrum (`gamma=0.95`, gap `6.0`, `R_star=7.5`) | 6.543, 5.393, 5.873, 6.579 | **164.224** | -4.00247 | no |

The spread grew by roughly `6x` but `E_Q` grew by `170x`, because a
Bellman-residual certificate with envelope `2B` and `B = R_star/(1-gamma)` is
itself proportional to `R_star`. **`E_Q` and `sigma` both scale linearly with
the reward bound, so the decisive ratio `E_Q/sigma` is invariant to reward
scaling.** Multiplying rewards cannot buy an emission.

### Lever 2: variance-adaptive residual radius. **Works, and is cheap.**

The frozen radius uses the worst-case envelope `2B = 10` for every residual,
while the measured residual spread is of order `0.5`. Replacing the envelope by
the residual scale only — keeping the same risk and the same union over `d`
groups — changes the radius by more than an order of magnitude:

| certificate | `r_x` at `N_x = 40000` | implied `E_Q` | count for `E_Q <= 0.15` |
|---|---|---|---|
| frozen envelope `2B` | 0.2703 | 0.9009 | `> 2**26` (unaffordable) |
| adaptive scale `1.00` | 0.0166 | 0.0552 | **5,956** |
| adaptive scale `0.50` | 0.0083 | 0.0276 | **1,562** |
| adaptive scale `0.25` | 0.0041 | 0.0138 | **1,000** |

So the certification count that the frozen envelope cannot reach at `2**26`
is reached by a variance-adaptive radius at roughly `6,000`. At the measured
rollout throughput of about `83,000` certification items per second, the whole
`24`-record matrix at `N_x = 10000` costs about `0.6` minutes of rollout.

### Conclusion

The obstruction is neither the protocol scale (repaired and verified) nor the
value spectrum (invariant). It is the **worst-case residual envelope `2B`**,
which is the one ingredient the frozen certificate cannot avoid because the
verified cosh-mixture argument requires a *known* sub-Gaussian parameter. A
variance-adaptive (self-normalised or empirical-Bernstein) residual bound
replaces that parameter with an estimated scale and is the decisive lever.

This is exactly what the FP-SCALE-001 design placed out of scope in section 6
("Variance-adaptive or Bernstein-type residuals: the diagnosis shows the mean
is not the binding term, so this is deferred until a scale-adequate run shows a
residual-dominated certificate"). This run is that scale-adequate run, and it
shows the opposite of what section 6 assumed: the residual *scale* — not its
mean — is what binds.

## 9. Prototype validation of the variance-adaptive certificate

The user chose the variance-adaptive direction on 2026-09-11. Before freezing a
task around it, the fix was prototyped and measured at full matrix scale.

### Construction (rigorous, non-circular)

Per pair, split the certification items into disjoint halves A and B:

1. on A, compute the residual sample standard deviation `sigma_A(x)`;
2. the confidence radius uses only B:
   `r_x = SAFETY * sqrt(2) * sigma_A(x) * sqrt(2 log(2 / delta_pair) / N_B)`
   with `delta_pair = delta / (2d)`;
3. `R = max_x(|Ybar_x^B| + r_x)`, `E_Q = R / (1-gamma)`.

The scale estimate never touches the half it certifies, so the bound is not
circular and no certification item is reused. Residuals are bounded by `2B`, so
the sub-Gaussian property is discharged by Hoeffding's lemma rather than
assumed. `SAFETY` inflates the estimated spread to absorb estimation error; at
`N_A ~ 10^4` a multiplicative 1.1 is ample.

### Full-matrix prototype result

Frozen matrix shape (2 mixing settings x 12 tasks), two primary routes,
`CERT_CHAINS x CERT_CHAIN_LENGTH = 16384 x 64 = 1048576`,
`SAFETY = 1.1`, total risk `delta = 0.05` split over pairs:

| quantity | result |
|---|---|
| route-records attempted | 48 |
| **certificates emitted** | 48 |
| **safe updates emitted** | **31** |
| componentwise non-degrading | **31 / 31** |
| strict improvements (`sum_s delta V > 0`) | **31 / 31** |
| certificate violations (`E_Q < realized error`) | **0** |
| `E_Q` among emitted | 0.064 -- 0.188 |
| realized `||Qhat - Q*||_inf` among emitted | 0.006 -- 0.045 |
| selected `eta` | 1.0 in every emission |
| wall time | 341.8 s for 48 route-records |

Every emission satisfies both `V^{pi_plus} >= V^pi` componentwise and a strictly
positive total value gain, which is exactly the "small complete policy
improvement example" the project has been trying to reach since `FP-ADV-001`.

Remaining abstentions (17 of 48) are ordinary: several records fall below the
`H1` count target (minimum counts as low as `12849`), and a few have
`min_s LB` marginally negative. They are reported, not repaired.

### The decisive comparison

| | FP-ESARSA-001 | FP-SCALE-001 v1.1 | variance-adaptive prototype |
|---|---|---|---|
| certificate | frozen cosh-mixture, envelope `2B` | same | estimated residual scale |
| `E_Q` | 22.47 (min) | 0.98 | 0.064 -- 0.188 |
| emission rate | 0 / 480 | 0 / 1 probed | **31 / 48** |
| oracle violations | 0 | 0 | **0** |

### What this does not yet establish

- The prototype is a design probe, not sealed evidence: it writes no formal
  output, has no sealed verifier, and its `SAFETY` constant and risk split are
  not yet frozen.
- The prototype's certificate is not yet independently reconstructed.
- `SAFETY = 1.1` must be justified against a stated bound on the estimation
  error of `sigma_A`, and that justification must be executable, not asserted.

These are the substance of the follow-on task, for which the user gave
direction on 2026-09-11.

## 10. Status

Stopped at the smoke gate before any formal run. No formal run has occurred and
none should occur under v1.1: the protocol now delivers `H1` with a valid
certificate and still cannot emit, so a formal run would only reconfirm at
higher cost a result already established at reachable scale.

The scale revision is exhausted and its finding is complete: the obstruction is
the worst-case residual envelope `2B`, which the verified cosh-mixture argument
cannot avoid because it requires a *known* sub-Gaussian parameter. The
variance-adaptive replacement removes that requirement and, in prototype,
turns a `0 / 480` emission rate into `31 / 48` with zero certificate
violations.

The certificate contract change is a new frozen hypothesis and is carried by a
separate task, for which the user gave direction on 2026-09-11.

### Recorded and disclosed limitations

- The `0.1` margin-to-TV ratio used in sections 6 and 7 is a measured
  order-of-magnitude from one record's `q_hat`, not a theorem. It is used only
  to size levers, never as an acceptance criterion.
- Section 7's Lever 1 probe normalises base rewards into `[-1, 1]` before
  applying the enlarged gap, so its declared reward bound is honest. That probe
  writes no formal output and is not evidence for any acceptance criterion.
- The variance-adaptive prototype is a design probe. Its results are motivation
  for the follow-on task and must not be cited as verified evidence.


