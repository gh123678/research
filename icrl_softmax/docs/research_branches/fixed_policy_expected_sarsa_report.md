# Fixed-policy Expected SARSA and certified relative-softmax improvement — main-route report (FP-ESARSA-001)

Claude main-route scientific report under the v1.1 responsibility exception.
Companion theory: `docs/research_branches/fixed_policy_expected_sarsa_theory.md`.
Preliminary until GPT post-seal acceptance or an explicit user exemption.

## Question

For a finite MDP and one frozen full-support policy, can standard normalized
softmax attention implement a state-action Expected SARSA residual iteration,
produce an observable cross-fitted Bellman-residual certificate for its final
`Q` estimate without an environment model, and use that certificate to emit
one pointwise non-degrading relative-softmax policy update?

## Construction

Three routes share one canonical Q-memory (one token per state-action pair),
one contiguous half split, one certificate, and one decision rule.

- `expected_exact` (primary): synchronous grouped Expected SARSA. The
  successor expectation replaces the sampled next action by
  `sum_b pi(b|S_{t+1}) Q(S_{t+1},b)`; the current head reads the unique pair
  token; the writer averages the residuals of transitions with the queried
  current pair. Unvisited pairs are unchanged. This equals direct batch
  Expected SARSA at `1e-12` on fixtures, including self-loops.
- `expected_finite` (primary): the same iteration with every exact one-hot
  retrieval replaced by a finite-sharpness softmax over the full token
  memory (`zeta=xi=tau=8`), with **no equality mask and no visited gate**, so
  off-group mass and unvisited-query leakage are reported rather than hidden.
  Its final estimate is certified only by the exact held-out residual; no
  no-bias claim is made.
- `sampled_exact` (control): the observed next action, no variance-dominance
  claim.

The exact-residual population operator `F_pi(Q)=Q+alpha M_pi^X(T_pi^X Q-Q)`
preserves `Q^pi` and, under the frozen diagonal-margin premise
`min_x M_xx >= (1+gamma)/2 + C`, has infinity-norm contraction
`<= 1-2 alpha C`. The theorem is conditional; empirical premise failures are
reported, never oracle-repaired.

## Certificate

Freeze `Qhat` before reading the held-out suffix. For each held-out
transition, `Y_t = R_{t+1} + gamma sum_b pi(b|S_{t+1}) Qhat(S_{t+1},b) -
Qhat(S_t,A_t)` is a bounded (`2B`, `B=R_star/(1-gamma)`) martingale difference
around `(T_pi^X Qhat - Qhat)(x)` conditional on the training filtration. Reusing
the verified 15-component geometric cosh-mixture event with exactly `d=|S||A|`
pair groups gives radius `r_x = 2B q_mix(N_x;d,delta)/N_x`, and

```
E_Q = max_x(|Ybar_x| + r_x)/(1-gamma)   bounds   ||Qhat - Q^pi||_infinity
```

model-free in its inputs. Only `Qhat`, the policy, held-out transitions,
`R_star`, `gamma`, `delta`, and public mixture constants enter.

## Relative-softmax improvement

`pi_eta^+ ∝ pi exp(eta Qhat)` over the frozen descending grid
`(1.0, 0.5, 0.2, 0.1, 0.05)`, emitting the first candidate whose every-state
lower bound `Ihat_s(eta) - E_Q ||pi_eta^+(.|s)-pi(.|s)||_1` is strictly
positive. Abstention returns the input policy bit-for-bit.

## Frozen experiment (480 records, three routes each)

Generator identity reproduces the frozen FP-TU-001 baseline in all 480 records
at `1e-12`.

- Certificates: `283/480` per route. Emission is support-limited — `0/30` at
  length 256, partial at 1024, `0.967`–`1.0` at 4096, `1.0` at 16384 — and
  never fails for a reason outside the frozen ordered list.
- `E_Q` among emitted: `22.47`–`134.01` (mean `57.69`), shrinking to mean
  `~27.3` at length 16384.
- Oracle audit: **0** certificate violations, **0** residual-event
  violations, **0** value decreases; reward bound satisfied in all records.
- Contraction premise: exact and sampled routes `480/480` satisfied (identity
  writeback); finite route `270/480`, failing exactly where a training pair is
  unvisited at lengths 256/1024.
- **No primary route emitted a safe update** in any record.

## Interpretation

The mandatory mathematics and construction hold and are verified; the
certificate is valid but loose (realized error `<= 2.19` against `E_Q >= 22.5`),
because the simultaneous `d=24` mixture radius dominates at moderate held-out
counts. Consequently the certified improvement lower bound stays nonpositive
and the algorithm abstains everywhere. This is the expected conservative
behaviour of the frozen design, not a defect: hypothesis 8 is a **verified
negative usefulness result**, and no formula, sharpness, eta grid, split, or
matrix was retuned to force an emission.
