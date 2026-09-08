# FP-ADV-001 Claude route theory

Status: independent construction before the blind first-result seal.
Author: Claude Code, branch `claude/FP-ADV-001`, isolated worktree.
Baseline: task `docs/research_tasks/FP-ADV-001.md` version 1.1 (`ACTIVE`),
scientific baseline `c579047950dfabb2600020cd2e53dd24b3e39c84`, common route
execution-start commit `4078f6911cbfb4654205772685f49896e4e8cad2`.

This document derives the frozen V-first local exact and finite-softmax
action-gap certificates, the complete-Q controls, and the single `theta=0.5`
safe policy update independently from the frozen task statement and the
verified `FP-MART-001` / `FP-TU-001` theory. No GPT FP-ADV-001 material was
read.

## 1. Setting and inherited assumptions

Assumption mapping (each inherited fact is cited, not re-proved):

- (A1) Fixed Markov policy `pi` on a finite `m`-state, `d`-action MDP; one
  stationary-start on-policy trajectory sampled in the order
  `S_t -> A_t -> (R_{t+1}, S_{t+1}) -> A_{t+1}`; deterministic edge rewards
  with the predeclared bound `|R| <= R_star`; `0 < gamma < 1`
  (FP-MART-001 verified theory, sections 1-2).
- (A2) `B = R_star/(1-gamma)` bounds `||V^pi||_infinity` and
  `||Q^pi||_infinity` (FP-MART-001 verified theory, section 1).
- (A3) Filtrations `G_t` (pre-action) and `H_t = sigma(G_t, A_t)`; the
  recovery residual
  `xi^R_t = R_{t+1} + gamma V^pi(S_{t+1}) - Q^pi(S_t, A_t)` is centered given
  `H_t` with conditional range width at most `2B`; predictable visit selection
  keeps the martingale transform valid (FP-MART-001 verified theory,
  sections 2-3).
- (A4) The verified `FP-TU-001` simultaneous event: over
  `G = m + 2d` pre-registered groups (`m` state Bellman, `d` pair Bellman,
  `d` recovery), with probability at least `1 - delta`, every group's
  fixed-target residual mean over its first `k` visits is within the frozen
  cosh-mixture radius `r_g(k)` for all `1 <= k <= n` simultaneously.
  Substituting the random observed count `N_g` is valid because the event is
  already uniform in `k`; no conditioning on counts is used (FP-TU-001
  verified theory, sections 1-2). This task spends no new risk: it reuses
  exactly this event.
- (A5) On the same event, whenever the exact (respectively finite-softmax)
  state route emits, `||V_hat - V^pi||_infinity <= E_V` (respectively
  `E_V^beta`), where the bounds are deterministic functions of observed
  counts, public hyperparameters, and iteration metadata (FP-TU-001 verified
  theory, section 4; serialized `state_value` route bounds).
- (A6) On the same event, whenever a complete-Q route emits, its
  `total_bound E_Q` covers `||q_hat - Q^pi||_infinity`, with the V-first
  compositions `E_Q^exact = gamma E_V + max_h r_h` and
  `E_Q^soft = gamma E_V^beta + max_h r_h + 2B(1 - kappa_min)`, where
  `kappa_min` is the minimum observed one-hot softmax diagonal over all
  pairs (FP-TU-001 verified theory, section 4; serialized route bounds).
- (A7) Every fixed-policy recovery target satisfies
  `|R_{t+1} + gamma V^pi(S_{t+1})| <= R_star + gamma B = B` (from A1-A2).
- (A8) Selective semantics: the only probability statement is over the joint
  event of emission and violation; there is no conditional-coverage claim and
  no high-probability support claim (FP-MART-001 section 1, FP-TU-001
  section 2).

The frozen task adds only observable combinatorial objects: empirical
successor rows, normalized one-hot softmax weights, and the declared transfer
fraction. No true kernel, occupancy, value, Q table, gap, realized error,
exact return, or oracle diagonal enters any certificate input.

## 2. Lemma (total variation times span)

For probability vectors `p, q` on a finite set and any real vector `e`,

```text
|(p - q)^T e| <= TV(p, q) span(e) <= 2 ||e||_infinity TV(p, q),
TV(p, q) = 0.5 ||p - q||_1,  span(e) = max e - min e.
```

Proof. `(p - q)^T 1 = 0`, so for any constant `c`,
`(p - q)^T e = (p - q)^T (e - c 1)`. Choose `c = (max e + min e)/2`; then
`||e - c 1||_infinity = span(e)/2` and Hoelder gives
`|(p - q)^T e| <= ||p - q||_1 * span(e)/2 = TV(p, q) span(e)`.
`span(e) <= 2||e||_infinity` is immediate. []

Consequence used below: common value offsets cancel exactly, because if
`e = c 1` then `(p - q)^T e = 0` for any rows `p, q`.

## 3. Exact V-first local action gap (task hypothesis 1)

For a visited pair `g = (s, a)` with observed count `N_g > 0`, define from the
trajectory only

```text
q_hat_g   = (1/N_g) sum_{t: (S_t,A_t)=g} [R_{t+1} + gamma V_hat(S_{t+1})],
P_bar_g   = empirical successor-state row over the same visits,
r_g       = frozen FP-TU-001 recovery radius at the observed count N_g.
```

Decomposition (exact, deterministic): with `e = V_hat - V^pi`,

```text
q_hat_g - Q^pi(g)
  = (1/N_g) sum [R_{t+1} + gamma V^pi(S_{t+1}) - Q^pi(g)]
    + gamma (P_bar_g)^T e
  = zeta_g + gamma (P_bar_g)^T e,
```

where `zeta_g` is the recovery residual mean of group `g`. On the simultaneous
event (A4), `|zeta_g| <= r_g` at the random count `N_g`.

For two actions `a, b` in the same state `s`, subtracting the decompositions
and applying the section-2 lemma with `||e||_infinity <= E_V` (A5) gives, on
the event, whenever the exact state route emits and both counts are positive,

```text
|(q_hat_sa - q_hat_sb) - (Q^pi(s,a) - Q^pi(s,b))|
  <= r_sa + r_sb + 2 gamma E_V TV(P_bar_sa, P_bar_sb)
  = U_exact(s,a,b),
```

hence `Q^pi(s,a) - Q^pi(s,b) >= LCB_exact(s,a,b)` with
`LCB_exact = q_hat_sa - q_hat_sb - U_exact`. Only the two compared pairs need
positive counts; unrelated missing pairs enter nowhere. The state bound
`E_V` is the only support-global ingredient. []

## 4. Finite-softmax V-first local action gap (task hypothesis 2)

For pair query `g`, the verified one-hot softmax weights over the trajectory
of length `n` are `w_t ∝ exp(beta 1{(S_t,A_t)=g})`. From the observed count
`N_g`, length `n`, and declared `beta` alone,

```text
Z_g     = N_g e^beta + (n - N_g),
kappa_g = N_g e^beta / Z_g        (total on-group mass).
```

The softmax recovery estimate and its effective successor row are

```text
q_hat_beta_g  = sum_t w_t [R_{t+1} + gamma V_hat(S_{t+1})],
P_bar_g^beta(s') = sum_t w_t 1{S_{t+1}=s'}
  = [e^beta cnt_g(s') + (cnt_tot(s') - cnt_g(s'))] / Z_g,
```

where `cnt_g` is the observed successor histogram of pair `g` and `cnt_tot`
the total successor histogram. `P_bar_g^beta` is finite, nonnegative, and sums
to one by construction; the implementation validates this explicitly.

Fixed-target split. Write each target as
`R + gamma V^pi(S') + gamma e(S')`. The on-group part contributes
`kappa_g * zeta_g` with `|zeta_g| <= r_g` on the event. The off-group part is
`(1 - kappa_g)(nu_off - Q^pi(g))`, where `nu_off` is a convex combination of
recovery targets; by (A7) every target lies in `[-B, B]`, and
`Q^pi(g) in [-B, B]` by (A2), so `|nu_off - Q^pi(g)| <= 2B`. Therefore

```text
|sum_t w_t [R_{t+1} + gamma V^pi(S_{t+1})] - Q^pi(g)|
  <= kappa_g r_g + 2B(1 - kappa_g) = C_g.
```

No true kernel, true target, or oracle diagonal enters: `kappa_g` uses only
`(N_g, n, beta)` and the contamination uses only the declared reward bound.

Pairwise. Subtracting the two queries' decompositions, the value-error terms
combine into `gamma (P_bar_sa^beta - P_bar_sb^beta)^T e`; the section-2 lemma
with `||e||_infinity <= E_V^beta` (A5) gives, on the event, whenever the
softmax state route emits and both counts are positive,

```text
U_soft(s,a,b) = C_sa + C_sb + 2 gamma E_V^beta TV(P_bar_sa^beta, P_bar_sb^beta),
LCB_soft      = q_hat_beta(s,a) - q_hat_beta(s,b) - U_soft(s,a,b),
```

and `Q^pi(s,a) - Q^pi(s,b) >= LCB_soft(s,a,b)`. []

## 5. Global controls and dominance (task hypothesis 4)

For any emitted complete-Q route bound `E_Q` (A6), on the same event

```text
|(q_hat(s,a) - q_hat(s,b)) - (Q^pi(s,a) - Q^pi(s,b))| <= 2 E_Q,
```

so `LCB_global(s,a,b) = q_hat(s,a) - q_hat(s,b) - 2 E_Q` is a valid lower
bound. Controls: V-first no-split exact/softmax global (same estimates as the
local routes) and Direct-Q exact/softmax global (own estimates). No local
Direct-Q recurrence is introduced.

Dominance, exact: when the V-first global exact control emits, pair support is
full, so `r_g <= max_h r_h` for both compared pairs, and `TV <= 1`; hence

```text
U_exact(s,a,b) <= 2 max_h r_h + 2 gamma E_V = 2 E_Q^exact.
```

Dominance, softmax: when the V-first global softmax control emits,
`kappa_g >= kappa_min > 0` and `r_g <= max_h r_h` for all pairs, so
`C_g = kappa_g r_g + 2B(1 - kappa_g) <= max_h r_h + 2B(1 - kappa_min)`;
hence

```text
U_soft(s,a,b) <= 2 max_h r_h + 4B(1 - kappa_min) + 2 gamma E_V^beta
              = 2 E_Q^soft.
```

Both inequalities are strict whenever the two compared successor rows overlap
(`TV < 1`) or a compared radius is below the maximum. The analyzer rejects any
record violating these inequalities when the corresponding control emits. []

## 6. Post-data selection (task hypotheses 3 and 6)

The receiver `a_star(s) = argmax_a q_hat(s,a)` (smallest-index tie break) and
the donor set are chosen after observing estimates. This is covered without
new risk: every statement used by any emitted update — the two recovery radii
at the observed counts, the state bound, the effective-row validity — is a
deterministic consequence of the single simultaneous event (A4-A5) over all
pre-registered groups. Selection among deterministic consequences of one
event is not a selection among separately calibrated certificates; no
route-level or pair-level resplit of `delta` occurs.

Required support is exactly: full state support for the emitted state bound,
and positive counts for the two compared actions. An unrelated missing pair
blocks nothing; a missing compared pair produces the ordered abstention
`candidate_pair_unvisited`. If the receiver itself is unvisited, every donor
comparison in that state abstains with `candidate_pair_unvisited` and the
state transfers nothing; safety is unaffected because eligibility requires
both counts positive.

## 7. One safe update (task hypothesis 5)

Freeze `theta = 1/2`. A donor `b != a_star(s)` is eligible only when both pair
counts are positive, the applicable `LCB(s, a_star, b) > 0`, and
`pi(b|s) > pi_min`. Transfer

```text
eta(s,b) = theta [pi(b|s) - pi_min]
```

from each eligible donor to `a_star(s)`. The update is row-local:

- row sums are preserved exactly (mass removed from donors equals mass added
  to the receiver, by construction in exact arithmetic on the same floating
  values: `pi_plus(b|s) = pi(b|s) - eta`, `pi_plus(a_star|s) += sum_b eta`);
- nonnegativity and the floor hold because `theta in (0,1)` gives
  `pi_plus(b|s) = (1-theta) pi(b|s) + theta pi_min > pi_min`;
- if no donor is eligible in any state, the original policy is returned and
  the record is an abstention.

Bellman identity. For `Q^pi(s,a) = r(s,a) + gamma P_sa^T V^pi`,

```text
(T_{pi_plus} V^pi)(s) - V^pi(s)
  = sum_a pi_plus(a|s) Q^pi(s,a) - sum_a pi(a|s) Q^pi(s,a)
  = sum_{eligible b} eta(s,b) [Q^pi(s,a_star) - Q^pi(s,b)]
  >= sum_{eligible b} eta(s,b) LCB(s,a_star,b) >= 0,
```

where the first inequality holds on the simultaneous event for every route's
applicable LCB (sections 3-5). `T_{pi_plus}` is monotone and a
`gamma`-contraction with fixed point `V^{pi_plus}`; iterating
`V^pi <= T_{pi_plus} V^pi` gives `V^{pi_plus} >= V^pi` componentwise.
Non-degradation is guaranteed; strict improvement is an oracle-described
empirical outcome, not a claim. []

## 8. Probability statement (exactly one)

For every emitted update,

```text
P(EmitUpdate and (
    any used action ordering is false
    or exists s: V^{pi_plus}(s) < V^pi(s))) <= delta.
```

Proof. On the simultaneous event (probability at least `1 - delta`), every
used LCB lower-bounds its true action difference (sections 3-5), so every
used ordering is true, and the section-7 Bellman lower bound is nonnegative
in every state, which implies `V^{pi_plus} >= V^pi` componentwise. The bad
joint event is therefore contained in the complement of the simultaneous
event, whose probability is at most `delta`. Emission itself is a
deterministic function of the trajectory and certificate; no statement
conditional on emission is made. []

## 9. Ordered reasons contract

Pair and update reasons use the frozen order

1. `algorithm_mode_mismatch`;
2. `divergence_guard_triggered`;
3. `state_certificate_not_emitted`;
4. `candidate_pair_unvisited`;
5. `recovery_radius_unavailable`;
6. `attention_mass_invalid`;
7. `effective_transition_row_invalid`;
8. `numerical_nonfinite`;
9. `gap_lcb_nonpositive`;
10. `no_transferable_mass`.

Applicability mapping used by this route (documented for review):

- Reasons 1-2 are route-level: mode flags come from the serialized certificate
  algorithm metadata; the divergence flag exists for the Direct-Q estimators
  and is `False` for the clipped V-first value iteration.
- Reason 3 applies to all four V-first routes (local and global) whenever the
  applicable state bound was not emitted; it does not apply to Direct-Q
  controls.
- Reason 5 covers: a missing local recovery radius (unvisited pair is caught
  earlier by reason 4; a `null` mixture radius at a positive count is caught
  here), and, for global controls, an unemitted complete-Q bound `E_Q`
  (for V-first globals this is reachable only when the state certificate
  emitted but full pair support or inversion failed).
- Reasons 6-7 apply to the finite-softmax local route only
  (`kappa_g` outside `(0,1]` or nonfinite; an effective row that fails the
  finite/nonnegative/sums-to-one validation).
- Reasons 9-10 are ordinary abstentions.

A failure in one state or pair never invalidates a valid comparison
elsewhere; no fallback to oracle values, other formulas, or other routes
exists.

## 10. Scope and limitations

- One fixed policy, one trajectory, one update; no repeated control, no
  online claim, no convergence claim.
- The bound validity is exactly the inherited selective statement; no
  conditional-on-emission coverage and no high-probability support claim.
- The exact route's TV factor can be small only if the two observed successor
  rows actually overlap; nothing is claimed when they do not.
- Theorem evidence is the proof above plus the deterministic verifier;
  empirical audits (`oracle_audit`) are structurally separated and never
  certificate inputs.
