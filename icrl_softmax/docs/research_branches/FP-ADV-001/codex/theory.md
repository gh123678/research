# FP-ADV-001 Codex route: frozen action-gap theory

Date: 2026-09-08  
Route: `codex/FP-ADV-001`  
Common scientific activation: `10a9a94e24ec92a59e7c756f9af6ce07b2f30e59`  
Common route execution baseline: `4078f6911cbfb4654205772685f49896e4e8cad2`

This note derives the new deterministic consequences of the verified
`FP-TU-001` simultaneous event before implementation or empirical output is
inspected. It introduces no new risk split. All selection below occurs after
the event has simultaneously bounded the pre-registered state Bellman, pair
Bellman, and recovery groups.

## 1. Inherited selective event

Let `E` denote the `FP-TU-001` event, with `P(E^c) <= delta`, on which every
emitted state-value, pair-value, and fixed-target recovery statement holds for
all registered groups at their observed counts. The action-gap builder is a
deterministic function of observable data and those simultaneous statements.
Thus an adaptively selected receiver or donor does not consume new risk: if an
emitted ordering is false, or if its resulting policy decreases value, then
`E` must have failed. Consequently

```text
P(EmitUpdate and (false used ordering or some value decrease)) <= delta.
```

There is no lower bound on emission probability and no conditional-on-emission
coverage claim.

## 2. Exact V-first local gap

For a visited pair `g=(s,a)`, let `Pbar_g` be its empirical successor row and
let `e=Vhat-V^pi`. The empirical recovery estimate and its fixed-target version
satisfy

```text
qhat_g - Q^pi_g
  = [qhat_g - Pbar_g(R + gamma V^pi)] + gamma Pbar_g e.
```

On `E`, the absolute value of the bracketed recovery error is at most `r_g`.
For two actions `a,b` in one state, subtraction yields a propagation term
`gamma(Pbar_sa-Pbar_sb)^T e`. For probability rows `p,q`,

```text
|(p-q)^T e| <= TV(p,q) span(e) <= 2 TV(p,q) ||e||_infinity.
```

Therefore, whenever the exact state route emits `||e||_infinity <= E_V`,

```text
|[(qhat_sa-qhat_sb) - (Q^pi_sa-Q^pi_sb)]|
 <= r_sa+r_sb+2 gamma E_V TV(Pbar_sa,Pbar_sb).
```

Subtracting this uncertainty from the estimated gap gives the frozen exact
LCB. Common successor-row error cancels; disjoint rows attain the worst-case
TV factor one.

## 3. Finite-softmax local gap

For a query group with count `c_g` in a prefix of length `n`, one-hot softmax
weights give

```text
kappa_g = c_g exp(beta) / [c_g exp(beta) + n-c_g].
```

The effective successor row is the normalized convex combination

```text
Pbar_g^beta
 = kappa_g Pbar_g
   + (1-kappa_g) Pbar_not-g,
```

with the off-group row omitted when `n=c_g`. Every fixed-policy recovery target
lies in `[-B,B]`, where the builder derives `B=R_star/(1-gamma)` from the
declared reward bound. On-group recovery contributes at most `kappa_g r_g`;
arbitrary off-group contamination contributes at most
`2B(1-kappa_g)`. Hence

```text
C_g = kappa_g r_g + 2B(1-kappa_g),
U_soft(a,b) = C_a+C_b
              + 2 gamma E_V^beta TV(Pbar_a^beta,Pbar_b^beta).
```

No true kernel, target, occupancy, value, Q table, realized error, or oracle
diagnostic appears in this calculation.

## 4. Global controls and dominance

An emitted complete-Q route gives the standard pairwise penalty `2E_Q`. For
exact no-split V-first,

```text
E_Q = gamma E_V + r_max.
```

Because `TV<=1` and each `r_g<=r_max`, the exact local uncertainty is at most
`2E_Q`.

For softmax no-split V-first, let `kappa_min` be the minimum observed diagonal
mass. Its global recovery component is
`r_max+2B(1-kappa_min)`. Since
`kappa_g r_g<=r_max` and `1-kappa_g<=1-kappa_min`, each local `C_g` is no larger
than that global component. Again `TV<=1`, so the local pair penalty is at most
the global `2E_Q`. This comparison is only asserted when the matching global
route emits. Direct-Q remains a global `2E_Q` control; no local Direct-Q
recurrence is introduced.

## 5. Deterministic selection and one safe update

For each route and state, choose the smallest-index maximizer of `qhat`. A donor
must differ from the receiver, both selected pairs must have positive counts,
its LCB must be strictly positive, and its policy mass must exceed `pi_min`.
Transfer exactly

```text
eta_b = 0.5 [pi(b|s)-pi_min]
```

from every eligible donor to the receiver. Donors remain at least `pi_min`,
all other entries are unchanged, and the receiver obtains precisely the mass
removed from donors. Thus rows retain unit mass and all entries remain
nonnegative and at least the declared floor.

On `E`, every used true gap dominates its positive LCB, so statewise

```text
(T_{pi_plus}V^pi)(s)-V^pi(s)
 = sum_b eta_b [Q^pi(s,a_star)-Q^pi(s,b)]
 >= sum_b eta_b LCB(s,a_star,b) >= 0.
```

Monotonicity of `T_{pi_plus}` gives
`V^pi <= T_{pi_plus}V^pi <= T_{pi_plus}^2V^pi <= ...`; contraction makes the
limit `V^{pi_plus}`. Therefore `V^{pi_plus}>=V^pi` componentwise. Abstention
returns the original policy and has equality.

## 6. Implementation consequences

- Receiver selection includes every action exactly as frozen. If the selected
  receiver is unvisited, every comparison involving it abstains with
  `candidate_pair_unvisited`; no visited-only fallback is allowed.
- Missing unrelated pairs cannot block another state's valid local comparison.
- State-certificate non-emission blocks local comparisons, while complete-Q
  non-emission blocks only its corresponding global control.
- Reasons are canonicalized in the task's ten-item order; numeric failure never
  invokes another route or oracle fallback.
- All new observable output is placed below `action_gap_certificate`; truth is
  computed later and only below its separate `oracle_audit` namespace.

The mandatory algebra is internally complete under the frozen assumptions, so
the route may proceed to contract tests without modifying the scientific task.
