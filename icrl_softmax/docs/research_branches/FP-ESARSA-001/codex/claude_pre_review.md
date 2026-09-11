# FP-ESARSA-001 v1.1 — Claude read-only pre-review

Date: 2026-09-11. Reviewer: Claude (read-only; no code executed, no files
changed by this review). Object under review: task `FP-ESARSA-001` v1.1 at
the REVIEW publication commit `61b5592` on `codex/FP-ESARSA-001`, with
approved design commit `75fc07217ec8a3e096804caade436ab8c2358659` and plan
`docs/superpowers/plans/2026-09-10-fixed-policy-expected-sarsa-relative-softmax-plan.md`.

Transparency note: the v1.1 revision text itself was recorded by Claude under
the user's direct ruling of 2026-09-11 (Codex execution quota exhausted).
The ruling changes responsibilities, baseline, and the read-only input list
only. This pre-review covers the full v1.1 task definition, including the
unchanged v1.0 scientific content. The user is the arbiter of this deviation
from the usual GPT-authored revision flow.

## Itemized checks

### 1. Problem definition and falsifiability

The research question has eight falsifiable hypotheses. H1–H7 are mandatory
mathematical/construction/software claims; H8 is the empirical usefulness
claim with an explicitly allowed negative result and a no-retuning rule.
PASS.

### 2. Mathematical targets (independently re-derived, no fitting)

- **Diagonal-margin contraction premise.** For
  `F_pi(Q)=Q+alpha M(T_pi^X Q-Q)` the linear part is
  `I-alpha M+alpha gamma M P_pi^X`. With `M`, `P_pi^X` row-stochastic and
  `0<alpha<=1`, row `x` of the linear part has nonnegative diagonal
  `1-alpha M_xx+alpha gamma (MP)_xx`, so its infinity norm is at most
  `1+alpha gamma-2 alpha min_x M_xx`. The frozen premise
  `min_x M_xx >= (1+gamma)/2+C` therefore yields the declared bound
  `1-2 alpha C < 1`. The derivation is valid; the premise is correctly
  conditional and must be checked empirically, not asserted. PASS.
- **Fixed-point identity.** `F_pi(Q^pi)=Q^pi` follows directly from
  `T_pi^X Q^pi=Q^pi`. PASS.
- **Bellman-residual certificate.** On the simultaneous event,
  `|(T_pi^X Qhat-Qhat)(x)| <= |Ybar_x|+r_x`, and the standard contraction
  argument gives `||Qhat-Q^pi||_inf <= ||T_pi^X Qhat-Qhat||_inf/(1-gamma)
  <= E_Q`. PASS.
- **Held-out residual bounds.** With `||Qhat||_inf<=B=R_star/(1-gamma)`,
  every held-out residual lies in `[-R_star-(1+gamma)B, R_star+(1+gamma)B]`
  = `[-2B, 2B]`, and its conditional mean given the training filtration and
  `X_t=x` is `(T_pi^X Qhat-Qhat)(x)`. The visit-indexed martingale and
  random-count substitution are inherited from VERIFIED FP-MART-001 /
  FP-TU-001. PASS.
- **Mixture grid reuse.** `build_mixture_grid(n_groups, delta, components=15)`
  in `time_uniform_mixture_certificate.py` is parametric in `n_groups`
  (verified in source at the v1.1 baseline); constructing the new event with
  exactly `d=24` held-out pair groups reuses the verified 15-component grid
  without code change. PASS.
- **Exact relative-softmax improvement.** For
  `pi_eta^+ proportional to pi exp(eta Q^pi)`,
  `sum_a (pi_eta^+ - pi) Q^pi = sum_a (pi_eta^+ - pi) A^pi
  = (1/eta)(KL(pi_eta^+||pi) + KL(pi||pi_eta^+)) >= 0` for every `eta>0`;
  `eta=0` is the identity. PASS.
- **Approximate-Q lower bound and improvement implication.**
  `|sum_a (pi_eta^+-pi)(Qhat-Q^pi)| <= E_Q ||pi_eta^+-pi||_1` (Hoelder), so
  on the certificate event `LB_s(eta)>=0` for all states implies
  `T_{pi_plus} V^pi >= V^pi` componentwise, and the policy improvement
  theorem gives `V^{pi_plus}>=V^pi`. The stated probability claim
  `P(EmitUpdate and exists s: V^{pi_plus}(s)<V^pi(s)) <= delta` follows
  from the simultaneous event. PASS.
- **Finite-route honesty.** The task explicitly makes no asymptotic no-bias
  claim for the full finite-logit route and certifies its estimate only via
  the exact held-out residual. This is the correct, non-overclaiming
  treatment. PASS.

### 3. Network boundary

The finite route forbids input-dependent equality masks and visited-query
gates, uses fixed one-hot identifiers and frozen sharpness 8, keeps exactly
one canonical Q-memory token per logical pair, and reports off-group mass
instead of hiding it. Self-loops are handled by separate heads, so no
candidate set contains the same logical pair twice. The boundary matches the
style verified in FP-EXPL-001. PASS.

### 4. Fair inputs and blindness

Frozen protocol fixes the 480-record matrix (30 tasks x 4 lengths x 2
mixings x 2 bonuses), seed 20260829, split, eta grid, delta, and sharpness
before any run. Oracle quantities are confined to `oracle_audit`; validation
transitions cannot influence Q construction or selection. Under the v1.1
exception, GPT does not read Claude's results before the author seal, and
its acceptance must be an independent reconstruction. PASS.

### 5. Completeness

Acceptance criteria 1–20 cover memory uniqueness, exact/finite equivalences,
proofs, certificate validity, emission safety, abstention bit-identity,
oracle separation, strict-JSON integrity, matrix identity, violation
enumeration, H8 without retuning, sealing, and reciprocal verification.
Failure and stopping conditions are explicit. PASS.

### 6. Tolerances

Deterministic fixture equivalences at 1e-12 and exact probability row sums
at 1e-12 are consistent with the verified predecessor tasks. PASS.

### 7. Resources

CPU-only, one smoke matrix and one frozen 480-record formal run, no new fee
category. Under the v1.1 single-route exception, Claude carries the main
4–7 hour estimate; GPT acceptance is a lighter reconstruction (pattern
proven in FP-EXPL-001). PASS.

### 8. Failure handling

The ordered 11-reason non-emission list forbids oracle fallbacks and second
risk budgets; H8 may fail without invalidating the task; premise failures
are reported, not oracle-repaired. PASS.

## Advisory notes (non-blocking)

A. **Baseline refresh**: v1.0 named c579047; v1.1 records 9d0994f with
   byte-identical inherited science files (verified by diff). Resolved.
B. **Responsibility exception**: the dual blind-construction requirement is
   replaced by the user-ruled FP-EXPL-001 pattern. Recorded in v1.1; the
   preliminary-results caveat stands until GPT acceptance or user exemption.
C. **H8 expectation**: the certificate carries a 1/(1-gamma)=3.33
   amplification, and FP-ADV-001 emitted zero updates on the same 480
   records; a zero-emission verified negative result is a realistic outcome
   and is allowed.
D. **Premise strength**: `min_x M_xx >= 0.85+C` is demanding for the
   finite writeback; empirical failure is a reportable conditional result,
   not a task defect.
E. **Emission rule determinism**: the descending eta grid with a
   first-passage emission rule is frozen and deterministic; no concern.

## Verdict

No blocking defect found in problem definition, mathematical targets,
network boundary, fair inputs/blindness, completeness, tolerances,
resources, or failure handling.

**APPROVED**
