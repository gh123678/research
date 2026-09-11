# FP-EXPL-001 (v1.1) — Claude route theory: fixed-batch grouped-mean Q iteration

Scope: theory for the Claude main-execution route of task FP-EXPL-001 v1.1
(`docs/research_tasks/FP-EXPL-001.md`, status ACTIVE). This document states the
operators and proves the bounds that `witness.py` and the independent
`verify.py` check numerically. It changes no protocol inputs.

## 1. Frozen task parameters

Two states and two actions, both in {0, 1}; state-action pairs are indexed
x = 2s + a in the canonical order 00, 01, 10, 11. Rewards depend only on the
current pair; there is no terminal state.

- gamma = 0.7, alpha = 0.5
- target_pi = [[0.75, 0.25], [0.25, 0.75]] (used by the successor average)
- behavior_pi = [[0.5, 0.5], [0.5, 0.5]] (used by the sampler only)
- rewards by pair (00, 01, 10, 11) = (1.0, -0.5, 0.25, 0.25)
- P_next_state by pair = [[0.75, 0.25], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25]]
  (sampler and audit only; never a network input)
- initial state 0; batch length N = 64; seed 20260911
  (numpy.random.Generator(numpy.random.PCG64(20260911)), exactly two scalar
  rng.random() draws per transition, action draw then transition draw)
- Q0 = (0, 0, 0, 0); 64 synchronous updates on the single frozen batch
- finite sharpness xi = zeta = tau = 8.0
- coverage gate: min_x n_x >= 1 over the four pairs, checked before any Q
  update; on failure the batch is rejected with COVERAGE_FAILURE and no Q
  iteration is run (no resampling, no seed choice).

The sampled batch is frozen once: all iterations reuse the same 64
transitions; there is no new data per round and no policy update.

## 2. Fixed-batch grouped-mean operator (exact route)

Let x_t = (s_t, a_t) be the observed pair and u_t = s_next,t the observed next
state in transition t = 0..N-1, and n_x the visit count of pair x. On a
covered batch (n_x >= 1 for all x), define the exact grouped matrices

    C0[t, y]      = 1{x_t = y}
    S0[t, (u, b)] = 1{u_t = u} * target_pi[b | u]
    W0[x, t]      = 1{x = x_t} / n_x

and the per-transition grouped TD residual

    d0_t(q) = r_t + gamma * sum_b target_pi[b | u_t] * q[u_t, b] - q[x_t].

The exact grouped-mean operator is

    F0(q) = q + alpha * W0 (r + gamma S0 q - C0 q) = q + alpha * W0 d0(q).

W0 performs a plain within-group arithmetic mean: row x places weight 1/n_x on
each occurrence of pair x. No visitation-frequency multiplier n_x/N appears
anywhere. The successor average uses target_pi only; behavior_pi appears only
in the sampler; P appears only in the sampler and the audit.

## 3. Exact and finite attention matrices

Exact route (declared equality masks; not counted as a finite-network
capability): the attention probabilities are the declared equality
distributions C0, S0, W0 above, and the exact grouped-attention operator
equals the direct grouped-mean reference F0 (checked as H2a).

Finite route (no content-equality mask, no visitation gate): with
xi = zeta = tau = 8.0,

    C[t, y]      = softmax_y( xi   * 1{x_t = y} )
    S[t, (u, b)] = softmax_(u,b)( zeta * 1{u_t = u} + log target_pi[b | u] )
    W[x, t]      = softmax_t( tau  * 1{x = x_t} )
    Ff(q)        = q + alpha * W (r + gamma S q - C q).

The indicator products are dot products of declared one-hot identity features
carried in the prompt; they do not replace the literal attention construction.
The literal witness implements Ff with a fixed-weight normalized-softmax
attention network: prompt matrix H (4 memory tokens, one per canonical pair;
N = 64 context tokens, one per batch row; 1 null token; 19 embedding fields),
explicit fixed WQ/WK/WV/WO per stage, scaled-dot-product softmax attention
with residual connections, a fixed linear feed-forward residual map
d = r + gamma*succ - cur, a fixed linear Q-update map q += alpha * wd, and a
fixed scratch-clearing projection that zeroes the four scratch fields after
each update. Only static role/position masks are used (context queries read
memory keys, memory queries read context keys, inactive queries read the null
token whose value is zero). Weights depend only on dimensions, the frozen
target_pi, gamma, alpha and the sharpness constants — never on sampled
rewards, visit counts, dynamic Q, or the true transition matrix.

## 4. Affine form of both operators

Because the matrices C0, S0, W0, C, S, W and the reward vector r are fixed by
the frozen batch, both operators are affine in q:

    F0(q) = G0 q + b0,   G0 = I + alpha W0 (gamma S0 - C0),   b0 = alpha W0 r
    Ff(q) = Gf q + bf,   Gf = I + alpha W  (gamma S  - C ),   bf = alpha W  r.

The witness derives G0, b0, Gf, bf directly from these formulas (not by
fitting traces) and separately verifies affinity on the four standard basis
vectors plus the zero vector Q0 = 0. These are the only one-step network
probes: q_pi is audit-only and never a network input (task section 4), and
q_hat and arbitrary non-basis vectors are not probed through the network
either. Since both maps are affine, exact reconstruction of G and b on this
probe set establishes the map on every input.

## 5. The three signed error terms (H3)

For any common input q, the finite-minus-exact one-step difference decomposes
exactly into three signed stage errors:

    e_current   = -alpha * W (C - C0) q
    e_successor =  alpha * gamma * W (S - S0) q
    e_write     =  alpha * (W - W0) (r + gamma S0 q - C0 q)
    Ff(q) - F0(q) = e_current + e_successor + e_write        (exact identity).

Derivation: Ff(q) - F0(q) = alpha [ W(r + gamma S q - C q) - W0(r + gamma S0 q
- C0 q) ]. Add and subtract W(r + gamma S0 q - C0 q): the first bracket gives
alpha W (gamma (S - S0) q - (C - C0) q) = e_current + e_successor; the second
gives alpha (W - W0)(r + gamma S0 q - C0 q) = e_write. The identity is a
vector equality evaluated at q = q_exact,k for each step k; the sum of the
three inf-norms is only an upper bound on ||Ff(q) - F0(q)||_inf.

Together with the finite contraction constant c_f (Section 6) this yields the
finite-step perturbation recursion

    E_0 = 0,
    E_(k+1) = c_f * E_k + ||Ff(q_exact,k) - F0(q_exact,k)||_inf,
    ||q_finite,k - q_exact,k||_inf <= E_k,

which follows by induction from ||Ff(q) - Ff(q')||_inf <= c_f ||q - q'||_inf
and the triangle inequality. Actual per-step errors are not required to be
monotone; only the bound is claimed.

## 6. Contraction condition (H4)

Exact operator. On any covered batch, row x of W0 is supported on transitions
with x_t = x, and there C0[t, y] = 1{y = x}. Hence sum_t W0[x, t] C0[t, y] =
1{y = x}, and with m_xy = sum_t W0[x, t] S0[t, y] >= 0, sum_y m_xy = 1:

    (G0)_xy = 1{y = x} (1 - alpha) + alpha gamma m_xy.

Every entry is nonnegative because 1 - alpha = 0.5 > 0, so each row's absolute
sum equals its plain sum:

    ||G0||_inf = 1 - alpha + alpha gamma = 1 - alpha (1 - gamma) = 0.85 < 1.

Therefore F0 is an inf-norm contraction with the proved constant
c0 = 0.85 on any covered batch; by Banach's theorem F0 has a unique fixed
point q_hat and

    ||q_exact,k - q_hat||_inf <= 0.85^k ||Q0 - q_hat||_inf.

Finite operator. c_f = ||Gf||_inf is a sufficient statistic computed from the
actual finite matrices; it is not assumed in advance. On this frozen batch the
computed value is c_f = 0.85 < 1, so Gf is an inf-norm contraction, Ff has a
unique fixed point q_f,inf, and

    ||q_finite,k - q_f,inf||_inf <= c_f^k ||Q0 - q_f,inf||_inf.

Caveats kept from the task: c_f >= 1 would not imply divergence (the
sufficient condition simply fails and the fixed-point quantities would be
reported as null with the reason); the spectral radius rho(Gf) is a
floating-point numerical diagnostic only and by itself proves no spectral
property, uniqueness, or convergence.

## 7. Fixed points and the decomposition (H5)

- q_hat: unique fixed point of F0 on the covered batch (Section 6); obtained
  by solving (I - G0) q = b0 and checked by the residual F0(q_hat) - q_hat.
- q_pi (audit only): solution of the true fixed-policy Bellman equation
  q_pi = r + gamma P_pi q_pi with P_pi[x, y] = P_next[x, s_y] target_pi[b_y | s_y],
  computed from ground truth by the auditor; never a network input.
  V_pi(s) = sum_a target_pi[a | s] q_pi[s, a].
- Nondegeneracy: the target-policy mean immediate rewards by state are
  r_bar = (0.625, 0.25). If V_pi(0) = V_pi(1) = v, the Bellman equation would
  force r_bar(s) = (1 - gamma) v for both states, contradicting
  r_bar = (0.625, 0.25); hence the two true state values differ.
- Data bias: reported as-is via ||q_hat - q_pi||_inf, with the deterministic
  audit upper bound

      ||q_hat - q_pi||_inf <= ||F0(q_pi) - q_pi||_inf / (1 - c0),  c0 = 0.85,

  Proof: q_hat = F0(q_hat), so with D = ||q_hat - q_pi||_inf the triangle
  inequality and the c0-contraction of F0 give

      D = ||F0(q_hat) - q_pi||_inf
        <= ||F0(q_hat) - F0(q_pi)||_inf + ||F0(q_pi) - q_pi||_inf
        <= c0 D + ||F0(q_pi) - q_pi||_inf,

  hence (1 - c0) D <= ||F0(q_pi) - q_pi||_inf and the bound follows because
  c0 = 0.85 < 1. (Repaired derivation: an earlier draft of this document
  asserted the unjustified intermediate inequality
  c0 D >= ||F0(q_pi) - q_pi||_inf - ||F0(q_hat) - F0(q_pi)||_inf; that line
  was invalid and has been replaced by the argument above. The bound itself
  is unchanged.)
  This bound uses audit ground truth; it is not a learner-computable
  statistical certificate, and a nonzero (or zero) data bias is a reported
  metric, not a success condition.
- q_f,inf: computed only because c_f < 1 holds here; checked via the residual
  Ff(q_f,inf) - q_f,inf and the steady-state bound

      ||q_f,inf - q_hat||_inf <= ||Ff(q_hat) - q_hat||_inf / (1 - c_f),

  by the same argument applied to the c_f-contraction Ff:
  ||q_f,inf - q_hat||_inf = ||Ff(q_f,inf) - q_hat||_inf
  <= c_f ||q_f,inf - q_hat||_inf + ||Ff(q_hat) - q_hat||_inf.

Three-term decomposition: for every k = 0..64 the signed vector identity

    q_finite,k - q_pi = (q_finite,k - q_f,inf)      (transient)
                      + (q_f,inf - q_hat)           (finite fixed-point shift)
                      + (q_hat - q_pi)              (data bias)

holds exactly; taking inf-norms on the right gives only an upper bound on the
left. All norms are inf-norms.

## 8. Tolerances (frozen by the task)

- probabilities and one-step equivalences: absolute tolerance 1e-12;
- repeated traces, linear solves, decompositions and bounds:
  1e-10 * (1 + max_abs(left, right)), with the tolerance added on the
  right-hand side of inequalities;
- raw sampling draws and the discrete trajectory: exact equality across
  routes (no tolerance).
