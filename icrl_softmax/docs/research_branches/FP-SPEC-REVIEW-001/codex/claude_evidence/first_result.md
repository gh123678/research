# FP-SPEC-REVIEW-001 — Claude independent first result

Route: `claude/FP-SPEC-REVIEW-001`, worktree `claude_worktree`, baseline `16ee0f652cde56b4f2ef6e3f1a43b58818bb0108`.
Date: 2026-09-15. Status: first result, sealed before reading any Codex report for this task.
Scope: correspondence audit only. No experiment, no model edit, no task-definition edit.

## 0. Question restated as audited here

Do the two implemented Expected-SARSA classes correspond to (a) an external paper, (b) a declared local
extension, or (c) neither? Sub-questions: H1 an explicit checkable mathematical specification for each
implemented operator; H2 each claimed paper attribution and theorem reference resolves and supports the
attributed claim; H3 every difference in memory, routing, sampling, update timing, logits and control is
explicitly accounted for.

Objects in scope (frozen by the task, §"Frozen inputs"):

- `icrl_softmax/model.py`: `EndToEndMaskedSoftmaxExpectedSARSA` (line 918), `EndToEndFiniteSoftmaxExpectedSARSA`
  (line 1027), `FixedPolicyActionExpectation` (line 888).
- Comparators: `fixed_policy_expected_sarsa.py` (`run_expected_exact` line 194, `run_expected_finite` line 291,
  `finite_successor_all` line 108, `finite_read_all` line 130, `finite_writeback_all` line 144);
  `fixed_policy_expected_sarsa_scaled.py` (imports the contract module byte-identically, lines 1–34, 93).
- External sources: `Xie_2026_beyond_linear_attention.pdf` (arXiv 2605.07333v2) and
  `Liang_Lai_2026_linear_attention_policy_improvement.pdf` (arXiv 2605.05755v1), via the page-indexed
  extracted text in `input/`.
- Local construction text: `论文_草稿/端到端_softmax_SARSA_构造性证明.md`.
- Local specification: `docs/research_tasks/FP-ESARSA-001.md` §"Frozen mathematical contract";
  lineage/scope evidence in `docs/research_tasks/FP-ATTN-001.md`.

## 1. Method

Each source operator was derived from the page-indexed text first (Sections 2–3), then the local declared
specification (Section 4), and only then was code mapped onto those operators (Section 5). No numerical
agreement with the in-repo NumPy reference is used anywhere as evidence of correspondence; numerical
equivalence is an implementation claim (already audited elsewhere), not a provenance claim.

## 2. What the external sources actually define

### 2.1 Xie et al., *Beyond Linear Attention* (arXiv 2605.07333v2)

| item | location | content |
|---|---|---|
| attention convention | p. 3, eq. (2)–(3) | `Z_{l+1} = Z_l + V_l Z_l softmax(Z_lᵀ A_l Z_l + M)`, softmax column-wise; `ẼK_l` the attention matrix |
| the mask | p. 3, eq. (1) | `M[i,j] = 0` for `i ≤ n`, `−∞` for `i = n+1`. A **fixed query-column mask**: the last column does not act as a source. No equality test, no input dependence |
| problem class | p. 4, §2.2 | finite **MRP** `(S, p, r, γ, p₀)` under a fixed policy. **No action space, no state-action pairs**; `v^π : S → ℝ` |
| target algorithm | p. 4, eqs. (8)–(9) and the boxed update | weighted softmax TD: `v_{t+1}(S_j) = v_t(S_j) + α_t Σ_{k=1}^{n} δ_k^{(t)} K(S_{k−1}, S_j)`, `δ_k = R_k + γ v_t(S_k) − v_t(S_{k−1})`, `K(S_{k−1},S_j) = exp(g(S_j,S_{k−1}))/Σ_m exp(g(S_j,S_{m−1}))` |
| construction | p. 5, eqs. (10)–(17) | prompt `Z_0 ∈ ℝ^{(d+3)×(n+1)}` with rows `[X; R; 0; 0]`; dual-head current-value/target-value with `V_l`, `A_l` as in eq. (12); shift matrix `Π`; **Theorem 1** states the forward pass realizes the weighted softmax TD update |
| reparameterization | p. 6, eqs. (18)–(19), Lemma 1 | single-head attention + parameter-free TSM equals the dual-head block |
| contraction premise | p. 7, **Assumption 5.1** | `∃ C_{5.1} ∈ (0, (1−γ)/2)` with `min_{s∈S} M_π(s,s) ≥ (1+γ)/2 + C_{5.1}`, `M_π` the population kernel in eq. (22) |
| convergence | p. 7, **Theorem 2** (eq. 23) | `‖v_L − v^π‖_∞ ≤ (1−C_{5.1})^L ‖v_0−v^π‖_∞ + C√(log(C/δ)/n)` |
| emergence | p. 8, **Theorem 3** (eq. 26–27) | the constructed `(V,A)` is a global minimizer of the NEU loss `J_{L,n}` |
| masks that do exist | p. 24, App. F.2, eqs. (52)–(54) | **parameter** sparsity masks on `V` and `A` (fixed throughout training) |
| diagonal mixing | p. 23, eq. (51) | `ẼK[i,j](ρ_t) = (1−ρ_t)·softmax + ρ_t·1{i=j}`; **`ρ_t ≡ 0` in all reported results** |

Facts that matter for this audit:

- **There is no "Theorem 3.1" in this paper.** Section 3 contains Theorem 1. Theorems are numbered 1, 2, 3.
- **There is no equality mask, and no input-dependent attention mask.** The only attention mask is the fixed
  query-column mask of eq. (1); the only other masks are fixed parameter sparsity patterns (App. F.2).
- **There is no action-value object, no SARSA, and no Expected SARSA.** The paper is state-value policy
  evaluation on an MRP; the successor term carries the successor *state's* own value, with no expectation
  over an action row.
- The softmax weights in Theorem 1 are genuinely finite and are the load-bearing object (the kernel
  `K(S_{k−1}, S_j)` is a smooth function of features `⟨x(S_j), x(S_{k−1})⟩`).

### 2.2 Liang & Lai, *Transformers Provably Implement ICRL with Policy Improvement* (arXiv 2605.05755v1)

| item | location | content |
|---|---|---|
| block | p. 2, eq. (1) | `H_out = H + (1/n)(VH)(HᵀPH)` — **linear** attention, no softmax, therefore no attention mask anywhere |
| target algorithm (primary) | p. 3, eq. (2) | batch **semi-gradient SARSA**: `w_SARSA = w + (α/n) Σ_{i=0}^{n−1} [r_{i+1} + γ wᵀφ(s_{i+1},a_{i+1}) − wᵀφ(s_i,a_i)] φ(s_i,a_i)`. Successor uses the **recorded next action** `a_{i+1}` |
| second algorithm | p. 3, eq. (3) | batch actor-critic with a softmax policy `π_λ(a|s) ∝ exp(λᵀφ_π(s,a))` and policy-gradient score `g_λ` |
| prompt | p. 4, eq. (4) | `H(z) = [[x_0 … x_{n−1} 0…0] ; [0…0 w̃]] ∈ ℝ^{D×(n+1)}`, `x_i = [φ_i; γφ⁺_i; r_{i+1}]`, `w̃ = [1; w]`, `D = 3d+2`. The last column is a **parameter column**, not a masked query |
| **Theorem 3.1** | p. 4, with eq. (5) | "Implement semi-gradient SARSA with a transformer block": for any `α>0` there is `θ⋆ = {(cP⋆, c⁻¹V⋆)}` such that `TF_{θ⋆}(H(z))` equals eq. (2) |
| invariant subblocks | p. 5, **Proposition 3.1**, eq. (8) | gradient-flow invariants `P11, P21, V11, V12` and the first rows of `V21, V22`; `P22`, `V̄22` stay zero |
| optimal manifold | p. 5–6, eqs. (9)–(12), **Theorem 3.2** | scaling manifold `(cP⋆₁₂, c⁻¹V̄⋆₂₁)`; local PL; gradient flow converges locally and exponentially |
| actor-critic theorem | p. 7, **Theorem 3.3**, eq. (14) | same construction for eq. (3) |
| readout identity | p. 14, Lemma C.1, eq. (15) | `TF_θ(H) = w + V̄21 Σ̂ P12 w̃ + (1/n) V̄22 w̃ w̃ᵀ P22 w̃` |
| proof of Thm 3.1 | p. 18–19, App. E, eqs. (32)–(33) | reduce to `V̄21, V̄22, P12, P22`; kill the cubic term with `P22 = V̄22 = 0`; match the affine `Δw_SARSA` |

Facts that matter for this audit:

- **Theorem 3.1 exists here, and it is the natural referent of the code's "Theorem 3.1" attribution.** But it is
  a **linear-attention, sampled-next-action semi-gradient SARSA** statement. It contains **no softmax**, therefore
  **no mask**, and definitely **no equality mask**.
- **"Expected SARSA" does not appear in this paper.** The successor term is the recorded next action.
- "Policy improvement" here means: the block outputs updated parameters `w` (or `λ, w`) that are then used to
  update the policy and resample. It does **not** mean the relative-softmax tilt `π_η⁺ ∝ π·exp(ηQ̂)`.

## 3. What the local construction text defines

`论文_草稿/端到端_softmax_SARSA_构造性证明.md` (exact SARSA, not Expected SARSA):

- §1 states the target operator with the **recorded successor pair** `X'_k`:
  `Q_{l+1}(x) = Q_l(x) + (α/n_x) Σ_{k:X_k=x}[R_k + γQ_l(X'_k) − Q_l(X_k)]` for `n_x>0`, unchanged otherwise.
  It explicitly declares: "等值路由 mask 由外部提供且依赖输入；已访问查询 / null token 的支持规则也由外部 gate
  提供" — **an externally supplied, input-dependent equality routing mask**, and labels the whole object an
  "oracle-routed 精确参考构造" that does *not* claim a generic pretrained Transformer discovers the routing.
- §2–§3 (eqs. 3.1–3.3): literal `d×L` prompt with type / current-ID / next-ID / scalar coordinates; two retrieval
  heads with masks `𝓜^cur`, `𝓜^next`.
- §4 (eq. 3.4): a ReLU FFN with `g = e_r + γe_v − e_u` forms `δ_k^S = R_k + γQ_l(X'_k) − Q_l(X_k)` exactly.
- §5 (eqs. 3.5–3.6): write-back head with mask; `n_x` equal logits → weight `1/n_x` → `(α/n_x)Σδ_k^S`; `n_x=0`
  reads the zero null token `Z` and writes exactly zero.
- §6: distinguishes the frozen-`Q_l` per-pair mean update `α/n_x` from a global `α/N` convention and from
  sequential per-transition updates.
- §7 (eqs. 3.7–3.9): removes the equality masks **but keeps the visited-query gate** — the document says so in
  words ("以下结果是 equality-mask-free but gate-assisted") — and derives
  `p_ζ = e^ζ/(e^ζ+m−1)`, `λ_R = (m−1)/(e^ζ+m−1)`,
  `K_η(k|x) = exp(η·1{X_k=x})/(n_x e^η + N − n_x)`, `ε_W(x) = 2(N−n_x)/(n_x e^η + N − n_x)`, giving
  `|Q^{soft}_{l+1}(x) − Q^{exact}_{l+1}(x)| ≤ α[E_R + B ε_W(x)]`.
- §8: explicitly warns that the literal witness and the compact control experiment "不能互换", and that the
  construction does not include learned routing or environment simulation.

## 4. What the local specification (`FP-ESARSA-001`) additionally declares

The task's frozen contract is where the **Expected** substitution and the **gate-free finite route** are declared:

- "Exact grouped Expected SARSA": `q̄_t^l = Σ_b π(b|S_{t+1}) Q_l(S_{t+1},b)`,
  `δ_t^l = R_{t+1} + γ q̄_t^l − Q_l(S_t,A_t)`,
  `Q_{l+1}(x) = Q_l(x) + (α/N_x^train)Σ_{t<m:X_t=x} δ_t^l` for `N_x^train>0`, unchanged otherwise, **synchronous**.
- Exact heads: "The exact action-expectation head masks to the successor state's unique action tokens and uses
  scores `log π(b|S_{t+1})`. The exact current-pair head admits only the unique current-pair token. The exact
  writer admits only transitions with the queried current pair, or a zero-value null token for an unvisited query."
- "Finite-logit standard-softmax route": `ζ = ξ = τ = 8`; successor score `ζ·1{u=s'} + log π(b|u)`;
  `κ_state = exp(ζ)/(exp(ζ)+|S|−1)` with **conditional action weights equal to `π(·|s')` exactly**; read score
  `ξ·1{x=y}` with `κ_read = exp(ξ)/(exp(ξ)+|S||A|−1)`; write score `τ·1{X_t=x}` normalized over every training
  transition; "**no equality mask and no visited-query gate**".
- Prohibited work: "No input-dependent equality mask or visited gate in `expected_finite`."
- Contraction premise: `min_x M_π^X(x,x) ≥ (1+γ)/2 + C`, `0<C<(1−γ)/2`, on the **induced pair-MRP kernel**
  `M_π^X`, with the conservative bound `‖I − αM_π^X + αγM_π^X P_π^X‖_∞ ≤ 1 − 2αC < 1`.
- `FixedPolicyActionExpectation` is explicitly inside the coverage ("The action-expectation helper is included
  in both-operator coverage", task §"Evidence and state history").

## 5. Code-to-operator mapping (exact line citations)

### 5.1 `EndToEndMaskedSoftmaxExpectedSARSA` (model.py:918–1024)

| operator part | code |
|---|---|
| one flat memory token per `(s,a)` pair | `memory_values = q_values.reshape(-1)` (line 983); `n_pairs = memory_values.numel()` (line 992) |
| exact current-pair retrieval | `_masked_singleton_retrieval(memory_values, current_pairs)` (lines 984–986, helper 960–971): `allowed = pair_queries[:,None] == pair_axis[None,:]` (965), zero scores with `-inf` on the complement (966–969), `softmax(scores, dim=-1)` (970) |
| exact successor policy expectation | `FixedPolicyActionExpectation()(q_values, next_states, policy)` (lines 987–989, helper 895–915) |
| signed residual | `residuals = rewards + self.gamma * qbar - current_q` (line 990) |
| visited / null token | `match` (994), `visited = match.any(dim=0)` (995), `allowed = torch.cat((match, (~visited)[None,:]), dim=0)` (1000) |
| write-back softmax over transitions | zero scores, `-inf` on the complement (1001–1004), `softmax(scores, dim=0)` (1005) |
| null value is zero | `source_values = torch.cat((residuals, torch.zeros(1, …)))` (1006–1008) |
| synchronous per-pair mean, `α/N_x` | `signed_write = (source_values[:,None] * write_attention).sum(dim=0)` (1009); `update = self.alpha * signed_write` (1010); `q_new = memory_values + update` (1011) |
| returned object | updated `Q` plus diagnostics (1013–1025); no certificate, no policy update |

### 5.2 `EndToEndFiniteSoftmaxExpectedSARSA` (model.py:1027–1100)

| operator part | code |
|---|---|
| one token per pair, no null token | `memory_values = q_values.reshape(-1)` (1059), `token_axis = torch.arange(n_pairs, …)` (1062) |
| successor score `log π(b|u) + ζ·1{u=s'}` | `log_policy = torch.log(policy).reshape(-1)` (1065); `successor_block = token_axis // n_actions` (1066); `successor_mask = (successor_block[None,:] == next_states[:,None])` (1067–1069); `successor_scores = log_policy[None,:] + self.zeta * successor_mask` (1070) |
| full-support softmax over all tokens | `action_attention = torch.softmax(successor_scores, dim=1)` (1071); `successor = action_attention @ memory_values` (1072) |
| read score `ξ·1{x=y}` | `read_mask = (token_axis[None,:] == current_pairs[:,None])` (1075); `read_attention = torch.softmax(self.xi * read_mask, dim=1)` (1076); `read = read_attention @ memory_values` (1077) |
| signed residual | `residuals = rewards + self.gamma * successor - read` (1079) |
| write softmax, explicit form | `match` (1082), `counts = match.sum(dim=0)` (1083), `numerator = torch.exp(self.tau * match)` (1084), `denominator = counts * math.exp(self.tau) + (n_transitions - counts)` (1085), `write_attention = numerator / denominator[None,:]` (1086) |
| **no visited gate** | `update = self.alpha * (write_attention.transpose(0,1) @ residuals)` (1087); `q_new = memory_values + update` (1088) — every pair moves, unvisited pairs by leakage |
| sharpness constants | `zeta = xi = tau = 8.0` defaults (1035–1045) |
| trainable parameters | none (`nn.Parameter` absent from both classes; the FP-MODEL-REVIEW-001 audit separately recorded empty `state_dict`) |

### 5.3 `FixedPolicyActionExpectation` (model.py:888–915)

`one_hot.scatter_` on the next state (906–909), `next_policy = policy[next_states]` (910),
`attention = (one_hot[:,:,None] * next_policy[:,None,:]).reshape(…)` (911–913),
`return attention @ memory_values, attention` (915). This is an exact one-hot⊗policy contraction; it is
"attention" by analogy — the weights are `π`, not a softmax output.

### 5.4 Algebraic checks performed by hand (no execution)

1. **Finite successor.** With token scores `log π(b|u) + ζ·1{u=s'}`,
   `Z = Σ_{u,b} π(b|u)e^{ζ·1{u=s'}} = e^ζ Σ_b π(b|s') + Σ_{u≠s'} Σ_b π(b|u) = e^ζ + (|S|−1)`
   provided every policy row sums to 1. Hence `κ_state = e^ζ/(e^ζ+|S|−1)` and the conditional weight on
   `(s',b)` is `π(b|s')` exactly. **Verified.** This is conditional on `Σ_b π(b|u)=1`; the policy-validity
   boundary is a known API gap (see §8).
2. **Finite read.** `finite_read_all` returns `(e^ξ q_x + (total − q_x))/(e^ξ + n − 1)` (fixed_policy_expected_sarsa.py:140),
   which is exactly the softmax of `ξ·1{token=x}` evaluated at the matched token. **Verified**, and it equals the
   code at model.py:1076–1077.
3. **Finite write.** `exp(τ·1{X_t=x}) / (N_x e^τ + (m−N_x))` (model.py:1084–1086) is the softmax of
   `τ·1{X_t=x}` over the `m` transitions; it is algebraically identical to
   `finite_writeback_all` (fixed_policy_expected_sarsa.py:166) and to the construction doc's `K_η(k|x)` (§7).
   **Verified.**
4. **Exact route.** Equality-masked softmax over a single admissible token gives weight 1 (read); over `N_x`
   equal logits gives `1/N_x` (write); the null row contributes exactly 0 for `n_x = 0`. The result is
   `Q_l + (α/N_x)Σ_{t:X_t=x}δ_t` with respect to the frozen `Q_l`. **Verified.**

## 6. Comparison table over the frozen dimensions

Legend: **E** = exact correspondence, **A** = explicit adaptation (declared, traceable), **M** = mismatch,
**U** = unresolved. "Local spec" = FP-ESARSA-001 + the construction doc. "External" = the two papers.

### 6.1 `EndToEndMaskedSoftmaxExpectedSARSA` (exact route)

| # | dimension | implemented | local spec | external | class |
|---|---|---|---|---|---|
| 1 | token contents / candidate multiplicity | one canonical token per `(s,a)` pair (983, 992); transition rows carry only the residual (1006–1008); one zero null token | identical (FP-ESARSA-001 "canonical memory"; doc §2 `M_x/T_k/Z`) | Xie has no pair tokens; Liang-Lai has a single parameter column | **E** (local) / **U** (external) |
| 2 | score, mask, softmax axis, normalization | score 0 with `-inf` **equality mask** over pairs, `softmax(dim=-1)` (965–970); score 0 with `-inf` equality mask + null, `softmax(dim=0)` over transitions (1000–1005) | declared verbatim in FP-ESARSA-001; doc §1 declares the mask is external and input-dependent | **Xie eq. (1) is a fixed query-column mask with no equality test; Liang-Lai has no mask at all** | **E** (local) / **U**→**no resolution** (external) |
| 3 | value projections / signed residual | arithmetic `r + γq̄ − Q` (990); sign preserved | doc §4 realizes the same residual through a ReLU FFN with weights `(1, γ, −1)` | Xie uses `V_l` with row `[1 1 −1]` (eq. 12) for the same signed combination | **E** (operator) / **A** (realization: arithmetic instead of projection/FFN) |
| 4 | simultaneous vs sequential / memory transport | synchronous: update built from the frozen `memory_values`, then added (983, 1009–1011) | FP-ESARSA-001 "All pair updates are synchronous"; doc §6 distinguishes the three conventions | Xie eq. (29)/(31) is likewise synchronous across columns | **E** |
| 5 | null / unvisited handling | unvisited pair admits only the zero null token → exactly zero update (1000, 1007–1011) | identical (FP-ESARSA-001 exact writer; doc §5 `n_x=0` → `Z` → zero) | not present in either paper | **E** (local) / **U** (external) |
| 6 | sampled vs expected successor | **expected**: `q̄ = Σ_b π(b|S_{t+1})Q(S_{t+1},b)` (987–989, 895–915) | the Expected substitution is declared in FP-ESARSA-001; the **construction doc uses the sampled pair `X'_k`** | Xie: successor state's own value, no action row. Liang-Lai eq. (2): recorded `a_{i+1}` | **A** (declared) |
| 7 | policy input vs learned parameters | `policy` is a call argument (973); class has no `nn.Parameter` (929–936) | FP-ESARSA-001: fixed policy probabilities are inputs, not learned | Liang-Lai returns learned `w` (parameters), a different object | **A** (declared; object type differs from both papers) |
| 8 | finite-logit leakage vs exact equality | exact equality; zero leakage by construction | exact route declared exact | n/a | **E** (local) |
| 9 | dependence on dimension / support | only `n_pairs = |S||A|`; no sharpness constant | `|S||A|` is the declared token count | Xie's kernel depends on features, not on an equality pool size | **E** (local) |
| 10 | outer certificate / policy update vs network-internal | class returns an updated `Q` only (1013–1025) | FP-ESARSA-001 keeps certificate (`fixed_policy_expected_sarsa.py:368`) and the relative-softmax decision (`:504`) outside the network | Liang-Lai's block outputs parameters consumed by an outer loop; Xie's block is evaluation-only | **E** (local) |

### 6.2 `EndToEndFiniteSoftmaxExpectedSARSA` (finite route)

| # | dimension | implemented | local spec | external | class |
|---|---|---|---|---|---|
| 1 | token contents / multiplicity | one token per pair (1059, 1062); no null token | FP-ESARSA-001 finite route (no gate, no null) | neither paper | **E** (local) / **A** vs doc §7 (which retains a null/gate) |
| 2 | score, mask, softmax axis, normalization | successor `log π(b|u) + ζ·1{u=s'}` over all tokens, `dim=1` (1070–1071); read `ξ·1{x=y}` over all pairs, `dim=1` (1075–1076); write `τ·1{X_t=x}` over all transitions, explicit ratio (1082–1086). No mask anywhere | successor/read/write formulas are stated in FP-ESARSA-001 verbatim; `κ_state`, `κ_read` derived in §5.4 above | Xie's weights are **feature-similarity** softmax, not equality-indexed; Liang-Lai has no softmax | **E** (local) / **U**→**no resolution** (external) |
| 3 | value projections / signed residual | arithmetic `r + γ·successor − read` (1079); sign preserved | doc §7 keeps signed `δ_k^S`; FP-ESARSA-001 requires signed residuals | Xie's `δ_k` also signed | **E** |
| 4 | simultaneous / memory transport | synchronous (1059, 1087–1088) | FP-ESARSA-001 "synchronous" | same as above | **E** |
| 5 | null / unvisited handling | **none**: unvisited pairs still move by the leakage term (1087–1088) | FP-ESARSA-001 requires this; **doc §7 keeps a visited gate** | neither paper has a visit concept | **A** (declared deviation from doc §7) |
| 6 | sampled vs expected successor | expected, via the finite successor softmax (1072) | FP-ESARSA-001 finite route | Liang-Lai eq. (2) is sampled | **A** |
| 7 | policy input vs learned parameters | `policy` is an argument (1047, 1057); `ζ,ξ,τ` are constructor floats (1035–1045), no `nn.Parameter` | FP-ESARSA-001 fixes `ζ=ξ=τ=8` | Liang-Lai's softmax policy has learned `λ` | **A** |
| 8 | finite-logit leakage vs exact equality | leakage present and reported: `κ_state`, `κ_read`, and the write denominator `N_x e^τ+(m−N_x)`; conditional successor weights are nevertheless **exact** `π(·|s')` | FP-ESARSA-001 declares `κ_state`, `κ_read` and the exactness of the conditional weights; doc §7 gives only an error **bound** `λ_R = (m−1)/(e^ζ+m−1)` with no `log π` bias term | — | **A** (declared refinement: off-group mass `|S|−1` instead of `m−1`, because the `log π` term makes non-successor rows sum to 1 each) |
| 9 | dependence on dimension / support | `κ_state` depends on `|S|`; `κ_read` on `|S||A|`; write on `N_x, m` | declared in FP-ESARSA-001 | — | **E** (local) |
| 10 | outer certificate / policy update vs network-internal | returns updated `Q` only (1090–1100) | outside the class | as above | **E** (local) |

### 6.3 `FixedPolicyActionExpectation` (helper)

| # | dimension | implemented | class |
|---|---|---|---|
| 1–2 | token/score/mask | exact one-hot ⊗ policy weights; **no softmax and no mask** (906–915) | **A** (declared as a head; it is a hard-selection contraction, and its own docstring says "无 mask" while FP-ESARSA-001 says it "masks to the successor state's unique action tokens" — vocabulary only, behaviour agrees) |
| 6 | sampled vs expected successor | this head *is* the Expected operator | **A** relative to both papers |
| 7 | policy input | policy passed in as a tensor (895) | **E** |

## 7. Findings

**F1 — severity: medium — the class-level paper attribution does not resolve.**
`model.py:926` states "布尔 mask 对应论文 Theorem 3.1 的结构化等值 mask" for
`EndToEndMaskedSoftmaxExpectedSARSA`, and `model.py:518` says the same for its SARSA sibling.
Neither paper supports this:

- Xie has **no Theorem 3.1**; its Section 3 contains Theorem 1, and its only attention mask is the fixed
  query-column mask of eq. (1), which contains no equality test and does not depend on the input.
- Liang-Lai **does** have a Theorem 3.1, and it is this codebase's evident lineage for the linear-attention
  operator family and for `Proposition 3.1` inert subblocks (mirrored in `model.py:6–11`, `LinearAttnICRL:18–41`),
  but that theorem is **linear attention, sampled-next-action semi-gradient SARSA**, with **no softmax and
  therefore no mask**.
  ⇒ `H2` fails for this claim: the reference resolves to a theorem that does not contain the attributed object.
  Correct statement: the equality mask is a **local** device, declared as externally supplied and input-dependent
  in the construction doc §1 and frozen in FP-ESARSA-001. Impact: provenance/documentation only — the operator
  itself is unaffected, and the exact route's mask is exactly what FP-ESARSA-001 declares.

**F2 — severity: medium — a second attribution of the same family is also unsupported.**
`model.py:633` says the SARSA sibling "保留论文声明的 visited-query gate". Neither paper declares a
visited-query gate; Xie has no visit/grouping concept at all. This is out of the two-class scope but shares
the same mis-attribution pattern, and the two-class block is where it matters (the correct contrast is that
`expected_finite` **has no gate**, per FP-ESARSA-001, whereas the doc §7 finite bound **is** gate-assisted).

**F3 — severity: low — one genuine, declared divergence from the construction doc's finite analysis.**
Doc §7 derives its finite-logit bound for a **gate-assisted** route without the `log π` bias term; the
implemented finite class has **no gate** (FP-ESARSA-001 explicitly forbids it) and **does** include `log π`.
Both are declared, and the doc's bound remains valid a fortiori because `|S|−1 ≤ |S||A|−1` (its `λ_R` is
looser than the implemented leakage). Consequence to state explicitly: **doc §7's equation (3.8) lemma does not
describe the implemented finite operator** — it bounds a related but different object. Not a scientific
mismatch; a citation-accuracy issue between two local documents.

**F4 — severity: low — "Softmax" is nominal for the exact route.**
In `EndToEndMaskedSoftmaxExpectedSARSA` every softmax is degenerate: the retrieval softmax has exactly one
admissible token (weight 1) and the write-back softmax has `N_x` equal logits (uniform `1/N_x`). The loading
is done entirely by the `−inf` equality mask. The finite route's softmax *is* finite, full-support and
load-bearing. This matters because the one dimension on which a paper correspondence could honestly be claimed
— genuinely finite softmax weights realizing an RL update — is exactly the dimension on which the exact route
does not correspond (Xie Thm 1). Naming must therefore distinguish the two routes.

**F5 — severity: none (confirmation) — the contraction premise corresponds exactly, and is declaredly lifted.**
FP-ESARSA-001's `min_x M_π^X(x,x) ≥ (1+γ)/2 + C`, `0<C<(1−γ)/2`, reproduces Xie **Assumption 5.1** verbatim
(p. 7), but (i) on the **induced pair-MRP kernel** `M_π^X` rather than Xie's state kernel `M_π`, and (ii) with a
declared conservative constant `2C` in `‖I − αM_π^X + αγM_π^X P_π^X‖_∞ ≤ 1 − 2αC`. Both the lift and the
conservatism are declared. The theorem remains conditional on the premise, as in Xie.

**F6 — severity: none (confirmation) — `H1` holds for both classes.**
An explicit, checkable specification exists: FP-ESARSA-001 §"Frozen mathematical contract" states both
operators in closed form; `fixed_policy_expected_sarsa.py` states the same formulas as executable pure
functions (`finite_successor_all:108`, `finite_read_all:130`, `finite_writeback_all:144`,
`run_expected_exact:194`, `run_expected_finite:291`); and
`verify_fixed_policy_expected_sarsa.py:44–48` imports exactly the three classes and asserts formula-level
equalities (its header claims 1e-12 on deterministic fixtures, including "no equality mask or visited gate"
for the finite route). Additionally, the FP-MODEL-REVIEW-001 audit recorded independently reconstructed
agreement on 60 fixtures. **This is agreement between the code and a local pure-NumPy reference, i.e. an
implementation claim; by the task's rule it is not used here as correspondence evidence.**

**F7 — severity: none (scope note) — the local doc §8 warns against exactly the substitution this audit avoids.**
Doc §8 states that the literal witness and the compact control experiment "不能互换". The two audited classes
are compact operator realizations (no `d×L` prompt, no type/ID coordinate block, no `W_Q/W_K/W_V/W_O`), not the
literal block of doc §2–§5. Therefore no claim that the model.py classes are the doc's literal construction is
made or implied here.

## 8. Answer: what network name and paper attribution are justified

- **Correspondence verdict.** Each implemented operator corresponds to a **declared local extension**, not to
  an external paper. Precisely: the *operator* (synchronous, per-pair grouped, `α/N_x`-averaged temporal-difference
  update on a complete one-token-per-pair Q-memory with equality-routed read/write and a zero null token) is
  declared in the local construction doc §1–§6 for the **sampled** successor; the **Expected** successor,
  the **finite-logit** successor score with the `log π` bias, and the **removal of the visited gate** are
  declared in FP-ESARSA-001. Neither paper contains the expected-successor operator, an equality mask, a
  pair-indexed memory, or a gate.
- **Justified lineage attributions (positive).**
  1. Liang-Lai **Theorem 3.1** and **Proposition 3.1** are the correct sources for the *linear-attention,
     sampled-next-action* SARSA operator family and the inert-subblock structure that this repository's older
     classes implement (`LinearAttnICRL:18–41`, `SoftmaxSARSA:109–147`, `LiangLaiOperatorBaseline:862–885`,
     whose docstring correctly disclaims reproducing the teacher-mimicking training dynamics).
  2. Xie **Assumption 5.1** is the correct source of the diagonal-margin contraction premise used by
     FP-ESARSA-001 (lifted from `M_π` to `M_π^X`).
  3. Xie **Theorem 1** is the correct source of the general *softmax-attention-executes-a-batch-TD-update*
     template. It is the nearest external neighbour of the finite route but is **not** its specification: Xie's
     weights are feature-similarity softmax, not equality-indexed finite logits.
- **Attributions that are not justified.** Any statement of the form "the equality mask comes from the paper's
  Theorem 3.1" (`model.py:518`, `model.py:926`), and "the paper declares a visited-query gate" (`model.py:633`).
  These should be re-worded to cite FP-ESARSA-001 and the local construction doc.
- **Permitted final terminology for the two classes.** Use, e.g.: "a fixed-weight, parameter-free softmax-attention
  operator that executes **one** synchronous grouped **fixed-policy Expected-SARSA** update on a complete
  Q-memory, once per layer, with an equality-routed exact variant and a finite-sharpness full-support variant;
  the routing mask, the Expected successor, and the gate-free finite route are **local declarations**
  (FP-ESARSA-001; construction doc §1–§7), not paper results." Do **not** say, without qualification:
  "the paper's Theorem 3.1 network"; "a pretrained/learned Transformer"; "the literal prompt construction of
  the construction doc"; "the network certifies its own error" (certificate and policy update are external);
  "Expected SARSA from Xie/Liang-Lai".
- **Caveats that must accompany the name.**
  - `EndToEnd` here means *from raw transition fields to the updated Q*; the modules have **zero trainable
    parameters**, so "end-to-end" must not be read as "trained end-to-end".
  - `Softmax` carries no expressive load in the exact route (F4).
  - The `κ_state`/`κ_read` exactness algebra presupposes row-normalized, strictly positive `policy`. The
    FP-MODEL-REVIEW-001 audit recorded that invalid policies are currently accepted at the API boundary
    (un-normalized / zero / negative entries) and that `math.exp(τ)` overflows for `τ = 1000`. Neither is
    triggered by the frozen configuration, but the exactness claim is conditional on that precondition.

## 9. Independence limitations (recorded, not claimed away)

1. **Same executor, prior exposure.** This route executed FP-ESARSA-001's main route (which wrote
   `fixed_policy_expected_sarsa.py` and the model.py additions) and FP-ATTN-001, and authored the
   FP-MODEL-REVIEW-001 audit. The specification under review here is therefore largely **self-authored**.
   This is *not* pristine blind discovery, and the audit cannot be blind to it. The correspondence question was
   nonetheless answered from the papers' text first and the local spec second; the external-paper findings
   (no Theorem 3.1 in Xie; no mask in Liang-Lai Thm 3.1) are checkable against the frozen extracted text by any
   reader, independently of this route's history.
2. **No Codex isolation guarantee.** I did not read any Codex result or conclusion for this task. I did read
   `ACTIVE_WORKSPACE.md`, which contains entries for unrelated tasks, including a GPT-authored review of
   FP-COMPOSE-002/003. Nothing there concerns the two-class correspondence question.
3. **Tooling failure in this session.** The shell was unavailable (every invocation failed with
   `EPERM: operation not permitted, mkdir 'C:\Users\Admin\.claude\session-env\…'`), and the grep/glob tools and
   all MCP tools were unavailable. Consequences: (a) I could not compute local file hashes; (b) I could not
   run targeted text searches, so both papers were read **sequentially in full** rather than query-filtered —
   this makes the "no equality mask / no Theorem 3.1 / no Expected SARSA" statements rest on complete reads of
   the frozen extracted text, but I could not produce a machine-checked absence proof.
4. **No execution.** No experiment, no formula check by execution, no construction re-derivation beyond the
   hand algebra in §5.4. This audit is a reading-and-mapping audit.
5. **Coverage.** Only the classes named in the task are assessed. The SARSA siblings
   (`EndToEndMaskedSoftmaxSARSA:510`, `EndToEndFiniteSoftmaxSARSA:628`) and the other model.py classes are
   touched only where they carry the same attribution text. Neither paper's theorems were re-proved.

## 10. Acceptance-criteria accounting

| requirement (task §"Deliverables and acceptance") | status |
|---|---|
| both classes covered across all listed dimensions | ✅ §6.1, §6.2 (10/10 rows each), plus §6.3 for the helper |
| original external paper claims separated from local propositions | ✅ §2 (papers) vs §3–§4 (local), with the split named per row |
| each row classified exact / explicit adaptation / mismatch / unresolved | ✅ §6 legend and per-row class |
| exact code-line citations | ✅ §5, §6, §7 |
| paper page/equation citations | ✅ §2 tables |
| findings with severity | ✅ §7 (F1 medium, F2 medium, F3 low, F4 low, F5–F7 none/confirmation) |
| permitted final terminology | ✅ §8 |
| supported answer on network name and paper attribution | ✅ §8 |
| no inference of correspondence from numerical agreement with the in-repo reference | ✅ stated in §1, §7-F6 |
| missing evidence kept unresolved | ✅ §6 (*external* mask rows marked unresolved on both classes), §9 |
| independence limitations recorded honestly | ✅ §9 |
| no experiment / no model edit / no task-definition edit / no push / no merge | ✅ complied |
| no reciprocal review yet | ✅ none performed |

**Route verdict for this first result:** the two-class correspondence question is answered with
`H1 = satisfied`, `H2 = fails for the two "论文 Theorem 3.1 / 论文声明的 gate" attribution statements`,
`H3 = satisfied` (all deviations enumerated and traced to FP-ESARSA-001 or the construction doc §7).
No task-level `OBJECTION`: the task explicitly permits "a mismatch, justified adaptation, or unresolved source"
as an acceptable result, and the defects found are documentation-level attribution errors inside the audited
code, not defects of the task definition.
