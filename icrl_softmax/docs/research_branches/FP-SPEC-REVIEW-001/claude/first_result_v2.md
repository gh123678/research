# FP-SPEC-REVIEW-001 — Claude independent first result, v2 (repaired)

Route: `claude/FP-SPEC-REVIEW-001`, worktree `claude_worktree`, baseline `16ee0f652cde56b4f2ef6e3f1a43b58818bb0108`.
Date: 2026-09-15. Status: **author repair of `first_result.md`** (v1, SHA256 `72e3d57046097f3507cccd36bf8b39cca51396d3788288e86b285ce575d159a9`) in response to GPT `verification.md` (`FAIL`, repairable report errors, no task objection).
Scope: correspondence audit only. No experiment, no model edit, no task-definition edit, no push, no merge.
The v1 file is **preserved unchanged**; this v2 supersedes it. Independence is complete (GPT verification §"Integrity and scope"), so v2 is permitted to read the Codex first result and the GPT verification.

## 0. Repair summary (what changed from v1)

| # | GPT item | v1 error | v2 disposition |
|---|---|---|---|
| R1 | P1 | F3 claimed the manuscript's finite-logit bound transfers **a fortiori** to the implemented Expected, gate-free class | **Withdrawn.** The old bound's exact target is *sampled* and *gate-assisted*; the implemented target is *Expected* and gate-free, so the objects differ. §8-F3 now derives a replacement statement with a common Expected target and explicit premises, and adds the separate unvisited term. |
| R2 | P2 | v1 stated Xie's weights are "feature-similarity softmax, **not** equality-indexed" as a difference | **Corrected.** With one-hot features `φ(x)=√τ e_x`, `⟨φ(x),φ(y)⟩=τ·1{x=y}`; the equality-indexed logit is a **special case** of Xie's score form. The real differences are memory/routing, successor/current retrieval roles, and the `log π` bias. |
| R3 | P2 | v1-F2 said Xie has "no visit/grouping concept at all" | **Corrected.** Xie assumes full coverage (p. 6) and groups weights by predecessor state (eq. 20). The defensible claim is absence of the *particular visited-query gate*. |
| R4 | P2 | v1-F1 resolved "论文 Theorem 3.1" to Liang–Lai and declared `H2` failed on that basis | **Corrected to the narrower finding.** The bare citation has an **identifiable LOCAL referent** (`preliminaries.md` Theorem 3.1, stated §3.1, proved §3.2–3.3; construction doc §5). The unsupported reading is the *external* one; the defect is the unnamed bare citation. |
| R5 | P2 | v1 grouped `SoftmaxSARSA` into "older classes implement Liang–Lai Theorem 3.1" | **Removed.** `SoftmaxSARSA` is a trainable softmax hybrid; only `LinearAttnICRL` and `LiangLaiOperatorBaseline` are Liang-lineage. |
| R6 | P2 | v1 said "every pair moves" and called the exact route's softmax "nominal" / "carries no expressive load" | **Corrected.** Every pair is *eligible* for a potentially nonzero update; the exact route's weights are degenerate (singleton/uniform) but it is still a softmax attention operation. |
| R7 | clarification | helper "no mask" phrasing blurred code representation and effective support | **Corrected** in §6.3: source selection is present; only the explicit `−inf` mask tensor is absent. |
| R8 | (user instruction) | v1 omitted explicit statements of the global `α/N` vs local `α/n_x` normalization difference, the memory/TSM differences, and the exact-residual vs full finite-leakage fixed-point caveat | **Added** as §7.1/§7.2 dimensions 11–12, §3.3, and §8-F5. |

## 1. Question restated as audited here

Do the two implemented Expected-SARSA classes correspond to (a) an external paper, (b) a declared local extension, or (c) neither? Sub-questions: H1 an explicit checkable mathematical specification for each implemented operator; H2 each claimed paper attribution and theorem reference resolves and supports the attributed claim; H3 every difference in memory, routing, sampling, update timing, logits and control is explicitly accounted for.

Objects in scope (frozen by the task, §"Frozen inputs"):

- `icrl_softmax/model.py`: `EndToEndMaskedSoftmaxExpectedSARSA` (line 918), `EndToEndFiniteSoftmaxExpectedSARSA` (line 1027), `FixedPolicyActionExpectation` (line 888).
- Comparators: `fixed_policy_expected_sarsa.py` (`run_expected_exact` line 194, `run_expected_finite` line 291, `finite_successor_all` line 108, `finite_read_all` line 130, `finite_writeback_all` line 144); `fixed_policy_expected_sarsa_scaled.py` (imports the contract module byte-identically, lines 1–34, 93).
- External sources: `Xie_2026_beyond_linear_attention.pdf` (arXiv 2605.07333v2) and `Liang_Lai_2026_linear_attention_policy_improvement.pdf` (arXiv 2605.05755v1), via the page-indexed extracted text in `input/`.
- Local construction text: `论文_草稿/端到端_softmax_SARSA_构造性证明.md` **and** `论文_草稿/preliminaries.md` (the latter is the authoritative numbered version of the same construction; it contains the local Theorem 3.1 — see §3.3).
- Local specification: `docs/research_tasks/FP-ESARSA-001.md` §"Frozen mathematical contract"; lineage/scope evidence in `docs/research_tasks/FP-ATTN-001.md`.

## 2. Method

Each source operator was derived from the page-indexed text first (§3), then the local declared specification (§4–§5), and only then was code mapped onto those operators (§6). No numerical agreement with the in-repo NumPy reference is used anywhere as evidence of correspondence; numerical equivalence is an implementation claim (already audited elsewhere), not a provenance claim.

**Repair note.** v2 adds a primary-source check that v1 missed: the local construction documents are not only the construction doc but also `preliminaries.md`, whose **Theorem 3.1** is the material referent of the docstring's "论文 Theorem 3.1". v2 also re-read Xie pp. 3–7 for the score form, the coverage assumption and eq. (20).

## 3. What the sources actually define (external papers; and the local referent)

### 3.1 Xie et al., *Beyond Linear Attention* (arXiv 2605.07333v2)

| item | location | content |
|---|---|---|
| attention convention | p. 3, eq. (2)–(3) | `Z_{l+1} = Z_l + V_l Z_l softmax(Z_lᵀ A_l Z_l + M)`, softmax column-wise; `ẼK_l` the attention matrix |
| the mask | p. 3, eq. (1) | `M[i,j] = 0` for `i ≤ n`, `−∞` for `i = n+1`. A **fixed positional mask**: the query token (column `n+1` of `Z`, i.e. state `S_n`) is excluded as a source. **No equality test, no input dependence** |
| problem class | p. 4, §2.2 | finite **MRP** `(S, p, r, γ, p₀)` under a fixed policy. **No action space, no state-action pairs**; `v^π : S → ℝ` |
| target algorithm | p. 4, eqs. (8)–(9) and the boxed update | weighted softmax TD: `v_{t+1}(S_j) = v_t(S_j) + α_t Σ_{k=1}^{n} δ_k^{(t)} K(S_{k−1}, S_j)`, `δ_k = R_k + γ v_t(S_k) − v_t(S_{k−1})`, `K(S_{k−1},S_j) = exp(g(S_j,S_{k−1}))/Σ_m exp(g(S_j,S_{m−1}))` |
| **score form** | p. 5, eq. (9) and the sentence after; eq. (12) | `g` is a general score function; the construction realizes `g(S_j,S_{k−1}) = ⟨x(S_j), x(S_{k−1})⟩`, i.e. a **feature inner product**. This is the family to which the writer kernel belongs (see R2) |
| construction | p. 5, eqs. (10)–(17) | prompt `Z_0 ∈ ℝ^{(d+3)×(n+1)}` with rows `[X; R; 0; 0]`; dual-head current-value/target-value with `V_l`, `A_l` as in eq. (12); shift matrix `Π`; **Theorem 1** states the forward pass realizes the weighted softmax TD update |
| reparameterization | p. 6, eqs. (18)–(19), Lemma 1 | single-head attention + **parameter-free TSM** (`U = I − e_{d+2}e_{d+2}ᵀ`, `W = γ e_{d+2}e_{d+3}ᵀ`, predecessor shift `Π`) equals the dual-head block. The last two prompt rows are the memory that is **rewritten and reused across layers** |
| coverage | p. 6, §5 opening | "we assume the trajectory τ_n = (S_0, R_1, …, S_n) visits every state, i.e. `{S_0,…,S_{n−1}} = S`" |
| predecessor grouping | p. 6, eq. (20) | `M̂_n(s,s') = Σ_k K(S_{k−1},s)·1{S_{k−1}=s'}` and `P̂_n(s,s') = Σ_k K(S_{k−1},s)·1{S_k=s'}`; the text says `M̂_n` "aggregates the softmax weights according to the predecessor state `S_{k−1}`" |
| contraction premise | p. 7, **Assumption 5.1** | `∃ C_{5.1} ∈ (0, (1−γ)/2)` with `min_{s∈S} M_π(s,s) ≥ (1+γ)/2 + C_{5.1}`, `M_π` the population kernel in eq. (22) |
| convergence | p. 7, **Theorem 2** (eq. 23) | `‖v_L − v^π‖_∞ ≤ (1−C_{5.1})^L ‖v_0−v^π‖_∞ + C√(log(C/δ)/n)` |
| emergence | p. 8, **Theorem 3** (eq. 26–27) | the constructed `(V,A)` is a global minimizer of the NEU loss `J_{L,n}` |
| parameter masks that do exist | p. 24, App. F.2, eqs. (52)–(54) | **parameter** sparsity masks on `V` and `A` (fixed throughout training) |
| diagonal mixing | p. 23, eq. (51) | `ẼK[i,j](ρ_t) = (1−ρ_t)·softmax + ρ_t·1{i=j}`; **`ρ_t ≡ 0` in all reported results** |

Facts that matter for this audit (corrected from v1):

- **There is no "Theorem 3.1" in this paper.** Section 3 contains Theorem 1. Theorems are numbered 1, 2, 3. (Unchanged; still correct.)
- **There is no equality mask and no input-dependent attention mask.** The only attention mask is the fixed positional mask of eq. (1). **Corrected:** the paper *does* have visit and grouping concepts (full-coverage assumption, p. 6; predecessor-state grouping, eq. (20)). What it does not have is a **visited-query gate** that zeroes unvisited queries.
- **The score form is shared with the implemented writer** (R2). With one-hot features `φ(x)=√τ e_x`, `⟨φ(x),φ(y)⟩ = τ·1{x=y}`. Hence a finite softmax over equality-indexed logits is *inside* Xie's score family, and the presence of such weights cannot by itself distinguish the implemented operator from Xie. The distinguishing facts are elsewhere (this section, §6).
- **There is no action-value object, no SARSA, and no Expected SARSA.** The paper is state-value policy evaluation on an MRP; the successor term carries the successor *state's* own value, with no expectation over an action row, and the query is a state, not a pair.

### 3.2 Liang & Lai, *Transformers Provably Implement ICRL with Policy Improvement* (arXiv 2605.05755v1)

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

- **`Theorem 3.1` exists here**, and it is *the same number* as the local document's theorem (§3.3). It is a **linear-attention, sampled-next-action semi-gradient SARSA** statement. It contains **no softmax**, therefore **no mask**, and definitely no equality mask. **Corrected from v1:** because a same-numbered external theorem exists, the *bare* docstring citation is genuinely ambiguous to an external reader — but that is an ambiguity defect, not evidence that this theorem is the intended referent. See §3.3.
- **"Expected SARSA" does not appear in this paper.** The successor term is the recorded next action.
- **Update coefficient.** Eq. (2) normalizes by the trajectory length `n`, so for a logical pair `x` the batch update is `(α/n)·Σ_{i:x_i=x} δ_i` — a **scaled sum**, not a mean. The implemented exact route uses `(α/n_x)·Σ_{t:x_t=x} δ_t` — a **mean over visits to that pair**. These coincide only when `n_x ≡ n/|X|` (or under an explicit per-pair learning-rate transformation). Equivalently, the implemented per-pair update is Liang's per-pair update multiplied by `n/n_x`. This is an explicit normalization difference and is listed as dimension 11 in §7 (also flagged by the Codex report §3.2).
- "Policy improvement" here means: the block outputs updated parameters `w` (or `λ, w`) that are then used to update the policy and resample. It does **not** mean the relative-softmax tilt `π_η⁺ ∝ π·exp(ηQ̂)`. Algorithm 1, p. 5, contains an external behaviour/control loop.

### 3.3 The local referent of "论文 Theorem 3.1" (new in v2)

`论文_草稿/preliminaries.md` is the authoritative numbered version of the construction (the construction doc `端到端_softmax_SARSA_构造性证明.md` is its independent reading edition). It contains, verbatim:

- §3.1: **"Theorem 3.1 (exact sampled-SARSA under structured equality routing)"** (line 51), with the update (3.6), explicitly conditional on "the supplied equality-routing masks and visited/null support rule";
- §3.2: "Proof of Theorem 3.1: retrieval and residual formation";
- §3.3: "Proof of Theorem 3.1: write-back and assembly", closing "This is exactly the theorem statement.";
- §3.3 also states: "Section 3.8 removes the equality masks at finite logits but still retains the visited-query gate."

The construction doc §5 closes with the same statement: "这就逐矩阵证明了定理 3.1" (line 182). Both documents declare the equality routing mask to be externally supplied and input-dependent (construction doc §1, line 26; `preliminaries.md` §3.1, line 13).

Therefore the docstring's "布尔 mask 对应论文 Theorem 3.1 的结构化等值 mask" (`model.py:926`, and the same sentence at `model.py:518`) has a **determinate local referent**: the local construction's Theorem 3.1, whose content *is* the structured equality mask. Two facts make this the only consistent reading:

1. The local Theorem 3.1 is about a **structured equality mask**; Liang–Lai's Theorem 3.1 has no softmax and hence no mask at all.
2. `model.py:633` says the finite SARSA sibling "保留论文声明的 visited-query gate". The local `preliminaries.md` §3.8 declares exactly that gate ("retaining the external visited-query gate … equality-mask-free but gate-assisted"); neither external paper declares any visited-query gate.

**Defect, narrowed.** The defect is not "the referent does not exist" or "the author intended Liang–Lai and got its theorem wrong". The defect is that the docstring cites a bare "论文" without naming the local document, while a same-numbered external theorem exists — so the citation is **inadequate/ambiguous as written** and should name the local source. `H2` is restated accordingly in §9.

**Also recorded (new in v2), to close the a-fortiori question honestly:** `preliminaries.md` §3.5 (Proposition 3.3, "Expected-SARSA identity") *does* contain an Expected-SARSA operator — but for a **Boltzmann policy `p_β(·|s;Q_l)` tied to the current Q-values**, with `A' ~ p_β`, not for a frozen external policy `π`. §3.7 integrates the action-sharpness term `γ log|A|/β` alongside `E_R`, `B ε_W` and `ε_D`. The implemented class instead uses the **frozen external `π`** (an argument, `model.py:1047,1057`, `log π` at `model.py:1065,1070`). So even the local document's own Expected-SARSA material is a **different expected operator** from the implemented one, and §3.8's bound (3.9) still has a **sampled** exact target. This is why R1's a-fortiori transfer fails for a second, independent reason.

## 4. What the local construction text defines (sampled SARSA)

`论文_草稿/端到端_softmax_SARSA_构造性证明.md` (exact SARSA, not Expected SARSA), congruent with `preliminaries.md`:

- §1 states the target operator with the **recorded successor pair** `X'_k`:
  `Q_{l+1}(x) = Q_l(x) + (α/n_x) Σ_{k:X_k=x}[R_k + γQ_l(X'_k) − Q_l(X_k)]` for `n_x>0`, unchanged otherwise.
  It explicitly declares an externally supplied, input-dependent equality routing mask and an external visited/null support gate, and labels the object an "oracle-routed 精确参考构造" that does *not* claim a generic pretrained Transformer discovers the routing.
- §2–§3 (eqs. 3.1–3.3): literal `d×L` prompt with type / current-ID / next-ID / scalar coordinates; two retrieval heads with masks `𝓜^cur`, `𝓜^next`.
- §4 (eq. 3.4): a ReLU FFN with `g = e_r + γe_v − e_u` forms `δ_k^S = R_k + γQ_l(X'_k) − Q_l(X_k)` exactly.
- §5 (eqs. 3.5–3.6): write-back head with mask; `n_x` equal logits → weight `1/n_x`; `n_x=0` reads the zero null token `Z` and writes exactly zero.
- §6: explicitly distinguishes the frozen-`Q_l` per-pair mean update `α/n_x` from a global `α/N` convention and from sequential per-transition updates. (This is the local statement of the normalization difference; the same distinction is drawn in `preliminaries.md` §3.3.)
- §7 (eqs. 3.7–3.9): removes the equality masks **but keeps the visited-query gate** ("以下结果是 equality-mask-free but gate-assisted"), and derives
  `p_ζ = e^ζ/(e^ζ+m−1)`, `λ_R = (m−1)/(e^ζ+m−1)`,
  `K_η(k|x) = exp(η·1{X_k=x})/(n_x e^η + N − n_x)`, `ε_W(x) = 2(N−n_x)/(n_x e^η + N − n_x)`, giving
  `|Q^{soft}_{l+1}(x) − Q^{exact}_{l+1}(x)| ≤ α[E_R + B ε_W(x)]` **for a visited query**.
  **Corrected reading (R1):** this `exact` is the document's own **sampled, gate-assisted** exact operator. Its writer formula `K_η` is *identical* to the implemented finite writer on **visited** queries (see §7.2 dimension 2 and §8-F3); what does not transfer is the bound's *target* and its *coverage*.
- §8: explicitly warns that the literal witness and the compact control experiment "不能互换", and that the construction does not include learned routing or environment simulation.

## 5. What the local specification (`FP-ESARSA-001`) additionally declares

The task's frozen contract is where the **Expected** substitution and the **gate-free finite route** are declared:

- "Exact grouped Expected SARSA": `q̄_t^l = Σ_b π(b|S_{t+1}) Q_l(S_{t+1},b)`,
  `δ_t^l = R_{t+1} + γ q̄_t^l − Q_l(S_t,A_t)`,
  `Q_{l+1}(x) = Q_l(x) + (α/N_x^train)Σ_{t<m:X_t=x} δ_t^l` for `N_x^train>0`, unchanged otherwise, **synchronous**.
- Exact heads: "The exact action-expectation head masks to the successor state's unique action tokens and uses scores `log π(b|S_{t+1})`. The exact current-pair head admits only the unique current-pair token. The exact writer admits only transitions with the queried current pair, or a zero-value null token for an unvisited query."
- "Finite-logit standard-softmax route": `ζ = ξ = τ = 8`; successor score `ζ·1{u=s'} + log π(b|u)`; `κ_state = exp(ζ)/(exp(ζ)+|S|−1)` with **conditional action weights equal to `π(·|s')` exactly**; read score `ξ·1{x=y}` with `κ_read = exp(ξ)/(exp(ξ)+|S||A|−1)`; write score `τ·1{X_t=x}` normalized over every training transition; "**no equality mask and no visited-query gate**".
- Prohibited work: "No input-dependent equality mask or visited gate in `expected_finite`."
- **Exact-residual vs full finite-leakage fixed point (added explicitly):** §"Exact-residual kernel population operator" defines `F_π(Q) = Q + α M_π^X (T_π^X Q − Q)` **for the exact fixed-policy Bellman residual** and requires `F_π(Q^π) = Q^π`. The same section states, in terms: *"This identity does not cancel successor-head or current-read leakage. The full finite-logit route can have a shifted population fixed point because its residual need not vanish at `Q^π`."* So the `Q^π`-preservation identity belongs to the **exact-residual population operator only**; no asymptotic no-bias claim is made for the full finite route, whose estimate is instead certified by the held-out Bellman residual. This caveat is restated as F5-extended in §8.
- Contraction premise: `min_x M_π^X(x,x) ≥ (1+γ)/2 + C`, `0<C<(1−γ)/2`, on the **induced pair-MRP kernel** `M_π^X`, with the conservative bound `‖I − αM_π^X + αγM_π^X P_π^X‖_∞ ≤ 1 − 2αC < 1`.
- `FixedPolicyActionExpectation` is explicitly inside the coverage ("The action-expectation helper is included in both-operator coverage").

## 6. Code-to-operator mapping (exact line citations)

### 6.1 `EndToEndMaskedSoftmaxExpectedSARSA` (model.py:918–1024)

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

### 6.2 `EndToEndFiniteSoftmaxExpectedSARSA` (model.py:1027–1100)

| operator part | code |
|---|---|
| one token per pair, no null token | `memory_values = q_values.reshape(-1)` (1059), `token_axis = torch.arange(n_pairs, …)` (1062) |
| successor score `log π(b|u) + ζ·1{u=s'}` | `log_policy = torch.log(policy).reshape(-1)` (1065); `successor_block = token_axis // n_actions` (1066); `successor_mask = (successor_block[None,:] == next_states[:,None])` (1067–1069); `successor_scores = log_policy[None,:] + self.zeta * successor_mask` (1070) |
| full-support softmax over all tokens | `action_attention = torch.softmax(successor_scores, dim=1)` (1071); `successor = action_attention @ memory_values` (1072) |
| read score `ξ·1{x=y}` | `read_mask = (token_axis[None,:] == current_pairs[:,None])` (1075); `read_attention = torch.softmax(self.xi * read_mask, dim=1)` (1076); `read = read_attention @ memory_values` (1077) |
| signed residual | `residuals = rewards + self.gamma * successor - read` (1079) |
| write softmax, explicit form | `match` (1082), `counts = match.sum(dim=0)` (1083), `numerator = torch.exp(self.tau * match)` (1084), `denominator = counts * math.exp(self.tau) + (n_transitions - counts)` (1085), `write_attention = numerator / denominator[None,:]` (1086) — algebraically `exp(τ·1{X_t=x})/(n_x e^τ + N − n_x)`, i.e. the local `K_η(k|x)` of §4 on visited queries |
| **no visited gate** | `update = self.alpha * (write_attention.transpose(0,1) @ residuals)` (1087); `q_new = memory_values + update` (1088). **Corrected from v1:** every pair is **eligible for a potentially nonzero update**; the update is exactly zero whenever the weighted residuals cancel, and for an unvisited pair `n_x=0` it is `α·mean_t δ_t` (leakage) |
| sharpness constants | `zeta = xi = tau = 8.0` defaults (1035–1045) |
| trainable parameters | none (`nn.Parameter` absent from both classes; the FP-MODEL-REVIEW-001 audit separately recorded empty `state_dict`) |

### 6.3 `FixedPolicyActionExpectation` (model.py:888–915)

`one_hot.scatter_` on the next state (906–909), `next_policy = policy[next_states]` (910),
`attention = (one_hot[:,:,None] * next_policy[:,None,:]).reshape(…)` (911–913),
`return attention @ memory_values, attention` (915). This is an exact one-hot⊗policy contraction whose **effective support** is exactly the declared masked-log-softmax support (`1{u=s'}π(b|s')`, exactly zero off the selected state). **Corrected from v1 (R7):** the helper *does* perform source selection; what it does not do is compute an explicit `−inf` mask tensor. Its own docstring "该头无 mask、无可训练参数" (`model.py:892`) describes the **code representation** (direct sparse weights), not the absence of **effective support**; FP-ESARSA-001 describes the same object by its effective support. The vocabulary differs; the operator agrees (on a row-normalized positive policy).

### 6.4 Algebraic checks performed by hand (no execution)

1. **Finite successor.** With token scores `log π(b|u) + ζ·1{u=s'}`,
   `Z = Σ_{u,b} π(b|u)e^{ζ·1{u=s'}} = e^ζ Σ_b π(b|s') + Σ_{u≠s'} Σ_b π(b|u) = e^ζ + (|S|−1)`
   provided every policy row sums to 1. Hence `κ_state = e^ζ/(e^ζ+|S|−1)` and the conditional weight on
   `(s',b)` is `π(b|s')` exactly. **Verified.** Conditional on `Σ_b π(b|u)=1`; the policy-validity boundary is a known API gap (§10).
2. **Finite read.** `finite_read_all` returns `(e^ξ q_x + (total − q_x))/(e^ξ + n − 1)` (fixed_policy_expected_sarsa.py:140), exactly the softmax of `ξ·1{token=x}` at the matched token. **Verified**, and equal to model.py:1076–1077.
3. **Finite write.** `exp(τ·1{X_t=x})/(N_x e^τ + (m−N_x))` (model.py:1084–1086) is the softmax of `τ·1{X_t=x}` over the `m` transitions; algebraically identical to `finite_writeback_all` and to the construction doc's `K_η(k|x)` §7. **Verified.**
4. **Exact route.** Equality-masked softmax over one admissible token gives weight 1 (read); over `N_x` equal logits gives `1/N_x` (write); the null row contributes exactly 0 for `n_x = 0`. Result `Q_l + (α/N_x)Σ_{t:X_t=x}δ_t` at frozen `Q_l`. **Verified.**
5. **Equality logits are one-hot feature inner products (new, R2).** With `φ(x)=√τ e_x`, `⟨φ(x),φ(y)⟩ = τ·1{x=y}`; likewise for the `ξ` and `ζ` blocks. Hence the implemented finite scores lie in Xie's score family. **Verified** (this is the statement that replaces v1's false contrast).

## 7. Comparison table over the frozen dimensions

Legend: **E** = exact correspondence, **A** = explicit adaptation (declared, traceable), **M** = mismatch, **U** = unresolved. "Local spec" = FP-ESARSA-001 + the construction doc + `preliminaries.md`. "External" = the two papers.

### 7.1 `EndToEndMaskedSoftmaxExpectedSARSA` (exact route)

| # | dimension | implemented | local spec | external | class |
|---|---|---|---|---|---|
| 1 | token contents / candidate multiplicity | one canonical token per `(s,a)` pair (983, 992); transition rows carry only the residual (1006–1008); one zero null token | identical (FP-ESARSA-001 "canonical memory"; doc §2 `M_x/T_k/Z`) | Xie has no pair tokens; Liang–Lai has a single parameter column | **E** (local) / **M** (external) |
| 2 | score, mask, softmax axis, normalization | score 0 with `-inf` **equality mask** over pairs, `softmax(dim=-1)` (965–970); score 0 with `-inf` equality mask + null, `softmax(dim=0)` over transitions (1000–1005) | declared verbatim in FP-ESARSA-001; doc §1 declares the mask is external and input-dependent | Xie eq. (1) is a fixed positional mask with no equality test and no input dependence; Liang–Lai has no attention mask at all | **E** (local) / **M** (external) |
| 3 | value projections / signed residual | arithmetic `r + γq̄ − Q` (990); sign preserved | doc §4 realizes the same residual through a ReLU FFN with weights `(1, γ, −1)` | Xie uses `V_l` with row `[1 1 −1]` (eq. 12) for the same signed combination | **E** (operator) / **A** (realization: arithmetic instead of projection/FFN) |
| 4 | simultaneous vs sequential / update timing | synchronous: update built from the frozen `memory_values`, then added (983, 1009–1011) | FP-ESARSA-001 "All pair updates are synchronous"; doc §6 distinguishes the three conventions | Xie eq. (18)–(19) is likewise synchronous across columns; Liang–Lai eq. (2) is a batch update | **E** |
| 5 | null / unvisited handling | unvisited pair admits only the zero null token → exactly zero update (1000, 1007–1011) | identical (FP-ESARSA-001 exact writer; doc §5 `n_x=0` → `Z` → zero) | neither paper has a null token | **E** (local) / **M** (external) |
| 6 | sampled vs expected successor | **expected**: `q̄ = Σ_b π(b|S_{t+1})Q(S_{t+1},b)` (987–989, 895–915) | the Expected substitution is declared in FP-ESARSA-001; the construction doc's **Theorem 3.1 uses the sampled pair `X'_k`**; `preliminaries.md` §3.5 has an Expected identity only for a Q-tied Boltzmann `p_β` (§3.3) | Xie: successor state's own value, no action row. Liang–Lai eq. (2): recorded `a_{i+1}` | **A** (declared) |
| 7 | policy input vs learned parameters | `policy` is a call argument (973); class has no `nn.Parameter` (929–936) | FP-ESARSA-001: fixed policy probabilities are inputs, not learned | Liang–Lai returns learned `w` (parameters), a different object | **A** (declared; object type differs from both papers) |
| 8 | finite-logit leakage vs exact equality | exact equality; zero leakage by construction | exact route declared exact | n/a | **E** (local) |
| 9 | dependence on dimension / support | only `n_pairs = |S||A|`; no sharpness constant | `|S||A|` is the declared token count | Xie's kernel depends on features, not on an equality pool size | **E** (local) |
| 10 | outer certificate / policy update vs network-internal | class returns an updated `Q` only (1013–1025) | FP-ESARSA-001 keeps certificate (`fixed_policy_expected_sarsa.py:368`) and the relative-softmax decision (`:504`) outside the network | Liang–Lai's block outputs parameters consumed by an outer loop; Xie's block is evaluation-only | **E** (local) |
| 11 | **update coefficient / normalization** | `α/N_x^train` **mean over visits to the queried pair** (1010 with 1005; `N_x = n_x`) | doc §6 / `preliminaries.md` §3.3 distinguish this from the global `α/N` convention | Liang–Lai eq. (2): `(α/n)·Σ` over the whole trajectory — a scaled **sum**, not a per-pair mean; equal only if `n_x ≡ n/|X|` | **E** (local) / **M** (external) |
| 12 | **memory object / transport** | one canonical token per pair, read and written **within one layer**; no cross-layer transport (983, 1009–1011) | FP-ESARSA-001 "canonical memory" is a flat, static Q-memory | Xie: trajectory-indexed feature/reward columns plus two memory rows rewritten across layers via TSM shift `Π`, `γ` (eq. 18–19), query token excluded as source; Liang–Lai: a single parameter column `w̃` whose output is the updated `w` | **E** (local) / **M** (external) |

### 7.2 `EndToEndFiniteSoftmaxExpectedSARSA` (finite route)

| # | dimension | implemented | local spec | external | class |
|---|---|---|---|---|---|
| 1 | token contents / multiplicity | one token per pair (1059, 1062); no null token | FP-ESARSA-001 finite route (no gate, no null) | neither paper | **E** (local) / **A** vs doc §7 (which retains a null/gate) |
| 2 | score, mask, softmax axis, normalization | successor `log π(b|u) + ζ·1{u=s'}` over all tokens, `dim=1` (1070–1071); read `ξ·1{x=y}` over all pairs, `dim=1` (1075–1076); write `τ·1{X_t=x}` over all transitions, explicit ratio (1082–1086). No mask anywhere | successor/read/write formulas are stated in FP-ESARSA-001 verbatim; `κ_state`, `κ_read` derived in §6.4 | **Corrected (R2):** Xie's scores are feature inner products (eq. 9, 12); the implemented equality logit is that score form at one-hot features `φ=√τ e_x` (§6.4.5). Shared score family, **different candidate set, query role and `log π` bias** | **A** (score form shared; operator not Xie's) |
| 3 | value projections / signed residual | arithmetic `r + γ·successor − read` (1079); sign preserved | doc §7 keeps signed `δ_k^S`; FP-ESARSA-001 requires signed residuals | Xie's `δ_k` also signed | **E** |
| 4 | simultaneous / memory transport | synchronous (1059, 1087–1088) | FP-ESARSA-001 "synchronous" | as in §7.1 dimension 12 | **E** |
| 5 | null / unvisited handling | **none**: unvisited pairs receive the leakage term (1087–1088), which may be zero | FP-ESARSA-001 requires this; doc §7 keeps a visited gate | neither paper has a visited-query gate (Xie does have coverage and predecessor grouping, §3.1) | **A** (declared deviation from doc §7) |
| 6 | sampled vs expected successor | expected, via the finite successor softmax (1072) | FP-ESARSA-001 finite route | Liang–Lai eq. (2) is sampled | **A** |
| 7 | policy input vs learned parameters | `policy` is an argument (1047, 1057); `ζ,ξ,τ` are constructor floats (1035–1045), no `nn.Parameter` | FP-ESARSA-001 fixes `ζ=ξ=τ=8` | Liang–Lai's softmax policy has learned `λ` | **A** |
| 8 | finite-logit leakage vs exact equality | leakage present and reported: `κ_state`, `κ_read`, and the write denominator `N_x e^τ+(m−N_x)`; conditional successor weights are nevertheless **exact** `π(·|s')` | FP-ESARSA-001 declares `κ_state`, `κ_read` and the exactness of the conditional weights; doc §7 gives only a visited-query error **bound** `λ_R = (m−1)/(e^ζ+m−1)` with no `log π` bias term | — | **A** (declared refinement: off-group mass `|S|−1` instead of `m−1`, because the `log π` term makes non-successor rows sum to 1 each) |
| 9 | dependence on dimension / support | `κ_state` depends on `|S|`; `κ_read` on `|S||A|`; write on `N_x, N, τ` | declared in FP-ESARSA-001 | — | **E** (local) |
| 10 | outer certificate / policy update vs network-internal | returns updated `Q` only (1090–1100) | outside the class | as above | **E** (local) |
| 11 | **update coefficient / normalization** | `α` times the **full-support finite kernel** over every transition (1086–1087); on a visited pair this is the `α/n_x` mean, on an unvisited pair the `1/N` uniform average | FP-ESARSA-001 finite route | Liang–Lai eq. (2) coefficient differs as in §7.1 dim. 11 | **E** (local) / **M** (external) |
| 12 | **memory object / transport** | as §7.1 dim. 12 | as §7.1 dim. 12 | as §7.1 dim. 12, with the added distinction that Xie's weights are over trajectory transitions, not pair tokens | **E** (local) / **M** (external) |

### 7.3 `FixedPolicyActionExpectation` (helper)

| # | dimension | implemented | class |
|---|---|---|---|
| 1–2 | token/score/mask | exact one-hot ⊗ policy **effective support** (`1{u=s'}π(b|s')`, exactly zero off the selected state) built by direct weight construction (906–915) — no `softmax` call and **no explicit `−inf` mask tensor** | **A** (declared as a head; the docstring's "无 mask" describes the code representation, FP-ESARSA-001 describes the effective support; the operator agrees on a row-normalized positive policy) |
| 6 | sampled vs expected successor | this head *is* the Expected operator | **A** relative to both papers |
| 7 | policy input | policy passed in as a tensor (895) | **E** |

## 8. Findings (repaired)

**F1 — severity: medium — the docstring's bare "论文 Theorem 3.1" is an inadequate citation; its referent is the local construction's Theorem 3.1.**
`model.py:926` and `model.py:518` state "布尔 mask 对应论文 Theorem 3.1 的结构化等值 mask".
The predicate is **true of the local document**: `preliminaries.md` §3.1 states a **Theorem 3.1** whose content is the structured equality routing, proved in §3.2–§3.3 and mirrored by the construction doc §5 ("这就逐矩阵证明了定理 3.1"). `model.py:633`'s "论文声明的 visited-query gate" has the same local referent (`preliminaries.md` §3.8). The defect is that the citation writes a bare "论文" while a **same-numbered external theorem exists** (Liang–Lai Theorem 3.1), so an external reader can misread it; and Liang–Lai's Theorem 3.1 is linear attention with no softmax and therefore cannot be the referent.
⇒ `H2` is restated in §9: the **external** attribution is unsupported; the **local** referent resolves; the citation as written is inadequate and should name the local source. Impact: provenance/documentation only; the operator is unaffected, and the exact route's mask is exactly what FP-ESARSA-001 declares.

**F2 — severity: medium (narrowed) — no visited-query gate exists in either external paper; the gate is a local declaration.**
Neither paper declares a visited-query gate. **Corrected from v1:** Xie *does* have visit and grouping concepts — the full-coverage assumption of §5 (p. 6) and the predecessor-state grouping of eq. (20) — so v1's "no visit/grouping concept at all" was false. The defensible claim is the **absence of that particular gate** (a support rule that zeroes unvisited queries), which the local documents declare (construction doc §1/§7, `preliminaries.md` §3.8) and `model.py:633` cites. The correct contrast remains: `expected_finite` **has no gate** (FP-ESARSA-001), whereas the local finite bound §7 **is** gate-assisted.

**F3 — severity: low — the v1 "a fortiori" inheritance is WITHDRAWN; a replacement statement with a common Expected target and explicit premises follows.**
v1 claimed doc §7's bound transfers a fortiori because `|S|−1 ≤ |S||A|−1`. That comparison controls only one leakage mass and, more importantly, compares **different objects**: the old bound's exact target is the document's **sampled, gate-assisted** operator, while the implemented class's target is the **Expected, gate-free** operator. GPT's counterexample is correct and I reproduce it: with `S=1, A=2, γ>0`, `Q=(0,1)`, `π=(1/2,1/2)`, one visit to action 0 whose recorded next action is 1, the old exact **sampled** residual is `γ`, while the **Expected** residual is `γ/2`; the implemented finite operator tends to the Expected residual as `ξ→∞` while the old bound `α[E_R + Bε_W]` tends to 0. So no a fortiori inheritance holds. **Also (new):** the local document's own Expected-SARSA material (`preliminaries.md` §3.5 Prop. 3.3, §3.7) is for a **Q-tied Boltzmann** policy, not the frozen external `π`, so it does not close the gap either.

*Replacement statement (derived here by hand; not executed or numerically verified; premises explicit).* Freeze `Q` with `sup_x|Q(x)| ≤ B`, reward bound `R*`, `B = R*/(1−γ)`, positive row-normalized `π`, `m=|S||A|`, `N` transitions, `n_x` visits, and let `B_δ` bound `|δ|` (e.g. `B_δ ≤ 2B`). Let `κ_ζ = e^ζ/(e^ζ+|S|−1)`, `κ_ξ = e^ξ/(e^ξ+m−1)`, and `ε_W(x) = 2(N−n_x)/(n_x e^τ+N−n_x)` (the local `K_η` deviation, eq. (3.8)). Comparing the implemented finite operator to the **implemented exact Expected** operator, for a **visited** query:
`|Δ^fin_Exp(x) − Δ^ex_Exp(x)| ≤ α[ E_R^Exp + B_δ·ε_W(x) ]`, with `E_R^Exp := (γ(1−κ_ζ) + (1−κ_ξ))·span(Q)`.
For an **unvisited** query the exact update is `0` while the finite update is `α·(1/N)Σ_t δ_t`, so a separate term is required: `|Δ^fin_Exp(x)| ≤ α·B_δ`.
Three things distinguish this from eq. (3.9): (i) both sides are Expected, so the `γ/2`-type gap of the counterexample does not appear; (ii) the retrieval term is state-level for the successor (`|S|−1` off-state mass, the `log π` rows each summing to 1) and pair-level for the read (`m−1`); (iii) unvisited queries need their own term, which (3.9) does not have because it is gate-assisted and visited-query-only.
*What is not withdrawn:* eq. (3.8) **is** the same writer formula as the implemented one **on visited queries** (`exp(τ·1{X_t=x})/(n_x e^τ+N−n_x)`, model.py:1084–1086), and its `K_η`/`ε_W` derivation is correct and descriptive there. v1's sentence "doc §7's equation (3.8) lemma does not describe the implemented finite operator" was too strong and is retracted; the correct statement is that (3.8) describes the **writer** on visited queries while (3.9)'s **bound** does not transfer.

**F4 — severity: low (corrected) — the exact route's softmax weights are degenerate, but that does not make its softmax nominal.**
In `EndToEndMaskedSoftmaxExpectedSARSA` the retrieval softmax has exactly one admissible token (weight 1) and the write-back softmax has `N_x` equal logits (uniform `1/N_x`); the loading is done by the `−inf` equality mask. That is still a **softmax attention operation** — a normalized weighted average over an admitted support — and v1's "Softmax is nominal" / "carries no expressive load" and "the one dimension … honestly" are retracted as unsupported categorical claims. The accurate, limited claim: **the exact route uses softmax with uniform weights on its admitted support, so it does not exercise the nonuniform finite-weight expressivity that Xie's Theorem 1 relies on**; the finite route's softmax weights are nonuniform, finite and full-support. The **sparse exact vs literal softmax distinction must be preserved**: the exact route's routing is supplied, not derived from logits.

**F5 — severity: none (confirmation) — the contraction premise corresponds, is declaredly lifted, and is explicitly not extended to the full finite route.**
FP-ESARSA-001's `min_x M_π^X(x,x) ≥ (1+γ)/2 + C`, `0<C<(1−γ)/2`, reproduces Xie **Assumption 5.1** (p. 7), but (i) on the **induced pair-MRP kernel** `M_π^X` rather than Xie's state kernel `M_π`, and (ii) with a declared conservative constant `2C` in `‖I − αM_π^X + αγM_π^X P_π^X‖_∞ ≤ 1 − 2αC`. **Added (exact-residual caveat):** the `Q^π`-preservation identity `F_π(Q^π)=Q^π` is stated in FP-ESARSA-001 **only for the exact-residual population operator** `F_π(Q)=Q+αM_π^X(T_π^XQ−Q)`; the same contract says the identity "does not cancel successor-head or current-read leakage" and that "the full finite-logit route can have a shifted population fixed point because its residual need not vanish at `Q^π`", with no asymptotic no-bias claim for that route. So the contraction statement must **not** be inherited by the implemented full finite class; the finite class's estimate is instead certified by the held-out Bellman residual. The theorem remains conditional on the premise, as in Xie.

**F6 — severity: none (confirmation) — `H1` holds for both classes.**
An explicit, checkable specification exists: FP-ESARSA-001 §"Frozen mathematical contract" states both operators in closed form; `fixed_policy_expected_sarsa.py` states the same formulas as executable pure functions (`finite_successor_all:108`, `finite_read_all:130`, `finite_writeback_all:144`, `run_expected_exact:194`, `run_expected_finite:291`); and `verify_fixed_policy_expected_sarsa.py:44–48` imports exactly the three classes and asserts formula-level equalities (header claims 1e-12 on deterministic fixtures, including "no equality mask or visited gate" for the finite route). FP-MODEL-REVIEW-001 additionally recorded reconstructed agreement on 60 fixtures. **This is agreement between the code and a local pure-NumPy reference, i.e. an implementation claim; by the task's rule it is not used here as correspondence evidence.**

**F7 — severity: none (scope note) — the local doc §8 warns against exactly the substitution this audit avoids.**
Doc §8 states that the literal witness and the compact control experiment "不能互换". The two audited classes are compact operator realizations (no `d×L` prompt, no type/ID coordinate block, no `W_Q/W_K/W_V/W_O`), not the literal block of doc §2–§5. Therefore no claim that the model.py classes are the doc's literal construction is made or implied here.

## 9. Answer: what network name and paper attribution are justified

- **Correspondence verdict (unchanged).** Each implemented operator corresponds to a **declared local extension**, not to an external paper. The *operator* (synchronous, per-pair grouped, `α/N_x`-averaged TD update on a one-token-per-pair Q-memory with equality-routed read/write and a zero null token) is declared in the local construction (§1–§6) for the **sampled** successor; the **Expected** successor, the **finite-logit** successor score with the `log π` bias, and the **removal of the visited gate** are declared in FP-ESARSA-001. Neither paper contains the expected-successor operator, an equality mask, a pair-indexed memory, a per-pair `α/n_x` mean, or a visited gate.
- **`H2`, restated (narrowed).** A blanket **external**-paper attribution is not supported. The docstring's "论文 Theorem 3.1" and "论文声明的 visited-query gate" have a determinate **local** referent (`preliminaries.md` Theorem 3.1, §3.1, proved §3.2–3.3; §3.8) and are *true of it*; but the citation is **inadequate as written** because it does not name the local document while a same-numbered external theorem exists. `H2` therefore reports: external attribution unsupported; local lineage resolves; bare citation inadequate. This is the same resolution independently reached by the Codex first result (finding 3) and by my own reciprocal verification of that report (§4.1) — see `verification.md` §4.1.
- **Justified lineage attributions (positive), corrected.**
  1. Liang–Lai **Theorem 3.1** and **Proposition 3.1** are the correct sources for the *linear-attention, sampled-next-action* SARSA operator family and the inert-subblock structure. In this repository that lineage is instantiated by `LinearAttnICRL:18–41` (linear-attention block, P12/V21 with the Proposition 3.1 inert subblocks frozen) and by `LiangLaiOperatorBaseline:862–885`, which **directly evaluates** `Δw = (α/n)Σδ_iφ_i` and whose docstring explicitly disclaims reproducing the teacher-mimicking training dynamics. **Removed (R5):** `SoftmaxSARSA:109–158` is **not** an implementation of Liang's linear-attention theorem — it is a trainable **hybrid** (softmax attention over feature rows + a SARSA teacher target, `nn.Parameter` `V`), and v1's grouping of it into that family is retracted.
  2. Xie **Assumption 5.1** is the correct source of the diagonal-margin contraction premise used by FP-ESARSA-001 (lifted from `M_π` to `M_π^X`), with the caveat in F5.
  3. Xie **Theorem 1** is the correct source of the general *softmax-attention-executes-a-batch-TD-update* template. It is the nearest external neighbour of the finite route, but is **not** its specification: the candidate set, the query role and the absence of a `log π` policy bias all differ. **Corrected (R2):** the score form is *not* a difference — the equality logit is a one-hot feature inner product and hence a special case of Xie's score family.
- **Attributions that are not justified.** Any statement that reads "论文 Theorem 3.1" as **Liang–Lai's** Theorem 3.1, or that reads "论文声明的 visited-query gate" as an external paper's declaration. Re-word to name the local source (FP-ESARSA-001 and the local construction / `preliminaries.md`).
- **Permitted final terminology for the two classes.** "A fixed-weight, parameter-free softmax-attention operator that executes **one** synchronous grouped **fixed-policy Expected-SARSA** update on a complete Q-memory, once per layer, with an equality-routed exact variant and a finite-sharpness full-support variant; the routing mask, the Expected successor, and the gate-free finite route are **local declarations** (FP-ESARSA-001; local construction §1–§7)." Do **not** say, without qualification: "the paper's Theorem 3.1 network" (unnamed); "a pretrained/learned Transformer"; "the literal prompt construction of the construction doc"; "the network certifies its own error" (certificate and policy update are external); "Expected SARSA from Xie/Liang–Lai".
- **Caveats that must accompany the name.**
  - `EndToEnd` here means *from raw transition fields to the updated Q*; the modules have **zero trainable parameters**, so "end-to-end" must not be read as "trained end-to-end".
  - The exact route's softmax weights are uniform on the admitted support; this does not exercise nonuniform finite-weight expressivity (F4), but it is still a softmax attention operation.
  - The update coefficient is the **local per-pair mean `α/N_x`**, not Liang–Lai's global `α/n` scaled sum, and not a global `α/N` convention (§7.1 dim. 11; local construction §6).
  - The memory object and its transport are local: a **static one-token-per-pair Q-memory updated within one layer**. They are not Xie's trajectory memory with TSM shift/`γ`-repopulation (eq. 18–19), and not Liang–Lai's parameter column `w̃` whose output is an updated `w` (§7.1 dim. 12).
  - The `Q^π`-preservation identity belongs to the **exact-residual population operator** only; the full finite route may have a shifted population fixed point and carries no no-bias claim (F5).
  - The `κ_state`/`κ_read` exactness algebra presupposes row-normalized, strictly positive `policy`. FP-MODEL-REVIEW-001 recorded that invalid policies are currently accepted at the API boundary and that `math.exp(τ)` overflows for `τ = 1000`. Neither is triggered by the frozen configuration, but the exactness claim is conditional on that precondition.

**Route verdict:** `H1 = satisfied`; `H2 = external attribution unsupported, local referent resolves, bare citation inadequate`; `H3 = satisfied` (all listed differences enumerated and traced to FP-ESARSA-001, the local construction, or the explicitly-added normalization/transport/fixed-point dimensions). No task-level `OBJECTION`.

## 10. Independence limitations (recorded, not claimed away)

1. **Same executor, prior exposure.** This route executed FP-ESARSA-001's main route (which wrote `fixed_policy_expected_sarsa.py` and the model.py additions) and FP-ATTN-001, and authored the FP-MODEL-REVIEW-001 audit. The specification under review is therefore largely **self-authored**; this is not pristine blind discovery. The correspondence question was nonetheless answered from the papers' text first and the local spec second.
2. **v2 reads both reports.** Initial independence is complete (GPT verification §"Integrity and scope"), so v2 read the Codex first result and the GPT verification. v1 read neither; the repairs do not rest on Codex's text — the primary sources were re-read for each item.
3. **Tooling failure in this session.** The shell was unavailable (every invocation failed with `EPERM: operation not permitted, mkdir 'C:\Users\Admin\.claude\session-env\…'`), and the grep/glob tools and all MCP tools were unavailable. Consequences: (a) I could not compute local file hashes; (b) the papers and local documents were read **sequentially** rather than query-filtered — the absence statements rest on complete reads of the frozen extracted text, but no machine-checked absence proof exists. Per GPT verification §"Integrity and scope", GPT separately hashed v1 and will archive a transport/integrity supplement; **this v2 is not independently hashed by me**.
4. **No execution.** No experiment, no formula check by execution. The F3 replacement bound and all §6.4 algebra are hand-derived; the F3 bound in particular is **not** numerically verified.
5. **Coverage.** Only the classes named in the task are assessed. The SARSA siblings (`EndToEndMaskedSoftmaxSARSA:510`, `EndToEndFiniteSoftmaxSARSA:628`) are touched only where they carry the same attribution text, and the report notes that the natural referent there is the local manuscript's gate-assisted §3.8 route. Neither paper's theorems were re-proved.

## 11. Acceptance-criteria accounting

| requirement (task §"Deliverables and acceptance") | status |
|---|---|
| both classes covered across all listed dimensions | ✅ §7.1, §7.2 (12/12 rows each), plus §7.3 for the helper |
| original external paper claims separated from local propositions | ✅ §3 (papers) vs §3.3/§4–§5 (local), with the split named per row |
| each row classified exact / explicit adaptation / mismatch / unresolved | ✅ §7 legend and per-row class |
| exact code-line citations | ✅ §6, §7, §8 |
| paper page/equation citations | ✅ §3 tables |
| findings with severity | ✅ §8 (F1 medium, F2 medium, F3 low, F4 low, F5–F7 none/confirmation) |
| permitted final terminology | ✅ §9 |
| supported answer on network name and paper attribution | ✅ §9, with corrected positive attributions (R5) |
| no inference of correspondence from numerical agreement with the in-repo reference | ✅ stated in §2, §8-F6 |
| missing evidence kept unresolved | ✅ §7 external cells; §10 |
| independence limitations recorded honestly | ✅ §10 (incl. the v2-reads-both disclosure) |
| no experiment / no model edit / no task-definition edit / no push / no merge | ✅ complied |
| repair of every GPT-supported item | ✅ R1–R7 in §0, each with a disposition and a location |

## 12. Paths

- This file: `C:/Users/Admin/Desktop/research/icrl_softmax/results/FP-SPEC-REVIEW-001/claude_worktree/icrl_softmax/docs/research_branches/FP-SPEC-REVIEW-001/claude/first_result_v2.md`
- Point-by-point reply to GPT: `C:/Users/Admin/Desktop/research/icrl_softmax/results/FP-SPEC-REVIEW-001/claude_worktree/icrl_softmax/docs/research_branches/FP-SPEC-REVIEW-001/claude/response_to_review.md`
- Preserved originals (unchanged): `.../claude/first_result.md` (v1, SHA256 `72e3d57046097f3507cccd36bf8b39cca51396d3788288e86b285ce575d159a9` as recorded by GPT) and `.../claude/verification.md`.
