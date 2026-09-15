# FP-SPEC-REVIEW-001 — Claude final report

Route `claude/FP-SPEC-REVIEW-001`; worktree `claude_worktree`; baseline `16ee0f652cde56b4f2ef6e3f1a43b58818bb0108`; date 2026-09-15.
Task v1.0 (`docs/research_tasks/FP-SPEC-REVIEW-001.md`). This self-contained report **supersedes** `first_result.md` (v1, SHA256 `72e3d57046097f3507cccd36bf8b39cca51396d3788288e86b285ce575d159a9`) and `first_result_v2.md`; both, and `response_to_review.md` and `verification.md`, are preserved unchanged. It repairs the report failures identified in GPT `verification_v2.md` (FAIL, report-level, no task objection).
Scope: correspondence audit only. No experiment, no model edit, no task-definition edit, no push, no merge. All algebra below is by hand; nothing was executed.

Independence: this route previously executed FP-ESARSA-001 and FP-ATTN-001 and authored FP-MODEL-REVIEW-001, so the specification under review is largely self-authored — not pristine blind discovery. This final report reads the Codex v2 report and GPT's verifications; the repairs do not rest on their text, and the primary sources were re-read for each item.

## 1. Corrections of earlier claims (this is the repair)

| # | earlier claim (v2 / response) | corrected statement |
|---|---|---|
| C-A | §3.2, §7.1 dim. 11, response item 1: the global `α/N` coefficient and the per-pair `α/n_x` coefficient "coincide only when `n_x ≡ N/\|X\|`" | **False.** With a common `α`, the coefficients are `(α/N)` and `(α/n_x)`. On a nonzero residual sum `S_x`, `Δ_impl(x) = (α/n_x)S_x = (N/n_x)·Δ_Liang(x)` (Liang–Lai eq. (2) is a scaled sum). At perfectly balanced counts `n_x=N/m` the factor is `m`, not 1. They agree for every pair with `S_x≠0` only if `n_x=N`; both vanish when `S_x=0`. Matching under balance requires the explicit convention `α_local=α_global/m` (= `α_global n_x/N`), which is a changed learning-rate convention, not the frozen same-`α` identity. The sampled-vs-Expected target difference is separate. |
| C-B | §7.2 dim. 11: on a visited pair the finite update "is the `α/n_x` mean" | **False at finite `τ`.** For `n_x>0`, `W^τ[t,x]=e^τ/(n_x e^τ+N−n_x)` on matching transitions and `1/(n_x e^τ+N−n_x)` on **each** non-matching transition (positive weight, `N−n_x` of them). Hence `Δ^fin(x)=α[w_+Σ_{match}δ_t + w_−Σ_{nonmatch}δ_t]`. This is the per-pair mean `α/n_x·Σ_{match}δ_t` only in the limit `τ→∞` with `n_x>0`, or exactly when `n_x=N` (no off-group transitions). The finite writer is a strict generalization, not the visited mean. For `n_x=0`, `W^τ=1/N` and `Δ^fin=α·(1/N)Σ_t δ_t` for every finite `τ`. |
| C-C | response §"precision note 1": "the Expected residual is exactly `γ/2`" and the counterexample holds for every finite `ξ,τ` | **Precision corrected.** Only the **exact** Expected residual is `γ/2`. The **finite** current-read residual in the one-transition example is `γ/2 − 1/(e^ξ+1)`, because the `ξ`-read is leaky; it equals `γ/2` only as `ξ→∞` (read-head convergence). The limit contradiction shows the blanket transfer fails; it does **not** show that every finite parameter value violates any numerical bound. That overstatement is removed. |
| C-D | §8-F3's replacement bound | **Withdrawn.** No replacement guarantee is asserted. The inequality compared two different operators through a bound on `|δ|`, and that `δ`-bound does not explicitly cover the residual being bounded (the finite-vs-exact Expected difference also carries off-group leakage of `Q`, not only `|δ|`). Withdrawal of the transfer plus the correct comparison (§5-F3) is sufficient; no new result is requested. |

Corrections C-A and C-B propagate to §4 dims. 11, §5-F3, and the naming caveats in §6. The other v2 repairs (R1–R8: local referent identified; blanket inherited bound withdrawn; one-hot kernel connection restored; sparse-support clarification; "every pair moves" corrected; helper wording) stand.

## 2. Objects, sources, method

Objects: `model.py` `FixedPolicyActionExpectation` (888–915), `EndToEndMaskedSoftmaxExpectedSARSA` (918–1024), `EndToEndFiniteSoftmaxExpectedSARSA` (1027–1100); comparators `fixed_policy_expected_sarsa.py` (`finite_successor_all`:108, `finite_read_all`:130, `finite_writeback_all`:144, `run_expected_exact`:194, `run_expected_finite`:291); contract `docs/research_tasks/FP-ESARSA-001.md`.
External sources (page-indexed frozen text in `results/FP-SPEC-REVIEW-001/input/`): Xie arXiv:2605.07333v2 — mask eq. (1), recursion eqs. (2)–(3) p.3; MRP p.4; score `g`, weight `K` eq. (9) pp.4–5; `A_l=diag(I,0)` and `g(S_j,S_{k−1})=⟨x(S_j),x(S_{k−1})⟩` eq. (12) and p.5 text; Theorem 1 eq. (17) p.5; TSM reparameterization eqs. (18)–(19) p.6; coverage assumption and predecessor grouping eq. (20) p.6; Assumption 5.1 and Theorem 2 eq. (23) p.7. Liang–Lai arXiv:2605.05755v1 — linear-attention block eq. (1) p.2; semi-gradient SARSA eq. (2) p.3; prompt eq. (4) and Theorem 3.1 p.4; Algorithm 1 p.5.
Local: `论文_草稿/端到端_softmax_SARSA_构造性证明.md` and `论文_草稿/preliminaries.md` (the numbered edition; **Theorem 3.1** stated §3.1, proved §3.2–§3.3; §3.8 = gate-assisted finite route).
Method: each source operator was derived from the sources first, then code was mapped onto it. Numerical agreement with the in-repo NumPy reference is never used as correspondence evidence.

## 3. Operators

**Exact routed operator** (implemented, `model.py:983–1011`). One canonical token per pair `y=(u,b)`, `m=|S||A|`; `N` transitions, `n_x=#{t:x_t=x}`.
`A_cur[t,y]=1{y=x_t}`; `A_exp[t,y]=1{u=s'_t}π(b|s'_t)`; `δ_t=R_t+γΣ_y A_exp[t,y]Q(y)−Σ_y A_cur[t,y]Q(y)`; `W[t,x]=1{x_t=x}/n_x` (`n_x>0`), a zero-valued null source of weight 1 (`n_x=0`); `Q_new(x)=Q(x)+αΣ_t W[t,x]δ_t`, synchronous. The null row contributes exactly zero for `n_x=0`.

**Finite-logit operator** (`model.py:1065–1088`), sharpness `ζ=ξ=τ=8`:
`A_exp^ζ[t,(u,b)]=π(b|u)e^{ζ1{u=s'_t}}/(e^ζ+|S|−1)`, `A_cur^ξ[t,y]=e^{ξ1{y=x_t}}/(e^ξ+m−1)`, `W^τ[t,x]=e^{τ1{x_t=x}}/(n_x e^τ+N−n_x)`; same residual/addition. Requested-state and requested-pair masses `κ_ζ=e^ζ/(e^ζ+|S|−1)`, `κ_ξ=e^ξ/(e^ξ+m−1)`; conditional successor action weights are exactly `π(·|s')` when every policy row sums to 1.

**External comparators.** Xie weighted softmax TD: `v_{t+1}(S_j)=v_t(S_j)+α_tΣ_k δ_k^{(t)}K(S_{k−1},S_j)`, `δ_k=R_k+γv_t(S_k)−v_t(S_{k−1})`, `K=exp(g(S_j,S_{k−1}))/Σ_m exp(g(S_j,S_{m−1}))` (eqs. (8)–(9)), realized as a feature inner product (eq. (12), p.5). Liang–Lai: `w_new=w+(α/n)Σ_t δ_t φ(S_t,A_t)` (eq. (2)), linear attention, no softmax, no mask.

## 4. Comparison over the frozen dimensions

Labels: **E** exact correspondence, **A** explicit adaptation, **M** mismatch, **U** unresolved. Dims. 1–10 are the task's frozen list; dims. 11–12 are the two added by this audit (§1 C-A/C-B; the local construction §6 / `preliminaries.md` §3.3 supply the local side of dim. 11).

### 4.1 Exact route

| # | implemented | local spec | external | class |
|---|---|---|---|---|
| 1 | one canonical token per pair (983, 992); transition rows carry the residual (1006–1008); one zero null token | identical ("canonical memory"; doc §2 `M_x/T_k/Z`) | Xie: trajectory states, no pair tokens; Liang–Lai: one parameter column | E / M |
| 2 | score 0 with `−∞` equality mask, `softmax(dim=−1)` (965–970); score 0 with `−∞` mask + null, `softmax(dim=0)` (1000–1005) | declared verbatim; doc §1 declares the mask external and input-dependent | Xie eq. (1) is a fixed positional mask, no equality test; Liang–Lai has no mask | E / M |
| 3 | arithmetic `r+γq̄−Q` (990), sign preserved | doc §4 paired-ReLU FFN `(1,γ,−1)`; Xie `V_l` row `[1 1 −1]` (eq. (12)) | — | E (operator) / A (realization) |
| 4 | synchronous: built from frozen `memory_values`, then added once (983, 1009–1011) | "All pair updates are synchronous"; doc §6 distinguishes conventions | Xie (18)–(19) synchronous; Liang–Lai eq. (2) batch | E |
| 5 | unvisited pair admits only the zero null token ⇒ exactly zero update (1000, 1007–1011) | identical (doc §5 `n_x=0→Z`) | neither paper | E / M |
| 6 | expected `q̄=Σ_b π(b\|s')Q(s',b)` (987–989, 895–915) | Expected substitution declared in FP-ESARSA-001; local Theorem 3.1 is **sampled**; `preliminaries.md` §3.5 Expected only for a `Q`-tied Boltzmann | Xie: successor state's own value; Liang–Lai: recorded `a_{i+1}` | A |
| 7 | `policy` is a call argument (973); no `nn.Parameter` | fixed policy is an input | Liang–Lai returns learned `w` | A |
| 8 | exact equality; zero leakage | exact route declared exact | — | E |
| 9 | only `n_pairs=\|S\|\|A\|`; no sharpness constant | `\|S\|\|A\|` declared token count | Xie kernel depends on features | E |
| 10 | returns updated `Q` only (1013–1025) | certificate (`:368`) and relative-softmax decision (`:504`) are outside the network | Liang–Lai block feeds an outer loop; Xie evaluation-only | E |
| 11 | `(α/n_x)` mean over visits (1005, 1010) | construction §6 / `preliminaries.md` §3.3 contrast it with a global `α/N` | Liang–Lai `(α/n)Σ` is a scaled sum; `Δ_impl=(N/n_x)Δ_Liang`; equal iff `n_x=N` or `S_x=0` | E / M |
| 12 | static one-token-per-pair `Q`-memory, read and written within one layer (983, 1009–1011) | canonical memory | Xie: trajectory columns plus two memory rows rewritten across layers by TSM `U,W,Π` (18)–(19); Liang–Lai: parameter column `w̃` → updated `w` | E / M |

### 4.2 Finite route

| # | implemented | local spec | external | class |
|---|---|---|---|---|
| 1 | one token per pair (1059, 1062); no null token | FP-ESARSA-001 finite route | neither paper | E / A (against doc §7, which keeps gate/null) |
| 2 | successor `log π(b\|u)+ζ1{u=s'}` over all tokens, `dim=1` (1070–1071); read `ξ1{x=y}`, `dim=1` (1075–1076); write explicit ratio (1082–1086); no mask | formulas stated in FP-ESARSA-001; `κ` derived §3 above | Xie's `g` is a **general** score (eq. (9)) realized as `⟨x(S_j),x(S_{k−1})⟩` (eq. (12), p.5); the write score `τ1{x_t=x}` is that inner product at `φ(x)=√τ e_x` | A (shared score family; different candidate set, query role, `log π` bias) |
| 3 | `r+γ·successor−read` (1079), signed | signed residual required | Xie `δ_k` signed | E |
| 4 | synchronous (1059, 1087–1088) | "synchronous" | as dim. 12 above | E |
| 5 | no gate; unvisited pairs receive `α(1/N)Σ_tδ_t`, possibly zero (1087–1088) | required by FP-ESARSA-001; doc §7 keeps the gate | neither paper has a visited-query gate (Xie has coverage and eq. (20) grouping) | A |
| 6 | expected, via the finite successor softmax (1072) | FP-ESARSA-001 finite route | Liang–Lai eq. (2) sampled | A |
| 7 | `policy` argument (1047, 1057); `ζ,ξ,τ` constructor floats (1035–1045); no `nn.Parameter` | `ζ=ξ=τ=8` fixed | Liang–Lai's softmax policy has learned `λ` | A |
| 8 | leakage reported: `κ_ζ`, `κ_ξ`, writer denominator `n_x e^τ+N−n_x`; conditional successor weights still exactly `π(·\|s')` | declared by FP-ESARSA-001; doc §7 gives only a visited-query bound with no `log π` term | — | A (refinement: off-group mass `\|S\|−1`, because `log π` rows each sum to 1) |
| 9 | `κ_ζ` on `\|S\|`; `κ_ξ` on `\|S\|\|A\|`; writer on `n_x,N,τ` | declared | — | E |
| 10 | returns updated `Q` only (1090–1100) | outside the class | as exact route | E |
| 11 | `α·W^τ` (1086–1088): **not** the `α/n_x` mean at finite `τ` (see §1 C-B); for `n_x=0` it is `α(1/N)Σ_tδ_t`, independent of `τ` | FP-ESARSA-001 finite route | Liang–Lai coefficient differs as dim. 11 above | E / M |
| 12 | as exact route dim. 12 | as above | as above, with the added difference that Xie's weights run over trajectory transitions, not pair tokens | E / M |

### 4.3 Helper

`FixedPolicyActionExpectation` (888–915) builds `1{u=s'}π(b|u)` by one-hot ⊗ policy (906–915): effective support is exactly the declared masked-log-softmax support with exact zeros off the selected state, but there is **no** `softmax` call and **no** explicit `−inf` mask tensor. The docstring's "无 mask" (892) describes the code representation, not the absence of source selection; the operator agrees with FP-ESARSA-001's masked log-softmax on a row-normalized positive policy. Class **A**.

## 5. Findings

**F1 (medium) — the bare "论文 Theorem 3.1" citation is inadequate; its referent is the local construction's Theorem 3.1.** `model.py:926` and `:518` read "布尔 mask 对应论文 Theorem 3.1 的结构化等值 mask"; `preliminaries.md` §3.1 states a Theorem 3.1 whose content is the structured equality routing (proved §3.2–§3.3; mirrored by the construction doc §5), and `model.py:633`'s "论文声明的 visited-query gate" matches `preliminaries.md` §3.8. Xie has no Theorem 3.1 at all (Theorems 1–3); Liang–Lai's Theorem 3.1 is linear attention with no softmax and hence no mask, so it cannot be the referent. The defect is the unnamed local source while a same-numbered external theorem exists. Documentation only; the operator is unaffected.

**F2 (medium, narrowed) — no visited-query gate exists in either external paper.** Xie **does** assume full coverage (p.6) and groups weights by predecessor state (eq. (20)); what is absent is the particular gate that zeroes unvisited queries — a local declaration (doc §1/§7, `preliminaries.md` §3.8, cited at `model.py:633`). The implemented `expected_finite` has no gate, whereas the local finite analysis §7 is gate-assisted.

**F3 (medium) — the "a fortiori" inheritance is WITHDRAWN, and no replacement bound is asserted.** v1 claimed the doc §7 bound transfers because `|S|−1 ≤ |S||A|−1`. That comparison controls one leakage mass, and the objects differ: the old bound's exact target is the local **sampled, gate-assisted** operator, while the implemented class is **Expected and gate-free**. The local documents' own Expected-SARSA material (`preliminaries.md` §3.5, §3.7) is for a `Q`-tied Boltzmann policy with `A'~p_β`, not the frozen external `π`, so it does not close the gap either.
*Correct comparison (no bound claimed).* One-transition example: `|S|=1`, `|A|=2`, `γ>0`, `Q=(0,1)`, `π=(1/2,1/2)`, one transition at `x=(s,a_0)` with `R=0` and successor state `s`. The successor term is `q̄=1/2` for the exact **and** the finite head (with `|S|=1` every token has `u=s'`, so the finite softmax returns `π` exactly, independently of `ζ`). Exact Expected residual `δ^ex=γ/2`; finite residual `δ^fin=γ/2−(e^ξQ(x)+Q(s,a_1))/(e^ξ+1)=γ/2−1/(e^ξ+1)` since `Q(x)=0` (`m=2`). So `δ^fin→γ/2` as `ξ→∞` while the old bound `α[E_R+Bε_W(x)]→0` as sharpness grows: the transfer fails uniformly. This establishes failure of the blanket inheritance; it does **not** establish that any finite parameter value violates a numerical bound, and no such claim is made.
*Not withdrawn:* eq. (3.8) **is** the same writer formula on visited queries (`e^{τ1{X_t=x}}/(n_x e^τ+N−n_x)`, `model.py:1084–1086`); what fails to transfer is the bound's target and coverage, not the writer formula.

**F4 (low) — the exact route's softmax weights are degenerate, but not "nominal".** Its retrieval softmax has one admissible token (weight 1) and its write-back softmax has `n_x` equal logits (uniform `1/n_x`), with routing supplied by the `−∞` mask — still a normalized softmax attention operation. The categorical claims "softmax is nominal"/"carries no expressive load" are retracted. Accurate limited claim: the exact route does not exercise the nonuniform finite-weight expressivity Xie's Theorem 1 relies on, and its routing is supplied rather than derived from logits.

**F5 (confirmation) — the contraction premise corresponds, is lifted, and is not inherited by the full finite route.** FP-ESARSA-001's `min_x M_π^X(x,x) ≥ (1+γ)/2+C`, `0<C<(1−γ)/2`, reproduces Xie Assumption 5.1 (p.7) on the induced **pair-MRP** kernel `M_π^X`, with a declared conservative constant. The identity `F_π(Q^π)=Q^π` is stated only for the **exact-residual population operator** `F_π(Q)=Q+αM_π^X(T_π^XQ−Q)`; the same contract says it "does not cancel successor-head or current-read leakage" and that the full finite route may have a shifted fixed point. The contraction statement must not be inherited by the implemented finite class; that class is certified externally by the held-out Bellman residual.

**F6 (confirmation) — H1 holds for both classes.** FP-ESARSA-001 states both operators in closed form; `fixed_policy_expected_sarsa.py` restates them as pure functions; `verify_fixed_policy_expected_sarsa.py:44–48` asserts formula-level equalities. This is code-vs-local-reference agreement — an implementation claim, not correspondence evidence.

**F7 (scope note) — the local doc §8 warns the literal witness and the compact control experiment "不能互换".** The audited classes are compact operator realizations (no `d×L` prompt, no type/ID block, no `W_Q/W_K/W_V`), and no claim that they are the literal construction is made here.

## 6. Answer: justified name and attribution

**Verdict.** Each implemented operator corresponds to a **declared local extension**, not to either external paper. The operator (synchronous, per-pair grouped TD update on a one-token-per-pair `Q`-memory with equality-routed read/write and a zero null token, `α/n_x`-averaged) is declared by the local construction (§1–§6) for a **sampled** successor; the **Expected** successor, the finite `log π` score, and the **removal of the visited gate** are declared by FP-ESARSA-001. Neither paper contains the expected-successor operator, an equality mask, pair-indexed memory, the per-pair `α/n_x` mean, or a visited gate.

**H1** satisfied. **H2**: external attribution unsupported; the local referent resolves; the bare citation is inadequate (F1). This is the same resolution the Codex route reached independently (its finding 3). **H3** satisfied — every listed difference is traced to the local documents or to FP-ESARSA-001. No task-level OBJECTION.

**Permitted terminology.** "A fixed-weight, parameter-free softmax-attention operator that executes one synchronous grouped fixed-policy Expected-SARSA update on a complete `Q`-memory per layer, with an equality-routed exact variant and a finite-sharpness full-support variant; the routing mask, the Expected successor and the gate-free finite route are local declarations (FP-ESARSA-001; local construction §1–§7)."
**Do not say**, unqualified: "the paper's Theorem 3.1 network"; "a pretrained/learned Transformer"; "the literal prompt construction"; "the network certifies its own error"; "Expected SARSA from Xie/Liang–Lai".
**Caveats that must accompany the name.** (i) `EndToEnd` means raw transition fields to updated `Q`; both classes have zero trainable parameters. (ii) The exact route's softmax weights are uniform on the admitted support; routing is supplied, not derived from logits (F4). (iii) The update coefficient is the local per-pair mean `α/n_x`, **not** Liang–Lai's global `(α/n)Σ` scaled sum and not a global `α/N` convention; a same-`α` identity with Liang–Lai holds only at `n_x=N` or a vanishing residual sum, and matching under balanced counts requires an explicit `α_local=α_global/m` convention (§1 C-A). (iv) At finite `τ` the writer is not the `α/n_x` mean; it approaches it only as `τ→∞` for a visited pair, or exactly when `n_x=N` (§1 C-B). (v) The `Q^π`-preservation identity belongs to the exact-residual population operator only (F5). (vi) The `κ`-exactness algebra presupposes a row-normalized, strictly positive `policy`; invalid policies are currently accepted at the API boundary and `math.exp(τ)` overflows at `τ=1000` — neither is triggered by the frozen configuration, but the claim is conditional on that precondition.

## 7. Limitations

1. Same-executor bias: the specification audited is largely self-authored (§2); this is not independent construction of the spec.
2. This report reads the other route's reports and both GPT verifications; initial independence for v1 was complete, but the final text is not blind.
3. No shell in this session: no hashes were computed and no machine-checked absence proof exists; absence statements rest on complete sequential reads of the frozen page-indexed text and `model.py`.
4. No execution: all algebra, including the F3 comparison and the §3/§4 formulas, is hand-derived and not numerically verified.
5. Coverage: only the classes named by the task (plus the SARSA siblings where they carry the same attribution text) and only the claims actually made; no external theorem was re-proved; no benchmark rerun, no learned-routing or full-prompt construction, no floating-point rigor.

## 8. Acceptance and paths

Both classes covered across all ten frozen dimensions plus the two added ones (§4.1, §4.2, 12 rows each), the helper covered (§4.3); external paper claims separated from local propositions (§2–§3); each row labelled E/A/M/U; code-line and page/equation citations given; findings carry severity (F1 medium, F2 medium, F3 medium, F4 low, F5–F7 confirmations); permitted terminology and caveats stated (§6); missing evidence kept unresolved; no correspondence inferred from numerical agreement (§2, F6); no experiment, model edit, task edit, push or merge. Both `first_result.md`, `first_result_v2.md`, `response_to_review.md` and `verification.md` are preserved.

- This file: `.../results/FP-SPEC-REVIEW-001/claude_worktree/icrl_softmax/docs/research_branches/FP-SPEC-REVIEW-001/claude/final_report.md`
- Reciprocal check of the Codex v2 report: `.../claude/verification_v2.md`
