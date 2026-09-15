# FP-SPEC-REVIEW-001 — Claude verification of the Codex first result

- Frozen task: `docs/research_tasks/FP-SPEC-REVIEW-001.md`, version 1.0, state `VERIFYING`, common baseline
  `16ee0f652cde56b4f2ef6e3f1a43b58818bb0108`.
- Object verified: `docs/research_branches/FP-SPEC-REVIEW-001/codex/first_result.md` (codex worktree), dated 2026-09-15.
- Verifier route: `claude/FP-SPEC-REVIEW-001`, worktree `claude_worktree`, date 2026-09-15.
- Method: the report was checked against the frozen task and the primary sources — the frozen page-indexed paper
  text, the local construction manuscript, the FP-ESARSA-001 frozen contract, and `model.py` at the baseline — and
  **not** against my own route's first result. Each equation, the hand counterexample, and each attribution claim
  was re-derived first, then compared. No experiment, no model edit, no task edit, no edit to any Codex file.

## 1. Inputs actually inspected

| input | location read | status |
|---|---|---|
| frozen task v1.0 | codex worktree `icrl_softmax/docs/research_tasks/FP-SPEC-REVIEW-001.md` | read in full |
| Xie et al. text | `results/FP-SPEC-REVIEW-001/input/Xie_2026_beyond_linear_attention.txt`, pp. 1–9 | read directly |
| Liang–Lai text | `results/FP-SPEC-REVIEW-001/input/Liang_Lai_2026_linear_attention_policy_improvement.txt`, pp. 1–5 | read directly |
| input manifest | `results/FP-SPEC-REVIEW-001/input/manifest.json` | read in full |
| local construction manuscript | `icrl_softmax/论文_草稿/端到端_softmax_SARSA_构造性证明.md` | read in full |
| local specification | `icrl_softmax/docs/research_tasks/FP-ESARSA-001.md` | read in full |
| primary code | `model.py` lines 862–1101, in **both** worktrees | read directly |
| external decision site | `docs/research_branches/FP-COMPOSE-002/claude/evaluate.py` lines 140–240, 360–399 | read directly |

Two provenance notes. (i) The frozen task file is present in the codex worktree but **not** in my worktree
(`claude_worktree/icrl_softmax/docs/research_tasks/FP-SPEC-REVIEW-001.md` does not exist), so I verified against
the codex copy; it is internally consistent with the state recorded by the user (v1.0, `VERIFYING`, both first
reports saved, baseline `16ee0f6`). (ii) `model.py` lines 886–926 and 950–1101 are byte-identical between
`claude_worktree` and `codex_worktree`, so the report's line citations are testable in the same frozen file,
and they are cited below as frozen-baseline line numbers.

## 2. Equations re-derived independently

### 2.1 Exact routed operator (report §4, first block)

Reported: `A_cur[t,y]=1{y=x_t}`; `A_exp[t,(u,b)]=1{u=s'_t}π(b|s'_t)`;
`δ_t = R_t + γΣ A_exp Q − Σ A_cur Q`; `W[t,x]=1{x_t=x}/n_x` for `n_x>0`, null source weight one for `n_x=0`;
`Q_new(x)=Q(x)+αΣ_t W[t,x]δ_t`.

My re-derivation from the code:

- `_masked_singleton_retrieval` (`model.py:960–971`): `allowed[t,y]=1{y=x_t}` (965), zero scores with `−∞` on the
  complement (966–969), `softmax(dim=−1)` (970). Exactly one admissible source per row ⇒ weight 1 ⇒
  `current_q[t]=Q(x_t)=Σ_y A_cur[t,y]Q(y)`. Matches.
- `FixedPolicyActionExpectation` (`model.py:888–915`): `one_hot` on the next state (906–909),
  `next_policy = policy[next_states]` (910), `attention = one_hot⊗next_policy` (911–913),
  `attention @ memory_values` (915) ⇒ `Σ_{(u,b)} 1{u=s'_t}π(b|u)Q(u,b) = Σ A_exp Q`. Matches. (No `softmax` call
  appears in this head; the report's implementation-level qualification is factually right.)
- `residuals = rewards + γ*qbar − current_q` (990). Matches.
- Writer (992–1011): `match[t,y]=1{x_t=y}` (994), `visited[y]=∃t match` (995),
  `allowed = cat(match, (~visited)[None,:])` (1000), zero scores with `−∞` (1001–1004), `softmax(dim=0)` (1005).
  A visited column has exactly `n_x` admissible transition rows, all with equal score 0 ⇒ weight `1/n_x`
  each; an unvisited column admits only the null row, whose `source_values` entry is 0 (1006–1008) ⇒ zero
  contribution. `signed_write` (1009), `update = α·signed_write` (1010), `q_new = memory_values + update` (1011).
  Matches the reported `W` and the synchronous single write, with `memory_values = q_values.reshape(-1)` (983)
  frozen for the whole layer.

Conclusion: the exact operator as stated is correct, including the `n_x=0` null rule and the `1/n_x` weight.

### 2.2 Finite-logit operator (report §4, second block)

Reported: `A_exp^ζ[t,y] = π(b|u)·e^{ζ·1{u=s'_t}}/(e^ζ+S−1)`; `A_cur^ξ[t,y] = e^{ξ·1{y=x_t}}/(e^ξ+m−1)`;
`W^τ[t,x] = e^{τ·1{x_t=x}}/(n_x e^τ + N − n_x)`; requested-state mass `e^ζ/(e^ζ+S−1)`;
requested-pair mass `e^ξ/(e^ξ+m−1)`.

My re-derivation:

- Successor (`model.py:1065–1072`): scores `log π(b|u) + ζ·1{u=s'_t}`. Denominator over all `m=S·A` tokens
  `= Σ_{u=s'}Σ_b π(b|s')e^ζ + Σ_{u≠s'}Σ_b π(b|u) = e^ζ·1 + (S−1)·1 = e^ζ+S−1`, using only row normalisation
  `Σ_b π(b|u)=1`. So the weight is exactly the reported expression, and the mass on the requested state is
  `e^ζ/(e^ζ+S−1)` with conditional action weights exactly `π(·|s')`. Matches; the report's remark that the
  denominator "follows by summing each normalized policy row, not by assuming uniform actions" is correct.
- Read (`1075–1077`): scores `ξ·1{y=x_t}` over `m` tokens ⇒ denominator `e^ξ+(m−1)`. Matches.
- Write (`1082–1086`): `numerator = exp(τ·match)`, `denominator = counts·e^τ + (N−counts)`. Matches. This is a
  column-normalised softmax over the `N` transitions, algebraically equal to the explicit ratio.
- `n_x=0 ⇒ numerator=1, denominator=N ⇒ W=1/N`, independent of `τ`. My recomputation confirms this, hence the
  report's negative claim "finite `τ→∞` does NOT converge to the exact no-update rule on unvisited queries"
  is correct (the limit is `α·mean(δ)` for every finite `τ`).
- The convergence claim "with all queried pairs visited … letting all three sharpness values grow recovers the
  exact operator" also checks out: for a visited column the denominator is dominated by `n_x e^τ` with
  `n_x ≥ 1`, so the writer converges uniformly over columns; read and successor converge unconditionally. The
  coverage caveat is necessary only for the writer, as stated.

### 2.3 Hand counterexample (report §4, last paragraph) — recomputed from scratch

Setup as reported: `S=1, A=2, γ=0`, `Q*=(0,1)`, batch = one visit per pair, finite `ξ`, finite `τ>0`,
`N=2=m`. My independent recomputation:

- read at the pair visited by transition 1: `(e^ξ·Q(y₁)+Q(y₂))/(e^ξ+1) = 1/(e^ξ+1)`; at transition 2:
  `(Q(y₁)+e^ξQ(y₂))/(e^ξ+1) = e^ξ/(e^ξ+1)`. Matches the report.
- `δ₁ = 0 − 1/(e^ξ+1) = −1/(e^ξ+1)`; `δ₂ = 1 − e^ξ/(e^ξ+1) = +1/(e^ξ+1)`. Matches.
- writer for `y₁`: `W[1,y₁]=e^τ/(e^τ+1)`, `W[2,y₁]=1/(e^τ+1)` (since `n_{y₁}=1`, denominator `e^τ+1`), so
  `update(y₁) = α[−e^τ/(e^τ+1) + 1/(e^τ+1)]/(e^ξ+1) = −α(e^τ−1)/[(e^τ+1)(e^ξ+1)]`; by symmetry `update(y₂)` is
  the positive of that. Matches the report's `±α(e^τ−1)/[(e^τ+1)(e^ξ+1)]`.
- `update ≠ 0` for every `τ>0`, so `Q*` is not a fixed point of the implemented finite operator even at full
  coverage. For the exact operator the same batch gives `δ=0` and `Q*` **is** a fixed point. The counterexample is
  therefore correct, is a real distinction between the two classes, and is anticipated by FP-ESARSA-001's own
  statement that "the full finite-logit route can have a shifted population fixed point because its residual need
  not vanish at `Q^π`". I did not need any numerical execution to check it.

## 3. Paper page/equation citations checked against the frozen text

| report claim | frozen-text location | verdict |
|---|---|---|
| Xie eqs. (1)–(4), (8)–(17) on pp. 3–5 | (1)–(4) p. 3; (8)–(10) pp. 4–5; (11)–(17) p. 5 | correct |
| fixed positional mask excludes the query column | p. 3 eq. (1): `M[i,j]=0` for `i≤n`, `−∞` for `i=n+1` | correct |
| scores are feature inner products with `A=diag(I,0)` | p. 5 eq. (12) and the sentence "the score matrix … depends only on the first `d` rows" | correct |
| `K(k\|s)=exp(⟨x(s),x(S_{k−1})⟩)/Σ_j exp(⟨x(s),x(S_{j−1})⟩)` | p. 4 eq. (9) with `g(S_j,S_{k−1})=⟨x(S_j),x(S_{k−1})⟩` | correct |
| `δ_k(v)=R_k+γv(S_k)−v(S_{k−1})`; all residuals use the same old `v` | p. 4 eq. (8) | correct |
| displayed step size `α=1` | p. 6: "For simplicity, we set `α_l=1` throughout" | correct |
| prompt has feature/reward/two memory rows, no Q table | p. 5 eq. (10); "the last two rows serve as memory" | correct |
| eqs. (13)–(15) shift into predecessor target-memory columns; (18)–(19) on p. 6 do update-current-then-repopulate with `γ` times the shifted new current row | p. 5 eqs. (13)–(15); p. 6 eqs. (18)–(19) with `U=I−e_{d+2}e_{d+2}^⊤`, `W=γe_{d+2}e_{d+3}^⊤` | correct |
| MRP induced by a fixed policy is the setting, p. 4 | p. 4 §2.2, including the marginalisation `p(s'\|s)=Σ_a π(a\|s)p_MDP`, `r(s)=Σ_a π(a\|s)r_MDP` | correct |
| Theorem 2 on p. 7 with coverage/ergodicity/diagonal-margin conditions and its own exact-residual recursion | p. 7 Thm 2 + Assumption 5.1; p. 4 ergodicity; p. 6 eq. (21) | correct |
| Xie's relevant representation theorem is Theorem 1 (not an equality-mask theorem) | p. 5 Thm 1; Theorems numbered 1, 2, 3 only | correct |
| Liang–Lai eq. (1) on p. 2 | p. 2: `H_out = H + (1/n)(VH)(H^⊤PH)`, no softmax | correct |
| its eq. (2), (4), (5), Thm 3.1 on pp. 3–4 | (2) p. 3; (4), (5), Thm 3.1 p. 4 | correct |
| `w_new = w + (α/N)Σ_t[R_t+γw^⊤φ(S'_t,A'_t)−w^⊤φ(S_t,A_t)]φ(S_t,A_t)` | p. 3 eq. (2), with `n` = trajectory length = the report's `N` | correct |
| query column contains `[1;w]`, transition columns carry features/rewards | p. 4 eq. (4): `x_i=[φ_i;γφ_i^+;r_{i+1}]`, `w̃=[1;w]` | correct |
| `P` supplies the signed TD scalar, `V` supplies `αφ` | p. 4 eq. (5): `P*₁₂` maps `w̃` to `(−w, w, 1)` and `V̄*₂₁=[αI_d,0,0]` | correct, checked block-by-block |
| successor action is an input; linear attention, no softmax normalisation; under one-hot features the per-pair update is `α/N` times the residual **sum**, not `α/n_x` times it | p. 3 eq. (2); `φ(s_{i+1},a_{i+1})` is recorded | correct |
| Algorithm 1 on p. 5 contains an external behaviour/control loop | p. 5 Algorithm 1, lines 6 and 12 (ε-greedy sampling, `w←w_SARSA`, `s₀←s_n`) | correct |
| Xie version/hash/pages, Liang–Lai version/hash/pages | `input/manifest.json` | correct (hashes and page counts match verbatim; see §6 limitation on independent re-hashing) |

The report's stated arXiv abstract-page check cannot be reproduced here (no network in this session); it is
labelled by the report itself as a bibliographic consistency check only, not an authenticity proof.

## 4. Architecture, attribution and code-line claims

- `model.py:888–915` (`FixedPolicyActionExpectation`), `model.py:987–1008`/`1009–1011`,
  `:1068–1086`, `:952–965`: all exist at the baseline file and are byte-identical across both worktrees. Note
  §5.1 on the ranges themselves.
- "the exact action-expectation helper directly constructs its probability matrix; it does not execute a
  logit/mask/softmax head" — verified at 906–915; there is no softmax call and no mask tensor in that head, and
  its operator equals the masked log-softmax specified in FP-ESARSA-001 only when `Σ_b π(b|u)=1`. The report's
  precondition is correct and correctly stated.
- "`A_exp` has exact zeros outside the selected state, so it cannot be a finite-logit, globally full-support
  softmax" — verified; a finite-logit softmax has strictly positive weights, whereas `one_hot⊗π` is exactly zero
  off the selected state. Consequently the report's reading of the docstring's "无 mask" as a coding description
  rather than the absence of support selection is the only reading consistent with the operator; the wording
  mismatch it flags (docstring `model.py:892` "无 mask" vs FP-ESARSA-001 "masks to the successor state's unique
  action tokens") is real.
- The report's separation of the two classes from both external architectures is correct: Xie's operator lives on
  an MRP state space with trajectory-indexed feature-similarity softmax weights and TSM memory transport, with no
  action row, no equality mask, no null token and no pair-indexed Q memory; Liang–Lai's operator is linear
  attention on a parameter column `w̃` returning an updated `w`. Neither contains the expected successor, the
  pair-indexed memory, the equality routing or the null token.
- Local lineage: the construction manuscript's equations (3.1)–(3.6) supply the routed-memory skeleton
  (`M_x/T_k/Z` prompt, 3.1; current head, 3.2; next head, 3.3; paired-ReLU residual, 3.4; write-back with
  `n_x` equal logits → `1/n_x`, 3.5; final update, 3.6), and its §7 finite analysis is explicitly
  gate-assisted ("仍保留外部 visited-query gate … equality-mask-free but gate-assisted"). FP-ESARSA-001 supplies
  the expected successor, the finite `log π(b|u)` score, `ζ=ξ=τ=8`, and the removal of the visited gate
  ("no equality mask and no visited-query gate"). Both report rows are accurate.
- External decision sites: `evaluate.py:151` `def l12_scalar(...)` (L12 defined outside the network; docstring
  "No kernel, no propagation, no central interval"), `:180` `def perstate_rows(...)` (per-state rule),
  `:220` `def network_qhat(...)` (repeated estimator calls, `for _ in range(LAYERS)`), `:375`
  `dec = perstate_rows(st["pi"], qh, cert["e_q"])` (external decision; the returned `pi_after` is applied at
  :396). All four citations are exact.
- "no trainable parameters": `__init__` (929–936, 1035–1045) stores only floats; no `nn.Parameter` in either
  class. Correct.

### 4.1 The one point where this report is better supported than my own route's sealed report

The report resolves the docstring's "论文 Theorem 3.1" (`model.py:926`, and the same sentence at `model.py:518`)
to the **local** sampled-SARSA construction rather than to an external paper, and requires the audit wording to
say so. I checked this specifically because my own route's sealed `first_result.md` took the opposite reading
(F1/F2: "the class-level paper attribution does not resolve"; "`H2` fails for this claim").

The evidence supports the Codex reading, by two independent routes:

1. `model.py:633` says the finite SARSA sibling "保留论文声明的 visited-query gate" (retains the gate **declared
   by the paper**). No visited-query gate exists anywhere in Xie (no visit or grouping concept at all) or in
   Liang–Lai (no attention mask of any kind). The local construction manuscript §7 declares exactly this gate
   ("仍保留外部 visited-query gate"), and §1 declares the support rule for visited queries / null token as
   externally supplied. Line 633 therefore cannot refer to either external paper.
2. The local construction manuscript has a 定理 3.1 (its §5 closes with "这就逐矩阵证明了定理 3.1") whose content is
   precisely a **structured equality mask** — §1 "等值路由 mask 由外部提供且依赖输入", §5 "`n_x` 个 logit 相等，每个
   权重为 `1/n_x`" — matching the docstring's "结构化等值 mask" ; Liang–Lai's Theorem 3.1 has no softmax and hence
   no mask at all.

So the report's finding 3 is correct and is the stronger reading; the report's hedge that the bare citation is
"ambiguous"/"inadequate" (because Liang–Lai also has a same-numbered theorem and the docstring does not name the
local document) is a fair, evidence-based qualification of the same recommendation my route reached — the
docstring must be re-worded. I record here, without editing it, that my route's sealed `first_result.md` F1/F2
stop at "does not resolve" and should be corrected to the local-manuscript referent when GPT reconciles the two
routes in `final_synthesis.md`. This does not change my route's operator findings, which agree with the report's.

## 5. Defects found in the report

None of the following affects any conclusion; all are evidence-precision or labelling defects.

**C1 (citation range).** "`model.py:952–965` creates current-pair equality mask and singleton softmax." The
equality mask is at 965, but the `torch.softmax` is at **970**, outside the cited range, and 952–958 are the tail
of `_validate_inputs`. The described content spans 960–971.

**C2 (citation range).** "`model.py:979–985` … forms the signed residual linearly." Lines 979–985 cover the dtype
conversions, `current_pairs`, `memory_values` and the `current_q` retrieval; the signed residual is formed at
**990**.

**C3 (citation range).** "`model.py:987–1008` … then adds one synchronous update." The match/visited/null/writer
parts are in range, but the single synchronous update is at **1009–1011**.

**C4 (trivial).** "`model.py:1068–1102`" runs past the class's last statement (the `return` is at 1100).

**N1 (labelling).** The task requires each comparison-table row to be classified as *exact correspondence,
explicit adaptation, mismatch,* or *unresolved*. The report's "Source correspondence" column uses descriptive
phrases ("Exact local contract; explicit adaptation from trajectory-memory Xie and parameter-query Liang",
"Distinct model classes, not interchangeable", "No universal dimension extrapolation") that convey those
categories but do not use the four required labels or their abbreviations. Content is unambiguous; the
vocabulary is not the frozen one.

**N2 (severity scale).** Findings are labelled `P1`/`P2` with `P2` used for two different findings and `P1` for a
conditional one ("P1 if generalized"); the scale is never defined. The severity requirement is met in substance,
but the labels are not self-explanatory.

**N3 (minor coverage).** `FixedPolicyActionExpectation` is covered in §1, §4 and §6 but has no table row, whereas
the task states it is "included in both-operator coverage". The substance is present.

## 6. Limitations of this verification

1. **Shell unavailable.** Every shell invocation failed with
   `EPERM: operation not permitted, mkdir 'C:\Users\Admin\.claude\session-env\09d7ddd7-7699-4f69-88db-be30823bd3e1'`,
   including with the sandbox disabled. Consequences: (a) I could not compute `git hash-object` for `model.py`, so
   the report's quoted blob hash `947542e60ff27aca5c4670982d9bf3fd966d2b23` is **unverified here** — I verified the
   cited code content instead, by reading both worktrees; (b) I could not re-hash the two PDFs, so the report's
   SHA256 values are verified only as a verbatim match to `input/manifest.json`, not independently recomputed;
   (c) I could not enumerate the codex results directory, so the claimed page renders under the codex route's
   results are **not** inspected — the matrix-orientation and equation-label claims were instead checked
   analytically against eq. (5)/(12) and eq. (1)/(9), which support them.
2. **No network.** The report's arXiv abstract-page metadata check cannot be reproduced; it is labelled by the
   report itself as a bibliographic check only.
3. **Read-vs-search.** Paper statements such as "no equality mask exists in Xie" and "Liang–Lai Theorem 3.1
   contains no mask" rest on direct reading of the frozen extracted text (Xie pp. 1–9 and Liang–Lai pp. 1–5 read
   in full; the extracted-text files are line-addressable). They are checkable by any reader of the frozen input,
   but I could not produce a machine-checked absence proof.
4. **No execution.** This is a reading-and-algebra verification; the hand counterexample was recomputed by hand,
   not by running the class.
5. **Coverage.** Only the two classes and the helper named by the task, and only the claims the report actually
   makes. I did not re-prove any external theorem.

## 7. Verdict

**PASS.**

Justification against AGENTS §七 and the task's acceptance clause:

- **H1 (explicit checkable specification for each implemented operator):** supported. Both operators were
  re-derived from the code and match the report's closed forms exactly, including the `n_x=0` null rule and the
  finite writer's `1/N` behaviour.
- **H2 (each claimed attribution and theorem reference resolves and supports the attributed claim):** the report's
  verdict — a blanket external-paper attribution is not supported, the local theorem lineage resolves, and the
  bare docstring citation is inadequate — is correct. I confirmed independently that Xie has no Theorem 3.1 and no
  equality mask, that Liang–Lai's Theorem 3.1 is linear attention with no mask, and that the docstring's referent
  is the local construction manuscript (`model.py:633` plus manuscript §1/§5/§7). The report's resolution is
  better supported than my own route's reading of the same sentences.
- **H3 (all differences in memory, routing, sampling, update timing, logits, control accounted for):** supported.
  All ten task dimensions are covered for both classes, and each difference (expected vs sampled successor,
  pair-indexed memory, equality routing, null token, gate removal, finite leakage, external certificate/decision)
  is traced to the local construction manuscript or to FP-ESARSA-001.
- The hand counterexample is algebraically correct and independently recomputed.
- The task's acceptance clause explicitly permits a negative scientific result; the report delivers an accurate
  negative attribution audit, and the defects in §5 above are documentary (three code-line ranges, one file-end
  overrun, one label vocabulary, one duplicated severity label, one helper row). None changes a conclusion, and
  none is a reasoning, implementation or evidence failure within the meaning of AGENTS §七's `FAIL`.

**Required corrections before the report is carried into `final_synthesis.md`** (author's own branch; not a
`FAIL` and not an objection): tighten the code-line ranges per C1–C3 (and C4); classify the table rows in the
task's four labels per N1; define the `P1`/`P2` scale and de-duplicate the label per N2. Recommended, not
required: add an explicit helper row per N3. **Note for reconciliation:** my route's sealed `first_result.md`
F1/F2 should be corrected to the local-manuscript referent of "论文 Theorem 3.1" / "论文声明的 gate" as
established in §4.1; I have not edited that sealed report.

No `OBJECTION`: the task definition, its acceptance basis and its four-label classification requirement are
internally consistent and testable; a source-fidelity mismatch is expressly not a task-level defect.

## 8. Path

`C:/Users/Admin/Desktop/research/icrl_softmax/results/FP-SPEC-REVIEW-001/claude_worktree/icrl_softmax/docs/research_branches/FP-SPEC-REVIEW-001/claude/verification.md`

## 9. Hash

**Not computed.** The shell was unavailable for every invocation in this session (§6.1, exact error recorded), so
no SHA256 of this file, of `model.py`, or of the two PDFs could be produced or re-derived here. What I can state
with certainty:

- This file was created by a single `Write` call and has not been modified since; the content above is final and
  reproducible by reading this path.
- The frozen input manifest's PDF hashes are
  Xie `04aef63b86239100c0ec836e6105486db125a4b4ec1dbfdbc0138bfc2f9647d3` and
  Liang–Lai `6f579b4fdc818d493350878ba042bba6ae324d8631c8ed9a8341298c4eed7f9b`; the report's quoted values match
  these verbatim (verification of the manifest, not of the PDFs).
- The report's claimed `model.py` blob hash `947542e60ff27aca5c4670982d9bf3fd966d2b23` remains **unverified** by me.

A reader with a working shell should record the SHA256 of this file (and re-derive the PDF and blob hashes) to
close the provenance gap that this session could not close.
