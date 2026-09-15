# Independent first result — FP-SPEC-REVIEW-001 / Codex

Date: 2026-09-15. Task v1.0; common baseline 16ee0f652cde56b4f2ef6e3f1a43b58818bb0108.
Status: preliminary independent result, awaiting reciprocal verification. No Claude audit results were read before this report was saved.

## 1. Answer

The implemented object is mathematically defined. It is a **local fixed-policy, synchronous per-pair Expected-SARSA construction with attention operators**, in an exact routed variant and a finite-logit approximation variant. It is **not an exact reproduction of either Xie et al.'s softmax ICTD architecture or Liang–Lai's linear-attention SARSA architecture**. This is an explicit local extension, not evidence that either paper was necessarily misunderstood.

The strongest supported local source is FP-ESARSA-001's frozen mathematical contract, not an external paper theorem. The docstring's bare “paper Theorem 3.1” is ambiguous: the local sampled-SARSA manuscript has a matching equality-mask construction, whereas Liang–Lai Theorem 3.1 is a different, linear-attention theorem. Xie's relevant representation theorem is Theorem 1, not a structured pair-equality-mask theorem.

An implementation-level qualification also remains: the exact action-expectation helper directly constructs its probability matrix; it does not execute a logit/mask/softmax head. On the frozen valid policy domain its operator equals that specified head. A statement that all three heads literally execute softmax is false for this code path. This distinction does not invalidate equality of its output with the specified mathematical operator.

## 2. Source identities and reading method

Frozen input manifest: host results/FP-SPEC-REVIEW-001/input/manifest.json. Page numbers below are 1-based PDF pages and agree with printed page numbers in the cited range.

- Xie et al., Beyond Linear Attention: Softmax Transformers Implement In-Context Reinforcement Learning, arXiv:2605.07333v2 (17 May 2026), 28 pages; PDF SHA256 04aef63b86239100c0ec836e6105486db125a4b4ec1dbfdbc0138bfc2f9647d3.
- Liang and Lai, Transformers Provably Implement In-Context Reinforcement Learning with Policy Improvement, arXiv:2605.05755v1 (7 May 2026), 25 pages; PDF SHA256 6f579b4fdc818d493350878ba042bba6ae324d8631c8ed9a8341298c4eed7f9b.
- Official metadata independently checked at https://arxiv.org/abs/2605.07333 and https://arxiv.org/abs/2605.05755. This checks bibliographic/version consistency, not byte-for-byte identity with a fresh server download.
- Page-indexed text extracted from the frozen PDFs with PyMuPDF. Xie pp.5–6 and Liang p.4 additionally rendered and visually checked for matrix orientation and equation labels; renders are in this route's results directory.
- Local primary construction: 论文_草稿/端到端_softmax_SARSA_构造性证明.md §§1–8 and 论文_草稿/preliminaries.md §§3.1–3.4, all at the baseline. The latter is retained local manuscript material, not a peer-reviewed external source.
- FP-ESARSA-001.md, “Canonical memory”, “Exact grouped Expected SARSA”, “Finite-logit standard-softmax route”, and “Exact-residual kernel population operator”. Historical task status or author assertions were not used to prove operator equivalence.
- model.py lines 888–1130; imported/called helpers included. Baseline Git blob of model.py: 947542e60ff27aca5c4670982d9bf3fd966d2b23.

Independence is limited to separate derivation and reporting in this task. I previously reviewed this repository and saw its reference implementations; no claim of pristine blind discovery is made. I derived the displayed source operators from paper/local-spec equations and inspected their code mapping; I did not use numerical agreement with the NumPy implementation as evidence of source fidelity.

## 3. Independent derivation from external sources

### 3.1 Xie: trajectory-memory weighted softmax TD

Xie pp.3–5, equations (1)–(4), (8)–(17): there is a column for each trajectory state, including a final query column. That query is excluded as a source by a fixed positional mask. Scores depend on fixed state features; with A=diag(I,0), they are feature inner products. For transition k and query state s,

K(k|s) = exp(<x(s),x(S[k-1])>) / sum_j exp(<x(s),x(S[j-1])>).

The signed value is δ_k(v)=R_k+γv(S_k)−v(S[k−1]), and v_new(s)=v(s)+α sum_k K(k|s)δ_k(v). The theorem's displayed choice is α=1. All residuals use the same old v. Values repeated at identical feature states remain consistent from the zero initialization because they get the same update.

The prompt contains feature, reward, target-memory and current-memory rows, not a separate complete Q table. Equations (13)–(15) shift the update into predecessor target-memory columns; equivalently equations (18)–(19), p.6, first update current memory then clear/repopulate target memory with γ times the shifted NEW current row. That is a temporal transport invariant, not retrieval from unique action-value candidates.

An MRP induced by a fixed policy is the setting (p.4). No action expectation head, policy improvement certificate or relative-softmax controller is implemented by this theorem. Theorem 2, p.7, needs its coverage/ergodicity/diagonal-margin conditions and its own exact-residual recursion; it cannot simply be inherited by an architecture with extra retrieval leakage.

### 3.2 Liang–Lai: global-batch semi-gradient SARSA

Liang–Lai p.2 equation (1) and pp.3–4 equations (2), (4), (5), Theorem 3.1:

H_out = H + (1/N)(V H)(H^T P H);
w_new = w + (α/N) sum_t [R_t+γ w^T φ(S'_t,A'_t)−w^T φ(S_t,A_t)] φ(S_t,A_t).

The sampled successor action is an input; the query contains [1;w], the transition columns contain features/rewards. P supplies the signed TD scalar and V supplies αφ. This is linear attention with no softmax normalization. Under one-hot tabular features, the update for pair x is α/N times the residual SUM over visits to x, not α/n_x times that sum. Unless counts happen to coincide with N (or a different per-pair learning-rate transformation is introduced), these are different operators.

The paper's learned parameter/teacher-mimicking claims concern this parameterized architecture and training protocol. The current no-trainable-parameter Expected-SARSA classes neither perform that training nor establish those learnability claims. Algorithm 1, p.5, includes an external behavior/control loop; calling the current estimator a reproduction of that algorithm would also conflate different loops.

## 4. Independent local specification and code mapping

Let m=S A, Q∈R^(S×A), N>0 transitions, x_t=(s_t,a_t), n_x=# {t:x_t=x}. Freeze Q and a strictly positive normalized policy π during each evaluation layer.

### Exact routed operator

A_cur[t,y]=1{y=x_t}. A_exp[t,(u,b)]=1{u=s'_t}π(b|s'_t).
δ_t=R_t+γ sum_(u,b) A_exp[t,(u,b)]Q(u,b)−sum_y A_cur[t,y]Q(y).
For n_x>0, W[t,x]=1{x_t=x}/n_x; for n_x=0 a zero-valued null source has weight one.
Q_new(x)=Q(x)+α sum_t W[t,x]δ_t.

Derivation: singleton softmax has weight one; equal logits on n_x admitted sources have weights 1/n_x. Since sum_b π(b|s')=1, softmax(log π(b|s')) on that state's actions returns π exactly. Thus these formulae follow from the FP-ESARSA contract without importing its NumPy reference.

Code:
- model.py:952–965 creates current-pair equality mask and singleton softmax.
- model.py:888–915 creates A_exp by one-hot state selection times policy, then multiplies by Q. This is exactly the above matrix, but no softmax is called there.
- model.py:979–985 reads from one frozen memory and forms the signed residual linearly.
- model.py:987–1008 creates transition match, visited flag, null support, column-normalized writer, then adds one synchronous update.
- Direct signed addition/subtraction is equivalent to a position-wise linear map (and to the paired-ReLU construction in local equation (3.4)); this compact code does not instantiate the entire manuscript H tensor and W_Q/W_K/W_V/FFN matrices.

For clarity, A_exp equals a global masked softmax with scores log π(b|u) if u=s' and −∞ otherwise. With S>1 and π>0 it has exact zeros outside the selected state, so it cannot be a finite-logit, globally full-support softmax. “No mask” in its docstring describes how the sparse matrix is coded, not absence of support selection.

### Finite-logit operator

For every canonical pair y=(u,b):

A_exp^ζ[t,y] = π(b|u) exp(ζ 1{u=s'_t})/(exp(ζ)+S−1).
A_cur^ξ[t,y] = exp(ξ 1{y=x_t})/(exp(ξ)+m−1).
W^τ[t,x] = exp(τ 1{x_t=x})/(n_x exp(τ)+N−n_x).

Use these matrices in the same signed-residual and residual-addition formula above. The successor denominator follows by summing each normalized policy row, not by assuming uniform actions. Requested-state and requested-pair masses are exp(ζ)/(exp(ζ)+S−1) and exp(ξ)/(exp(ξ)+m−1).

Code model.py:1068–1102 realizes these three matrices: two torch.softmax calls and an explicit exp/normalizer writer algebraically equal to column softmax. Positive finite policy entries and safe finite sharpness are preconditions for the finite-score description; frozen sharpness 8/CPU satisfies the audited scope. Boolean equality is used as a numeric score feature; it does not set support to zero or invoke a visited gate.

If n_x=0, W^τ[t,x]=1/N and the update is α times the mean residual, irrespective of τ. Therefore finite τ→∞ does NOT converge to the exact no-update rule on unvisited queries. With all queried pairs visited and finite fixed batch/Q, letting all three sharpness values grow recovers the exact operator. No dimension-uniform approximation follows at fixed sharpness.

An elementary fixed-point counterexample (algebra, no experiment): S=1,A=2, γ=0, deterministic reward vector Q*= (0,1), ξ finite. At Q*, current reads are 1/(e^ξ+1) and e^ξ/(e^ξ+1). For a batch containing one visit to each action, residuals are (-1,+1)/(e^ξ+1). The finite writer at τ>0 sends a nonzero update to each action, ±α(e^τ−1)/[(e^τ+1)(e^ξ+1)]. Thus the true Q* need not be fixed even with full coverage. FP-ESARSA explicitly anticipates this; Xie's exact-residual fixed-point argument is not automatically applicable.

## 5. Coverage and classification table

| Dimension | Exact class | Finite class | Source correspondence |
|---|---|---|---|
| Tokens/multiplicity | One complete Q-table entry per pair; repeated transitions remain samples; null source | Same canonical memory; no null writer | Exact local contract; explicit adaptation from trajectory-memory Xie and parameter-query Liang |
| Score/support | Current singleton, successor state support, visited pair writer | ζ state score + log π; ξ pair score; τ write score, full candidate support | Exact local operators; external architectures differ |
| Normalization/axis | Current/expected normalize over memory; writer over transitions/null (dim=0) | Read/successor dim=1; writer column formula | Exact at operator level; sparse expectation is a compiled closed form |
| Values/sign | R+γ expectation−current read, may be negative | Same form using leaky reads | Local exact specification; not value clipping or positive-only TD |
| Timing/memory | All read old Q, add once; scratch reconstructed next call | Same | Exact local spec; no TSM transport |
| Unvisited | Null support enforces zero | Uniform residual average can update | Explicit local finite extension; differs from old manuscript's gate-assisted finite route |
| Successor | Known-policy expectation over all actions of next state | Policy expectation conditional on each state plus inter-state leakage | Explicit change from sampled next action in local old theorem and Liang; not Xie's trajectory shift |
| Policy/parameters | π given as input, no trainable tensors | π enters scores through log; fixed sharpness | Local contract exact; no pretrained emergence evidence |
| Finite/exact | Equality mask is an architectural assumption | Finite scores and leakage, no input-dependent support pruning | Distinct model classes, not interchangeable |
| Dimension/support | Width/complete Q domain known; routed zero handling | Leakage depends on S,m,counts and sharpness | No universal dimension extrapolation |
| Outer loop | Returns Q and diagnostics only | Same | Certificate and policy update external, neither external paper's controller reproduced |

The old local sampled-SARSA construction (equations (3.1)–(3.6)) supplies the routed-memory skeleton. The replacement of the successor singleton by a known-policy expectation and removal of the finite visited gate are explicitly specified by the later FP-ESARSA task. The old theorem alone does not prove those changes; the derivations above establish their operator meaning separately.

FP-COMPOSE-002/claude/evaluate.py:151 defines L12 outside the network; :180 defines the per-state policy rule; :220 repeatedly calls the estimator; :375 applies the external decision. Hence a certified policy-improvement SYSTEM uses these network outputs; the estimator class alone does not return an improved policy.

## 6. Findings, terminology and acceptance

1. **P1, attribution/claim boundary:** Calling these classes direct implementations of Xie Theorem 1 or Liang–Lai Theorem 3.1 is unsupported. Their operators, memory and update conventions differ as shown above. Use “local construction inspired by these papers”, with explicit differences. This is not a discovered error in either external paper.
2. **P2, executable-description mismatch:** model.py:892 says the action expectation head has no mask; FP-ESARSA specifies masked log-softmax. The code directly builds a sparse matrix. Numerical/operator equivalence is valid on normalized positive policies, but a literal all-head softmax execution claim needs this qualification.
3. **P2, ambiguous theorem reference:** model.py:926's “paper Theorem 3.1” should be understood as a reference to the local sampled-SARSA construction, with its later Expected-SARSA adaptation separately cited. It must not be read as Liang–Lai's same-numbered theorem. Record the correction in audit wording; model.py remains unchanged under user instruction.
4. **P1 if generalized, currently explicitly bounded:** finite retrieval adds residual bias, and unvisited handling differs even at infinite sharpness. Do not import exact Bellman fixed-point, coverage or convergence statements without their premises. Existing local FP-ESARSA contract already states the bias boundary; this is not a new invalidation of sealed composition evidence.

Permitted precise name: “本仓库构造的、固定策略下按状态—动作分组同步更新的 Expected-SARSA 注意力算子实现（精确路由版／有限-logit 近似版）”. For the exact route append “动作期望采用与 masked softmax 代数等价的直接权重构造”. For composition: “该实现产生 Q 估计，由外部 L12 证书与按状态保守规则决定策略更新”.

It is unnecessary to prove fidelity to an external algorithm in order to give a NEW explicitly defined operator a mathematical name. The meaningful restriction is attribution and architectural scope. The object's code/specification is determinate; its identity as either paper's particular network is refuted, not merely awaiting confirmation.

H1: supported on the declared domain. H2: a blanket external-paper attribution is not supported; the local theorem lineage resolves but the bare docstring citation is inadequate. H3: differences are accounted for above. Both classes and the shared action head are covered. Audit deliverables can pass with this scientific negative attribution verdict.

Limits: no benchmark reruns; no learned-routing/full-prompt compiler construction; no comprehensive proof audit of external convergence/training theorems; no arbitrary API/CUDA validation; no change to FP-COMPOSE task states or conclusions. Real-number operator algebra does not provide floating-point rigor. Bibliographic verification is not a fresh-download authenticity proof.

Awaiting reciprocal review.
