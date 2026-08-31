# Method and Experiments

## 3. Softmax Policy-Improvement Operators

### 3.1 End-to-end construction and compact experimental operator

Section 3 gives the full token computation in a literal $d\times L$ prompt. Raw transition tokens contain pair identifiers and rewards; two structured-routing retrieval heads copy the required current Q-values from Q-memory tokens; an explicit two-unit ReLU FFN forms the signed TD residual; and a final state--action head writes it back. All $W_Q,W_K,W_V,W_O,W_1,W_2$ matrices are fixed coordinate maps. No residual or current Q-value is supplied as an initial transition-token field.

For efficiency, the large control sweep bypasses the two prompt-level retrieval heads: direct table indexing supplies the same two scalars, the same fixed linear expression forms the residual, and the standard-softmax kernel performs the write-back. It therefore tests the compact write-back/control operator, not execution of the full prompt network. A separate literal matrix witness instantiates the complete prompt, every projection, all query-by-source routing masks, the null-token gate, residual scratch states and the write-back head, then compares the result with both the compact implementation and the tabular reference. The experiments use one-hot state--action identifiers, so neither route tests learned representations or function approximation.

For a visited query x, the compact finite-logit write-back uses

K_eta(k|x)=\frac{\exp(eta\langle psi(X_k),psi(x)\rangle)}{\sum_j\exp(eta\langle psi(X_j),psi(x)\rangle)}

and writes

Q_{l+1}(x)
=Q_l(x)+alpha_l\sum_k K_eta(k|x)delta_k.

The attention weights are positive and normalised, but delta_k is signed; negative TD corrections therefore remain available. Queries absent from the context receive no update. The exact construction in Theorem 3.1 uses externally supplied equality-routing masks and an externally gated zero-value null token for unvisited queries. Removing the equality masks gives the explicit finite-logit approximation analysed jointly with finite-logit Q retrieval in Section 3.8; that approximation still retains the visited-query gate.

### 3.2 Route I: sampled SARSA control

For each batch, draw A'_k from the current behaviour policy at S'_k. The two retrieval heads and the fixed residual map of Section 3 form

delta_k^S
=R_k+gamma Q_l(S'_k,A'_k)-Q_l(X_k).

The state--action head then performs the write-back above. Under exact matching this is precisely

Q_{l+1}(x)
=Q_l(x)+\frac{alpha_l}{n_x}
  \sum_{k:X_k=x}delta_k^S,

the simultaneous per-pair mean mini-batch SARSA update at frozen Q_l. With a single queried transition it is the usual sampled SARSA update. It is not a summed update over repeated visits and not a sequential update in which Q changes within the batch.

To close the control loop, the next behaviour policy is derived from Q_{l+1}. We use an $\varepsilon$-greedy policy in the sampled experiment. The TD target itself contains no maximum: the next action is the action sampled from the current policy. In the theorem and literal witness, Q retrieval, residual formation and Q write-back occur in the constructed blocks. In the large control sweep, retrieval is replaced by direct table indexing and only the compact residual/write-back kernel is run. Environment interaction, exact $\varepsilon$-greedy argmax and random action sampling are external interfaces. Theorem 3.1 is an operator identity, Corollary 3.2 concerns the different singleton online protocol, and the present frozen-batch protocol is evaluated empirically rather than assigned either theorem's conclusion.

### 3.3 Route II: Boltzmann Expected SARSA and approximate greedification

The stronger route adds a standard softmax head over the action tokens at each next state. Its scores are beta Q_l(S'_k,b), its values are Q_l(S'_k,b), and a group mask restricts attention to actions sharing S'_k. The result is

M_beta Q_l(S'_k)=\sum_b\frac{\exp(beta Q_l(S'_k,b))}{\sum_c\exp(beta Q_l(S'_k,c))}Q_l(S'_k,b).

The next linear projection forms

delta_k^beta
=R_k+gamma M_beta Q_l(S'_k)-Q_l(X_k),

and the common state--action head writes this residual to Q-tokens. No exact maximum is used inside the proposed operator. At finite beta the target is the conditional Expected-SARSA target for the current Boltzmann policy. Viewed instead as an approximation to greedy control, its error is bounded by gamma log(|A|)/beta at the Bellman-target level, yielding Theorem 3.6.

### 3.4 Scope of the construction

The exact sampled-SARSA result concerns a two-block residual attention--FFN network with standard exponential normalisation, fixed coordinate projections, bidirectional structured attention, no LayerNorm and no dropout; it is not a decoder-only causal Transformer. Its matrices are fixed for fixed $m,N,\gamma,\alpha$. It starts from raw transitions and persistent Q-memory tokens and does not assume a precomputed TD residual. Exact retrieval and write-back rely on externally supplied, input-dependent equality-routing masks and an external visited-query/null-token gate. Proposition 3.7 removes the equality masks but retains the gate. The claims do not assert that a generic pretrained Transformer discovers the layout or learns routing, that finite logits perform exact content selection, or that simulator action execution and random sampling occur inside a deterministic attention head. A variable step-size schedule requires a family of blocks with different fixed output scales or an additional multiplication mechanism.

## 4. Experiments

### 4.1 Questions, environments, and policy metric

The experiments ask three questions tied directly to the central claim:

1. Does the compact one-stage softmax write-back operator match its sampled-SARSA reference, and does the separate literal witness match that compact operator?
2. Does closing that operator with its behaviour policy improve policy return?
3. Does the two-stage operator retain policy improvement while its internal action aggregation approaches greedy control?

Each task is a finite random MDP. Transition rows and the initial-state distribution are sampled from symmetric Dirichlet distributions, and transition rewards are uniform on [-1,1]. For a policy pi and initial distribution \mu, we compute the discounted return exactly,

J_\mu(pi)
=\mu^T(I-gamma P_pi)^{-1}r_pi,

so policy comparisons contain no rollout-evaluation noise. Unless noted otherwise, reported dispersions are sample standard deviations across independently sampled tasks.

### 4.2 Sampled SARSA: matched write-back and closed-loop improvement

We use 30 MDPs with 9 states, 4 actions, and gamma=0.5. The standard-softmax learner starts from a small random Q-table and uses an $\varepsilon=0.1$-greedy behaviour policy derived from its current Q. At each of 200 control updates, it generates a 128-transition trajectory, freezes that batch's Q and policy, and applies the per-pair mean sampled-SARSA update with eta=12 and

alpha_l=0.35/\sqrt{1+0.02(l-1)}.

The matched reference receives the same transitions and step sizes but uses the exact-match kernel U. It isolates the effect of finite softmax matching; it is not a second independently sampled control run.

**Table 1. Exact policy return for sampled-SARSA control (30 MDPs, mean with sample standard deviation in parentheses).**

| update rule | initial return | final return | return gain | positive J_mu gain |
|---|---:|---:|---:|---:|
| standard-softmax SARSA | 0.104 (0.211) | 0.533 (0.117) | 0.429 (0.190) | 30/30 |
| exact-match write-back | 0.104 (0.211) | 0.532 (0.117) | 0.428 (0.192) | 30/30 |

The two-sided Student-t 95% confidence interval for the softmax learner's mean gain is [0.358,0.500]. Its greedy-policy return rises from 0.110 to 0.587, compared with an optimal return of 0.621. At the final update, its Q-table differs from the shared-trajectory comparator by only 2.19e-4 in infinity norm (SD 5.27e-5), and their greedy policies agree on 0.996 of states (SD 0.020). A same-frozen-Q one-step check has mean discrepancy 1.54e-4; every checked step lies below its finite-kernel bound. The largest observed non-matching attention mass is 7.80e-4. These paired controlled results show positive final J_mu gain on every sampled task and operator fidelity, not statewise policy dominance or monotonicity of every stochastic SARSA step. Indeed, intermediate evaluated returns sometimes decrease.

![](figures/softmax-sarsa-policy-improvement.png)

**Figure 1. Standard-softmax sampled SARSA gives positive final policy-return gain.** Left: mean exact return change across control updates; shading is a 95% confidence interval. Right: final return gain in every matched task, with the shared-trajectory identity comparator overlaid.

### 4.3 Two-stage route: policy gain and action sharpness

The second experiment isolates internal action aggregation from sampling noise. On 20 new MDPs with the same state, action, and discount sizes, every state--action pair is updated from its exact conditional expectation. We start from Q_0=0, use alpha=0.5 and state--action sharpness eta=20, and run 200 synchronous updates. Greedy argmax ties, including the initial all-zero tie, select the lowest-index action. The exact-max kernel operator is an oracle reference; the proposed operator uses beta in {1,2,5,10,20}. Final policies are greedy with respect to the resulting Q-tables using the same rule.

The mean initial policy return is -0.0569 (SD 0.262), while the oracle reaches 0.5209 (SD 0.151), a gain of 0.5777 (SD 0.224). Every softmax temperature gives positive final J_mu gain on all 20 tasks.

**Table 2. Two-stage softmax control after 200 expected updates (means over 20 MDPs).**

| operator | return gain | optimal-return gap | Q infinity error | greedy agreement |
|---|---:|---:|---:|---:|
| exact-max oracle | 0.57774 | 0 | 7.48e-13 | 1.000 |
| softmax, beta=1 | 0.57682 | 9.25e-4 | 0.2465 | 0.972 |
| softmax, beta=2 | 0.57766 | 7.51e-5 | 0.1905 | 0.989 |
| softmax, beta=5 | 0.57766 | 7.51e-5 | 0.08758 | 0.989 |
| softmax, beta=10 | 0.57774 | 0 | 0.03441 | 1.000 |
| softmax, beta=20 | 0.57773 | 8.98e-6 | 0.01131 | 0.994 |

The action-value error falls as beta increases. The observed statewise softmax-to-max gap falls from 0.3618 at beta=1 to 0.02007 at beta=20, below the corresponding bounds 1.386 and 0.06931. Greedy policies are piecewise constant in Q, so policy return can already match the oracle before the action-value error vanishes; the table therefore reports both quantities rather than treating Q error as a substitute for policy improvement.

![](figures/two-stage-temperature.png)

**Figure 2. The two-stage operator gives positive policy-return gain and approaches greedy action selection.** Left: mean return gain with sample-standard-deviation bars; the dashed line is the exact-max oracle. Right: the observed softmax-to-max gap remains below log(|A|)/beta.

### 4.4 End-to-end and finite-sharpness checks

The literal matrix witness uses a $32\times20$ prompt on the deterministic fixture and instantiates every stated projection and full query-by-source mask. Its masks have nonempty support, all softmax rows are finite and normalised, retrieved values and signed residual coordinates match the intended token states, and unvisited queries route to the zero-value null token. Its final Q update differs from both the compact implementation and the explicit frozen-batch tabular reference by $5.55\times10^{-17}$; six successive singleton updates also match sequential tabular SARSA. Eight unvisited pairs are unchanged. This is an implementation witness for the complete construction, distinct from the large control sweep. With deliberately low retrieval and write-back sharpness $(\zeta,\eta)=(2.4,2.1)$, the observed retrieval error is 0.425 below its 0.999 bound, and the complete end-to-end update error is 0.250 below the Proposition 3.7 bound 1.170.

The earlier compact direct formula check gives exact-match per-pair write-back to $1.11\times10^{-16}$. At the control experiment's $\eta=12$, the observed synthetic one-step finite-kernel error is $8.55\times10^{-5}$, below its computed bound $2.00\times10^{-4}$. In the two-stage experiment's 20-update sharpness sweep, reducing non-matching mass from 0.928 at $\eta=1$ to 0.00159 at $\eta=10$ reduces mean Q error from 0.547 to 0.01158; $\eta=20$ changes it only to 0.01158. These checks support the finite-step retrieval and write-back distortion analysis. They are not used to claim that leakage is an unavoidable asymptotic error floor.

## 5. Discussion

### 5.1 Direct answer to the policy-improvement question

Yes, standard normalised softmax attention can support policy improvement in the precise oracle-routed constructive sense established here. In Route I, two equality-routed retrieval heads read the current and sampled next-pair values from Q-memory, a fixed ReLU map forms the signed SARSA residual, and a final matching head performs the per-pair mean write-back. The exact identity assumes the external masks and visited/null gate. Equality-mask-free finite-logit retrieval and write-back approximate their exact counterparts while retaining the gate. Closing the external operator--policy loop gives positive final $J_\mu$ gain in the controlled tasks, whose large sweep uses the compact direct-indexing implementation. In Route II, an action-softmax head supplies the current Boltzmann expected continuation value and provides a controlled approximation to greedy action selection. The empirical claim rests on before-versus-after $J_\mu$, not on decreasing TD error or statewise dominance.

This answer does not require Q-learning. SARSA already becomes a control method when its behaviour policy is updated from its current action values [6]. The maximum-based route is retained only because it proves that standard softmax can also carry out internal approximate greedification and because Bellman optimality supplies a stronger neighbourhood guarantee.

### 5.2 Relation to linear-attention SARSA

Liang and Lai [4] use linear attention to update a shared parameter vector, which requires a sample-dependent product between a TD residual and a feature vector. Our construction stores Q directly at state--action memory tokens. Standard softmax heads retrieve the two required scalar values and later route the resulting signed residual back to the matching memory token; no dynamic residual--feature product is required. The representations differ, but both implement SARSA control updates. Relative to fixed-policy softmax TD constructions [5], persistent state--action memory, explicit Q retrieval and closure of the behaviour-policy loop are the essential changes.

### 5.3 Limitations

The theory is tabular and assumes one persistent memory token per state--action pair. Exact retrieval and write-back use externally supplied, input-dependent structured equality masks and an external visited-query/null-token gate; finite equality-mask-free logits are approximate and remain gate-assisted. The exact network omits LayerNorm and dropout and is not a decoder-only causal Transformer. The sampled experiment freezes Q and the behaviour policy within each batch, directly indexes the two Q-values and uses per-pair mean updates; classical online SARSA convergence instead applies to singleton sequential updates under GLIE and per-pair Robbins--Monro conditions. A fixed block supplies a fixed step size, so a variable schedule requires a block family or another multiplication mechanism. The two-stage expected experiment has full coverage and isolates operator bias rather than finite-trajectory concentration. Exact epsilon-greedy argmax, random policy sampling and environment action execution remain outside the attention blocks. Extending the construction to learned representations must address approximate identifiers, learned or mask-free routing, support gating, coverage and persistent-memory scaling.
