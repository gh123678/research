# Introduction

Can standard normalised softmax attention support policy improvement in-context? Fixed-weight Transformers can execute learning algorithms in their forward pass [1--3], and recent work gives constructive reinforcement-learning examples. Liang and Lai [4] show that linear attention can implement parametric semi-gradient SARSA and actor--critic updates. Xie et al. [5] show that standard softmax attention can implement weighted TD policy evaluation for a fixed policy. Together these two lines leave open whether the latter, normalised attention mechanism can support action-value control rather than evaluation alone.

Policy improvement does not require a maximum in every TD target. SARSA forms the on-policy residual

delta_k^S = r_k + gamma Q(s'_k,a'_k) - Q(s_k,a_k)

using the action a'_k actually selected by the current behaviour policy. Updating Q and then deriving the next behaviour policy from the updated values gives an on-policy control loop [6]. A greedy Bellman backup is a different, stronger route: it places action selection inside the target through max_b Q(s',b). The two routes answer complementary questions and should not be conflated.

We represent each state--action pair x=(s,a) by a persistent Q-memory token that stores Q(x), while a raw transition token contains current- and next-pair identifiers and reward. Two fixed masked heads retrieve the two Q-values required by sampled SARSA, a position-wise map forms the signed residual, and a final standard-softmax head performs scalar write-back. This tokenisation avoids updating a global parameter vector through a sample-dependent product of delta and phi. The resulting write-back is

Q^+(x)=Q(x)+alpha \sum_k K_eta(x_k,x) delta_k,

where K_eta is a normalised state--action matching kernel and the signed TD residual is carried by the value vectors. Positive attention weights therefore do not remove negative TD corrections.

Our first construction proves the full raw-token computation rather than assuming sampled SARSA residuals as inputs. It is a precisely specified two-block residual attention--FFN network with standard exponential softmax, bidirectional structured attention, and no LayerNorm or dropout; it is not a decoder-only causal Transformer. Externally supplied, input-dependent equality-routing masks retrieve Q(s,a) and Q(s',a'), a fixed two-unit ReLU map forms the residual, and an external visited-query/null-token gate completes the simultaneous per-pair mean tabular SARSA write-back. The singleton version generates the usual sequential SARSA iterates; coupling those iterates to a behaviour policy gives the standard on-policy control loop under the usual external environment and sampling interface. When the equality masks are removed, finite logits approximate retrieval and write-back with an explicit one-step bound while the external visited-query gate remains.

Our second construction adds an action-attention stage. For the current Boltzmann policy

p_beta(b|s) = \exp(beta Q(s,b)) / \sum_c \exp(beta Q(s,c)),

the stage returns

M_beta Q(s) = \sum_b p_beta(b|s) Q(s,b).

At finite beta this is exactly the conditional Expected-SARSA target of the current Boltzmann policy. It also approximates greedy action selection, with the uniform bound

0 \le \max_b Q(s,b)-M_beta Q(s) \le \log(|A|)/beta.

The Expected-SARSA and approximate-greedy interpretations are both valid. We use the second interpretation only to derive a stronger optimal-control consequence: in the ideal tabular, synchronous setting, the resulting update is a bounded perturbation of relaxed Bellman optimality. The proof uses the contraction of T^{\star}, not an unproved contraction property of the smooth operator. Kernel mismatch and finite-context error enter as separate perturbations.

Controlled tabular experiments test both control routes through policy behaviour. For scale, the sampled-SARSA sweep uses a compact implementation in which direct table indexing supplies the two Q-values and softmax implements the write-back kernel; it does not execute the literal prompt network. It is compared with a shared-trajectory exact-match write-back comparator, which isolates finite softmax matching without being mislabelled as an independent on-policy learner. A separate literal matrix witness tests the complete prompt construction. The two-stage compact operator is compared with exact greedy control while action sharpness and kernel sharpness are varied. These experiments are designed to answer whether the corresponding softmax operators improve policies.

Our contributions are:

1. We give an end-to-end fixed-weight standard-softmax construction that retrieves current action values from Q-memory tokens, forms sampled-SARSA residuals and writes them back, and we separate its exact externally routed identity from its equality-mask-free but gate-assisted finite-logit approximation.
2. We show that the two-stage finite-temperature target is exactly Boltzmann Expected SARSA and simultaneously a uniformly bounded approximation to greedy action selection.
3. We derive action-value and policy-performance guarantees for the stronger approximate-greedy route, with explicit action-sharpness, kernel and finite-context terms.
4. We evaluate both routes through matched control experiments and policy return.

The result is constructive and deliberately scoped. Exact content selection uses externally supplied, input-dependent structured equality masks and an external visited-query/null-token gate. Finite equality-mask-free logits remain gate-assisted and are approximate. Environment interaction, exact epsilon-greedy argmax and random action sampling remain external. We do not claim that every stochastic TD step monotonically improves a policy, that a generic pretrained Transformer automatically learns the proposed operators or routing, or that the main control sweep executes the full token network. The main guarantees concern finite tabular problems with explicit coverage conditions.
