# End-to-End Construction and Guarantees

### 3.1 Exact model class and complete prompt

Let $\mathcal X=\mathcal S\times\mathcal A$, let $m=|\mathcal X|$, and fix an enumeration of $\mathcal X$. Pair $x$ has one-hot identifier $e_x\in\mathbb R^m$. One update receives a frozen table $Q_l$, fixed scalars $\gamma\in[0,1)$ and $\alpha>0$, and $N$ sampled transitions

$(S_k,A_k,R_k,S'_k,A'_k),\qquad k=1,\ldots,N,$

where $A'_k$ is the next action actually selected by the current behaviour policy. Write $X_k=(S_k,A_k)$, $X'_k=(S'_k,A'_k)$ and $n_x=\sum_{k=1}^N\mathbf 1\{X_k=x\}$.

We prove the result for a precisely specified residual attention network. It has two attention--FFN blocks, standard exponential softmax, no LayerNorm and no dropout. Tokens are columns. It uses bidirectional structured attention and is not a decoder-only causal Transformer. The projection matrices are fixed once $m,N,\gamma,\alpha$ are fixed. The structured equality-routing masks are externally supplied and input-dependent. An external support gate supplies the visited-query/null-token rule from the discrete pair identifiers in the sampled batch. Thus the theorem is an exact oracle-routed reference construction. It does not prove or assert that a generic pretrained Transformer learns or discovers routing.

Take token width $d=2m+8$ and the direct-sum convention

$$
\mathbb R^d
=\underbrace{\mathbb R^3}_{\text{type}}
\oplus\underbrace{\mathbb R^m}_{\text{current ID}}
\oplus\underbrace{\mathbb R^m}_{\text{next ID}}
\oplus\underbrace{\mathbb R^5}_{(r,q,u,v,\delta)}.
$$

Let $\tau_Q,\tau_T,\tau_Z\in\mathbb R^3$ be the three type basis vectors. The scalar coordinates store reward $r$, persistent memory $q$, retrieved current value $u$, retrieved next value $v$, and TD residual $\delta$. Define

$$
\begin{aligned}
M_x&=(\tau_Q,e_x,0_m,0,Q_l(x),0,0,0)^\top,\\
T_k&=(\tau_T,e_{X_k},e_{X'_k},R_k,0,0,0,0)^\top,\\
Z&=(\tau_Z,0_m,0_m,0,0,0,0,0)^\top,\\
H^{(0)}
&=[M_{x_1},\ldots,M_{x_m},T_1,\ldots,T_N,Z]
\in\mathbb R^{d\times L},\qquad L=m+N+1 .
\end{aligned}
\tag{3.1}\label{eq:prompt}
$$

The null token $Z$ is not the zero vector because it carries its type coordinate, but every coordinate read as an attention value below is zero. In particular, a raw transition token contains neither $Q_l(X_k)$, $Q_l(X'_k)$ nor a precomputed residual.

Let $P_c,P_n\in\mathbb R^{m\times d}$ select the current- and next-ID blocks. Let $\mathbf e_r,\mathbf e_q,\mathbf e_u,\mathbf e_v,\mathbf e_\delta\in\mathbb R^d$ be basis columns for the five scalar coordinates. For a head with $W_Q,W_K\in\mathbb R^{d_h\times d}$, $W_V\in\mathbb R^{d_v\times d}$ and $W_O\in\mathbb R^{d\times d_v}$, define query-by-source weights

$$
A_{ij}
=\frac{\exp\!\left((W_Qh_i)^\top(W_Kh_j)/\sqrt{d_h}+\mathcal M_{ij}\right)}
{\sum_{s=1}^{L}\exp\!\left((W_Qh_i)^\top(W_Kh_s)/\sqrt{d_h}+\mathcal M_{is}\right)} .
$$

In matrix form the head output is $W_O(W_VH)A^\top\in\mathbb R^{d\times L}$. Every mask row below has nonempty support. Non-target queries are routed to $Z$, so no row is an all-$-\infty$ softmax.

**Theorem 3.1 (exact sampled-SARSA under structured equality routing).** For fixed $m,N,\gamma,\alpha$ and the supplied equality-routing masks and visited/null support rule below, the two-block residual attention--FFN network maps the raw prompt in (3.1) to Q-memory satisfying, for every $x\in\mathcal X$,

$$
Q_{l+1}(x)=
\begin{cases}
Q_l(x)+\dfrac{\alpha}{n_x}\displaystyle\sum_{k:X_k=x}
[R_k+\gamma Q_l(X'_k)-Q_l(X_k)],&n_x>0,\\[3mm]
Q_l(x),&n_x=0.
\end{cases}
$$

### 3.2 Proof of Theorem 3.1: retrieval and residual formation

**Step 1: current-pair retrieval.** The first head uses

$$
\begin{gathered}
W_Q^{\rm cur}=P_c,\qquad W_K^{\rm cur}=P_c,\qquad
W_V^{\rm cur}=\mathbf e_q^\top,\qquad W_O^{\rm cur}=\mathbf e_u,\\
\mathcal M^{\rm cur}_{ij}=0
\quad\Longleftrightarrow\quad
\bigl(i=T_k,\ j=M_{X_k}\bigr)
\ \text{or}\ 
\bigl(i\notin\{T_1,\ldots,T_N\},\ j=Z\bigr).
\end{gathered}
\tag{3.2}\label{eq:block1-current}
$$

All other mask entries are $-\infty$. For transition query $T_k$, the only admitted source is $M_{X_k}$. Hence its softmax row has weight one at that source, independently of the numerical logit, and

$$
W_O^{\rm cur}\sum_jA^{\rm cur}_{T_kj}W_V^{\rm cur}h_j
=\mathbf e_u\,\mathbf e_q^\top M_{X_k}
=Q_l(X_k)\mathbf e_u .
$$

A Q-memory or null query admits only $Z$; since $\mathbf e_q^\top Z=0$, its head output is zero.

**Step 2: next-pair retrieval.** The second head is parallel to the first and reads the next-ID block only at its query:

$$
\begin{gathered}
W_Q^{\rm next}=P_n,\qquad W_K^{\rm next}=P_c,\qquad
W_V^{\rm next}=\mathbf e_q^\top,\qquad W_O^{\rm next}=\mathbf e_v,\\
\mathcal M^{\rm next}_{ij}=0
\quad\Longleftrightarrow\quad
\bigl(i=T_k,\ j=M_{X'_k}\bigr)
\ \text{or}\ 
\bigl(i\notin\{T_1,\ldots,T_N\},\ j=Z\bigr),\\
W_O^{\rm next}\sum_jA^{\rm next}_{T_kj}W_V^{\rm next}h_j
=Q_l(X'_k)\mathbf e_v .
\end{gathered}
\tag{3.3}\label{eq:block1-next}
$$

Again all unspecified entries are $-\infty$, and every non-transition query receives zero from the head. Both heads read the same $H^{(0)}$, as required for parallel multi-head attention. Their output projections write to disjoint coordinates, so the residual state after the attention sublayer is

$$
\begin{aligned}
H^{(a)}&=H^{(0)}+O^{\rm cur}+O^{\rm next},\\
M_x^{(a)}&=M_x,\qquad Z^{(a)}=Z,\\
T_k^{(a)}
&=(\tau_T,e_{X_k},e_{X'_k},R_k,0,
Q_l(X_k),Q_l(X'_k),0)^\top .
\end{aligned}
$$

**Step 3: exact position-wise residual map.** Set

$$
\begin{gathered}
g^\top=\mathbf e_r^\top+\gamma\mathbf e_v^\top-\mathbf e_u^\top,\qquad
W_1=\begin{bmatrix}g^\top\\-g^\top\end{bmatrix}\in\mathbb R^{2\times d},\\
W_2=\mathbf e_\delta\begin{bmatrix}1&-1\end{bmatrix}
\in\mathbb R^{d\times2},\qquad
W_2\operatorname{ReLU}(W_1h)=\mathbf e_\delta g^\top h .
\end{gathered}
\tag{3.4}\label{eq:residual-ffn}
$$

The last identity follows from $\operatorname{ReLU}(z)-\operatorname{ReLU}(-z)=z$. Consequently the first block's FFN residual gives

$$
H^{(1)}=H^{(a)}+W_2\operatorname{ReLU}(W_1H^{(a)}),
\qquad
\delta_k^S=R_k+\gamma Q_l(X'_k)-Q_l(X_k).
$$

For Q-memory and null tokens, the $r,u,v$ coordinates are zero, so the FFN output is zero. Thus Block I preserves every identifier, reward and Q-memory value, while it writes exactly the two retrieved values and the signed sampled-SARSA residual to each transition token.

### 3.3 Proof of Theorem 3.1: write-back and assembly

**Step 4: complete write-back head.** The second block has one active head:

$$
\begin{gathered}
W_Q^{\rm wr}=P_c,\qquad W_K^{\rm wr}=P_c,\qquad
W_V^{\rm wr}=\mathbf e_\delta^\top,\qquad
W_O^{\rm wr}=\alpha\mathbf e_q,\\
\mathcal M^{\rm wr}_{ij}=0
\quad\Longleftrightarrow\quad
\begin{cases}
i=M_x,\ j=T_k,\ X_k=x,&n_x>0,\\
i=M_x,\ j=Z,&n_x=0,\\
i\in\{T_1,\ldots,T_N,Z\},\ j=Z.&
\end{cases}
\end{gathered}
\tag{3.5}\label{eq:write-back}
$$

All other entries are $-\infty$. For a visited memory query $M_x$ and every admitted source $T_k$,

$$
\frac{(P_cM_x)^\top(P_cT_k)}{\sqrt m}
=\frac{e_x^\top e_{X_k}}{\sqrt m}
=\frac1{\sqrt m}.
$$

The $n_x$ admitted logits are therefore equal, so their weights are exactly $1/n_x$. Since the value projection selects the signed scalar $\delta_k^S$, the head output at $M_x$ is

$$
\frac{\alpha}{n_x}\sum_{k:X_k=x}\delta_k^S\,\mathbf e_q .
$$

Positive softmax weights do not remove negative TD corrections: the sign resides in the value coordinate, not in the attention weight. If $n_x=0$, the only source is $Z$ and $\mathbf e_\delta^\top Z=0$, so the output is zero. Transition and null queries also read only $Z$ and remain unchanged.

**Step 5: assemble the two blocks.** Set every unused projection block to zero and set the second block's FFN to zero. Relative to the initial states in (3.1), the complete token-state changes are

| token | after Block I attention | after Block I FFN | after Block II |
|---|---|---|---|
| $M_x$ | unchanged | unchanged | set $q=Q_{l+1}(x)$ |
| $T_k$ | set $u=Q_l(X_k)$ and $v=Q_l(X'_k)$ | set $\delta=\delta_k^S$ | unchanged |
| $Z$ | unchanged | unchanged | unchanged |

Every coordinate not listed in the table remains exactly as it was in $H^{(0)}$.

Reading the Q-memory coordinate of $H^{(2)}$ gives

$$
Q_{l+1}(x)=
\begin{cases}
Q_l(x)+\dfrac{\alpha}{n_x}\displaystyle\sum_{k:X_k=x}
\left[R_k+\gamma Q_l(X'_k)-Q_l(X_k)\right],&n_x>0,\\[3mm]
Q_l(x),&n_x=0.
\end{cases}
\tag{3.6}\label{eq:final-update}
$$

This is exactly the theorem statement. Every displayed matrix is a fixed coordinate selector or fixed scalar multiple for the declared $m,N,\gamma,\alpha$; all data dependence is confined to the prompt and supplied masks. Both Q-lookups, residual formation and Q write-back are therefore included in the two-block computation. $\square$

**Exact scope of the identity.** Equation (3.6) is a simultaneous per-pair mean update at frozen $Q_l$. It is not the global-batch semi-gradient convention

$$
Q_l(x)+\frac{\alpha}{N}\sum_{k:X_k=x}\delta_k^S,
$$

and repeated occurrences of a pair are not processed sequentially within a frozen batch. A singleton call has $n_x=1$ and does recover the usual sequential tabular SARSA step. The construction requires input-dependent equality routing and external visited/null support detection; neither is a causal mask learned by softmax. It does not show that a generic pretrained Transformer discovers the layout. The exact network is fixed for fixed $m,N,\gamma,\alpha$; a varying Robbins--Monro step size requires the corresponding family of output projections or an additional multiplication mechanism. At each control round we rebuild the prompt, or equivalently clear and reinitialize the $u,v,\delta$ scratch coordinates. Section 3.8 removes the equality masks at finite logits but still retains the visited-query gate.

### 3.4 From an update operator to control

**Corollary 3.2 (sequential SARSA equivalence).** If one sampled transition is processed per control round and only its visited pair receives a non-null write-back query, the construction instantiated with step size $\alpha_t$ produces

Q_{t+1}(S_t,A_t)
=Q_t(S_t,A_t)
+\alpha_t[R_{t+1}+\gamma Q_t(S_{t+1},A_{t+1})-Q_t(S_t,A_t)],

with all other state--action values unchanged. A sequence of blocks whose fixed output scales are the prescribed $\alpha_t$ therefore generates exactly the tabular sequential sampled-SARSA iterates. A single fixed block corresponds to a constant step size.

For a finite tabular MDP with bounded rewards, the sequential construction inherits the classical almost-sure convergence result under infinite state--action visitation, per-pair Robbins--Monro step sizes and a greedy-in-the-limit with infinite exploration (GLIE) policy sequence [6]. The frozen mini-batch experiment in Section 5.2 uses a different update convention and fixed exploration, so that convergence theorem is not asserted for the experimental protocol.

An individual SARSA residual is not a policy-improvement theorem. The usual exact policy-improvement bridge is logically separate. If $\pi$ is $\varepsilon$-soft, $Q^\pi$ has been evaluated exactly, and $\pi'$ is $\varepsilon$-greedy with respect to $Q^\pi$, then writing $\pi=\varepsilon/|\mathcal A|+(1-\varepsilon)\widetilde\pi$ gives

\sum_a\pi'(a|s)Q^\pi(s,a)
=\frac{\varepsilon}{|\mathcal A|}\sum_aQ^\pi(s,a)
+(1-\varepsilon)\max_aQ^\pi(s,a)
\ge V^\pi(s).

Hence $V^{\pi'}\ge V^\pi$ by the policy-improvement theorem. With an estimate $\widehat Q$ satisfying $\|\widehat Q-Q^\pi\|_\infty\le\xi$, the worst-case statement weakens to $V^{\pi'}\ge V^\pi-2\xi/(1-\gamma)$; monotonicity requires a sufficient advantage margin. Theorem 3.1 establishes representational equivalence, Corollary 3.2 supplies a conditional online convergence route, and the frozen-batch experiment supplies empirical before--after policy evidence. These are three distinct claims.

### 3.5 Internal Boltzmann action attention

The sampled construction conditions on the next action $A'_k$ supplied by the behaviour interface. Standard softmax can also form a Boltzmann behaviour distribution internally. For $\beta>0$, group the Q-memory tokens by state and use score $\beta Q_l(s,b)$ for action token $b$. The action head returns

p_\beta(b|s;Q_l)
=\frac{\exp(\beta Q_l(s,b))}{\sum_c\exp(\beta Q_l(s,c))}.

An external sampler may draw $A'\sim p_\beta(\cdot|S';Q_l)$ and append it to a raw transition, after which Theorem 3.1 implements sampled SARSA. Alternatively, using $Q_l(s,b)$ as the action-head values gives the internal conditional expectation

M_\beta Q_l(s)
=\sum_b p_\beta(b|s;Q_l)Q_l(s,b).

The associated conditional Bellman target is

(T_\beta Q)(s,a)
=\mathbb E[R+\gamma M_\beta Q(S')\mid s,a].

**Proposition 3.3 (Expected-SARSA identity).** If $A'$ conditional on $S'$ is drawn from $p_\beta(\cdot|S';Q)$, then

\mathbb E[R+\gamma Q(S',A')\mid s,a]=(T_\beta Q)(s,a).

**Proof.** Conditional on $S'$, the inner expectation over $A'$ is $\sum_b p_\beta(b|S';Q)Q(S',b)=M_\beta Q(S')$. Taking the remaining conditional expectation proves the identity. $\square$

At finite $\beta$, this is exactly the conditional Expected-SARSA operator for the Boltzmann policy tied to the current Q-values. It can also be analysed as approximate greedification; finite $\beta$ is not exact Q-learning.

### 3.6 Softmax approximation of greedy action selection

**Lemma 3.4 (entropy bound for softmax action aggregation).** For every $q\in\mathbb R^d$ and $\beta>0$,

0\le\max_aq_a-\sum_a\operatorname{softmax}(\beta q)_aq_a
\le\frac{\log d}{\beta}.

**Proof.** Let $p=\operatorname{softmax}(\beta q)$ and define $\operatorname{LSE}_\beta(q)=\beta^{-1}\log\sum_a\exp(\beta q_a)$. The Gibbs entropy identity gives $\operatorname{LSE}_\beta(q)=\sum_ap_aq_a+H(p)/\beta$. Since $\max_aq_a\le\operatorname{LSE}_\beta(q)$, $\sum_ap_aq_a\le\max_aq_a$ and $0\le H(p)\le\log d$, the result follows. $\square$

**Corollary 3.5 (greedy-target deviation).** For every $Q$,

\|T_\beta Q-T^\star Q\|_\infty
\le\frac{\gamma\log|\mathcal A|}{\beta},

where

(T^\star Q)(s,a)=\mathbb E[R+\gamma\max_bQ(S',b)\mid s,a].

### 3.7 Stronger guarantee for the approximate-greedy route

Consider the synchronous full-coverage update with exact state--action matching,

Q_{l+1}=(1-\alpha)Q_l+\alpha T_\beta Q_l,

where $0<\alpha\le1$, and define $\rho=1-\alpha(1-\gamma)$.

**Theorem 3.6 (bounded perturbation of Bellman optimality).** The iterates satisfy

\|Q_{l+1}-Q^\star\|_\infty
\le\rho\|Q_l-Q^\star\|_\infty
+\frac{\alpha\gamma\log|\mathcal A|}{\beta}.

Consequently,

\limsup_{l\to\infty}\|Q_l-Q^\star\|_\infty
\le\frac{\gamma\log|\mathcal A|}{\beta(1-\gamma)}.

**Proof.** Add and subtract $T^\star Q_l$. The relaxed identity contributes $1-\alpha$, the contraction of $T^\star$ contributes $\alpha\gamma$, and Corollary 3.5 contributes the additive target deviation. Iterating the resulting scalar recurrence gives the limit-superior bound. The proof does not assume that $T_\beta$ itself is a contraction. $\square$

If $\beta_l\to\infty$ with depth, the additive term tends to zero and the same stable scalar recurrence yields convergence to $Q^\star$. Fixed $\beta$ gives a controlled neighbourhood rather than exact convergence.

### 3.8 Finite-logit deviations without equality masks

The exact theorem delegates pair equality to masks. We now remove the two retrieval masks and the write-back equality mask, while retaining the external visited-query gate. The result is therefore equality-mask-free but gate-assisted.

For retrieval sharpness $\zeta>0$, scale the one-hot projections so that the correct Q-memory logit is $\zeta$ and every incorrect logit is zero. Softmax is now taken over all $m$ Q-memory tokens. Its correct mass, total off-target mass, and induced residual error obey

$$
\begin{gathered}
p_\zeta=\frac{e^\zeta}{e^\zeta+m-1},\qquad
\lambda_R=1-p_\zeta=\frac{m-1}{e^\zeta+m-1},\\
|\widetilde Q_l(x)-Q_l(x)|
\le\lambda_R\operatorname{span}(Q_l),\\
|\widetilde\delta_k^S-\delta_k^S|
\le(1+\gamma)\lambda_R\operatorname{span}(Q_l)
=:E_R ,
\end{gathered}
\tag{3.7}\label{eq:finite-retrieval}
$$

where $\operatorname{span}(Q_l)=\max_xQ_l(x)-\min_xQ_l(x)$. The first value bound holds because a probability mass $\lambda_R$ is moved from the correct table entry to values lying within the table span. The residual uses one current and one discounted next retrieval, producing the factor $1+\gamma$.

For write-back sharpness $\eta>0$, use logit $\eta\langle e_{X_k},e_x\rangle$. For a visited query $x$, the finite attention kernel and the exact group-mean kernel are

$$
\begin{gathered}
K_\eta(k|x)
=\frac{\exp(\eta\mathbf1\{X_k=x\})}
{n_xe^\eta+N-n_x},\qquad
U(k|x)=\frac{\mathbf1\{X_k=x\}}{n_x},\\
\varepsilon_W(x)
:=\sum_{k=1}^N|K_\eta(k|x)-U(k|x)|
=\frac{2(N-n_x)}{n_xe^\eta+N-n_x}.
\end{gathered}
\tag{3.8}\label{eq:finite-write-back}
$$

To verify the last identity, matching tokens contribute total absolute difference
$(N-n_x)/(n_xe^\eta+N-n_x)$ and nonmatching tokens contribute the same amount.

**Proposition 3.7 (finite-logit end-to-end update error).** Suppose $|\delta_k^S|\le B$ for every transition. For every visited query,

$$
|Q_{l+1}^{\mathrm{soft}}(x)-Q_{l+1}^{\mathrm{exact}}(x)|
\le\alpha[E_R+B\varepsilon_W(x)].
\tag{3.9}\label{eq:finite-end-to-end-error}
$$

**Proof.** Write the finite update with approximate residuals as
$\alpha\sum_kK_\eta(k|x)\widetilde\delta_k^S$ and the exact update as
$\alpha\sum_kU(k|x)\delta_k^S$. Add and subtract
$\alpha\sum_kK_\eta(k|x)\delta_k^S$. Since $K_\eta(\cdot|x)$ is a probability vector, (3.7) bounds the residual-replacement term by $\alpha E_R$. Since $|\delta_k^S|\le B$, (3.8) bounds the kernel-replacement term by $\alpha B\varepsilon_W(x)$. Summing the two bounds proves (3.9). $\square$

For a frozen policy and frozen $Q_l$, define the finite-context residual deviation

\varepsilon_D=\max_{x:n_x>0}\left|\frac1{n_x}\sum_{k:X_k=x}\delta_k^S-[(T^\pi Q_l)(x)-Q_l(x)]\right|.

The implemented update differs from the corresponding visited population update by at most $\alpha(E_R+B\varepsilon_W+\varepsilon_D)$. In the approximate-greedy full-coverage recurrence, these terms enter additively:

\|Q_{l+1}-Q^\star\|_\infty\le\rho\|Q_l-Q^\star\|_\infty+\alpha\left[\frac{\gamma\log|\mathcal A|}{\beta}+E_R+B\varepsilon_W+\varepsilon_D\right].

This is a one-step operator-deviation statement. It does not turn an arbitrary finite trajectory into a global Bellman recursion, and it does not supply a concentration rate without an independent-sampling, mixing or martingale assumption. Retrieval sharpness, write-back sharpness and action sharpness control three different operations.

### 3.9 Policy consequence

**Corollary 3.8.** If $\|Q-Q^\star\|_\infty\le\epsilon$ and $\pi_Q$ is greedy with respect to $Q$, then

\|V^\star-V^{\pi_Q}\|_\infty
\le\frac{2\epsilon}{1-\gamma}.

Thus the approximate-greedy action-value guarantee implies a conditional policy-performance guarantee. The sampled-SARSA route instead obtains exact operator equivalence from Theorem 3.1, a conditional asymptotic result from Corollary 3.2, and empirical final-return evidence under the separate frozen-batch protocol.
