# FP-ITER-001 independent Codex construction and analysis

Frozen protocol: task v1.1 at 6d2376968aeb9379c978bd1a7af7929b70fdeb09.
This proof and implementation were constructed without seeing Claude's route.

## Literal network

Use row tokens H in R^((4+N+1) x 15). Rows 0..3 are canonical pair memories,
rows 4..4+N-1 are observed transitions, and the last row is a zero null token.
Columns 0..3 hold pair one-hots, 4..5 memory state one-hots, 6..7 successor
state one-hots, 8 log pi on memory, 9 a constant one on non-null tokens,
10 the observed reward on context, 11 Q on memory, and 12..14 scratch u,v,d.
Only the four initialized memory rows contain Q. P and q_pi are never inputs.

Each head computes Qh=H WQ, Kh=H WK, Vh=H WV, A=softmax(Qh Kh^T/sqrt(dk)
with the declared role mask), and H <- H+A Vh WO. All matrices and the full
first-step projected tensors are saved in JSON. Current/read and write have
dk=4; successor has dk=3. The sqrt(dk) factor is explicitly absorbed in WQ.

Current head: WQ selects pair times xi*sqrt(4), WK selects pair, WV selects Q,
WO writes to u. Context reads all four memories; all other queries read null.
Its eligible scores are exactly xi times the pair dot product.

Successor head: WQ selects next-state times zeta*sqrt(3) and constant sqrt(3);
WK selects memory state and log pi. WV selects Q and WO writes to v. The
eligible scores are zeta times the state dot product plus log pi. Therefore
within each state the action weights equal pi exactly in real arithmetic.

The signed residual uses two fixed ReLU units: g=r+gamma*v-u,
d=ReLU(g)-ReLU(-g). This is valid for both signs, with no clipping. Memory
and null scratch remain zero. Writer WQ selects pair times tau*sqrt(4), WK
selects context pair, WV selects d, and WO writes alpha*d into the Q column.
Each memory query reads all N contexts, including an absent pair query. Other
queries read null. Thus the residual connection yields Ff(q) exactly.

A final fixed diagonal projection zeros u,v,d; all immutable fields and Q are
retained. The next call receives this H directly, without indexing Q to create
targets, rebuilding context targets, or consulting audit values. Output
extraction is the declared positional selection H[0:4,Q]. Role masks are a
static interface, not learned routing; there is no causal/online claim.

For the exact reference only, replace role masks with current-pair equality,
successor-state equality and matching-context masks. Use zero sharpness and
the log pi term. Missing writer rows read null. These explicit content masks
are external reference interfaces, never attributed to the finite network.
This gives C0,S0,W0, hence exactly the scalar synchronous batch operator.

## Affine map and stage errors

All attention scores are independent of q, so Gf=I+alpha W(gamma S-C) and
bf=alpha W r. This is derived from the network, not estimated from traces.
Add and subtract alpha W(gamma S0-C0)q and alpha W(r+gamma S0q-C0q):

Ff(q)-F0(q) = -alpha W(C-C0)q + alpha gamma W(S-S0)q
             + alpha(W-W0)(r+gamma S0q-C0q).

The three signed vectors telescope; their norms merely bound the total norm.
The action distribution conditional on state is exactly pi, so it contributes
no fourth approximation term. The two-state routing mass is exp(zeta)/(1+exp(zeta))
on the intended state and 1/(1+exp(zeta)) on the other state.

## Exact iteration and coverage

For C every W0 row averages observations of that pair; W0 C0=I. Therefore
G0=(1-alpha)I+alpha gamma T_hat with T_hat=W0 S0 nonnegative and row-stochastic.
Its infinity norm is exactly 1-alpha(1-gamma)=0.85. Banach contraction gives
a unique q_hat and ||F0^k(q0)-q_hat|| <= .85^k ||q0-q_hat||.

For M, row 11 of W0 is zero, so F0(q)[11]=q[11] and the corresponding G0 row
is an identity row. Its infinity norm is one; there is no strict full-table
contraction. Fixing q[11] leaves a contractive observed-coordinate problem
(observed submatrix norm at most .85), so a family of fixed points is indexed
by the preserved coordinate. We do not return one as a unique empirical q_hat.

## Finite iteration bounds

Let c=||Gf||_infinity and epsilon_k=||Ff(q_exact_k)-F0(q_exact_k)||_infinity.
Subtract the two recurrences and use the induced-norm inequality:
||q_finite_(k+1)-q_exact_(k+1)|| <= c ||q_finite_k-q_exact_k||+epsilon_k.
Induction proves the task's E_k bound for every finite k, including c>=1.
This is a deterministic audit bound, not a computable policy safety certificate.

If c<1, the affine map has a unique fixed point q_inf and iterates converge
geometrically. Comparing that equation with q_hat gives
||q_inf-q_hat|| <= ||Ff(q_hat)-q_hat||/(1-c). Add
c^k||q0-q_inf|| for the empirical-distance transient bound.
If c>=1 we do not infer divergence or use these steady-state bounds. The
reported spectral radius is numerical evidence and is not itself a proof of
any stronger claim. No unqualified inverse is taken in these cases.

At zero sharpness all current rows are uniform, all successor rows equal
pi.flatten()/2 and every writer row is uniform. Thus Gf=I+1*v^T with
v^T 1=alpha*(gamma-1)=-0.15. Its eigenvalues are .85 and 1 (multiplicity 3).
Every update adds the same scalar to all coordinates, preserving all pairwise
differences. The scalar update contracts by .85, so each initial point approaches
some fixed point, with no common unique limit. This proves that a failed
infinity-norm contraction test need not imply divergence.

## Data discrepancy is separate

For C, the audit population point solves (I-gamma T_pi)q_pi=r. Its difference
from q_hat is due to these particular deterministic transition frequencies,
not finite attention. Where c<1 and C applies, the exact signed identity is
q_finite_k-q_pi=(q_finite_k-q_inf)+(q_inf-q_hat)+(q_hat-q_pi).
No claim about a random sampling distribution or sample complexity follows.
Missing coverage cannot be repaired by normalization of W; finite writer
leakage into pair 11 is not evidence of having observed its transitions.

Numerical applicability and limitations are recorded in report.md. This is
an independent preliminary derivation until reciprocal verification succeeds.
