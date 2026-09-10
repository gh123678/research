# FP-ITER-001: fixed-policy finite-softmax Expected SARSA iteration witness

## Metadata and scope

- Date: 2026-09-10. Author: GPT. Version: 1.1. Status: `ACTIVE`.
- Scientific baseline: `c579047950dfabb2600020cd2e53dd24b3e39c84`.
- Inherited code baseline: `fd0b99826026af6868779f1c27fec4e43325df09`.
- Preserved v1.0 draft/review commit: `3b9d7b0`.
- GPT branch: `codex/FP-ITER-001`; Claude branch: `claude/FP-ITER-001`.
  Both execution routes must start from the same activation commit, to be
  recorded in an execution-start manifest after this version is APPROVED.
- Design: `docs/superpowers/specs/2026-09-10-fixed-policy-softmax-iteration-witness-design.md`.
- Predecessor: CTRL-PREFLIGHT-001 is VERIFIED. FP-ESARSA-001 stays DRAFT.
- Long, conclusion-critical construction task: CPU only, no random draws,
  estimated 30--60 minutes per actor; no large experiment matrix.
- Scope is policy evaluation. No policy improvement, training, concentration
  certificate, online control, parameter search, or claim of publication novelty.

## Question and falsifiable checks

Can explicit fixed-weight normalized softmax attention repeatedly implement
a synchronous batch Expected SARSA update with fixed policy? Which errors
come from attention, finite iteration, and the observed transition batch?

H1: exact grouped attention equals a separately written direct reference.
H2: literal finite attention equals an independently written scalar formula.
H3: signed stage errors telescope to the finite-minus-exact update.
H4: exact full-coverage iteration obeys its contraction bound; missing coverage
is separately analyzed and never assigned a strict full-table contraction.
H5: the frozen finite iterations obey a proved, checked perturbation bound.
A failure of a sufficient contraction test alone is not evidence of divergence.
A negative stability or coverage result is scientifically admissible.

## Frozen inputs and evaluation protocol

All computations: CPU float64. States/actions: 0,1; pair order 00,01,10,11.
No terminal transitions. gamma=0.7, alpha=0.5. Fixed full-support policy:

```text
pi = [[0.75, 0.25], [0.25, 0.75]]
Q0_Z = [[0, 0], [0, 0]]
Q0_A = [[1, -1], [0.5, 0.5]]
```

Audit-only population MDP: deterministic rewards by pair
r=[1,-0.5,0.25,0.75]; next-state probabilities by pair:

```text
P = [[0.75, 0.25], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25]]
```

Complete batch C has these ordered rows (s,a,reward,next_state):

```text
(0,0, 1.00,0)
(0,0, 1.00,1)
(0,1,-0.50,1)
(1,0, 0.25,0)
(1,0, 0.25,1)
(1,1, 0.75,0)
```

Missing batch M is exactly C's first five rows; pair 11 is absent.
These are deterministic admissible batches, not a sampled contiguous
trajectory. Population P and population Q are audit-only and never network
inputs. Their comparison measures this batch's discrepancy, without a
probabilistic or sample-complexity interpretation.

Freeze all three finite sharpness parameters jointly at
(xi,zeta,tau)=(0,0,0) and (8,8,8). Save every step k=0,...,64.
Four direct/exact sequences: C/M times Z/A.
Eight finite sequences: C/M times Z/A times the two sharpness triples.
No early stopping, clipping, tuning, new data, or random seeds.
An algebraic verifier may additionally use basis vectors and malformed inputs;
these are unit checks, not additional research fixtures.

## Three independently checkable operators

Let m=4 canonical memory tokens, one per pair, and N batch rows.
C0[t,y]=1{x_t=y}. S0[t,(u,b)]=1{u_t=u}pi(b|u).
W0[x,t] equals 1/n_x for matching rows when n_x>0, and zero otherwise.
The direct reference must compute grouped residuals without calling an
attention implementation:

```text
d0_t(q) = r_t + gamma sum_b pi(b|u_t) q(u_t,b) - q(x_t)
F0(q) = q + alpha W0 d0(q)
```

The finite operator has NO visited-query gate or content equality mask:

```text
C[t,y] = softmax_y(xi 1{x_t=y})
S[t,(u,b)] = softmax_(u,b)(zeta 1{u_t=u} + log pi(b|u))
W[x,t] = softmax_t(tau 1{x=x_t})
Ff(q) = q + alpha W (r + gamma S q - C q)
```

Indicator products in these equations specify dot products of declared
one-hot identity features, not masks. In particular W has all N eligible rows
even when pair x is absent. Conditional action weights within a state are
exactly pi; report their error as zero up to rounding. Cross-state leakage is
a separate successor-routing error.

The literal witness must define a prompt matrix H and explicit fixed
WQ/WK/WV/WO matrices for ordinary scaled-dot-product normalized softmax,
residual connections, and fixed linear or ReLU feed-forward maps as needed.
Every dynamic Q read and pair write must pass through these matrices.
No externally indexed TD target, hidden oracle copy, or hardcoded output.
Weights may depend on dimensions, frozen policy, gamma, alpha and sharpness;
not on dynamic Q, observed rewards, or audit population transitions.

Allowed interfaces: prompt initialization from the frozen rows, positional
extraction of the four Q-memory outputs, and static token-role/position
attention masks (context reads memory, memory reads context, inactive queries
read a null token). These masks must not depend on observed pair equality.
Exact reference attention may use explicitly declared equality and absent-row
null masks; this is not counted as a finite-network capability.
Repeated application carries only Q between steps, resets scratch fields by
explicit fixed maps, and retains immutable prompt fields. No externally
recomputed targets between steps. No layernorm, dropout, or causal/online claim.
Expose projected queries, keys, values, scores, probabilities, write outputs,
and residuals so literalness is inspectable.

## Error and stability analysis

Derive affine maps Gf=I+alpha W(gamma S-C), bf=alpha W r and G0,b0
from formulas, not a fitted trace. At the SAME input q, use the signed terms:

```text
e_current   = -alpha W(C-C0)q
e_successor = alpha gamma W(S-S0)q
e_write     = alpha(W-W0)(r + gamma S0 q - C0 q)
Ff(q)-F0(q) = e_current + e_successor + e_write
```

Their infinity norms supply a triangle bound, not an equality of norms.
Compute c_f=||Gf||_infinity and spectral radius rho(Gf). A sufficient condition
c_f<1 proves a unique fixed point and geometric convergence; c_f>=1 alone
does not prove instability. Any stronger spectral claim requires justification.

For C, F0 has infinity-norm contraction factor at most
1-alpha(1-gamma)=0.85. Compute its unique empirical fixed point q_hat.
For M the absent coordinate is unchanged by F0; strict full-table contraction
fails. Do not invent unobserved transitions or claim a unique full-table
empirical fixed point. A value conditioned on its preserved coordinate may
be analyzed if clearly labeled.

For each paired trajectory start E_0=0 and check the finite-horizon bound

```text
E_(k+1) = c_f E_k + ||Ff(q_exact_k)-F0(q_exact_k)||_infinity
||q_finite_k-q_exact_k||_infinity <= E_k.
```

This bound remains meaningful when c_f>=1; report looseness.
When C and c_f<1, solve q_f_inf and verify the steady-state bound
||q_f_inf-q_hat|| <= ||Ff(q_hat)-q_hat||/(1-c_f), as well as
||q_finite_k-q_hat|| <= c_f^k ||Q0-q_f_inf|| + ||q_f_inf-q_hat||.
Solve audit population q_pi separately. Where all fixed points are justified,
report the signed decomposition

```text
q_finite_k-q_pi =
(q_finite_k-q_f_inf) + (q_f_inf-q_hat) + (q_hat-q_pi).
```

For singular/noncontractive cases return unavailable values with reasons,
unless a separately justified fixed-point analysis is provided. Never silently
apply an inverse or claim uniqueness. No norm identity from signed telescoping.

## Exact artifact scope and reproducibility

Paths below are relative to icrl_softmax. Each actor creates only:

- docs/research_branches/FP-ITER-001/<actor>/witness.py
- docs/research_branches/FP-ITER-001/<actor>/verify.py
- docs/research_branches/FP-ITER-001/<actor>/theory.md
- docs/research_branches/FP-ITER-001/<actor>/report.md
- docs/research_branches/FP-ITER-001/<actor>/verification_of_other.md
- results/FP-ITER-001/<actor>/ (ignored raw evidence, CLI logs, manifests).

GPT additionally owns this task, the linked design, the task-scoped
implementation plan, pre-review records and synthesis under its document
directory, and ACTIVE_WORKSPACE.md. No shared scientific module is modified.
No changes to FP-ADV, FP-TU, FP-ESARSA, main, or the user's learning records.

From icrl_softmax, record the absolute Python executable and run:

```text
python -B docs/research_branches/FP-ITER-001/<actor>/verify.py
python -B docs/research_branches/FP-ITER-001/<actor>/witness.py
python -m ruff check docs/research_branches/FP-ITER-001/<actor>/witness.py docs/research_branches/FP-ITER-001/<actor>/verify.py
```

witness.py writes strict JSON to stdout, saved as results.json. verify.py
writes check counts and maximum errors, saved as verification.json.
Both scripts must run without importing the other actor's code.
Raw evidence records environment, code commit, input protocol, matrices,
per-step values, errors, fixed points, bounds and unavailable-value reasons.
No NaN/Infinity JSON literals. Report all actual failed runs and fixes.

Absolute tolerance 1e-12 for one-step equivalence and normalized probabilities.
For repeated traces, linear solves, decompositions and bound checks use
1e-10*(1+max_abs(left,right)). Report raw errors as well as pass/fail.
Verify basis-vector affine equality, signed residuals, synchronization,
self-loops/repeats, absent-pair behavior, policy preservation, malformed pi,
nonfinite inputs and duplicate canonical identity rejection.
Schema wrappers may differ but fixture IDs, steps and mathematical quantities
must be unambiguous and cross-reconstructible.

## Roles, blind execution, lifecycle and acceptance

AGENTS.md long-task rule applies: GPT and Claude each independently derive
the construction/proof, implement both scripts, run the fixed protocol, and
seal their first results in their own branch and result directory before
reading the other's implementation or conclusions. GPT keeps the routine
scope small to respect the user's quota preference. The predecessor's
delegated diagnostic does not establish a waiver of this task's dual-route rule.

After both seals, GPT executes and reviews Claude's route; Claude executes
and reviews GPT's route. Verification covers source literalness, formulas,
raw outputs, reproduction, frozen input identity, all H1--H5 checks and claims.
Each actor writes verification_of_other.md ending PASS, FAIL or OBJECTION
with actual executable evidence; text agreement is insufficient.
The task then progresses ACTIVE -> VERIFYING -> VERIFIED only with reciprocal
PASS, all acceptance checks evidenced, and explained discrepancies.
An implementation failure returns to ACTIVE for its original author to fix.
Task-definition OBJECTION stops affected work for user ruling.

A scientific negative result can pass if it is reproducible and interpreted
within the protocol. A false theorem, undeclared oracle, unexplained bound
violation, or unreported failed run cannot pass. Stop rather than retune the
fixture or policy. If CLI/permissions/quota fails, record the blocker and the
exact continuation; never substitute a Codex subagent for Claude.
No push, merge or main-branch change is authorized.

## Review history and current next step

v1.0 was created as DRAFT and moved to REVIEW after the user approved
continuation. Its historical Claude CLI summary is
docs/research_branches/FP-ITER-001/codex/claude_review.md (APPROVED).
That approval concerned the earlier incomplete draft; it is not v1.1 approval.

GPT's prior claim that no placeholders remained was incorrect: complete rows,
Q0, steps, sharpness, coverage-dependent claims and exact actor/file scope
were not frozen. v1.1 supplies those details before any execution, keeping the
approved fixed-policy iteration scope. No experiment results informed them.
The historical claim that no callable Claude CLI exists is obsolete.
The earlier permission/quota branch blocker is resolved: codex/FP-ITER-001
was created successfully, with prior records preserved at 3b9d7b0.
Three unrelated user documents remain untouched and untracked.

Claude's read-only v1.1 pre-review returned APPROVED on 2026-09-10; full report:
docs/research_branches/FP-ITER-001/codex/claude_review_v11.md. The protocol is
now ACTIVE. This activation commit is the common execution baseline; its full
hash is recorded in each actor's execution-start manifest before implementation.
No research protocol changed after review. Existing CLI metadata reports
kimi-k2.6/k3; no model-provider identity is inferred from the CLI brand.

Execution record: common baseline 6d2376968aeb9379c978bd1a7af7929b70fdeb09.
Codex source/theory is cb26a339b3a9160cc508f0558c507786e79b8304; its bounded
protocol and 6,332 self-checks pass. Codex report.md records preliminary
results and all limitations, including zero C population/data discrepancy.
Claude's isolated branch is created at the common baseline, but its execution
launch was rejected by automatic approval review over private payload transfer
authorization and broad permissions. No execution process started. Read-only
inspection identifies the existing destination as HTTPS api.kimi.com:443.
This is an operational approval blocker, not a task-definition OBJECTION.
Task remains ACTIVE pending explicit scoped approval and independent execution.
No reciprocal verification or VERIFIED claim. Continuation: Codex handoff.md.

User ruling, 2026-09-10: in direct response to the explicit request to transmit
this task's governance files, task/design, related private source and results
to the existing Kimi backend (api.kimi.com), for Claude Code independent
execution in its isolated branch and reciprocal verification, the user replied
"授权". This authorizes those scoped transfers and execution/verification
continuations using the existing configuration, resolving the prior approval
blocker. It does not authorize unrelated payloads, provider changes, new fee
categories, main merge or push. Scientific v1.1 protocol and common baseline
are unchanged; Claude's blind phase receives only that baseline.
