# Fixed-policy finite-softmax Expected SARSA iteration witness

## Purpose

This design isolates the next scientific question after `CTRL-PREFLIGHT-001`:
can a standard normalized softmax attention construction execute and repeat a
fixed-policy Expected SARSA evaluation update on a small finite MDP?

The policy is frozen throughout evaluation. The output of this task is an
evaluation witness and an error decomposition, not a policy-improvement claim.

## Scope

The witness has one canonical Q-memory token for every state-action pair and
four explicit stages: current-pair read, fixed-policy successor-action
expectation, signed Expected SARSA residual, and synchronous pair writeback.
It must expose the exact grouped route, the finite-logit route, and a direct
tabular reference. The same frozen policy and transition data are used for a
short deterministic iteration sequence.

The finite route uses ordinary finite scores and normalized softmax. Its
state-selection, current-read, successor-action, and pair-write leakage are
reported separately. The task must state exactly which masks, if any, are part
of the exact reference route and must not describe an externally supplied
index or mask as a learned attention capability.

The iteration comparison reports, at every step, the direct reference value,
exact attention value, finite attention value, their one-step discrepancies,
and distance to the fixed-policy Bellman solution. A deterministic fixture
must include a self-loop, repeated visits, an unvisited pair, and a successor
state whose two actions have unequal values.

## Out of scope

This task does not implement relative-softmax policy improvement, an action-gap
certificate, a held-out concentration bound, a learned routing mechanism,
online policy changes, a formal 480-record matrix, or a publication claim. The
existing `FP-ESARSA-001` draft remains unchanged and is not activated by this
design.

## Decision gate

The route passes only if:

1. the exact grouped attention output equals the direct Expected SARSA
   operator and repeated exact iterations use synchronous updates;
2. every finite attention output is finite, normalized, and reproducible from
   its declared score formula;
3. the finite-vs-exact discrepancy is reported by stage and does not silently
   include sampling error;
4. the iteration record preserves the fixed policy and contains no oracle
   values in the construction inputs;
5. a stability statement is made only under an explicitly checked contraction
   or perturbation condition, with a counterexample or abstention when the
   condition is not met.

A failure is useful: it identifies the first broken interface or shows that
finite attention error grows under repeated evaluation. It does not authorize
retuning sharpness or changing the fixture after inspection.

## Proposed evidence

The task should produce a pure reference implementation, one literal matrix
witness, a finite-score implementation, a verifier with strict JSON output,
and a short report. Each output must include the commit, environment, command,
raw per-step values, discrepancies, limitations, and a final `PASS`, `FAIL`,
or `OBJECTION` verdict. No existing scientific result is overwritten.

## Next transition

After the user reviews this design and the corresponding `DRAFT` task, GPT may
move the task to `REVIEW`. Only a Claude pre-review `APPROVED` can activate it.
