# Control-route comparison: source review

Date: 2026-09-10. Baseline: db9d63043489f4ec5660e71c53e7e842a280d04f.
Status: preliminary source assessment; diagnostic execution awaits pre-review.
No new algorithm, numerical fixture, or inherited verifier was run in this task.

## Comparison of actual computation interfaces

| Interface | Existing two-stage control | Fixed-policy Expected SARSA draft |
|---|---|---|
| Persistent value memory | Q table; reusable canonical pair representation | One canonical token per pair, explicitly specified |
| Current Q retrieval | Direct indexing in model.py, TwoStageSoftmaxQControl.forward | A finite-score current-pair head is specified, not implemented in the draft |
| Successor action set | q_values[next_states] externally supplies the exact action row | All canonical tokens with state similarity plus log policy probability |
| Action weighting | GroupedSoftmaxMax implements beta*Q scores and Q values | Fixed policy log probabilities supply action weights |
| Residual formation | Arithmetic after exact indexing | Fixed linear combination after two attention reads |
| Writeback | KernelizedSoftmaxQTD updates all passed queries, including unvisited pairs | Finite writer explicitly updates all canonical queries |
| Policy change during evaluation | Current Boltzmann continuation changes with Q | Policy stays fixed through evaluation |
| Final policy | Existing performance analysis/experiments use external policy extraction; actual output head needs an explicit witness | Relative-softmax output is specified; literal realization remains to be constructed |
| Full literal witness | Existing sampled-SARSA witness is reusable, but does not establish this entire finite two-stage route | No Expected-SARSA literal witness in the inspected source |
| Environment execution | External sampling and simulator | External sampling and simulator |
| Guarantee scope | Ideal approximate-greedy analysis plus separately stated perturbations | Fixed-policy evaluation and one selective safety test; current certificate preflight is adverse |

Sources: model.py classes GroupedSoftmaxMax, KernelizedSoftmaxQTD,
TwoStageSoftmaxQControl; verify_end_to_end_sarsa.py;
docs/research_tasks/FP-ESARSA-001.md;
docs/superpowers/specs/2026-09-09-fixed-policy-expected-sarsa-relative-softmax-design.md;
manuscript method_experiments.md sections 3.1--3.4.

The existing sampled finite-logit SARSA class still uses a visited-query gate.
It must not be conflated with the two-stage writer, which passes all Q entries
as queries. Reusing a module requires checking its actual support semantics.

## Candidate-specific obstacles

For two-stage control, replacing pre-grouped action rows by all-memory
attention is not just exchanging indexing for a fixed state score. The
proposed finite score combines state similarity with beta*Q. Values in other
states can compete with the requested-state margin. Thus action sharpness and
state-routing sharpness must be analyzed together. Sharpening action choice
alone is not a valid remedy for state leakage.

For fixed-policy Expected SARSA, action scores are log pi(b|u). Each policy
row sums to one, so the draft's requested-state mass has a simpler expression.
This is a real architectural advantage to test. It does not remove current-
read leakage, writer leakage, or the need for a literal policy-output head.

The previously disclosed certificate obstruction remains a separate issue.
The preflight task will reproduce its arithmetic, but will not change the
FP-ESARSA certificate. A working finite head would not cure an always-rejecting
safety test, and a rejecting test would not establish that the head is useless.

## Provisional choice and exact next gate

Existing code maturity favors two-stage control as the initial candidate.
The fixed-policy candidate has cleaner successor-state normalization. There
is not yet enough evidence to decide the winner of the finite routing test.

The approved task must test both successor heads on the same fixed small
fixtures, including the separated-state-value case. It must separately report
agreement with the finite formula and disagreement with the grouped formula.
The selected next implementation should minimize remaining external content
operations while giving an explicit error condition that is nonvacuous on a
declared case. If both fail that gate, report no selection.

Potential publication value is not validated by this audit. A later task must
specify the full permitted architecture, policy output, unvisited-pair behavior,
and a matched literal-network/control reference. It must not call compact
operator tests end-to-end control evidence or infer per-update safety from
an approximate-optimality bound.

## Execution record

- Read-only source inspection completed; user-owned untracked learning record
  preserved.
- New isolated GPT branch and task created. DRAFT seal: ebe6bd3.
- Claude pre-review launch was rejected by automatic approval review before
  process creation. No private files were sent by that attempted launch.
- Reason: authorization did not explicitly cover transfer of AGENTS.md and
  this new task to the configured external Claude Code service.
- No diagnostic implementation or experiment proceeded past the review gate.
- Next action requires explicit scoped external-data authorization; after
  approval, resume the same task rather than create another plan or matrix.
