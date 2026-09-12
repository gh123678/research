# Independent verification handoff

Date: 2026-09-12.
Author: Claude, acting on the user's instruction to close the outstanding items.
Status: **handoff document. It verifies nothing.** It exists so that a second actor
— or the user personally — can check this line's claims without re-deriving what
"same-actor verification" does and does not already cover.

## 1. Why this document exists

Every result in this repository's `FP-*` line is **same-actor only**. The user's
standing instruction of 2026-09-11 ("验证先不管") made that a deliberate choice
rather than an oversight, and every task sheet and report states it. Codex is
unavailable, so a genuine second actor is not currently obtainable.

The gap is therefore not going to close by itself, and repeating "verification is
outstanding" in each report adds nothing. What is missing is a **statement of what
would actually count**, specific enough to be executed. That is this document.

## 2. The starting position

The same-actor checks are not thin. As of this date the repository carries:

| asset | count |
|---|---:|
| `verify_*.py` checkers | `29` |
| `analyze_*.py` analyzers | `18` |
| `evaluate_*.py` evaluators | `14` |
| sealed `results/*/task_results.json` bundles | `55` |
| result task directories | `20` |

and every checker exits `0`. They independently re-derive headline numbers from the
sealed bundles, replay the sealed programs, verify frozen-file hashes, confirm
analyzer determinism by byte-comparison on re-run, and — since `FP-CENSUS-001` —
verify that a certification batch reproduces the sealed pair-count vector rather
than merely claiming to.

That is a real amount of checking. It is also, structurally, **not verification of
the claim**, for the reason in the next section.

## 3. What same-actor verification cannot catch, precisely

The central architectural fact is that the two "independent" paths share far more
than they differ. `expected_exact` and `expected_finite` differ in `model.py`'s
finite-expansion handling; everything else is common:

| shared component | role | consequence |
|---|---|---|
| `fixed_policy_variance_certificate.py` | computes `E_Q` | both paths inherit the same bound |
| `fixed_policy_expected_sarsa_scaled.py` | routes and the decision rule | both paths inherit the same rule |
| `evaluate_fixed_policy_q_routes.py` (`policy_quantities`) | the oracle "truth" | the audit's ground truth is the same code the census measures against |
| task definitions, hypotheses, protocol constants | what is being tested | both paths test the same reading of the question |

So numpy-versus-network agreement establishes that **the network implements the
same recursion as the numpy reference to within `1e-5`**. It does not establish
that the recursion is the one the claim requires. Specifically, these failure modes
survive every check currently in the repository:

1. **A wrong certificate.** If the two-half Hoeffding construction, the risk split
   `δ/(2d)`, or the union bound were mis-derived, both paths would compute the same
   wrong `E_Q` and agree perfectly.
2. **A wrong norm or comparator in the decision rule.** `LB_s = Î_s − E_Q·‖Δπ_s‖₁`
   with `min_s LB_s > 0` — if the claim needs a different norm or a non-strict
   inequality, no amount of internal replay detects it.
3. **A self-referential oracle.** The oracle audit compares `Qhat` against
   `policy_quantities(mdp, π)`. If that routine were wrong, the audit would be wrong
   in the same direction on both paths. **Partially closed** — see V2 below; a
   coding error is now ruled out by a different method, though not by a different
   actor.
4. **A mis-specified risk statement.** See V1 — the highest-value item here.
5. **A wrong yardstick.** The census's `sigma_min` was a choice; a second actor
   should ask whether it measures what the narrative says it measures.
6. **A task-definition error.** One was already found by accident: `FP-CENSUS-001`'s
   `H4` is phrased universally in the task sheet and operationalised as a majority
   claim in the plan. A verifier whose job is to read the *definition* catches that
   class; a verifier who replays the code does not.

Items 1, 2 and 4 are the reason a second actor must **re-derive, not re-run**.

## 4. The claims, ranked by value per unit of effort

Each entry states the claim, what an independent check looks like, and what would
falsify it. V1 is first because it is the only one where a plausible defect would
change the headline rather than the detail.

### V1 — What exactly does `δ` control, and is the violation count the right statistic?

**Claim under test**: "every emitted certified error bounds the realized oracle
error", reported as `0` violations across all six steps on both paths.

**Why it needs an outsider**: both paths inherit the same reading of `δ`. The
protocol freezes `δ = 0.05`. If `δ` is the per-certificate failure probability,
then the run performs many certificate evaluations — `48` route-records × up to `6`
steps × `2` paths — and the reported `0` violations is then a statement about a
quantity for which `0.05` is not the relevant budget. If instead `δ` is a per-run
budget applied through the `δ/(2d)` split, the arithmetic should be shown to cover
the number of certificates actually issued.

**Independent check**: state in one paragraph what event `δ` bounds, count the
certificate evaluations the protocol performs, and determine whether the reported
`0`-violation count is the statistic that the risk statement licenses. A
seed-sweep (`FP-SCALE-002`'s seed-sensitivity machinery already exists) gives an
empirical violation rate to compare against `δ`.

**Falsified if**: the `0`-violation result is reported as evidence for a statement
whose risk budget it does not exhaust, or if the seed-sweep rate is incompatible
with the stated `δ`. Note that `0` observed violations is *also* consistent with a
bound that is very loose — the census's own finding is that much of `E_Q` comes
from a worst-case assumption — so "0 violations" and "the bound is informative"
are different claims, and the reports should not let the first stand in for the
second.

### V2 — Is `policy_quantities` correct? — **DONE, different method, same actor**

**Claim**: `Q^π` and `v^π` as returned by `evaluate_fixed_policy_q_routes.py` are
the fixed point of the policy-evaluation Bellman operator, and the routes' target
is `Q^{π_{k-1}}` rather than `Q^{π_0}`.

**Check performed** (`icrl_softmax/verify_policy_quantities_by_solve.py`, exit `0`):
for six records spanning both mixings and the plateau/never/dropout groups, the
returned `Q^π`, `v^π` and `mu_state` were compared against routes that share no
code with the routine —

- a **direct linear solve** of `(I − γP_π)Q = R_π` as a `(d·A)×(d·A)` system,
- **value iteration** of `v` under `π` to `1e-14`,
- the **stationary law** by solving `μ = μP_π` with a normalisation row,

plus two cross-identities that catch a self-consistent but wrong convention
(`Q` recomputed from `v`, and `v` recomputed from `Q`). All six records agree to
`≤ 2.3e-14`, with the `Q` gap at `≤ 1.4e-15`.

**What this closes**: failure mode 3 above. A coding error in the audit's ground
truth is now ruled out, and by a different method rather than by a re-run — which
is the property all `29` existing checkers lacked.

**What it does not close**: I derived the Bellman system from my own reading of the
same `P`/`R` semantics, so a shared misreading of what the MDP arrays *mean* would
survive. Closing that needs a second actor reading the MDP definition and the
protocol independently. It also remains same-actor, so it does not upgrade any
result's verification status.

**Still to check**: that the step-`k` route target is `π_{k-1}`. That is a
code-reading question, not a numerical one; `evaluate_fp_iter2_001.py` carries the
comment "The certificate bounds `||Q_k - Q^{π_{k-1}}||`, not `||Q_k - Q^{π_0}||`"
and the fix that made it true, but no checker asserts it structurally.

### V3 — Does the decision rule's guarantee follow from the certificate?

**Claim**: `P(an update is emitted AND some state's value decreases) ≤ δ`.

**Independent check**: a proof read, not a code read. Verify that
`LB_s = Î_s − E_Q‖Δπ_s‖₁` is a valid lower bound on the true improvement at state
`s` given `‖Qhat − Q^π‖∞ ≤ E_Q`, that `min_s LB_s > 0` implies componentwise
non-degradation, and that the union bound over states is where `d` enters. Then
compare the code's constants against that derivation line by line.

**Falsified if**: the derivation needs an assumption the code does not enforce, or
a constant differs.

### V4 — Is the oracle audit independent of the thing it audits? — **partly done via V2**

**Claim**: the audit's "realized error" is ground truth, not a restatement.

**Check**: V2 established that `policy_quantities` is correct to machine precision,
which is the foundation this needs. What remains is to recompute the sealed
`oracle_audit.realized_q_sup_error_vs_current_target_pi` values from V2's
independent `Q^π` and confirm they match — i.e. to verify not just that the ground
truth is right, but that the audit *used* it correctly. That is a short,
mechanical, and fully decisive check, and it has not been done.

### V5 — Does the census's `H0` really reproduce the sealed iteration?

**Claim**: `48/48` route-records exact on decision, `E_Q`, `eta`, ordered reasons
and row count.

**Independent check**: do not re-run the analyzer. Read the sealed
`FP-ITER5-001` bundle and the census bundle and compare the fields directly — this
is what `verify_fp_census_001_same_actor.py:rederive_h0` does, and a second actor
should write that comparison themselves rather than trust it.

**Falsified if**: any per-step field differs.

### V6 — Is `sigma_min` the right yardstick?

**Claim**: the within-state spread `min_s ptp_a Q^π(s,a)` is the "action-relevant
value spread" the narrative refers to.

**Independent check**: partially done already, and reported post-hoc in
`analyze_fp_census_001.py` section 11 — the top-2 gap scores `14/48` against
`1/48` for the full spread, so the choice is supported on the sealed data. What
remains for an outsider is the conceptual question: is the *binding* quantity for
the gate really the spread, given the gate is `Î_s > E_Q‖Δπ_s‖₁` and `Î_s` is a
first-order functional of `Qhat` weighted by the policy movement? The census's own
finding that the spread barely moves while `E_Q` moves more (mean CV `0.018` vs
`0.026`) suggests the spread is closer to a record constant than to a state
reading, which is what makes the ratio a poor predictor of *when* a record stops.

**Falsified if**: a different functional of `Q^π` separates the groups better, or
if the spread is shown not to bound `Î_s/‖Δπ_s‖₁` in the regime the narrative uses.

### V7 — Read the task definitions, not just the code

**Claim**: the hypotheses as written are the hypotheses as tested.

**Independent check**: for each task sheet, compare the hypothesis wording against
the analyzer's operationalisation. `FP-CENSUS-001`'s `H4` is a known instance
(universal in the sheet, majority in the plan, both reported). Others have not been
audited this way.

**Falsified if**: a hypothesis is scored on a weaker or different condition than
its wording.

## 5. What the user can do without a second actor

These are real independent checks, and they do not require a research actor:

1. **Run the existing checkers on a fresh clone.** Every `verify_*.py` exits `0`
   and is deterministic. This tests reproducibility, not correctness — worth
   stating plainly, because a green run feels stronger than it is.
2. **Hand-check V2 for one record.** Already done in full by
   `verify_policy_quantities_by_solve.py` (six records, `≤ 2.3e-14`), but it is a
   short file and worth reading rather than trusting: a `4×3` system is small
   enough to follow by eye.
3. **Hand-check V1's arithmetic.** It is a bookshelf calculation — count
   certificates, compare against `δ` — and it is the item most likely to change a
   conclusion.
4. **Ask a second reader to attempt V3** as a proof read. It needs no code and no
   compute.

## 6. What a real second actor would need

- **Scope**: re-derive V1–V4 independently rather than re-running the checkers.
  V1–V3 need no compute at all; V4 needs one exact solve per record.
- **Isolation**: their own branch `codex/VERIFY-<task>` and their own result
  directory, per `AGENTS.md` §6, reading the sealed bundles but never editing them.
- **Verification report**: one of `PASS` / `FAIL` / `OBJECTION` per `AGENTS.md` §7,
  pointing at checkable evidence rather than agreement in prose.
- **Reciprocity**: the same actor should then be verified in return, or the user
  should rule the exception.

## 7. A separate, more urgent risk than the verification gap

`results/` is **git-ignored**. The `55` sealed `task_results.json` bundles, the
frozen prediction file, and every `summary.json` and `environment.json` exist
**only on this machine**. All `29` checkers read them. If this working copy is
lost, the checkers cannot run and most of the line's evidence becomes
unreproducible in practice, even though every script to regenerate it survives in
git — regeneration would cost hours of compute and would produce *new* bundles
rather than the sealed ones the reports cite.

Independent verification is a scientific gap. This is a durability gap, and it is
the one that would be expensive rather than merely unsatisfying. It is recorded
here because it surfaced while writing this handoff, and it is outside the scope of
what this document was asked to do.

## 8. What this document does not claim

It does not claim any verification has occurred. It does not upgrade any result
beyond "same-actor derived verification". Its own content is not itself verified.
