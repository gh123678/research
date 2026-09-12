# FP-TIGHT-001 same-actor pre-review

Date: 2026-09-12.
Reviewer: Claude, acting as both executor and verifier under the user's standing
instruction of 2026-09-11 ("验证先不管") and the scope instruction of 2026-09-12
("好的你去做").

**This is not an independent review.** There is no second actor.

Task under review: `docs/research_tasks/FP-TIGHT-001.md` v1.0.

Outcome: `APPROVED`, with four conditions carried into the implementation.

## 1. Scope and consistency

| check | result |
|---|---|
| The sheet carries every element `AGENTS.md` section 3 requires | PASS |
| Each prediction's **basis** is stated per prediction | PASS |
| Falsifying `H3`--`H7` is declared non-invalidating | PASS |
| Prohibited work forbids re-tuning the new certificate after the run | PASS |
| Stopping conditions make a coverage violation fatal to the run | PASS |
| The step-1-only scope is stated, with the iteration explicitly out of scope | PASS |

## 2. Condition 1 (mandatory): the sealed corpus must not be touched

`fixed_policy_variance_certificate.py` is part of the scientific corpus and its hash
is recorded by several sealed bundles. Editing it to "write the constants tight"
would invalidate every one of those records and destroy the sealed baselines this
task needs for comparison.

Condition: implement the repairs in a **new** module
(`fixed_policy_bernstein_certificate.py`), never by editing, monkey-patching or
shadowing the frozen one; reproduce the split rule and the residual definition
byte-identically so the arms sit on the same items; and verify after the run that
every sealed module's hash is unchanged.

## 3. Condition 2 (mandatory): the decision rule must be provably untouched

Both arms are scored through the unmodified `fs.improvement_for`. The only channel
by which a new certificate can change a decision is the `e_q` field it carries,
since the rule reads `certificate["e_q"]` and `certificate["reasons"]` and nothing
else.

Condition: the analyzer and verifier must confirm that no rule constant, eta grid or
reason order changed, and the pre-review requires the substitution to be described
as a *substitution of the bound*, not as a new decision rule.

## 4. Condition 3 (mandatory): the unsound arm must never be read as a certificate

`H6` asks whether deleting the envelope is a bigger lever than correcting the
inequality. Answering it needs an arm with the envelope deleted — and that arm has
no high-confidence statement left, so its `e_q` is a floor, not a bound.

Condition: label it `counterfactual_no_envelope`, set an explicit `sound: false`
flag on every one of its entries, exclude it from the coverage audit, and report how
often it would have failed coverage, so the label is backed by a number rather than
by a comment.

## 5. Condition 4 (mandatory): the risk allocation must sum to `delta`

The frozen certificate spends `delta/(2d)` on each of `2d` bounds. The repairs need
three bounds per pair — the second-moment bound, and one one-sided mean bound in each
direction, because `||.||_inf` is two-sided — so spending `delta/(2d)` on each of
`3d` bounds would silently overspend the risk by `1.5x` and make the repair
anti-conservative in exactly the way this task exists to check.

Condition: `delta/(3d)` per bound over `3d` bounds, stated explicitly in the module
and in its output as `delta_each`, so the arithmetic can be re-added by a reader.

## 6. An error this review caught, and a correction to an earlier estimate

**The range.** Bernstein's inequality needs the RANGE of the bounded variable. The
frozen envelope is `2B = 10` and residuals lie in `[-10, 10]`, so the range is `20`,
not `10`. The exploratory diagnostics that produced the first lever table passed
`envelope` where the range was required, which under-counted the `(4/3) R t` term by
a factor of two. `Y_RANGE = 2 * ENVELOPE = 20` is now explicit and named.

**The estimate I gave the user was too optimistic.** The first lever table reported
empirical Bernstein at `-56%`, because the scratch Bernstein radius used
`t^2 = (2V + (2/3) R t) log(1/delta)/N` with the observed half-B variance. The
rigorous Maurer-Pontil form carries the constant `7/3` and `log(2/delta)`, and the
sealed risk is spread over `3d = 36` bounds rather than `2d = 24`, both of which cost
tightness. The pilot measured `-24.1%`. The task sheet's `H4` band is set from the
pilot, not from the earlier table, and the earlier figure is recorded here as
withdrawn so it is not quoted again.

The general lesson is written into the task sheet: a pilot is what licenses a band,
and a back-of-envelope Bernstein calculation is not a pilot.

## 7. Inheritance fidelity

| element | status |
|---|---|
| protocol dimensions, lengths, mixing, gap, seed | inherited from FP-SCALE-002 |
| certified quantity (`||Qhat - Q^pi||_inf`) | unchanged |
| residual definition and pair split | reproduced exactly from the frozen module |
| `(1 - gamma)` division | unchanged |
| decision rule, eta grid, ordered reasons | untouched, frozen code |
| certificate | the only change, and it is additive |

## 8. Residual risks

| risk | assessment |
|---|---|
| A repair is unsound and its `E_Q` no longer covers | This is `H1`, mandatory, and a violation stops the run. |
| Risk is overspent by the extra one-sided bound | Addressed by condition 4. |
| The range is mis-set | Caught in review; `Y_RANGE` is explicit and named. |
| The unsound floor is read as a result | Addressed by condition 3. |
| Bands are back-fitted to the result | Bands are in the task sheet, written before the formal run, and anchored on a pilot whose scope is disclosed. |
| Step-1-only scope is overstated as an iteration result | Stated in the task sheet and in the result record. |
| Same-actor verification | Not mitigated; disclosed. |

## 9. Verdict

**`APPROVED`**, contingent on the four conditions above being enforced executably,
which the implementation and verifier do.

Same-actor approval; correspondingly less assurance than an independent review.

## Objections

None.
