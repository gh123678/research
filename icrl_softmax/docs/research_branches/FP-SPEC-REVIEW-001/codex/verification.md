# GPT verification of Claude independent first result

Task FP-SPEC-REVIEW-001, 2026-09-15. Reviewed first_result.md SHA256 72e3d57046097f3507cccd36bf8b39cca51396d3788288e86b285ce575d159a9, independently sealed before comparison. Primary paper equations and local source checked. This is a report-validation FAIL, not a task-definition OBJECTION.

## Supported

The two implemented operators match the local Expected-SARSA specification at the operator level. Neither is a direct reproduction of the external architectures. Canonical memory, positive normalized policy, synchronous timing, sparse exact versus full-support finite weighting, and external certification/control are correctly separated in the main mapping.

## Required repairs

1. **P1: F3's inherited-bound assertion is false as stated** (lines 256–260). Comparing |S|−1 and SA−1 controls one leakage mass, not the whole operator error. The old bound compares a finite sampled-SARSA operator to an exact sampled-SARSA operator, with a visited gate; the current operator uses a policy expectation and no gate. Counterexample on a visited pair: S=1, A=2, Q=(0,1), π=(1/2,1/2), one transition from action 0 to the same state with recorded next action 1, R=0 and γ>0. The old exact sampled residual is γ; the Expected residual tends to γ/2 as ξ→∞. Thus discrepancy tends to αγ/2 while the old finite retrieval bound tends to zero. Unvisited queries require a separate term/rule as well. Withdraw the global “a fortiori” inheritance; a newly derived bound with a common Expected target and explicit premises is a different statement. Old eq.(3.8) actually IS the same writer formula on VISITED queries; do not call that formula non-descriptive merely because the full operator differs.

2. **P2: equality-indexed finite logits are feature-similarity logits** (lines 213, 310–312). Taking φ(x)=sqrt(τ)e_x gives <φ(x),φ(y)>=τ1{x=y}. Therefore this contrast cannot establish a difference from Xie's kernel family. The substantive differences remain memory/routing, successor/current retrieval and the policy bias; state the writer kernel is a special case of the score form. Also Xie p.6–7 explicitly discusses visits/coverage and groups weights by predecessor states in eq.(20); “no visit/grouping concept at all” (line 251) is false. Absence of the particular visited gate is the defensible claim.

3. **P2: paper referent over-resolved** (lines 82, 234–251). The SAME local construction document cited in the report explicitly says its eq.(3.6) proves “定理 3.1” (line 182); preliminaries.md §3.3 also names that theorem. It declares the very equality mask and visited gate at issue. The bare docstring is ambiguous/inadequate attribution, not proof that the author intended Liang–Lai and got its theorem wrong. Resolve local lineage and distinguish that from unsupported EXTERNAL-paper attribution. H2 must reflect this narrower finding.

4. **P2: incorrect positive attribution of older classes** (lines 304–307). SoftmaxSARSA is a hybrid trainable softmax class, model.py:109–158, not an implementation of Liang's LINEAR attention theorem; LiangLaiOperatorBaseline:862–885 directly evaluates the formula and explicitly disclaims training dynamics. Remove the broad “older classes implement” grouping; keep any mention tightly scoped and evidenced. No new all-model audit requested.

5. **P2: report overstates actual updates and softmax significance.** “Every pair moves” (line 159) is false when residuals average to zero; say every pair is eligible for a potentially nonzero update. F4 may accurately say exact admitted weights are singleton/uniform and routing is supplied, but “Softmax is nominal/carries no expressive load” and “the one dimension … honestly” are unsupported categorical claims. Averaging is still an attention operation; lack of nonuniform finite weights limits a claim, not its mathematical name. Preserve the sparse expectation versus literal softmax distinction.

6. **Clarification, not an independent failure:** in §6 helper source selection is present even though no explicit -inf mask is computed. Keep code representation and effective support distinct. Main line citations were checked against the baseline source.

## Integrity and scope

Claude reported its shell/hashing unavailable and did not invent hashes. GPT separately hashed the delivered report and will archive a transport/integrity supplement; this is not represented as Claude independently hashing. The authored first report remains unchanged; write a revised report and response. Revisions may now read both reports because initial independence is complete. No task-definition changes or model/experiment work needed.

FAIL
