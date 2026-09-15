# GPT final verification of Claude route

FP-SPEC-REVIEW-001, 2026-09-15. Reviewed final_report.md SHA256 d52a83e293241516c5c5b5966c7b57bbbc074c9e2e6477689e3e6611a52c6da9. This supersedes the two FAIL verdicts on earlier report versions, which remain archived.

## Primary-evidence checks

1. Exact read, expectation, signed residual and null/visited writer were re-derived against model.py:888–1011 and FP-ESARSA's frozen contract. The reported operator is correct on positive normalized policies. Direct sparse weights and actual softmax execution are properly distinguished.
2. Finite matrices were re-derived against model.py:1065–1088. Successor denominator e^ζ+S−1 uses normalization of ALL policy rows. Read denominator e^ξ+SA−1 and writer denominator n_x e^τ+N−n_x are correct. n_x=0 yields 1/N weights; n_x>0 does not make finite τ exact. Accidental identical updates can occur when residuals cancel or coincide, but the claimed operator identity is correctly limited.
3. Normalization correction accepted. With the SAME residual sum S_x, Δ_local=(N/n_x)Δ_global at the same α. At balanced n_x=N/m the factor is m. Sampled versus Expected residuals are a separate difference; these coefficient comparisons never identify the actual whole operators unless that difference is also removed. Read the table's shorthand under that explicit premise in §1 C-A.
4. The non-transfer counterexample is correct with the recorded next action fixed to action 1 as in earlier verification: old sampled residual γ, exact Expected residual γ/2, finite residual γ/2−1/(e^ξ+1), one visited transition. Difference from the old sampled target tends to αγ/2 while the old claimed finite-vs-sampled bound tends to zero. Final §5-F3's short comparison should be read with this recorded-action premise; the complete evidence is retained here and in verification.md. No uniform replacement guarantee is accepted or needed.
5. Xie original pp.3–7 confirms its fixed positional source mask, trajectory-memory/TSM construction, score family and exact-residual convergence conditions. One-hot finite writer is a score-family special case. In Claude's table, 'pair tokens versus trajectory sources' describes its READ/SUCCESSOR heads; BOTH writers normalize over transition samples. This role distinction is explicit in the matrices, and prevents treating that shorthand as a claim that the current writer uses Q tokens as sources.
6. Liang–Lai pp.2–5 confirms linear attention, global α/N normalization, sampled successor and the parameter-query/training architecture. The local manuscript Theorem 3.1 explicitly supplies equality routing and the old visited gate. The corrected attribution is local lineage plus an inadequate bare citation, not demonstrated external-paper misreading.
7. Exact-residual fixed-point identity and conditional diagonal premise are not extended to the full finite class. The independent GPT two-action γ=0 counterexample was also recomputed by Claude in verification_v2.md. The source contract explicitly anticipates finite residual bias.

## Acceptance accounting

- Both primary classes and shared helper: covered; canonical candidates, support, axes, signs, timing, memory transport, unvisited handling, sampled/Expected distinction, policy input, finite leakage, dimension dependence and external control all evidenced.
- Paper/local theorem distinction, matched/different operators, and permitted wording: supported by primary sources, not NumPy agreement alone.
- Repairs: all material v1/v2 errors withdrawn or corrected; incidental alternative bound removed rather than silently inherited.
- Report limits: no experiment/full-prompt construction, no re-proof of external convergence/training theorems, historical exposure, and shell/hash limitations disclosed.
- Provenance: actual Claude Code wrote its reports in its own worktree. GPT separately checked both code trees match the baseline and hashed the reports. Claude's independent verification is reading/algebra; independent hash execution is not claimed. Archive manifest includes raw and LF-normalized hashes so Git CRLF conversion is detectable without confusing it with a content change.
- No task-level objection or unresolved material disagreement remains. This PASS validates an accurate, bounded audit with a negative direct-paper-attribution result; it does not certify a learned Transformer, original-paper reproduction, or previous experiments.

PASS
