# FP-EXPL-001 v1.1 — GPT acceptance audit of the Claude route

Final decision: **PASS** (see the governing last section). This document
records the full audit history: the resumed audit below found the pre-repair
route incomplete (then FAIL); the author repairs, post-seal replay, and the
final 2589-check acceptance are recorded in the last section. The former
20-check PASS is superseded. It omitted saved stage errors,
bounds and literal projections and used the wrong probability tolerance.
Its files remain in results/FP-EXPL-001/codex/superseded_acceptance_20260911.
No scientific inputs or acceptance thresholds have changed.

The revised independent reference imports no Claude author module. It
reconstructs all PCG64 draws, discrete transitions, exact/finite formula
matrices, affine maps, complete traces, stage errors and bounds, fixed points
and signed total-error decomposition. A generic matrix evaluator additionally
checks the saved prompt, fixed parameter sparsity, static role masks,
feedforward maps, first-step projections, and 64 steps with nonzero Q.
Probability/one-step checks use absolute 1e-12; repeated checks use the
frozen elementwise scaled tolerance. Missing trace records, shape broadcasting,
nonfinite/duplicate-key JSON, altered draws/projections/errors/bounds are
rejected. Synthetic evidence corruption is a verifier unit check, not a new
research sample or parameter scan.

Source review found that finite dynamic Q reads and writes follow the saved
projection matrices; there is no n_x/N multiplier or true P in those weights.
However, the original one-step probe lists include q_pi as network input,
contrary to the frozen audit-only boundary. This does not contaminate the
main Q0=0 trajectory, but the original author must remove these probes and
retain the permitted basis/zero and invalid-input checks. The old data-bias
bound has the correct conclusion with an invalid written derivation; the
author must repair that proof. These are implementation/proof repairs, not
an objection to the task definition.

## Analytic contraction audit (no new experiment)

For a covered N=64 batch let p_C=3/(exp(8)+3),
p_S=1/(exp(8)+1), and p_W=max_x (64-n_x)/(n_x exp(8)+64-n_x).
The row-L1 routing errors are 2p_C, 2p_S, 2p_W. Write

Gf-G0 = alpha W[gamma(S-S0)-(C-C0)]
       + alpha(W-W0)[gamma S0-C0].

All probability matrices have infinity matrix norm 1. Therefore

||Gf||_inf <= 17/20 + (7/10)p_S + p_C + (17/10)p_W.

For n_x>=1, p_W<=63/(exp(8)+63). From the first five exponential
series terms e>8/3 and (8/3)^8>2000, an exact rational upper bound is

17/20 + (7/10)/2001 + 3/2003 + (17/10)*63/2063 < 1.

The reference verifies this inequality with rational arithmetic. This
conservative certificate justifies finite contraction for the present
covered batch without treating floating-point spectral estimates as proof.
For this batch min n_x=12. Reconstructing G0 with rational arithmetic gives
min_(x,y) G0[x,y] = 21/1360 > 0.01544. The same perturbation argument
with p_W <= 52/(12*2000+52) bounds ||Gf-G0||_inf by 0.005523.
Thus every Gf entry is strictly positive. Its row sums are exactly
1+alpha*(gamma-1)=17/20 because C,S,W are row-stochastic. Consequently
||Gf||_inf=17/20 exactly for this batch, justifying the 0.85 rate as well
as uniqueness. The code checks the rational inequalities, without relying
on floating-point eigenvalues. No sample-complexity or policy
improvement claim follows.

## Post-seal acceptance outcome (2026-09-11, governing section)

The two required author repairs were sealed at
`6512252934614807916953a65eb5e6ba7ee4a5a5` (audit-truth probes removed from
the network probe lists; data-bias proof repaired; all scientific outputs
bit-identical to pre-repair). GPT copied the seal to the independent snapshot
`results/FP-EXPL-001/codex/postseal_replay/6512252934614807916953a65eb5e6ba7ee4a5a5/`,
replayed the three task-prescribed commands from the snapshot (verify.py,
witness.py, Ruff; all exit code 0, scientific payloads field-identical), and
ran the revised independent acceptance above against the sealed evidence:
**2589 checks, 0 failures, PASS** — executable record
`results/FP-EXPL-001/codex/verification.json` (acceptance source SHA-256
`8a37c790b2fc9a5dcca6dbe7400c135a2761e681ea49d9177ae1b00d5c3435c7`,
sealed evidence `results.json` SHA-256
`8eea2216c6c7df21c5a25d46bd2adcf4067a8a9189db977d1909c38550302990`),
with replay manifest and output hashes in the snapshot directory. Every FAIL
condition from the resumed audit is resolved: stage errors, bounds and saved
literal projections are now compared; probability checks use the absolute
1e-12 tolerance; the probe-boundary violation is removed at the source.

Claude's renewed executable reciprocal review re-ran this acceptance against
the snapshot (exit 0, identical 2589-check record), reproduced every manifest
hash, confirmed the mutation guards (13/13), and found no q_pi network input,
hidden Q lookup, or visitation-frequency multiplier in the sealed source;
its report ends PASS at
`docs/research_branches/FP-EXPL-001/claude/verification_of_other.md`
(commit `5025cdf535b8f1c9d1460930ccc5ffdd92d3bd87`).

Provenance note: GPT's Codex session exhausted its execution quota before it
could rewrite this document; the user then explicitly authorized Claude to
take over all closure work (2026-09-11). This final section was recorded by
Claude under that authorization. The GPT acceptance verdict itself stands on
GPT's own executable artifact (`verification.json` above) and replay
manifest, not on this text.

PASS
