# GPT verification of Claude v2

The principal v1 repairs are accepted: local theorem referent identified, blanket inherited bound withdrawn, one-hot kernel connection restored, sparse attention support clarified, actual finite updates allowed to be zero, and exact-residual versus full finite fixed-point caveat added.

Two newly introduced mathematical statements still require correction:

1. v2 §3.2 and §7.1 dimension 11 (also response item 1): global α/N and per-pair α/n_x do NOT coincide when n_x=N/m. On a nonzero residual sum that condition gives a factor m difference. Same α and same residual target: coefficients equal only if n_x=N; updates can coincidentally agree if the sum is zero. Under perfectly balanced counts one may choose α_local=α_global/m to match, but this is an explicit changed learning-rate convention, not the frozen same-α identity. Furthermore sampled versus Expected targets remain a separate difference.
2. v2 §7.2 dimension 11: at FINITE τ, a visited writer is NOT the α/n_x mean. Each matching transition has weight e^τ/(n_x e^τ+N−n_x), and nonmatching transitions retain positive weight. The local per-pair mean is the τ→∞ limit when n_x>0, or a special case with no off-group transitions (n_x=N); other accidental equalities do not prove identity.

Also correct response_to_review's precision note 1: finite current-read residual is γ/2−1/(e^ξ+1), not γ/2, in the one-transition example. Only the exact Expected residual is γ/2. The limit contradiction establishes failure of the blanket transfer; it does not establish that every finite parameter value violates a numerical bound. Remove that overstatement.

Keep an added replacement bound only if its δ bound explicitly covers the residual being bounded. Simplest final audit need not introduce any replacement guarantee: withdrawal of transfer plus the correct comparison is sufficient. No new result or task definition is requested.

Preserve v1/v2 and response history. Write a concise final_report.md superseding them, with the complete required dimensions and corrected equations, then request recheck. These are report repair failures, not an OBJECTION.

FAIL
