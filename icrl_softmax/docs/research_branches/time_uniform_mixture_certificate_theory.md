# Time-uniform mixture certificate: consolidated theory

FP-TU-001 replaces the count-wise Hoeffding union with a fixed, two-sided
15-component geometric mixture for each of `G=m+2d` residual groups.  For a
group sum `S_k`, normalize `Z_k=S_k/B`, with the declared pre-sampling bound
`B=R_star/(1-gamma)`.  The inherited pre/post-action filtrations make each
selected residual conditionally centered with range width at most `2B`.
Interval-form Hoeffding therefore gives

```text
E[exp(a (Z_k-Z_{k-1})) | F_{k-1}] <= exp(a^2/2),
```

so `exp(a Z_k-a^2 k/2)` is a nonnegative supermartingale.  Mixing the fixed
positive and negative rates with preregistered weights produces

```text
M(k,q) = sum_j w_j exp(-a_j^2 k/2) cosh(a_j q),
```

with `M(0,0)=1`.  Ville's inequality and a union over the `G` groups give one
event, uniform over all visit counts, with failure probability at most
`delta`.  Consequently random observed counts may be substituted directly;
the selective statement remains only
`P(Emit and an emitted bound is violated) <= delta`, with no conditional-support
claim.

For every integer count, continuity and strict increase on the nonnegative
half-line give a unique crossing root.  The signed line-stitching envelope is
proved separately and supplies an analytic upper bracket only; it is never the
reported certificate.  Stable log-domain evaluation plus bisection returns the
conservative upper endpoint and emits ordered deterministic non-emission
reasons on any bracketing, finiteness, convergence, or conservativeness
failure.  The implementation uses no oracle inputs and derives `B` only from
the declared reward bound.

The Direct-Q and V-first recurrences are unchanged and monotone in the
residual radius.  Thus the exhaustive numerical audit can establish
`r_mix(k) <= r_old(k;n)` for every frozen trajectory length and count without
altering route gates or legacy fields.

Transition variance remains feasibility-only.  A confidence set over successor
rows would still need a simultaneous, observable optimization over every
`V in [-B,B]^m`, including unseen mass and Bellman coupling.  The present work
records this exact confidence-set gap and does not claim a global impossibility
or spend risk budget on it.

Primary probability references are Hoeffding (1963), Azuma (1967), and Howard,
Ramdas, McAuliffe, and Sekhon (2020), as listed in the route theory files.
