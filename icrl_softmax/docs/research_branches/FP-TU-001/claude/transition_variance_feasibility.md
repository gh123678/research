# FP-TU-001 Claude route: observable transition-variance feasibility

Feasibility-only assessment required by plan Task 8. Nothing in this document
enters the mandatory certificate, receives risk allocation, or modifies the
frozen mixture. The mandatory certificate remains the width-`2B` conditional
Hoeffding supermartingale of `theory.md` section 4 with variance proxy 1 for
the normalized increments.

## Question

Under the frozen protocol (fixed policy, deterministic edge rewards with
declared bound `R_star`, `B = R_star/(1-gamma)`, successor observations from
one trajectory), can observed data yield a confidence set that uniformly
upper-bounds

```text
Var_P[ R(s,a,S') + gamma V(S') ]
```

simultaneously for all candidate targets `V in [-B, B]^m`, including mass on
unseen successors, by strictly less than the trivial range bound `B^2`?

## Answer: no (deterministic obstruction)

The certificate must hold for the unknown fixed target `V^pi`, over which the
data carry no information beyond `V^pi in [-B, B]^m`. Hence any valid
observable bound must dominate

```text
sup over P consistent with the data, sup over V in [-B,B]^m of Var_P[...].
```

Two configurations show this supremum equals `B^2` exactly, so no observable
strict tightening exists.

1. Fully observed support, aligned corners. Take two successors `s1, s2`,
   both observed with empirical frequency one half, deterministic edge
   rewards at the declared extremes `R(s,a,s1) = +R_star`,
   `R(s,a,s2) = -R_star` (permitted by the declared bound), and the
   admissible target `V(s1) = +B`, `V(s2) = -B`. Then
   `R + gamma V` takes the values `+(R_star + gamma B) = +B` and `-B`, each
   with probability `1/2`, and

   ```text
   Var = B^2,
   ```

   saturating the Popoviciu bound for a width-`2B` variable. Every ingredient
   is compatible with arbitrarily large observed counts, so no amount of data
   excludes it.

2. Unseen-successor mass. Take observed successors carrying identical values
   (observed conditional variance zero) and an unobserved successor with
   probability `p > 0` carrying the aligned corner value `B`. The true
   variance is `p(1 - p) B^2 > 0` while every observable functional of the
   seen samples is consistent with variance zero. Observed data therefore
   cannot even upper-bound the contribution of unseen mass below the trivial
   range bound.

By Popoviciu's inequality no assignment within the width-`2B` range exceeds
`B^2`, so the two configurations exactly sandwich the uniform supremum at
`B^2`.

## Consequence for FP-TU-001

The variance proxy `sigma^2 = (range/2)^2 = B^2` (normalized proxy 1) used by
the mandatory mixture is already the exact uniform optimum over
`V in [-B,B]^m`; there is nothing observable left to estimate. This is
distinct from Waudby-Smith and Ramdas (2024,
https://doi.org/10.1093/jrsssb/qkad009), whose variance-adaptive empirical
bounds require the variance of the *observed* sequence to be identifiable;
here the relevant variance is that of an *unobservable* fixed-target residual
under adversarial `V`, and the observed-sequence construction does not apply.

## Numeric verification

`verify_time_uniform_mixture_certificate.py::verify_transition_variance_obstruction`
checks the corner saturation (`Var = B^2` at the aligned configuration), the
unseen-mass invisibility fixture (observed variance zero, true variance
`p(1-p) B^2`), and 2000 random six-successor assignments confirming the
Popoviciu ceiling `B^2`. Status: PASS.
