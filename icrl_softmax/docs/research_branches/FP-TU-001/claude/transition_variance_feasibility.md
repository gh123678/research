# FP-TU-001 Claude route: observable transition-variance feasibility

Feasibility-only assessment required by plan Task 8. Nothing in this document
enters the mandatory certificate, receives risk allocation, or modifies the
frozen mixture. The mandatory certificate remains the width-`2B` conditional
Hoeffding supermartingale of `theory.md` section 4 (interval form) with
variance proxy `(width/2)^2 <= 1` for the normalized increments.

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

## Answer: the structure-free uniform worst case equals `B^2`

Without extra structure beyond the width-`2B` range, the answer is no: the
certificate must hold for the unknown fixed target `V^pi`, over which the
data carry no information beyond `V^pi in [-B, B]^m`. Hence any valid
bound that uses only the width-`2B` range must dominate

```text
sup over P consistent with the data, sup over V in [-B,B]^m of Var_P[...].
```

Two configurations show this structure-free uniform supremum equals `B^2`
exactly.

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
   cannot upper-bound the contribution of unseen mass at a fixed sample size.
   Note, however, that the unseen mass itself shrinks with sample confidence:
   a successor of mass `p` stays unseen after `n` observations with
   probability `(1 - p)^n`, and standard high-probability ceilings on the
   total unseen mass (of order `log(1/delta)/n`) decrease in `n`. This
   configuration therefore does not by itself rule out data-dependent
   tightening at large `n`; the binding obstruction is configuration 1.

By Popoviciu's inequality no assignment within the width-`2B` range exceeds
`B^2`, so the configurations exactly sandwich the structure-free uniform
supremum at `B^2`.

## Scope of the conclusion

This is a statement about the uniform worst case without extra structure. It
does NOT claim that data-dependent adaptive tightening is impossible in
general. A construction that couples an empirical confidence set for the
transition law (and rewards) with the Bellman fixed-point constraint on the
admissible targets `V` could in principle restrict `V` below the full cube
`[-B, B]^m` and yield a strict tightening below `B^2`; that confidence-set +
Bellman-coupling proof is currently not completed, and this task neither
relies on it nor rules it out.

## Consequence for FP-TU-001

The variance proxy `sigma^2 = (range/2)^2 = B^2` (normalized proxy 1) used by
the mandatory mixture is the exact optimum among constructions that use only
the width-`2B` range without extra structure: configuration 1 is compatible
with arbitrarily large observed counts, so no structure-free observable
bound can go below `B^2`. Whether a data-dependent construction exploiting
additional structure (confidence sets, Bellman coupling) can strictly tighten
remains open and unproved here. This is distinct from Waudby-Smith and
Ramdas (2024, https://doi.org/10.1093/jrsssb/qkad009), whose
variance-adaptive empirical bounds require the variance of the *observed*
sequence to be identifiable; here the relevant variance is that of an
*unobservable* fixed-target residual under adversarial `V`, and the
observed-sequence construction does not apply.

## Numeric verification

`verify_time_uniform_mixture_certificate.py::verify_transition_variance_obstruction`
checks the corner saturation (`Var = B^2` at the aligned configuration), the
unseen-mass invisibility fixture (observed variance zero, true variance
`p(1-p) B^2`), the shrinkage of unseen mass with sample size
(`(1-p)^n` and `log(1/delta)/n` both decreasing in `n`), and 2000 random
six-successor assignments confirming the Popoviciu ceiling `B^2`.
Status: PASS.
