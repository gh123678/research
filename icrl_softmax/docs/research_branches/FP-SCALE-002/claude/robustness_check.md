# FP-SCALE-002 robustness check (read-only, post-seal)

Date: 2026-09-11.
Scope: read-only robustness evidence for the sealed formal result. This check
does **not** amend, replace, or rerun the frozen formal matrix.

## Why it was run

The headline `22/48` emission rate came from a single seed schedule. A
single-actor result with one draw should be checked for whether the rate is a
property of the construction or an accident of the batch draws. The task sheet
forbids rerunning the frozen formal matrix, so this check moves only the two
batch RNG streams and leaves the task seed schedule, matrix shape, dimensions,
and every frozen constant untouched. It writes no result files.

## Seed sensitivity

Tasks `0--5` per mixing setting (`24` primary route-records per variant), same
`16384 x 64` certification batches, four variants:

| variant | attempted | emitted | rate | non-degrading | strict | violations | min control/adaptive ratio |
|---|---|---|---|---|---|---|---|
| frozen schedule subset | 24 | 16 | 0.667 | 16 | 16 | 0 | 3.520 |
| training seed +1000 | 24 | 13 | 0.542 | 13 | 13 | 0 | 3.554 |
| certification seed +1000 | 24 | 14 | 0.583 | 14 | 14 | 0 | 3.448 |
| both +1000 | 24 | 12 | 0.500 | 12 | 12 | 0 | 3.512 |

Reading:

- the emission rate stays in `0.500`--`0.667` across seed variants, so `22/48`
  is not an artefact of one lucky draw;
- **every** emission in **every** variant is componentwise non-degrading and
  strictly improving, with **zero** certificate violations across all `96`
  route-records;
- the `H5` attribution survives: the envelope control is worse in every record
  of every variant, minimum ratio `3.448`.

Note on comparability: the frozen-schedule-subset row (`0.667`) is not a
re-measurement of `22/48`, because it uses only tasks `0--5`. On the full formal
matrix that subset contributes `13/24 = 0.542` while tasks `6--11` contribute
`9/24 = 0.375`, giving the sealed `22/48 = 0.458`. The same subset therefore
ranges `0.542`--`0.667` across seed variants, so the sealed value sits at the
low end of, but inside, the observed spread.

## A stronger explanatory factor than the seed: the mixing setting

The task-level breakdown of the sealed formal bundle is not uniform:

| mixing | primary emissions |
|---|---|
| `0.08` | **19 / 24 (0.792)** |
| `0.5` | **3 / 24 (0.125)** |

The mixing setting dominates the seed by a wide margin. This is an exploratory
observation from the sealed data, not a pre-registered hypothesis: FP-SCALE-002
froze no hypothesis about mixing, and this check did not test one. It is
recorded because it is the obvious next question and because it would be wrong
to leave it out of a robustness note.

## Status

Robustness evidence only. The frozen formal matrix was executed exactly once
and its single execution stands as sealed; the sealed module hashes were
re-verified before and after this check and were unchanged:

- `fixed_policy_expected_sarsa.py` `ebea85c455c8c2eb62fcf0ccdf93ce3e3b005b105b58198721ed0e0d181d8ea0`
- `fixed_policy_variance_certificate.py` `1410dfda1d37e54624fa5bbdbcc252eafbd6bd74117931d6d063d0971a4e315d`

`main` was not modified. This check adds no acceptance-criterion evidence and
changes no verdict; it strengthens confidence in `H3`, `H4`, and `H5` only.

## Limitation

Same actor as the execution. This is not independent verification and must not
be described as such.
