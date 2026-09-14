"""FP-COMPOSE-002 / claude: dimension-scaling analysis (READ-ONLY, no experiment).

The composition's conclusion is scoped to S*A in {24, 12}. The load-bearing assumption that
let every step emit is "at least one row satisfies LB_s > 0", and it has never been tested at
another dimension. This asks where that assumption breaks as S*A grows and what it costs to
hold it.

WHAT THIS IS NOT: not a run, not a new arm, and not a measurement at any dimension other than
the two that were run. It is an extrapolation calibrated on exactly TWO points, and the
calibration is an interpolation between them, so it has no residual by construction and
cannot validate its own functional form. That is stated in the output, not just here.

FIRST ATTEMPT WAS WRONG and is recorded: an earlier version hardcoded kappa = 0.114 from an
off-hand estimate while the same script MEASURED kappa at 0.438 and 0.299 from the sealed
data, and it then predicted that support would already have failed at d=12 and d=24 -- which
the sealed pair sizes contradict (n_min 11464 and 7432, both far above the 2000 threshold).
A script that measures a quantity and then uses a different number for it is worse than no
script. This version uses only measured quantities.

Two regimes, behaving oppositely, with L = 64 and C = 16384 both frozen:
  d <= L : a chain of L steps can reach every pair, so coverage per pair IMPROVES with d
  d >  L : a chain has only L steps, so coverage per pair FALLS
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402
from evaluate_fp_xfam_001 import FAMILIES, build_family_task  # noqa: E402
import fixed_policy_expected_sarsa as es  # noqa: E402

C_CHAINS, L_CHAIN, MIN_VISITS = 16384, 64, 2000
DELTA_STEP, GAMMA = 0.0125, 0.70
ETA_GRID = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
B = PROJECT / "results/FP-COMPOSE-002/claude/formal/task_results.json"
bundle = json.loads(B.read_text(encoding="utf-8"))


def row_margins(pi, q_hat):
    S = pi.shape[0]
    h = np.full(S, -np.inf)
    for eta in ETA_GRID:
        cand = es.relative_softmax_candidate(pi, q_hat, eta)
        dd = cand - pi
        num = np.einsum("sa,sa->s", dd, q_hat)
        den = np.abs(dd).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(den > 0, num / np.where(den > 0, den, 1.0), -np.inf)
        h = np.maximum(h, ratio)
    return h


# ------------------------------------------------ measured coverage at the two real d ---- #
meas = {}
for rec in bundle["records"]:
    fam_name = rec["family"]
    e = rec["routes"][bundle["routes"][0]]["numpy|perstate|L12"]["steps"][0]
    obs = np.asarray(e["pair_sizes"], float)
    d = obs.size
    fam = FAMILIES[fam_name]
    mdp, beh, rng = build_family_task(fam, float(rec["mixing"]), int(rec["task_index"]))
    mu = np.asarray(policy_quantities(mdp, beh)["mu_pair"], np.float64)
    meas.setdefault(d, {"n_min": [], "n_mean": [], "kappa_min": [], "family": fam_name})
    meas[d]["n_min"].append(float(obs.min()))
    meas[d]["n_mean"].append(float(obs.mean()))
    meas[d]["kappa_min"].append(float(mu.min() * d))

measured = {}
for d, v in meas.items():
    measured[d] = {"family": v["family"], "tasks": len(v["n_min"]),
                   "n_min_median": float(np.median(v["n_min"])),
                   "n_mean_median": float(np.median(v["n_mean"])),
                   "coverage_min": float(np.median(v["n_min"]) / C_CHAINS),
                   "coverage_mean": float(np.median(v["n_mean"]) / C_CHAINS),
                   "kappa_min_median": float(np.median(v["kappa_min"])),
                   "headroom_over_min_visits": float(np.median(v["n_min"]) / MIN_VISITS)}
print("=== measured coverage per pair (sealed pair_sizes, step 1, median over tasks) ===")
for d in sorted(measured):
    m = measured[d]
    print("  %s d=%2d  n_min %6.0f  n_mean %6.0f  coverage_min %.4f  headroom x%.2f  kappa %.3f"
          % (m["family"], d, m["n_min_median"], m["n_mean_median"], m["coverage_min"],
             m["headroom_over_min_visits"], m["kappa_min_median"]))

# two-point log-log slope of coverage in d
d1, d2 = sorted(measured)
cov1, cov2 = measured[d1]["coverage_min"], measured[d2]["coverage_min"]
slope = math.log(cov2 / cov1) / math.log(d2 / d1)
print()
print("  two-point log-log slope of coverage_min in d: %.4f  (coverage ~ d^%.3f)"
      % (slope, slope))
print("  NOTE: an exponent fitted to two points interpolates them exactly and validates")
print("        nothing. It is a bracket, not a law.")


def coverage_min_of(d, base_d=d2, base_cov=cov2, s=slope):
    return base_cov * (d / base_d) ** s


def n_mean_of(d, base_d=d2, base_n=None, s=None):
    s = slope if s is None else s
    return C_CHAINS * (measured[base_d]["coverage_mean"]) * (d / base_d) ** s


# ------------------------------------------------ measured anchors --------------------------------- #
vars_, yrs, hs, eqs = [], [], [], []
eq_by_fam = {}
for rec in bundle["records"]:
    for route, cs in rec["routes"].items():
        for c, s in cs.items():
            for e in s["steps"]:
                if not e["emitted"]:
                    continue
                vars_.append(float(np.median(e["pair_vars"])))
                yrs.append(float(e["y_range"]))
                eqs.append(float(e["e_q"]))
                eq_by_fam.setdefault(rec["family"], []).append(float(e["e_q"]))
                hs.append(float(np.max(row_margins(np.asarray(e["pi_before"], float),
                                                   np.asarray(e["q_hat"], float)))))
V_MED, YR_MED, H_MED, EQ_MED = (float(np.median(vars_)), float(np.median(yrs)),
                                float(np.median(hs)), float(np.median(eqs)))
print()
print("=== anchors: per-pair variance %.5f | y_range %.4f | gate margin h %.4f | E_Q %.5f ==="
      % (V_MED, YR_MED, H_MED, EQ_MED))


def radius_at(d, n):
    lt = math.log(2.0 / (DELTA_STEP / d))
    return (math.sqrt(2.0 * V_MED * lt / n)
            + (7.0 / 3.0) * YR_MED * lt / max(n - 1.0, 1.0))


def EQ_at(d, n):
    return radius_at(d, n) / (1.0 - GAMMA)


sweep = []
ANCHOR_D = d2                       # anchor every E_Q prediction to the measured level
ANCHOR_EQ = float(np.median(eq_by_fam[measured[d2]["family"]]))
ANCHOR_N = n_mean_of(ANCHOR_D)
ANCHOR_R = radius_at(ANCHOR_D, ANCHOR_N)
N_REF = measured[ANCHOR_D]["n_mean_median"]
for d in (12, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256, 384, 512, 768, 1000):
    cov = coverage_min_of(d)
    n_min = C_CHAINS * cov
    n_mean = n_mean_of(d)
    # ANCHORED prediction: scale the measured E_Q by the model's radius RATIO. The anchor
    # absorbs the max-over-pairs and |mean| effects that the pure-radius formula omits, so
    # only the change is extrapolated. An unanchored version of this script was 1.5x low.
    eq_pred = ANCHOR_EQ * radius_at(d, n_mean) / ANCHOR_R
    chains_support = MIN_VISITS / cov
    chains_hold_n = N_REF / (measured[ANCHOR_D]["coverage_mean"] * (d / ANCHOR_D) ** slope)
    sweep.append(dict(
        d=d, coverage_min_predicted=round(cov, 5),
        n_min_predicted=round(n_min, 0), n_mean_predicted=round(n_mean, 0),
        support_ok_at_frozen_chains=bool(n_min >= MIN_VISITS),
        chains_needed_for_support=round(chains_support, 0),
        chains_needed_to_hold_per_pair_n=round(chains_hold_n, 0),
        E_Q_anchored_prediction=round(eq_pred, 5),
        E_Q_over_h_anchored=round(eq_pred / H_MED, 4),
        median_gate_would_close=bool(eq_pred >= H_MED),
        batch_cost_multiple=round(chains_hold_n / C_CHAINS, 2),
    ))

print()
print("=== sweep: coverage by the two-point slope; E_Q anchored to the measured level ===")
print("   d   cov_min   n_min  n_mean  supp?  C_supp  C_holdn  EQ/h   closes?  cost x")
for s in sweep:
    print("  %4d  %7.4f %6.0f %7.0f  %5s %7.0f %8.0f %6.3f  %6s  %5.2f"
          % (s["d"], s["coverage_min_predicted"], s["n_min_predicted"], s["n_mean_predicted"],
             "yes" if s["support_ok_at_frozen_chains"] else "NO",
             s["chains_needed_for_support"], s["chains_needed_to_hold_per_pair_n"],
             s["E_Q_over_h_anchored"], "YES" if s["median_gate_would_close"] else "no",
             s["batch_cost_multiple"]))

fails = [s["d"] for s in sweep if not s["support_ok_at_frozen_chains"]]
closes = [s["d"] for s in sweep if s["median_gate_would_close"]]
out = {
    "framing": ("READ-ONLY extrapolation from the sealed K=12 bundle. Not a run, not a new "
                "arm, and not a measurement at any dimension other than d=24 and d=12."),
    "frozen": {"chains": C_CHAINS, "chain_length": L_CHAIN, "min_visits": MIN_VISITS,
               "delta_step": DELTA_STEP, "gamma": GAMMA},
    "measured": measured,
    "coverage_power_law": {"slope_in_d": slope, "fitted_on": [d1, d2],
                           "warning": ("fitted to two points, so it interpolates them "
                                       "exactly and validates nothing; a bracket, not a law")},
    "anchors": {"per_pair_variance_median": V_MED, "y_range_median": YR_MED,
                "gate_margin_h_median": H_MED, "E_Q_median": EQ_MED,
                "E_Q_by_family_median": {k: float(np.median(v)) for k, v in eq_by_fam.items()},
                "anchor_d": ANCHOR_D, "anchor_E_Q": ANCHOR_EQ},
    "sweep": sweep,
    "first_d_in_sweep_where_support_fails_at_frozen_chains": (fails[0] if fails else None),
    "first_d_in_sweep_where_median_gate_closes_at_frozen_chains": (closes[0] if closes else None),
    "corrected_earlier_errors": [
        "an earlier version hardcoded kappa=0.114 while MEASURING 0.438 and 0.299 in the same "
        "run, and thereby predicted support failure at d=12 and d=24 -- contradicted by the "
        "sealed n_min of 11464 and 7432. The hardcode is gone.",
        "an earlier version predicted E_Q absolutely from the radius formula, which omits the "
        "max over pairs and the |mean| term, and came out about 1.5x below the measured "
        "E_Q/h. The prediction is now anchored to the measured E_Q and extrapolates only the "
        "radius RATIO.",
    ],
    "caveats": [
        "TWO calibration points. kappa differs between them (0.438 vs 0.299), so even the "
        "occupancy concentration is not established as constant in d.",
        "variance, y_range and h are HELD at measured medians; all three depend on the policy "
        "and Qhat, which this analysis cannot predict at another dimension. The E_Q column is "
        "conditional on them not moving.",
        "E_Q is compared against a MEDIAN h, so 'gate closes' means the median cell.",
        "The power law's exponent is an interpolation between two points.",
        "Nothing here has been run. Every number beyond d=24 and d=12 is a prediction.",
    ],
}
outp = PROJECT / "results/FP-COMPOSE-002/claude/verification/dimension_scaling.json"
outp.parent.mkdir(parents=True, exist_ok=True)
print()
print("=== sensitivity to the coverage exponent (the ONE thing two points cannot pin) ===")
print("  coverage_min ~ d^s. The 2-point fit gives s=%.3f by construction." % slope)
print("  Mechanistically, coverage_min ~ 1-exp(-L*mu_min); with mu_min ~ kappa/d and small")
print("  mu_min this tends to s = -1. The fit gives -0.625, i.e. the fit is GENTLER.")
print("  Reporting only the fit would therefore OVERSTATE the ceiling, so both are swept.")
print()
print("  s      d where support fails   d where median gate closes   chains x at d=192")
for s_try in (slope, -0.75, -0.85, -1.0):
    def cov_s(d, s=s_try):
        return cov2 * (d / d2) ** s

    def nmean_s(d, s=s_try):
        return C_CHAINS * measured[d2]["coverage_mean"] * (d / d2) ** s

    ds = [12, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 320, 384, 512, 768, 1000]
    sf = next((d for d in ds if C_CHAINS * cov_s(d) < MIN_VISITS), None)
    gc = next((d for d in ds
               if ANCHOR_EQ * radius_at(d, nmean_s(d)) / ANCHOR_R >= H_MED), None)
    c192 = MIN_VISITS / cov_s(192) / C_CHAINS
    print("  %+.3f        %-6s                    %-6s                 %.2f"
          % (s_try, sf if sf else ">1000", gc if gc else ">1000", c192))

out_extra = {}
for s_try in (slope, -0.75, -0.85, -1.0):
    def cov_s(d, s=s_try):
        return cov2 * (d / d2) ** s

    def nmean_s(d, s=s_try):
        return C_CHAINS * measured[d2]["coverage_mean"] * (d / d2) ** s

    ds = [12, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 320, 384, 512, 768, 1000]
    out_extra["s=%.3f" % s_try] = {
        "first_d_support_fails": next((d for d in ds if C_CHAINS * cov_s(d) < MIN_VISITS), None),
        "first_d_median_gate_closes": next(
            (d for d in ds if ANCHOR_EQ * radius_at(d, nmean_s(d)) / ANCHOR_R >= H_MED), None),
        "chains_multiple_needed_at_d192": round(MIN_VISITS / cov_s(192) / C_CHAINS, 2),
    }

out["exponent_sensitivity"] = out_extra

# --- what does it COST to reach a given dimension? solve for chains such that E_Q = h --- #
def chains_to_keep_gate_open(d, s):
    """Smallest C such that the anchored E_Q(d, n(d,C)) stays at or below h."""
    cov_m = measured[d2]["coverage_mean"] * (d / d2) ** s

    def eq_of(C):
        return ANCHOR_EQ * radius_at(d, max(C * cov_m, 1.0)) / ANCHOR_R

    if eq_of(C_CHAINS) < H_MED:
        return None                      # already open at the frozen budget
    lo, hi = C_CHAINS, 10 ** 9
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if eq_of(mid) >= H_MED:
            lo = mid
        else:
            hi = mid
    return hi


cost = {}
for s_try in (slope, -1.0):
    row = {}
    for d in (24, 48, 64, 96, 128, 192, 256, 384, 512, 1000):
        c = chains_to_keep_gate_open(d, s_try)
        row[d] = {"chains_needed": (None if c is None else round(c, 0)),
                  "multiple_of_frozen": (None if c is None else round(c / C_CHAINS, 2))}
    cost["s=%.3f" % s_try] = row
print()
print("=== chains needed to keep the MEDIAN gate open at dimension d (anchored) ===")
print("   d     s=-0.625            s=-1.0")
for d in (24, 48, 64, 96, 128, 192, 256, 384, 512, 1000):
    a = cost["s=%.3f" % slope][d]["multiple_of_frozen"]
    b = cost["s=-1.000"][d]["multiple_of_frozen"]
    print("  %4d   %-18s %s" % (d, "frozen budget OK" if a is None else "%.2fx frozen" % a,
                               "frozen budget OK" if b is None else "%.2fx frozen" % b))
out["chains_to_keep_gate_open"] = cost
out["caveats"].append(
    "The model captures the n and log(2/delta_dir) effects but NOT the growth of "
    "E[max_x eps_x] with d, which is absorbed into the anchor. That omitted effect works in "
    "the SAME direction as d, so the ceilings here are OPTIMISTIC: the true ceiling is likely "
    "lower than the table says.")
out["caveats"].append(
    "Variance, y_range and h are held at measured medians. A larger MDP plausibly has larger "
    "variance and a larger ||Vhat||, both of which would also tighten the gate. Every "
    "omitted effect points the same way, so read the ceiling as an upper bound.")

outp.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
print()
print(json.dumps({"coverage_slope_fitted": round(slope, 4),
                  "exponent_sensitivity": out_extra}, indent=2))

