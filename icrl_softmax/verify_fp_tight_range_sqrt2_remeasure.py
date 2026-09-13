"""Re-measure the two figures invalidated by the ``sqrt(2)`` defect, and prove the
re-runs are faithful by self-validation.

Background
----------
``fixed_policy_bernstein_certificate.py``'s second-moment step wrote
``(E^2/2) sqrt(log(1/delta)/n)`` where Hoeffding on ``Z = Y^2`` requires
``(E^2/2) sqrt(2 log(1/delta)/n)`` -- too TIGHT by ``sqrt(2)``. The defect reached
two published figures:

* ``FP-TIGHT-001`` section 4 / ``H3``: the ``bernstein`` arm's ``-15.0%``;
* ``FP-RANGE-001`` section 2: the ``data_range`` ladder
  (``+6.9% / -24.5% / -43.4% / -56.4%``).

Both records carry correction banners forbidding those numbers from being quoted
again until re-measured. This script re-measures them from fresh sealed bundles
(``formal_v2``) and checks the re-runs against the originals.

Why the comparison is trustworthy: in each task only ONE arm calls the defective
slack. ``frozen`` / ``frozen_same_sample``, ``empirical_bernstein`` and
``counterfactual_no_envelope`` are untouched by the fix and are recomputed by the
re-run from the SAME deterministic batch. If every one of their per-record-route
numbers is bit-identical between the two bundles, the re-run reproduced the sealed
experiment exactly and the only thing that moved is the corrected formula.

Read-only: reads both bundles, writes only ``tmp/``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parent
TIGHT = PROJECT / "results" / "FP-TIGHT-001" / "claude"
RANGE = PROJECT / "results" / "FP-RANGE-001" / "claude"

UNAFFECTED_TIGHT = ("frozen", "empirical_bernstein", "counterfactual_no_envelope")
UNAFFECTED_RANGE = ("frozen_same_sample", "empirical_bernstein", "counterfactual_no_envelope")
TIGHT_BAND_H3 = (-0.25, -0.10)
TIGHT_BAND_H4 = (-0.35, -0.18)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def route_records(bundle: dict):
    for rec in bundle["records"]:
        for route in bundle["routes"] if "routes" in bundle else ("expected_exact", "expected_finite"):
            pass
    return None


# --------------------------------------------------------------------------
# FP-TIGHT-001
# --------------------------------------------------------------------------
def tight_rows(bundle: dict):
    """((record-route, arm), entry) -- the arm MUST be in the key, since every
    record-route carries all four arms."""
    for rec in bundle["records"]:
        for route, payload in rec["routes"].items():
            for arm, entry in payload["arms"].items():
                key = (round(float(rec["mixing"]), 6), int(rec["task_index"]), route)
                yield (key, arm), entry


def tight_compare():
    old = load(TIGHT / "formal" / "task_results.json")
    new = load(TIGHT / "formal_v2" / "task_results.json")
    old_map = dict(tight_rows(old))
    new_map = dict(tight_rows(new))
    if set(old_map) != set(new_map):
        raise SystemExit("the two bundles do not cover the same record-route-arms")

    report: dict = {"task": "FP-TIGHT-001"}

    # ---- self-validation on the untouched arms ----
    self_val = {}
    for arm in UNAFFECTED_TIGHT:
        worst = 0.0
        mismatched = 0
        n = 0
        for key, e_o in old_map.items():
            if key[1] != arm:
                continue
            n += 1
            e_n = new_map[key]
            for field in ("e_q", "epsilon_res", "eta_selected", "min_lb"):
                a, b = e_o.get(field), e_n.get(field)
                if a is None or b is None:
                    if a != b:
                        mismatched += 1
                    continue
                worst = max(worst, abs(float(a) - float(b)))
            for field in ("status", "status_certificate", "emitted", "covers_realized",
                          "ordered_reasons"):
                if e_o.get(field) != e_n.get(field):
                    mismatched += 1
        self_val[arm] = {"n": n, "max_abs_numeric_delta": worst, "field_mismatches": mismatched}
    report["self_validation_untouched_arms"] = self_val

    # ---- per-arm summaries ----
    def arm_summary(bundle_map, arm):
        e_qs, emitted, covers = [], 0, 0
        for key, e in bundle_map.items():
            if key[1] != arm:
                continue
            if e.get("status_certificate") == "certificate_emitted":
                e_qs.append(float(e["e_q"]))
            if e.get("emitted"):
                emitted += 1
            if e.get("covers_realized"):
                covers += 1
        return {
            "n_records": sum(1 for k in bundle_map if k[1] == arm),
            "mean_e_q": float(np.mean(e_qs)) if e_qs else None,
            "emitted": emitted,
            "covers_realized": covers,
        }

    frozen_before = arm_summary(old_map, "frozen")["mean_e_q"]
    frozen_after = arm_summary(new_map, "frozen")["mean_e_q"]
    table = {}
    for arm in ("frozen", "bernstein", "empirical_bernstein", "counterfactual_no_envelope"):
        before = arm_summary(old_map, arm)
        after = arm_summary(new_map, arm)
        row = {"before": before, "after": after}
        if before["mean_e_q"] is not None:
            row["pct_before"] = before["mean_e_q"] / frozen_before - 1.0
            row["pct_after"] = after["mean_e_q"] / frozen_after - 1.0
        table[arm] = row
    report["arms"] = table

    # ---- flips, recomputed on the corrected run (compare RECORD-ROUTES, not
    # (record-route, arm) keys, or the sets can never intersect) ----
    frozen_emits = {k[0] for k, e in new_map.items() if k[1] == "frozen" and e.get("emitted")}
    flipped = {}
    for arm in ("bernstein", "empirical_bernstein"):
        emitted = {k[0] for k, e in new_map.items() if k[1] == arm and e.get("emitted")}
        flipped[arm] = {
            "frozen_emits": len(frozen_emits),
            "arm_emits": len(emitted),
            "flipped_to_emission": sorted(emitted - frozen_emits),
            "flipped_to_abstention": sorted(frozen_emits - emitted),
        }
    report["flips_vs_frozen_after_fix"] = flipped
    report["verdicts"] = {
        "H3_band": list(TIGHT_BAND_H3),
        "H3_before": table["bernstein"].get("pct_before"),
        "H3_after": table["bernstein"].get("pct_after"),
        "H3_after_in_band": bool(
            TIGHT_BAND_H3[0] <= table["bernstein"]["pct_after"] <= TIGHT_BAND_H3[1]
        ),
        "H4_band": list(TIGHT_BAND_H4),
        "H4_after": table["empirical_bernstein"].get("pct_after"),
        "H4_after_in_band": bool(
            TIGHT_BAND_H4[0] <= table["empirical_bernstein"]["pct_after"] <= TIGHT_BAND_H4[1]
        ),
    }
    return report


# --------------------------------------------------------------------------
# FP-RANGE-001
# --------------------------------------------------------------------------
def range_rows(bundle: dict):
    for rec in bundle["records"]:
        for route, payload in rec["routes"].items():
            for rung, entry in payload["rungs"].items():
                yield (round(float(rec["mixing"]), 6), int(rec["task_index"]), route), int(rung), entry


def range_compare():
    old = load(RANGE / "formal" / "task_results.json")
    new = load(RANGE / "formal_v2" / "task_results.json")
    rungs = [int(r) for r in new["rungs"]]
    old_map = {(k, r): e for k, r, e in range_rows(old)}
    new_map = {(k, r): e for k, r, e in range_rows(new)}
    report: dict = {"task": "FP-RANGE-001", "rungs": rungs}

    self_val = {}
    for arm in UNAFFECTED_RANGE:
        worst, mismatched = 0.0, 0
        for (key, rung), e_o in old_map.items():
            e_n = new_map[(key, rung)]
            if arm not in e_o or arm not in e_n:
                continue
            for field in ("e_q", "epsilon_res", "eta_selected"):
                a, b = e_o[arm].get(field), e_n[arm].get(field)
                if a is None or b is None:
                    if a != b:
                        mismatched += 1
                    continue
                worst = max(worst, abs(float(a) - float(b)))
            for field in ("status_certificate", "emitted", "covers_realized", "ordered_reasons"):
                if e_o[arm].get(field) != e_n[arm].get(field):
                    mismatched += 1
        self_val[arm] = {"max_abs_numeric_delta": worst, "field_mismatches": mismatched}
    report["self_validation_untouched_arms"] = self_val

    # The analyzer's baseline is the frozen certificate AT RUNG 1, used for every
    # rung (the frozen arm is only computed at the first rung).
    def frozen_base(bundle_map):
        vals = [
            float(e["frozen_same_sample"]["e_q"])
            for (key, r), e in bundle_map.items()
            if r == rungs[0]
            and e.get("frozen_same_sample", {}).get("e_q") is not None
        ]
        return float(np.mean(vals)) if vals else None

    base_new, base_old = frozen_base(new_map), frozen_base(old_map)

    ladder = {}
    for rung in rungs:
        row = {}
        for arm in ("frozen_same_sample", "data_range", "empirical_bernstein",
                    "counterfactual_no_envelope"):
            e_qs, emitted = [], 0
            for (key, r), e in new_map.items():
                if r != rung or arm not in e:
                    continue
                if e[arm].get("e_q") is not None:
                    e_qs.append(float(e[arm]["e_q"]))
                if e[arm].get("emitted"):
                    emitted += 1
            row[arm] = {
                "mean_e_q": float(np.mean(e_qs)) if e_qs else None,
                "emitted": emitted,
                "n": len(e_qs),
            }
        old_e_qs = [
            float(e["data_range"]["e_q"])
            for (key, r), e in old_map.items()
            if r == rung and e.get("data_range", {}).get("e_q") is not None
        ]
        row["data_range_before"] = {
            "mean_e_q": float(np.mean(old_e_qs)) if old_e_qs else None,
            "emitted": sum(
                1
                for (key, r), e in old_map.items()
                if r == rung and e.get("data_range", {}).get("emitted")
            ),
        }
        row["pct_before"] = (
            row["data_range_before"]["mean_e_q"] / base_old - 1.0
            if base_old and row["data_range_before"]["mean_e_q"] is not None
            else None
        )
        row["pct_after"] = (
            row["data_range"]["mean_e_q"] / base_new - 1.0
            if base_new and row["data_range"]["mean_e_q"] is not None
            else None
        )
        ladder[rung] = row
    report["baseline_frozen_rung1"] = {"before": base_old, "after": base_new}
    report["ladder"] = ladder

    # price decomposition mechanism (record section 4): tail mass exact, bias grows
    mech = {}
    for rung in rungs:
        tails, biases, concs = [], [], []
        for (key, r), e in new_map.items():
            if r != rung:
                continue
            pd = e["data_range"].get("price_decomposition")
            if not pd:
                continue
            for name, sink in (
                ("empirical_tail_mass", tails),
                ("cauchy_schwarz_bias", biases),
                ("concentration", concs),
            ):
                if name in pd:
                    sink.append(float(pd[name]))
        mech[rung] = {
            key: {
                "mean": float(np.mean(vals)) if vals else None,
                "max": float(np.max(vals)) if vals else None,
                "n": len(vals),
            }
            for key, vals in (("tail_mass", tails), ("bias", biases), ("concentration", concs))
        }
    report["price_decomposition"] = mech
    return report


def main() -> int:
    out = {}
    print("=" * 92)
    print("FP-TIGHT-001: re-measuring the bernstein arm after the sqrt(2) fix")
    print("=" * 92)
    t = tight_compare()
    out["tight"] = t
    for arm, sv in t["self_validation_untouched_arms"].items():
        print(f"  self-validation {arm:<28} max|delta| {sv['max_abs_numeric_delta']:.3e}  "
              f"field mismatches {sv['field_mismatches']}")
    print()
    print(f"  {'arm':<28}{'mean E_Q before':>17}{'after':>12}{'% before':>12}{'% after':>11}"
          f"{'emit before':>13}{'emit after':>12}")
    for arm, row in t["arms"].items():
        b, a = row["before"], row["after"]
        pb = f"{row['pct_before']:+.2%}" if "pct_before" in row else "--"
        pa = f"{row['pct_after']:+.2%}" if "pct_after" in row else "--"
        print(f"  {arm:<28}{b['mean_e_q']:>17.4f}{a['mean_e_q']:>12.4f}{pb:>12}{pa:>11}"
              f"{b['emitted']:>13}{a['emitted']:>12}")
    v = t["verdicts"]
    print()
    print(f"  H3 band {v['H3_band']}: {v['H3_before']:+.2%} -> {v['H3_after']:+.2%}  "
          f"in band after fix: {v['H3_after_in_band']}")
    print(f"  H4 band {v['H4_band']}: {v['H4_after']:+.2%}  in band: {v['H4_after_in_band']}")
    for arm, d in t["flips_vs_frozen_after_fix"].items():
        print(f"  vs frozen (emits {d['frozen_emits']}), {arm} emits {d['arm_emits']}: "
              f"{len(d['flipped_to_emission'])} route-records gain emission, "
              f"{len(d['flipped_to_abstention'])} lose it")

    print()
    print("=" * 92)
    print("FP-RANGE-001: re-measuring the data_range ladder after the sqrt(2) fix")
    print("=" * 92)
    r = range_compare()
    out["range"] = r
    for arm, sv in r["self_validation_untouched_arms"].items():
        print(f"  self-validation {arm:<28} max|delta| {sv['max_abs_numeric_delta']:.3e}  "
              f"field mismatches {sv['field_mismatches']}")
    print()
    print(f"  {'rung':>5}{'frozen@1x':>12}{'data_range before':>20}{'after':>11}"
          f"{'% before':>11}{'% after':>11}{'emit b/a':>11}")
    for rung, row in r["ladder"].items():
        fm = row["frozen_same_sample"]["mean_e_q"]
        print(f"  {rung:>5}{(f'{fm:.4f}' if fm is not None else '--'):>12}"
              f"{row['data_range_before']['mean_e_q']:>20.4f}"
              f"{row['data_range']['mean_e_q']:>11.4f}"
              f"{row['pct_before']:>11.2%}{row['pct_after']:>11.2%}"
              f"{str(row['data_range_before']['emitted']) + '/' + str(row['data_range']['emitted']):>11}")
    print(f"  (baseline: frozen at 1x, before {r['baseline_frozen_rung1']['before']:.4f} "
          f"/ after {r['baseline_frozen_rung1']['after']:.4f})")
    print()
    print("  price decomposition (means, corrected run):")
    for rung, m in r["price_decomposition"].items():
        print(f"    rung {rung:>2}: tail_mass {m['tail_mass']['mean']}  "
              f"bias {m['bias']['mean']:.5f}  concentration {m['concentration']['mean']:.5f}  "
              f"(n={m['bias']['n']})")

    (ROOT / "tmp").mkdir(exist_ok=True)
    (ROOT / "tmp" / "sqrt2_remeasure.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
