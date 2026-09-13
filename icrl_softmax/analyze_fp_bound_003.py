"""FP-BOUND-003 analyzer: the lopsided-risk-split hypotheses H1-H6.

Bundle schema matches FP-BOUND-002, so this only differs in the registered bands
and in the two extra mandatory clauses about `L12M(0.05)` versus `L12` and
versus `L12M(0.5)`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
LEVERS = ("frozen", "L1", "L12", "L123", "L12M")
D = 12


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=PROJECT / "results" / "FP-BOUND-003" / "claude" / "c64k",
    )
    parser.add_argument(
        "--bound-001-dir",
        type=Path,
        default=PROJECT / "results" / "FP-BOUND-001" / "claude",
    )
    parser.add_argument(
        "--bound-002-dir",
        type=Path,
        default=PROJECT / "results" / "FP-BOUND-002" / "claude",
    )
    args = parser.parse_args()
    bundle = json.loads((args.result_dir / "task_results.json").read_text(encoding="utf-8"))
    rung = f"c{int(bundle['chains']) // 1024}k"
    delta_step = float(bundle["delta_step"])
    rows = [
        (float(r["mixing"]), int(r["task_index"]), name, rt)
        for r in bundle["records"]
        for name, rt in r["routes"].items()
    ]
    out: list[str] = []
    failures = 0

    def say(ok: bool, text: str) -> None:
        nonlocal failures
        out.append(f"  {'PASS' if ok else 'FAIL'}  {text}")
        if not ok:
            failures += 1

    out.append(f"FP-BOUND-003 analysis at {rung} (delta_prop_fraction="
               f"{bundle['delta_prop_fraction']}, {len(rows)} record-routes)")
    out.append("=" * 92)

    def eq(rt, lv):
        return rt["arms"][lv]["e_q"]

    means = {
        lv: float(np.mean([eq(rt, lv) for _, _, _, rt in rows if eq(rt, lv) is not None]))
        for lv in LEVERS
    }
    emitted = {lv: sum(1 for _, _, _, rt in rows if rt["arms"][lv]["emitted"]) for lv in LEVERS}
    viol = {
        lv: sum(
            1 for _, _, _, rt in rows
            if eq(rt, lv) is not None and not rt["arms"][lv]["covers_realized"]
        )
        for lv in LEVERS
    }

    out.append("\n1. H1 (mandatory, three clauses)")
    say(all(v == 0 for v in viol.values()), f"coverage violations {viol}")
    ex_l12 = sum(
        1 for _, _, _, rt in rows
        if eq(rt, "L12M") is not None and eq(rt, "L12") is not None
        and eq(rt, "L12M") > eq(rt, "L12") + 1e-15
    )
    say(ex_l12 == 0, f"L12M(0.05) <= L12 on every record-route (exceptions {ex_l12}) "
                     f"-- the clause FP-BOUND-002 falsified")
    ref2_path = args.bound_002_dir / rung / "task_results.json"
    if ref2_path.exists():
        ref2 = json.loads(ref2_path.read_text(encoding="utf-8"))
        ref2_map = {
            (float(r["mixing"]), int(r["task_index"]), n): rt
            for r in ref2["records"]
            for n, rt in r["routes"].items()
        }
        ex_l12m = sum(
            1 for m, t, n, rt in rows
            if eq(rt, "L12M") is not None
            and ref2_map[(m, t, n)]["arms"]["L12M"]["e_q"] is not None
            and eq(rt, "L12M") > ref2_map[(m, t, n)]["arms"]["L12M"]["e_q"] + 1e-15
        )
        say(ex_l12m == 0, f"L12M(0.05) <= L12M(0.5) on every record-route (exceptions {ex_l12m})")
    else:
        say(False, f"FP-BOUND-002 bundle missing at {ref2_path}")

    out.append("\n2. magnitudes")
    out.append(f"  {'lever':<10}{'mean E_Q':>11}{'vs frozen':>12}{'emitted':>9}")
    for lv in LEVERS:
        out.append(f"  {lv:<10}{means[lv]:>11.5f}{means[lv] / means['frozen'] - 1:>+12.2%}"
                   f"{emitted[lv]:>9}")
    red = means["L12M"] / means["frozen"] - 1.0
    say(-0.45 <= red <= -0.30, f"H2 L12M reduction {red:+.1%} in [-45%, -30%]")

    out.append("\n3. H3: recovery of the kernel arm's gain")
    rec = (means["L12"] - means["L12M"]) / (means["L12"] - means["L123"])
    out.append(f"  L12 {means['L12']:.5f} -> L12M {means['L12M']:.5f} -> L123 {means['L123']:.5f}"
               f"   recovery {rec:.2%}")
    say(rec >= 0.80, f"H3 recovery {rec:.2%} >= 80%")

    out.append("\n4. H4: emissions")
    ref_emit = None
    if ref2_path.exists():
        ref_emit = sum(
            1 for r in ref2["records"] for rt in r["routes"].values()
            if rt["arms"]["L12M"]["emitted"]
        )
    say(
        emitted["L12M"] >= (ref_emit if ref_emit is not None else 0),
        f"L12M(0.05) emits {emitted['L12M']} vs L12M(0.5) {ref_emit}",
    )

    out.append("\n5. H5: risk accounting")
    bad = 0
    for _, _, _, rt in rows:
        prop = rt["arms"]["L12M"].get("propagation")
        if not prop:
            bad += 1
            continue
        if abs(float(prop["delta_eps"]) + float(prop["delta_prop_fraction"]) * delta_step
               - delta_step) > 1e-12:
            bad += 1
        if abs(D * float(rt["arms"]["L12M"]["delta_each"]) - float(prop["delta_eps"])) > 1e-15:
            bad += 1
        spent = float(prop["delta_eps"]) + (
            float(prop["delta_per_iteration"]) * int(prop["n_iter"]) * D
            + float(prop["delta_final"]) * D
        )
        if abs(spent - delta_step) > 1e-12:
            bad += 1
    say(bad == 0, f"delta_eps + delta_prop = delta_step and d*delta_each = delta_eps (bad {bad})")

    out.append("\n6. H6: shared arms reproduce FP-BOUND-001")
    ref_path = args.bound_001_dir / rung / "task_results.json"
    if not ref_path.exists():
        say(False, f"FP-BOUND-001 bundle missing at {ref_path}")
    else:
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        ref_map = {
            (float(r["mixing"]), int(r["task_index"]), n): rt
            for r in ref["records"]
            for n, rt in r["routes"].items()
        }
        worst, mism = 0.0, 0
        for m, t, n, rt in rows:
            for lv in ("frozen", "L1", "L12", "L123"):
                a = ref_map[(m, t, n)]["arms"][lv]
                b = rt["arms"][lv]
                if a["e_q"] is None or b["e_q"] is None:
                    if a["e_q"] != b["e_q"]:
                        mism += 1
                    continue
                worst = max(worst, abs(float(a["e_q"]) - float(b["e_q"])))
                if a["emitted"] != b["emitted"]:
                    mism += 1
        say(worst == 0.0 and mism == 0,
            f"max |delta E_Q| {worst:.3e}, decision mismatches {mism}")

    out.append("\n" + "=" * 92)
    out.append(f"SUMMARY  failures: {failures}")
    text = "\n".join(out)
    print(text)
    (args.result_dir / "analysis_report.md").write_text(text, encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
