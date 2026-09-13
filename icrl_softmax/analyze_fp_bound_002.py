"""FP-BOUND-002 analyzer: the pre-registered H1-H7 for the model-free propagation arm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

LEVERS = ("frozen", "L1", "L12", "L123", "L12M")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=PROJECT / "results" / "FP-BOUND-002" / "claude" / "c64k",
    )
    parser.add_argument(
        "--bound-001-dir",
        type=Path,
        default=PROJECT / "results" / "FP-BOUND-001" / "claude",
    )
    args = parser.parse_args()
    bundle = json.loads((args.result_dir / "task_results.json").read_text(encoding="utf-8"))
    rows = [
        (float(r["mixing"]), int(r["task_index"]), name, rt)
        for r in bundle["records"]
        for name, rt in r["routes"].items()
    ]
    rung = f"c{int(bundle['chains']) // 1024}k"
    out: list[str] = []
    failures = 0

    def say(ok: bool, text: str) -> None:
        nonlocal failures
        out.append(f"  {'PASS' if ok else 'FAIL'}  {text}")
        if not ok:
            failures += 1

    out.append(f"FP-BOUND-002 analysis at {rung} ({len(rows)} record-routes)")
    out.append("=" * 92)

    means = {
        lv: float(
            np.mean([float(rt["arms"][lv]["e_q"]) for _, _, _, rt in rows
                     if rt["arms"][lv]["e_q"] is not None])
        )
        for lv in LEVERS
    }
    emitted = {
        lv: sum(1 for _, _, _, rt in rows if rt["arms"][lv]["emitted"]) for lv in LEVERS
    }
    viol = {
        lv: sum(
            1
            for _, _, _, rt in rows
            if rt["arms"][lv]["e_q"] is not None and not rt["arms"][lv]["covers_realized"]
        )
        for lv in LEVERS
    }

    out.append("\n1. H1 (mandatory): coverage and L12M <= L12")
    say(all(v == 0 for v in viol.values()), f"coverage violations {viol}")
    worse = sum(
        1
        for _, _, _, rt in rows
        if rt["arms"]["L12M"]["e_q"] is not None
        and rt["arms"]["L12"]["e_q"] is not None
        and rt["arms"]["L12M"]["e_q"] > rt["arms"]["L12"]["e_q"] + 1e-15
    )
    say(worse == 0, f"L12M <= L12 on every record-route (exceptions {worse})")

    out.append("\n2. magnitudes")
    out.append(f"  {'lever':<10}{'mean E_Q':>11}{'vs frozen':>12}{'emitted':>9}")
    red = {}
    for lv in LEVERS:
        red[lv] = means[lv] / means["frozen"] - 1.0
        out.append(f"  {lv:<10}{means[lv]:>11.5f}{red[lv]:>+12.2%}{emitted[lv]:>9}")
    say(-0.45 <= red["L12M"] <= -0.15, f"H2 L12M reduction {red['L12M']:+.1%} in [-45%, -15%]")

    out.append("\n3. H3: how much of the kernel arm's gain does the model-free arm recover?")
    rec = (means["L12"] - means["L12M"]) / (means["L12"] - means["L123"])
    out.append(f"  L12 {means['L12']:.5f} -> L12M {means['L12M']:.5f} -> L123 {means['L123']:.5f}"
               f"   recovery {rec:.1%}")
    say(rec >= 0.70, f"H3 recovery {rec:.1%} >= 70%")

    out.append("\n4. H4: is the kernel arm at least as tight?")
    n_le = sum(
        1
        for _, _, _, rt in rows
        if rt["arms"]["L123"]["e_q"] is not None
        and rt["arms"]["L12M"]["e_q"] is not None
        and rt["arms"]["L123"]["e_q"] <= rt["arms"]["L12M"]["e_q"] + 1e-15
    )
    frac = n_le / len(rows)
    say(frac >= 0.90, f"L123 <= L12M on {n_le}/{len(rows)} = {frac:.1%} (registered >= 90%)")

    out.append("\n5. H5: emissions")
    say(
        emitted["L12M"] >= emitted["L12"] + 3,
        f"L12M emits {emitted['L12M']} vs L12 {emitted['L12']} (registered: at least +3)",
    )

    out.append("\n6. H6: risk accounting")
    bad = 0
    for _, _, _, rt in rows:
        prop = rt["arms"]["L12M"].get("propagation")
        if not prop:
            bad += 1
            continue
        total = float(prop["delta_eps"]) + (
            float(prop["delta_per_iteration"]) * int(prop["n_iter"]) * 12
            + float(prop["delta_final"]) * 12
        )
        if abs(total - float(bundle["delta_step"])) > 1e-12:
            bad += 1
        if abs(float(prop["delta_eps"]) - 12.0 * float(rt["arms"]["L12M"]["delta_each"])) > 1e-15:
            bad += 1
    say(bad == 0, f"delta_eps + delta_prop = delta_step and delta_eps = d*delta_each (bad {bad})")

    out.append("\n7. H7: the four shared arms reproduce FP-BOUND-001 bit-for-bit")
    other = args.bound_001_dir / rung / "task_results.json"
    if not other.exists():
        say(False, f"FP-BOUND-001 bundle missing at {other}")
    else:
        ref = json.loads(other.read_text(encoding="utf-8"))
        ref_map = {
            (float(r["mixing"]), int(r["task_index"]), name): rt
            for r in ref["records"]
            for name, rt in r["routes"].items()
        }
        worst = 0.0
        mism = 0
        for mixing, ti, name, rt in rows:
            for lv in ("frozen", "L1", "L12", "L123"):
                a = ref_map[(mixing, ti, name)]["arms"][lv]
                b = rt["arms"][lv]
                if a["e_q"] is None or b["e_q"] is None:
                    if a["e_q"] != b["e_q"]:
                        mism += 1
                    continue
                worst = max(worst, abs(float(a["e_q"]) - float(b["e_q"])))
                if a["emitted"] != b["emitted"] or a["eta_selected"] != b["eta_selected"]:
                    mism += 1
        say(worst == 0.0 and mism == 0,
            f"max |delta E_Q| {worst:.3e}, decision mismatches {mism} over "
            f"{len(rows)} record-routes x 4 arms")

    out.append("\n" + "=" * 92)
    out.append(f"SUMMARY  failures: {failures}")
    text = "\n".join(out)
    print(text)
    (args.result_dir / "analysis_report.md").write_text(text, encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
