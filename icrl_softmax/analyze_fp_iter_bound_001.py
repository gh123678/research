"""FP-ITER-BOUND-001 analyzer: the pre-registered H1-H7."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
ARMS = ("frozen", "L12", "L12M", "L123")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=PROJECT / "results" / "FP-ITER-BOUND-001" / "claude" / "fresh_K16",
    )
    args = parser.parse_args()
    bundle = json.loads((args.result_dir / "task_results.json").read_text(encoding="utf-8"))
    horizon = int(bundle["horizon"])
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

    out.append(f"FP-ITER-BOUND-001 analysis (K={horizon}, {len(rows)} record-routes, "
               f"fresh tasks {bundle['fresh_tasks'][0]}..{bundle['fresh_tasks'][-1]})")
    out.append("=" * 92)

    def steps(arm):
        return [e for _, _, _, rt in rows for e in rt[arm]["steps"]]

    stats = {
        arm: {
            "emitted": sum(1 for e in steps(arm) if e["emitted"]),
            "length": float(np.mean([len(rt[arm]["steps"]) for _, _, _, rt in rows])),
            "reach": sum(1 for _, _, _, rt in rows if rt[arm]["stopped_at"] is None),
            "gain": float(np.mean([rt[arm]["total_value_gain"] for _, _, _, rt in rows])),
            "cov": sum(1 for e in steps(arm) if not e["covers_realized"]),
            "deg": sum(
                1 for e in steps(arm) if e["emitted"] and not e["componentwise_nondegrading"]
            ),
        }
        for arm in ARMS
    }

    out.append("\n1. H1 (mandatory): no degradation, no coverage violation")
    say(
        all(s["deg"] == 0 and s["cov"] == 0 for s in stats.values()),
        f"degradations {[stats[a]['deg'] for a in ARMS]}, "
        f"coverage violations {[stats[a]['cov'] for a in ARMS]}",
    )

    out.append("\n2. the four arms")
    out.append(f"  {'arm':<8}{'emitted':>9}{'mean len':>10}{'reach K':>9}"
               f"{'mean value gain':>18}{'coverage viol':>15}{'degrading':>11}")
    for arm in ARMS:
        s = stats[arm]
        out.append(f"  {arm:<8}{s['emitted']:>9}{s['length']:>10.3f}{s['reach']:>9}"
                   f"{s['gain']:>18.4f}{s['cov']:>15}{s['deg']:>11}")

    out.append("\n3. H2: trajectory length")
    say(
        stats["L12M"]["length"] > stats["frozen"]["length"]
        and stats["L123"]["length"] >= stats["L12M"]["length"],
        f"L12M {stats['L12M']['length']:.3f} > frozen {stats['frozen']['length']:.3f}, "
        f"L123 {stats['L123']['length']:.3f} >= L12M",
    )

    out.append("\n4. H3: total emissions")
    say(
        stats["L12M"]["emitted"] >= 1.10 * stats["frozen"]["emitted"],
        f"L12M {stats['L12M']['emitted']} >= 1.10 x frozen {stats['frozen']['emitted']}",
    )

    out.append("\n5. H4: total value gain")
    say(
        stats["L12M"]["gain"] >= 1.20 * stats["frozen"]["gain"],
        f"L12M {stats['L12M']['gain']:.4f} >= 1.20 x frozen {stats['frozen']['gain']:.4f}",
    )

    out.append("\n6. H5: does any trajectory reach the horizon?")
    say(
        stats["L12M"]["reach"] >= 5,
        f"L12M trajectories reaching K={horizon}: {stats['L12M']['reach']} "
        f"(registered >= 5); longest run = {max(len(rt[a]['steps']) for _, _, _, rt in rows for a in ARMS)}",
    )

    out.append("\n7. H6: are the step-1 stoppers revived?")
    pop = [
        (m, t, n, rt)
        for m, t, n, rt in rows
        if rt["frozen"]["stopped_at"] == 1
    ]
    revived = sum(1 for _, _, _, rt in pop if rt["L12M"]["stopped_at"] != 1)
    frac = revived / len(pop) if pop else 0.0
    say(frac >= 0.50, f"{revived}/{len(pop)} = {frac:.0%} revived (registered >= 50%)")

    out.append("\n8. H7: the mechanism -- E_Q/h")
    med = {}
    for arm in ARMS:
        vals = [float(e["e_q_over_h"]) for e in steps(arm) if e["e_q_over_h"] is not None]
        med[arm] = float(np.median(vals)) if vals else float("nan")
    drop = med["L12M"] / med["frozen"] - 1.0
    say(
        drop <= -0.20,
        f"median E_Q/h  frozen {med['frozen']:.4f} -> L12M {med['L12M']:.4f} "
        f"({drop:+.1%}, registered <= -20%)",
    )

    out.append("\n" + "=" * 92)
    out.append(f"SUMMARY  failures: {failures}")
    text = "\n".join(out)
    print(text)
    (args.result_dir / "analysis_report.md").write_text(text, encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
