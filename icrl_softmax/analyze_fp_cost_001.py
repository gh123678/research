"""FP-COST-001 analyzer: equal-total-data comparison, H1-H5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=PROJECT / "results" / "FP-COST-001" / "claude" / "all",
    )
    parser.add_argument(
        "--xrule-dir",
        type=Path,
        default=PROJECT / "results" / "FP-XRULE-002" / "claude" / "f1f2_K12",
    )
    args = parser.parse_args()
    bundle = json.loads((args.result_dir / "task_results.json").read_text(encoding="utf-8"))
    ref = bundle["perstate_reference"]
    cell = bundle["cells"][0]
    out: list[str] = []
    failures = 0

    def say(ok: bool, text: str) -> None:
        nonlocal failures
        out.append(f"  {'PASS' if ok else 'FAIL'}  {text}")
        if not ok:
            failures += 1

    # ---- per-ladder-run aggregates
    rows: dict[str, list[dict]] = {}
    for run in bundle["runs"]:
        rr = [rt[cell] for r in run["records"] for rt in r["routes"].values()]
        steps = [e for r in run["records"] for rt in r["routes"].values()
                 for e in rt[cell]["steps"]]
        rows.setdefault(run["family"], []).append({
            "chains": run["chains"],
            "items": sum(rt["items_if_run_alone"] for rt in rr),
            "gain": float(np.mean([rt["total_value_gain"] for rt in rr])),
            "emitted": sum(rt["emitted_steps"] for rt in rr),
            "mean_sim": float(np.mean([rt["simulated_steps"] for rt in rr])),
            "cov": sum(1 for e in steps if not e["covers_realized"]),
            "deg": sum(1 for e in steps if e["emitted"] and not e["componentwise_nondegrading"]),
            "wall": run["wall_seconds"],
            "records": run["records"],
        })
    for fam in rows:
        rows[fam].sort(key=lambda d: d["chains"])

    out.append(f"FP-COST-001: the conjunctive rule's data-cost curve vs the per-state arm "
               f"(K={bundle['horizon']}, cell {cell})")
    out.append("=" * 100)

    out.append("\n1. H1 (mandatory)")
    allcov = [d["cov"] for fam in rows for d in rows[fam]]
    alldeg = [d["deg"] for fam in rows for d in rows[fam]]
    say(all(v == 0 for v in alldeg), f"degradations {alldeg}")
    say(all(v == 0 for v in allcov), f"coverage violations {allcov}")

    out.append("\n2. the conjunctive ladder, and the per-state reference")
    for fam in rows:
        out.append(f"  family {fam}: per-state|frozen at 16384 chains spends "
                   f"{ref[fam]['items']:,} items and gains {ref[fam]['gain']:.4f}")
        out.append(f"    {'chains':>8}{'items':>16}{'x perstate cost':>17}"
                   f"{'mean sim steps':>16}{'emitted':>9}{'mean value gain':>18}{'wall s':>8}")
        for d in rows[fam]:
            out.append(f"    {d['chains']:>8}{d['items']:>16,}"
                       f"{d['items'] / ref[fam]['items']:>17.2f}{d['mean_sim']:>16.3f}"
                       f"{d['emitted']:>9}{d['gain']:>18.4f}{d['wall']:>8.0f}")

    out.append("\n3. H2: does the 16384 rung reproduce FP-XRULE-002's conj|frozen?")
    ref_path = args.xrule_dir / "task_results.json"
    if not ref_path.exists():
        say(False, f"FP-XRULE-002 bundle missing at {ref_path}")
    else:
        other = json.loads(ref_path.read_text(encoding="utf-8"))
        ref_map = {
            (r["family"], float(r["mixing"]), int(r["task_index"]), n): rt
            for r in other["records"] for n, rt in r["routes"].items()
        }
        worst = 0.0
        mism = 0
        for run in bundle["runs"]:
            if run["chains"] != 16384:
                continue
            for r in run["records"]:
                for name, rt in r["routes"].items():
                    a = ref_map[(r["family"], float(r["mixing"]), int(r["task_index"]), name)][
                        "conj|frozen"
                    ]
                    b = rt[cell]
                    worst = max(worst, abs(a["total_value_gain"] - b["total_value_gain"]))
                    if a["emitted_steps"] != b["emitted_steps"]:
                        mism += 1
        say(worst == 0.0 and mism == 0,
            f"max |delta value| {worst:.3e}, emission mismatches {mism}")

    out.append("\n4. H3: at EQUAL TOTAL DATA, does the per-state rule still win?")
    for fam in rows:
        target = ref[fam]["items"]
        below = [d for d in rows[fam] if d["items"] <= target]
        above = [d for d in rows[fam] if d["items"] >= target]
        if not below or not above:
            say(False, f"{fam}: the ladder does not bracket the per-state cost {target:,} "
                       f"(rungs {[d['items'] for d in rows[fam]]})")
            continue
        lo, hi = below[-1], above[0]
        if lo["items"] == hi["items"]:
            interp = lo["gain"]
        else:
            w = (target - lo["items"]) / (hi["items"] - lo["items"])
            interp = lo["gain"] + w * (hi["gain"] - lo["gain"])
        out.append(f"  {fam}: bracket {lo['chains']} chains ({lo['items']:,}, gain {lo['gain']:.4f})"
                   f" .. {hi['chains']} chains ({hi['items']:,}, gain {hi['gain']:.4f})")
        out.append(f"       linear interpolation at {target:,} items -> {interp:.4f}; "
                   f"per-state measured {ref[fam]['gain']:.4f} "
                   f"({ref[fam]['gain'] / interp:.2f}x)")
        say(interp < ref[fam]["gain"],
            f"{fam}: interpolated conjunctive {interp:.4f} < per-state {ref[fam]['gain']:.4f}")

    out.append("\n5. H4: marginal value per item")
    for fam in rows:
        ds = sorted(rows[fam], key=lambda d: d["items"])
        if len(ds) < 2:
            continue
        margins = []
        for a, b in zip(ds, ds[1:]):
            di = b["items"] - a["items"]
            dg = b["gain"] - a["gain"]
            margins.append((a["chains"], b["chains"], dg / di * 1e8))
        avg = ref[fam]["gain"] / ref[fam]["items"] * 1e8
        txt = ", ".join(f"{a}->{b}: {m:+.4f}" for a, b, m in margins)
        out.append(f"  {fam}: conjunctive marginal value per 1e8 items: {txt}; "
                   f"per-state average {avg:+.4f} per 1e8 items")
        say(all(m < avg for _, _, m in margins),
            f"{fam}: every conjunctive marginal step is below the per-state average "
            f"({avg:.4f} per 1e8)")

    out.append("\n" + "=" * 100)
    out.append(f"SUMMARY  failures: {failures}")
    text = "\n".join(out)
    print(text)
    (args.result_dir / "analysis_report.md").write_text(text, encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
