"""FP-XRULE-001 analyzer: the registered H1-H6."""

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
        default=PROJECT / "results" / "FP-XRULE-001" / "claude" / "f1f2_K4",
    )
    args = parser.parse_args()
    bundle = json.loads((args.result_dir / "task_results.json").read_text(encoding="utf-8"))
    cells = list(bundle["cells"])
    fams = list(bundle["families"])
    out: list[str] = []
    failures = 0

    def say(ok: bool, text: str) -> None:
        nonlocal failures
        out.append(f"  {'PASS' if ok else 'FAIL'}  {text}")
        if not ok:
            failures += 1

    def rows(fam):
        return [
            (float(r["mixing"]), int(r["task_index"]), name, rt)
            for r in bundle["records"] if r["family"] == fam
            for name, rt in r["routes"].items()
        ]

    st: dict = {}
    for fam in fams:
        rr = rows(fam)
        st[fam] = {
            c: {
                "emitted": sum(1 for _, _, _, rt in rr for e in rt[c]["steps"] if e["emitted"]),
                "gain": float(np.mean([rt[c]["total_value_gain"] for _, _, _, rt in rr])),
                "cov": sum(1 for _, _, _, rt in rr for e in rt[c]["steps"]
                           if not e["covers_realized"]),
                "deg": sum(1 for _, _, _, rt in rr for e in rt[c]["steps"]
                           if e["emitted"] and not e["componentwise_nondegrading"]),
                "items": sum(rt[c]["items_if_run_alone"] for _, _, _, rt in rr),
                "length": float(np.mean([len(rt[c]["steps"]) for _, _, _, rt in rr])),
                "n": len(rr),
            }
            for c in cells
        }

    out.append(f"FP-XRULE-001 analysis (K={bundle['horizon']}, families {fams}, "
               f"L12S(f={bundle['split_fraction']}, p={bundle['delta_prop_fraction']}), "
               f"wall {bundle['wall_seconds_total'] / 60:.1f} min)")
    out.append("=" * 100)

    out.append("\n1. H1 (mandatory): no degradation; E_Q covers the realized error")
    deg = {f"{f}/{c}": st[f][c]["deg"] for f in fams for c in cells}
    cov = {f"{f}/{c}": st[f][c]["cov"] for f in fams for c in cells}
    say(all(v == 0 for v in deg.values()), f"degradations {deg}")
    say(all(v == 0 for v in cov.values()), f"coverage violations {cov}")

    out.append("\n2. the cells")
    for fam in fams:
        out.append(f"  family {fam} ({st[fam][cells[0]]['n']} route-records)")
        out.append(f"    {'cell':<16}{'emitted':>9}{'mean len':>10}{'mean value gain':>18}"
                   f"{'items':>16}{'cov':>5}{'deg':>5}")
        for c in cells:
            s = st[fam][c]
            out.append(f"    {c:<16}{s['emitted']:>9}{s['length']:>10.3f}{s['gain']:>18.4f}"
                       f"{s['items']:>16,}{s['cov']:>5}{s['deg']:>5}")

    out.append("\n3. H2: does the per-state rule transfer to the new families?")
    for fam in fams:
        for cert in bundle["certificates"]:
            a = st[fam][f"conj|{cert}"]["gain"]
            b = st[fam][f"perstate|{cert}"]["gain"]
            say(b > a, f"{fam} / {cert}: perstate {b:.4f} > conj {a:.4f} "
                       f"({b / a - 1:+.1%})")

    out.append("\n4. H3: does the tightened certificate transfer (direction only)?")
    for fam in fams:
        for rule in bundle["rules"]:
            a = st[fam][f"{rule}|frozen"]["gain"]
            b = st[fam][f"{rule}|L12S"]["gain"]
            say(b > a, f"{fam} / {rule}: L12S {b:.4f} > frozen {a:.4f} ({b / a - 1:+.1%})")

    out.append("\n5. H4: no reversed interaction")
    for fam in fams:
        best = st[fam]["perstate|L12S"]["gain"]
        base = st[fam]["conj|frozen"]["gain"]
        say(best >= base, f"{fam}: perstate|L12S {best:.4f} >= conj|frozen {base:.4f}")

    out.append("\n6. H5: cost and the same-total-data truncation")
    for fam in fams:
        rr = rows(fam)
        m_min = min(len(rt[c]["steps"]) for _, _, _, rt in rr for c in cells)
        out.append(f"  family {fam}: per-step budget {bundle['items_per_step']:,} items, "
                   f"K={bundle['horizon']}; truncating to {m_min} step(s) per record")
        for c in cells:
            em = 0
            gains = []
            for _, _, _, rt in rr:
                em += sum(1 for e in rt[c]["steps"][:m_min] if e["emitted"])
                gains.append(float(sum(e.get("total_value_gain", 0.0) for e in rt[c]["steps"][:m_min])))
            out.append(f"    {c:<16} items/family {st[fam][c]['items']:>14,}  "
                       f"truncated emitted {em:>4}  value {np.mean(gains):.4f}")

    out.append("\n7. H6: is the construction dimension-generic?")
    say(all(st[f][c]["cov"] == 0 for f in fams for c in cells),
        "F1 (6x4 = 24 pairs) and F2 (4x3) both have 0 coverage violations in all cells")

    out.append("\n" + "=" * 100)
    out.append(f"SUMMARY  failures: {failures}")
    text = "\n".join(out)
    print(text)
    (args.result_dir / "analysis_report.md").write_text(text, encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
