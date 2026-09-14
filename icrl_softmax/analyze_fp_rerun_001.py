"""FP-RERUN-001 analyzer: the registered H1-H6, plus the cost and truncation tables
the audit asked for."""

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
        default=PROJECT / "results" / "FP-RERUN-001" / "claude" / "fresh_K16",
    )
    args = parser.parse_args()
    bundle = json.loads((args.result_dir / "task_results.json").read_text(encoding="utf-8"))
    cells = list(bundle["cells"])
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

    def steps(c):
        return [e for _, _, _, rt in rows for e in rt[c]["steps"]]

    st = {
        c: {
            "emitted": sum(1 for e in steps(c) if e["emitted"]),
            "length": float(np.mean([len(rt[c]["steps"]) for _, _, _, rt in rows])),
            "gain": float(np.mean([rt[c]["total_value_gain"] for _, _, _, rt in rows])),
            "cov": sum(1 for e in steps(c) if not e["covers_realized"]),
            "deg": sum(1 for e in steps(c) if e["emitted"] and not e["componentwise_nondegrading"]),
            "items": sum(rt[c]["items_if_run_alone"] for _, _, _, rt in rows),
            "stop_median": float(np.median(
                [float(e["e_q_over_h"]) for e in steps(c)
                 if not e["emitted"] and e["e_q_over_h"] is not None] or [float("nan")]
            )),
        }
        for c in cells
    }

    out.append(f"FP-RERUN-001 analysis (K={bundle['horizon']}, {len(rows)} record-routes, "
               f"L12S(f={bundle['split_fraction']}, p={bundle['delta_prop_fraction']}), "
               f"wall {bundle['wall_seconds_total'] / 60:.1f} min)")
    out.append("=" * 104)

    out.append("\n1. H1 (mandatory): no degradation; the certificate covers each cell's OWN producer")
    say(all(st[c]["deg"] == 0 for c in cells), f"degradations {[st[c]['deg'] for c in cells]}")
    say(all(st[c]["cov"] == 0 for c in cells), f"coverage violations {[st[c]['cov'] for c in cells]}")

    out.append("\n2. the eight cells")
    out.append(f"  {'cell':<18}{'emitted':>9}{'mean len':>10}{'mean value gain':>18}"
               f"{'cov':>5}{'deg':>5}{'items (run alone)':>20}")
    for c in cells:
        s = st[c]
        out.append(f"  {c:<18}{s['emitted']:>9}{s['length']:>10.3f}{s['gain']:>18.4f}"
                   f"{s['cov']:>5}{s['deg']:>5}{s['items']:>20,}")

    out.append("\n3. H2/H3: what the SOUND arm buys in the loop")
    nf, ns, n12 = st["numpy|frozen"], st["numpy|L12S"], st["numpy|L12"]
    say(ns["emitted"] >= 1.30 * nf["emitted"],
        f"H2 emissions {ns['emitted']} >= 1.30 x frozen {nf['emitted']} ({ns['emitted'] / nf['emitted']:.2f}x)")
    say(ns["gain"] >= 1.30 * nf["gain"],
        f"H2 value gain {ns['gain']:.4f} >= 1.30 x frozen {nf['gain']:.4f} ({ns['gain'] / nf['gain']:.2f}x)")
    say(ns["emitted"] >= n12["emitted"],
        f"H3 L12S {ns['emitted']} >= L12 {n12['emitted']}")

    out.append("\n4. H4: producer agreement under the same certificate")
    for lv in bundle["levers"]:
        n = st[f"numpy|{lv}"]["emitted"]
        m = st[f"network|{lv}"]["emitted"]
        rel = abs(m - n) / n if n else float("inf")
        say(rel <= 0.20, f"{lv}: network {m} vs numpy {n} (gap {rel:.1%} <= 20%)")
    for lv in bundle["levers"]:
        agree = tot = 0
        for _, _, _, rt in rows:
            a, b = rt[f"numpy|{lv}"]["steps"], rt[f"network|{lv}"]["steps"]
            if not a or not b:
                continue
            agree += int(bool(a[0]["emitted"]) == bool(b[0]["emitted"]))
            tot += 1
        say(agree / tot >= 0.80 if tot else False,
            f"{lv}: step-1 decisions agree {agree}/{tot} = {(agree / tot if tot else 0):.0%}")

    out.append("\n5. H5: cost, and the same-total-data comparison")
    base_items = st["numpy|frozen"]["items"]
    out.append(f"  {'cell':<18}{'items':>16}{'x frozen':>10}{'emitted':>9}{'gain':>10}")
    for c in cells:
        out.append(f"  {c:<18}{st[c]['items']:>16,}{st[c]['items'] / base_items:>10.2f}"
                   f"{st[c]['emitted']:>9}{st[c]['gain']:>10.4f}")
    # truncate each L12S cell to the steps `frozen` actually simulated, per record-route
    trunc = {}
    for c in cells:
        em = 0
        gains = []
        for _, _, _, rt in rows:
            m = len(rt["numpy|frozen"]["steps"])
            em += sum(1 for e in rt[c]["steps"][:m] if e["emitted"])
            gains.append(float(sum(e.get("total_value_gain", 0.0) for e in rt[c]["steps"][:m])))
        trunc[c] = {"emitted": em, "gain": float(np.mean(gains))}
    out.append(f"  truncated to frozen's per-record step count ({nf['length']:.3f} mean):")
    for c in ("numpy|L12S", "network|L12S", "numpy|L12"):
        out.append(f"    {c:<18} emitted {trunc[c]['emitted']:>3}  "
                   f"mean value gain {trunc[c]['gain']:.4f}  "
                   f"(frozen {nf['emitted']} / {nf['gain']:.4f})")
    say(trunc["numpy|L12S"]["gain"] >= nf["gain"],
        f"H5 same-total-data: L12S gain {trunc['numpy|L12S']['gain']:.4f} >= "
        f"frozen {nf['gain']:.4f}")

    out.append("\n6. H6: relation to the WITHDRAWN L12M number")
    say(ns["emitted"] <= 99, f"L12S {ns['emitted']} <= withdrawn L12M's 99")

    out.append("\n7. what stops the loop (audit item: per-step decomposition)")
    for c in ("numpy|frozen", "numpy|L12S", "numpy|L123"):
        vals = [float(e["e_q_over_h"]) for e in steps(c)
                if not e["emitted"] and e["e_q_over_h"] is not None]
        if vals:
            near = sum(1 for v in vals if v <= 1.25)
            out.append(f"  {c:<18} stops {len(vals):>3}  median E_Q/h {np.median(vals):.4f}  "
                       f"<=1.25: {near} ({near / len(vals):.0%})")

    out.append("\n" + "=" * 104)
    out.append(f"SUMMARY  failures: {failures}")
    text = "\n".join(out)
    print(text)
    (args.result_dir / "analysis_report.md").write_text(text, encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
