"""FP-NET-BOUND-001 analyzer: the registered H1-H6 for the network cells."""

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
        default=PROJECT / "results" / "FP-NET-BOUND-001" / "claude" / "fresh_K16",
    )
    args = parser.parse_args()
    bundle = json.loads((args.result_dir / "task_results.json").read_text(encoding="utf-8"))
    cells = list(bundle["cells"])
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

    out.append(f"FP-NET-BOUND-001 analysis (K={horizon}, {len(rows)} record-routes, "
               f"fresh tasks {bundle['fresh_tasks'][0]}..{bundle['fresh_tasks'][-1]})")
    out.append("=" * 96)

    def steps(cell):
        return [e for _, _, _, rt in rows for e in rt[cell]["steps"]]

    st = {
        c: {
            "emitted": sum(1 for e in steps(c) if e["emitted"]),
            "length": float(np.mean([len(rt[c]["steps"]) for _, _, _, rt in rows])),
            "gain": float(np.mean([rt[c]["total_value_gain"] for _, _, _, rt in rows])),
            "cov": sum(1 for e in steps(c) if not e["covers_realized"]),
            "deg": sum(1 for e in steps(c) if e["emitted"] and not e["componentwise_nondegrading"]),
            "trunc": sum(1 for _, _, _, rt in rows if rt[c]["stopped_at"] is None),
        }
        for c in cells
    }

    out.append("\n1. H1 (mandatory): no degradation, and the certificate covers the "
               "producer's OWN error")
    say(
        all(s["deg"] == 0 for s in st.values()),
        f"degradations {[st[c]['deg'] for c in cells]}",
    )
    say(
        all(s["cov"] == 0 for s in st.values()),
        f"coverage violations {[st[c]['cov'] for c in cells]}",
    )

    out.append("\n2. the eight cells")
    out.append(f"  {'cell':<18}{'emitted':>9}{'mean len':>10}{'reached K':>11}"
               f"{'mean value gain':>18}{'cov':>6}{'deg':>6}")
    for c in cells:
        s = st[c]
        out.append(f"  {c:<18}{s['emitted']:>9}{s['length']:>10.3f}{s['trunc']:>11}"
                   f"{s['gain']:>18.4f}{s['cov']:>6}{s['deg']:>6}")

    out.append("\n3. H2/H3: does the tightening transfer to the network?")
    net_f, net_m = st["network|frozen"], st["network|L12M"]
    say(
        net_m["emitted"] >= 1.10 * net_f["emitted"],
        f"network: L12M emits {net_m['emitted']} vs frozen {net_f['emitted']} (>= 1.10x)",
    )
    say(
        net_m["gain"] >= 1.20 * net_f["gain"],
        f"network: value gain {net_m['gain']:.4f} vs frozen {net_f['gain']:.4f} (>= 1.20x)",
    )

    out.append("\n4. H4: producer comparison under the same certificate")
    for lever in bundle["levers"]:
        n = st[f"numpy|{lever}"]["emitted"]
        m = st[f"network|{lever}"]["emitted"]
        rel = abs(m - n) / n if n else float("inf")
        say(rel <= 0.20, f"{lever}: network {m} vs numpy {n} (relative gap {rel:.1%} <= 20%)")

    out.append("\n5. H5: decision-level agreement at step 1")
    for lever in bundle["levers"]:
        agree = tot = 0
        for _, _, _, rt in rows:
            a = rt[f"numpy|{lever}"]["steps"]
            b = rt[f"network|{lever}"]["steps"]
            if not a or not b:
                continue
            agree += int(bool(a[0]["emitted"]) == bool(b[0]["emitted"]))
            tot += 1
        frac = agree / tot if tot else 0.0
        say(frac >= 0.80, f"{lever}: step-1 decisions agree on {agree}/{tot} = {frac:.0%} (>= 80%)")

    out.append("\n6. H6: the network's realized error against numpy's")
    worst = 0.0
    for _, _, _, rt in rows:
        for i in range(min(len(rt["numpy|L12M"]["steps"]), len(rt["network|L12M"]["steps"]))):
            a = float(rt["numpy|L12M"]["steps"][i]["realized_q_sup_error"])
            b = float(rt["network|L12M"]["steps"][i]["realized_q_sup_error"])
            if a > 0:
                worst = max(worst, b / a)
    say(worst <= 3.0, f"max ratio network/numpy realized error = {worst:.3f} (<= 3)")

    out.append("\n7. the mechanism, on the same yardstick as the numpy line")
    for c in ("numpy|L12M", "network|L12M"):
        vals = [float(e["e_q_over_h"]) for e in steps(c) if e["e_q_over_h"] is not None]
        if vals:
            out.append(f"  {c:<18} E_Q/h median {np.median(vals):.4f}  "
                       f"range [{min(vals):.4f}, {max(vals):.4f}]")
    for c in ("numpy|frozen", "network|frozen"):
        vals = [float(e["e_q_over_h"]) for e in steps(c) if e["e_q_over_h"] is not None]
        if vals:
            out.append(f"  {c:<18} E_Q/h median {np.median(vals):.4f}  "
                       f"range [{min(vals):.4f}, {max(vals):.4f}]")

    out.append("\n" + "=" * 96)
    out.append(f"SUMMARY  failures: {failures}")
    text = "\n".join(out)
    print(text)
    (args.result_dir / "analysis_report.md").write_text(text, encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
