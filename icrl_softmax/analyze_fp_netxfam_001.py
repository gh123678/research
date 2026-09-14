"""FP-NETX-002 analyzer: the registered H1-H6 for the network across families."""

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
        default=PROJECT / "results" / "FP-NETX-002" / "claude" / "f1f2_K4",
    )
    parser.add_argument(
        "--xrule-dir",
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

    def rr(fam):
        return [
            (float(r["mixing"]), int(r["task_index"]), n, rt)
            for r in bundle["records"] if r["family"] == fam
            for n, rt in r["routes"].items()
        ]

    st = {
        f: {
            c: {
                "emitted": sum(1 for _, _, _, rt in rr(f) for e in rt[c]["steps"] if e["emitted"]),
                "gain": float(np.mean([rt[c]["total_value_gain"] for _, _, _, rt in rr(f)])),
                "cov": sum(1 for _, _, _, rt in rr(f) for e in rt[c]["steps"]
                           if not e["covers_realized"]),
                "deg": sum(1 for _, _, _, rt in rr(f) for e in rt[c]["steps"]
                           if e["emitted"] and not e["componentwise_nondegrading"]),
                "items": sum(rt[c]["items_if_run_alone"] for _, _, _, rt in rr(f)),
                "partial": float(np.mean([
                    1.0 if e["states_updated"] < next(
                        v["n_states"] for k, v in __import__(
                            "evaluate_fp_xfam_001").FAMILIES.items() if k == f
                    ) else 0.0
                    for _, _, _, rt in rr(f) for e in rt[c]["steps"] if e["emitted"]
                ] or [0.0])),
            }
            for c in cells
        }
        for f in fams
    }

    out.append(f"FP-NETX-002 analysis (K={bundle['horizon']}, families {fams}, per-state rule, "
               f"{bundle['network_calls_total']} network calls, "
               f"wall {bundle['wall_seconds_total'] / 60:.1f} min)")
    out.append("=" * 100)

    out.append("\n1. H1 (mandatory): no degradation; the certificate covers each producer's OWN error")
    say(all(st[f][c]["deg"] == 0 for f in fams for c in cells),
        f"degradations {[st[f][c]['deg'] for f in fams for c in cells]}")
    say(all(st[f][c]["cov"] == 0 for f in fams for c in cells),
        f"coverage violations {[st[f][c]['cov'] for f in fams for c in cells]}")

    out.append("\n2. the cells")
    for f in fams:
        out.append(f"  family {f}")
        out.append(f"    {'cell':<16}{'emitted':>9}{'mean value gain':>18}{'items':>16}"
                   f"{'partial':>9}{'cov':>5}{'deg':>5}")
        for c in cells:
            s = st[f][c]
            out.append(f"    {c:<16}{s['emitted']:>9}{s['gain']:>18.4f}{s['items']:>16,}"
                       f"{s['partial']:>9.0%}{s['cov']:>5}{s['deg']:>5}")

    out.append("\n3. H2: producer agreement across the new families")
    for f in fams:
        for cert in bundle["certificates"]:
            n = st[f][f"numpy|{cert}"]["emitted"]
            m = st[f][f"network|{cert}"]["emitted"]
            rel = abs(m - n) / n if n else float("inf")
            say(rel <= 0.20, f"{f} / {cert}: network {m} vs numpy {n} (gap {rel:.1%} <= 20%)")
    for f in fams:
        for cert in bundle["certificates"]:
            agree = tot = 0
            for _, _, _, rt in rr(f):
                a, b = rt[f"numpy|{cert}"]["steps"], rt[f"network|{cert}"]["steps"]
                if not a or not b:
                    continue
                agree += int(bool(a[0]["emitted"]) == bool(b[0]["emitted"]))
                tot += 1
            say(tot > 0 and agree / tot >= 0.80,
                f"{f} / {cert}: step-1 decisions agree {agree}/{tot} "
                f"= {(agree / tot if tot else 0):.0%}")

    out.append("\n4. H3: the network's realized error against numpy's")
    for f in fams:
        worst = 0.0
        for _, _, _, rt in rr(f):
            for i in range(min(len(rt["numpy|L12S"]["steps"]), len(rt["network|L12S"]["steps"]))):
                a = float(rt["numpy|L12S"]["steps"][i]["realized_q_sup_error"])
                b = float(rt["network|L12S"]["steps"][i]["realized_q_sup_error"])
                if a > 0:
                    worst = max(worst, b / a)
        say(worst <= 3.0, f"{f}: max ratio network/numpy realized error = {worst:.3f} (<= 3)")

    out.append("\n5. H4: built-in cross-check against FP-XRULE-001")
    ref_path = args.xrule_dir / "task_results.json"
    if not ref_path.exists():
        say(False, f"FP-XRULE-001 bundle missing at {ref_path}")
    else:
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        ref_map = {
            (r["family"], float(r["mixing"]), int(r["task_index"]), n): rt
            for r in ref["records"] for n, rt in r["routes"].items()
        }
        worst_gain = 0.0
        mismatches = 0
        for r in bundle["records"]:
            for name, rt in r["routes"].items():
                for cert in bundle["certificates"]:
                    key = (r["family"], float(r["mixing"]), int(r["task_index"]), name)
                    a = ref_map[key][f"perstate|{cert}"]
                    b = rt[f"numpy|{cert}"]
                    worst_gain = max(
                        worst_gain,
                        abs(a["total_value_gain"] - b["total_value_gain"]),
                    )
                    if a["emitted_steps"] != b["emitted_steps"]:
                        mismatches += 1
        say(worst_gain == 0.0 and mismatches == 0,
            f"numpy cells reproduce FP-XRULE-001's perstate|* : max |d value| {worst_gain:.3e}, "
            f"emission mismatches {mismatches}")

    out.append("\n6. H6: does the float32 gap grow with the pair count?")
    gaps = {}
    for f in fams:
        worst = 0.0
        for _, _, _, rt in rr(f):
            for cert in bundle["certificates"]:
                a = rt[f"numpy|{cert}"]["steps"]
                b = rt[f"network|{cert}"]["steps"]
                for i in range(min(len(a), len(b))):
                    worst = max(worst, float(np.max(np.abs(
                        np.asarray(a[i]["q_hat"], dtype=np.float64)
                        - np.asarray(b[i]["q_hat"], dtype=np.float64)
                    ))))
        gaps[f] = worst
        say(worst <= 1e-4, f"{f}: max |q_net - q_numpy| = {worst:.3e} (<= 1e-4)")
    if len(gaps) == 2:
        f1, f2 = fams
        out.append(f"  pair counts: {f1} = 24 pairs, {f2} = 12 pairs; "
                   f"gap ratio {gaps[f1] / max(gaps[f2], 1e-300):.2f} "
                   f"(a joint growth with the pair count would show here)")

    out.append("\n" + "=" * 100)
    out.append(f"SUMMARY  failures: {failures}")
    text = "\n".join(out)
    print(text)
    (args.result_dir / "analysis_report.md").write_text(text, encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
