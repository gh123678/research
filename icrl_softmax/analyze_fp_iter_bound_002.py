"""FP-ITER-BOUND-002 analyzer: the registered H1-H7 for the K schedule."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
D = 12


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=PROJECT / "results" / "FP-ITER-BOUND-002" / "claude" / "fresh_k4k8k16",
    )
    parser.add_argument(
        "--iter-001-dir",
        type=Path,
        default=PROJECT / "results" / "FP-ITER-BOUND-001" / "claude" / "fresh_K16",
    )
    args = parser.parse_args()
    bundle = json.loads((args.result_dir / "task_results.json").read_text(encoding="utf-8"))
    labels = list(bundle["arms"])
    spec = bundle["arm_spec"]
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

    out.append(f"FP-ITER-BOUND-002 analysis ({len(rows)} record-routes, "
               f"horizons {bundle['horizons']}, L12M delta_prop_fraction="
               f"{bundle['delta_prop_fraction']})")
    out.append("=" * 94)

    def steps(lb):
        return [e for _, _, _, rt in rows for e in rt[lb]["steps"]]

    st = {
        lb: {
            "emitted": sum(1 for e in steps(lb) if e["emitted"]),
            "length": float(np.mean([len(rt[lb]["steps"]) for _, _, _, rt in rows])),
            "truncated": sum(1 for _, _, _, rt in rows if rt[lb]["truncated"]),
            "gain": float(np.mean([rt[lb]["total_value_gain"] for _, _, _, rt in rows])),
            "cov": sum(1 for e in steps(lb) if not e["covers_realized"]),
            "deg": sum(
                1 for e in steps(lb) if e["emitted"] and not e["componentwise_nondegrading"]
            ),
        }
        for lb in labels
    }

    out.append("\n1. H1 (mandatory): no degradation, no coverage violation")
    say(
        all(s["deg"] == 0 and s["cov"] == 0 for s in st.values()),
        f"degradations {[st[lb]['deg'] for lb in labels]}, "
        f"coverage violations {[st[lb]['cov'] for lb in labels]}",
    )

    out.append("\n2. the cells")
    out.append(f"  {'arm':<14}{'lever':<8}{'K':>4}{'emitted':>9}{'mean len':>10}"
               f"{'reached K':>11}{'mean value gain':>18}")
    for lb in labels:
        s = st[lb]
        out.append(f"  {lb:<14}{spec[lb]['lever']:<8}{spec[lb]['horizon']:>4}{s['emitted']:>9}"
                   f"{s['length']:>10.3f}{s['truncated']:>11}{s['gain']:>18.4f}")

    out.append("\n3. H2: length falls as K grows")
    say(
        st["L12M05@K4"]["length"] > st["L12M05@K8"]["length"] > st["L12M05@K16"]["length"],
        f"len K4 {st['L12M05@K4']['length']:.3f} > K8 {st['L12M05@K8']['length']:.3f} "
        f"> K16 {st['L12M05@K16']['length']:.3f}",
    )

    out.append("\n4. H3: do trajectories reach K=4?")
    say(
        st["L12M05@K4"]["truncated"] >= 15,
        f"reached K=4: {st['L12M05@K4']['truncated']}/48 (registered >= 15)",
    )

    out.append("\n5. H4: what stops the loop -- the bound, or convergence?")
    stops = [
        float(e["e_q_over_h"])
        for e in steps("L12M05@K16")
        if not e["emitted"] and e["e_q_over_h"] is not None
    ]
    if stops:
        near = sum(1 for v in stops if v <= 1.25)
        frac = near / len(stops)
        out.append(f"  stops {len(stops)}: median E_Q/h {np.median(stops):.4f}, "
                   f"<=1.25: {near} ({frac:.0%}), >1.25: {len(stops) - near}")
        say(frac >= 0.40, f"H4 near-line stop share {frac:.0%} >= 40%")
    else:
        say(False, "H4 not evaluable: no stops in L12M05@K16")

    out.append("\n6. H5: the final arm beats FP-ITER-BOUND-001's L12M(0.5) at K=16")
    ref_path = args.iter_001_dir / "task_results.json"
    ref_emit = None
    if ref_path.exists():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        ref_emit = sum(
            rt["L12M"]["emitted_steps"] for r in ref["records"] for rt in r["routes"].values()
        )
    say(
        ref_emit is not None and st["L12M05@K16"]["emitted"] >= ref_emit,
        f"L12M(0.05)@K16 emits {st['L12M05@K16']['emitted']} vs L12M(0.5)@K16 {ref_emit}",
    )

    out.append("\n7. H6: the K=4 horizon buys value")
    ratio = st["L12M05@K4"]["gain"] / st["L12M05@K16"]["gain"]
    say(ratio >= 1.20, f"value gain ratio K4/K16 = {ratio:.3f} (registered >= 1.20)")

    out.append("\n8. H7: is the K=8 vs K=16 difference material?")
    diff = abs(st["L12M05@K8"]["length"] - st["L12M05@K16"]["length"])
    say(diff < 0.5, f"|len(K8) - len(K16)| = {diff:.3f} < 0.5")

    out.append("\n" + "=" * 94)
    out.append(f"SUMMARY  failures: {failures}")
    text = "\n".join(out)
    print(text)
    (args.result_dir / "analysis_report.md").write_text(text, encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
