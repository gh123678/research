"""FP-XRULE-002 analyzer: the registered H1-H6 at `K = 12`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT))
from evaluate_fp_xfam_001 import FAMILIES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=PROJECT / "results" / "FP-XRULE-002" / "claude" / "f1f2_K12",
    )
    args = parser.parse_args()
    bundle = json.loads((args.result_dir / "task_results.json").read_text(encoding="utf-8"))
    cells = list(bundle["cells"])
    fams = list(bundle["families"])
    K = int(bundle["horizon"])
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
                "sim": float(np.mean([len(rt[c]["steps"]) for _, _, _, rt in rr(f)])),
                "reach": sum(1 for _, _, _, rt in rr(f) if rt[c]["stopped_at"] is None),
                "gain": float(np.mean([rt[c]["total_value_gain"] for _, _, _, rt in rr(f)])),
                "cov": sum(1 for _, _, _, rt in rr(f) for e in rt[c]["steps"]
                           if not e["covers_realized"]),
                "deg": sum(1 for _, _, _, rt in rr(f) for e in rt[c]["steps"]
                           if e["emitted"] and not e["componentwise_nondegrading"]),
                "items": sum(rt[c]["items_if_run_alone"] for _, _, _, rt in rr(f)),
                "partial": float(np.mean([
                    1.0 if e["states_updated"] < FAMILIES[f]["n_states"] else 0.0
                    for _, _, _, rt in rr(f) for e in rt[c]["steps"] if e["emitted"]
                ] or [0.0])),
            }
            for c in cells
        }
        for f in fams
    }

    out.append(f"FP-XRULE-002 analysis (K={K}, families {fams}, "
               f"L12S(f={bundle['split_fraction']}, p={bundle['delta_prop_fraction']}), "
               f"wall {bundle['wall_seconds_total'] / 60:.1f} min)")
    out.append("=" * 100)

    out.append("\n1. H1 (mandatory)")
    say(all(st[f][c]["deg"] == 0 for f in fams for c in cells),
        f"degradations {[st[f][c]['deg'] for f in fams for c in cells]}")
    say(all(st[f][c]["cov"] == 0 for f in fams for c in cells),
        f"coverage violations {[st[f][c]['cov'] for f in fams for c in cells]}")

    out.append("\n2. the cells")
    for f in fams:
        out.append(f"  family {f} ({len(rr(f))} route-records, S={FAMILIES[f]['n_states']})")
        out.append(f"    {'cell':<16}{'emitted':>9}{'mean sim steps':>15}{'reached K':>11}"
                   f"{'mean value gain':>18}{'items':>16}{'partial':>9}")
        for c in cells:
            s = st[f][c]
            out.append(f"    {c:<16}{s['emitted']:>9}{s['sim']:>15.3f}{s['reach']:>11}"
                       f"{s['gain']:>18.4f}{s['items']:>16,}{s['partial']:>9.0%}")

    out.append("\n3. H2/H3: is the per-state loop self-terminating within K?")
    for f in fams:
        say(st[f]["perstate|frozen"]["reach"] + st[f]["perstate|L12S"]["reach"] >= 1,
            f"{f}: per-state trajectories reaching K={K}: "
            f"frozen {st[f]['perstate|frozen']['reach']}, L12S {st[f]['perstate|L12S']['reach']}")
    for f in fams:
        for c in ("perstate|frozen", "perstate|L12S"):
            say(st[f][c]["sim"] >= 8.0,
                f"{f} / {c}: mean simulated steps {st[f][c]['sim']:.3f} >= 8")

    out.append("\n4. H4: the certificate's marginal value at long K")
    for f in fams:
        a = st[f]["perstate|frozen"]["gain"]
        b = st[f]["perstate|L12S"]["gain"]
        say(b >= a, f"{f}: perstate L12S {b:.4f} >= frozen {a:.4f} ({b / a - 1:+.1%})")

    out.append("\n5. H5: cost and the same-total-data truncation")
    for f in fams:
        base_cell = "conj|frozen"
        base_steps = [len(rt[base_cell]["steps"]) for _, _, _, rt in rr(f)]
        m = int(np.min(base_steps))
        out.append(f"  family {f}: per-step budget {bundle['items_per_step']:,}; "
                   f"truncating to conj|frozen's shortest run ({m} step(s))")
        for c in cells:
            em, gains = 0, []
            for _, _, _, rt in rr(f):
                em += sum(1 for e in rt[c]["steps"][:m] if e["emitted"])
                gains.append(float(sum(e.get("total_value_gain", 0.0)
                                       for e in rt[c]["steps"][:m])))
            out.append(f"    {c:<16} items {st[f][c]['items']:>14,}  "
                       f"truncated emitted {em:>4}  value {np.mean(gains):.4f}")

    out.append("\n6. H6: is the advantage from PARTIAL updates?")
    for f in fams:
        for c in ("perstate|frozen", "perstate|L12S"):
            say(st[f][c]["partial"] >= 0.50,
                f"{f} / {c}: {st[f][c]['partial']:.0%} of emitted steps updated fewer than "
                f"S={FAMILIES[f]['n_states']} states")

    out.append("\n" + "=" * 100)
    out.append(f"SUMMARY  failures: {failures}")
    text = "\n".join(out)
    print(text)
    (args.result_dir / "analysis_report.md").write_text(text, encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
