"""FP-BOUND-001 analyzer: the pre-registered hypotheses H1-H7.

Reads a sealed `evaluate_fp_bound_001.py` bundle and prints/writes the verdicts
registered in `docs/research_tasks/FP-BOUND-001.md` before the confirmatory run.
Falsified predictions are reported as falsified, never reinterpreted.

    python analyze_fp_bound_001.py --result-dir results/FP-BOUND-001/claude/c64k
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

TASK_ID = "FP-BOUND-001"
GAMMA = 0.70
R_STAR = 1.5
D = 12
LEVERS = ("frozen", "L1", "L12", "L123", "support_range")
REPORT: list[str] = []


def check(ok: bool, text: str) -> None:
    REPORT.append(f"  {'PASS' if ok else 'FAIL'}  {text}")
    if not ok:
        check.failures += 1  # type: ignore[attr-defined]


check.failures = 0  # type: ignore[attr-defined]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_of(bundle: dict):
    for rec in bundle["records"]:
        for route_name, route in rec["routes"].items():
            yield float(rec["mixing"]), int(rec["task_index"]), route_name, route


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=PROJECT / "results" / "FP-BOUND-001" / "claude" / "c64k",
    )
    args = parser.parse_args()
    bundle = load(args.result_dir / "task_results.json")
    rows = list(rows_of(bundle))
    rung = f"c{int(bundle['chains']) // 1024}k"
    delta_step = float(bundle["delta_step"])

    REPORT.append(f"{TASK_ID} analysis at {rung} ({len(rows)} record-routes, "
                  f"fresh tasks {bundle['fresh_tasks'][0]}..{bundle['fresh_tasks'][-1]})")
    REPORT.append("=" * 92)

    def e_qs(lever: str) -> np.ndarray:
        return np.array(
            [
                float(r["arms"][lever]["e_q"])
                for _, _, _, r in rows
                if r["arms"][lever]["e_q"] is not None
            ]
        )

    def emitted(lever: str) -> int:
        return sum(1 for _, _, _, r in rows if r["arms"][lever]["emitted"])

    # ---------------------------------------------------------------- H1
    REPORT.append("\n1. H1 (mandatory): soundness and monotonicity")
    viol = {
        lever: sum(
            1
            for _, _, _, r in rows
            if r["arms"][lever]["e_q"] is not None and not r["arms"][lever]["covers_realized"]
        )
        for lever in LEVERS
    }
    check(
        all(v == 0 for v in viol.values()),
        f"every lever covers the realized error on every record-route "
        f"(violations {viol})",
    )
    chain = {"frozen>L1": 0, "L1>L12": 0, "L12>L123": 0}
    worst_gain = 0.0
    for _, _, _, r in rows:
        vals = {lv: r["arms"][lv]["e_q"] for lv in ("frozen", "L1", "L12", "L123")}
        if any(v is None for v in vals.values()):
            continue
        if vals["L1"] > vals["frozen"] + 1e-15:
            chain["frozen>L1"] += 1
        if vals["L12"] > vals["L1"] + 1e-15:
            chain["L1>L12"] += 1
        if vals["L123"] > vals["L12"] + 1e-15:
            chain["L12>L123"] += 1
        worst_gain = max(worst_gain, vals["frozen"] - vals["L123"])
    check(
        all(v == 0 for v in chain.values()),
        f"monotone chain frozen >= L1 >= L12 >= L123 on every record-route "
        f"(exceptions {chain}); worst-case gain {worst_gain:.3e}",
    )
    # the propagation bound must dominate the uniform bound it replaces
    prop_ok = all(
        r["arms"]["L123"]["e_q"] <= r["arms"]["L123"]["uniform_bound"] + 1e-15
        for _, _, _, r in rows
        if r["arms"]["L123"]["e_q"] is not None
    )
    check(prop_ok, "L123 <= max_x eps_x/(1-gamma) (the bound it replaces) everywhere")

    # ------------------------------------------------------------ H2 / H3
    base = float(np.mean(e_qs("frozen")))
    means = {lv: float(np.mean(e_qs(lv))) for lv in LEVERS}
    red = {lv: means[lv] / base - 1.0 for lv in LEVERS}
    REPORT.append("\n2. magnitudes (mean E_Q over the emitting-eligible record-routes)")
    REPORT.append(f"  {'lever':<15}{'mean E_Q':>11}{'vs frozen':>12}{'emitted':>9}"
                  f"{'coverage viol':>15}")
    for lv in LEVERS:
        REPORT.append(f"  {lv:<15}{means[lv]:>11.5f}{red[lv]:>+12.2%}"
                      f"{emitted(lv):>9}{viol[lv]:>15}")
    REPORT.append(f"  realized ||Qhat-Q^pi||_inf mean: "
                  f"{np.mean([r['realized_q_sup_error'] for _, _, _, r in rows]):.5f}")
    check(-0.25 <= red["L12"] <= -0.05, f"H2 model-free reduction {red['L12']:+.1%} in [-25%, -5%]")
    check(-0.55 <= red["L123"] <= -0.20, f"H3 all-lever reduction {red['L123']:+.1%} in [-55%, -20%]")

    # ---------------------------------------------------------------- H4
    REPORT.append("\n3. H4: the decision consequence")
    check(
        emitted("L123") >= emitted("frozen") + 3,
        f"L123 emits {emitted('L123')} vs frozen {emitted('frozen')} "
        f"(registered: no fall, and at least +3)",
    )

    # ---------------------------------------------------------------- H5
    REPORT.append("\n4. H5: the FP-SHORT-001 yardstick E_Q/h")
    stop_pop = [
        r for _, _, _, r in rows if not r["arms"]["frozen"]["emitted"]
    ]
    ratios = [
        float(r["arms"]["L123"]["e_q_over_h"])
        for r in stop_pop
        if r["arms"]["L123"]["e_q_over_h"] is not None
    ]
    frozen_ratios = [
        float(r["arms"]["frozen"]["e_q_over_h"])
        for r in stop_pop
        if r["arms"]["frozen"]["e_q_over_h"] is not None
    ]
    if ratios:
        med = float(np.median(ratios))
        REPORT.append(f"  frozen-abstaining population: {len(stop_pop)} record-routes")
        REPORT.append(f"  E_Q/h frozen: median {np.median(frozen_ratios):.4f} "
                      f"range [{min(frozen_ratios):.4f}, {max(frozen_ratios):.4f}]")
        REPORT.append(f"  E_Q/h L123  : median {med:.4f} "
                      f"range [{min(ratios):.4f}, {max(ratios):.4f}]")
        revived = sum(1 for v in ratios if v < 1.0)
        REPORT.append(f"  revived by L123 in that population: {revived}/{len(ratios)}")
        check(med < 1.0, f"H5 median E_Q/h under L123 = {med:.4f} < 1.0")
    else:
        check(False, "H5 not evaluable: empty abstaining population")

    # ---------------------------------------------------------------- H6
    REPORT.append("\n5. H6: is the range lever already captured without the kernel?")
    extra = means["support_range"] / means["L1"] - 1.0
    check(
        extra >= -0.10,
        f"oracle support range adds {extra:+.1%} over L1 (registered: no more than -10%)",
    )

    # ---------------------------------------------------------------- H7
    REPORT.append("\n6. H7: additivity of the three levers")
    a = 1.0 - means["L1"] / means["frozen"]
    b = 1.0 - means["L12"] / means["L1"]
    c = 1.0 - means["L123"] / means["L12"]
    total = 1.0 - means["L123"] / means["frozen"]
    REPORT.append(f"  L1 {a:+.1%}, L2 {b:+.1%}, L3 {c:+.1%}; sum {a + b + c:+.1%}, "
                  f"combined {total:+.1%} (multiplicative would be "
                  f"{1 - (1 - a) * (1 - b) * (1 - c):+.1%})")
    check(total >= 0.8 * (a + b + c), f"H7 combined {total:+.1%} >= 80% of the sum")

    REPORT.append("\n" + "=" * 92)
    REPORT.append(f"SUMMARY  failures: {check.failures}")
    text = "\n".join(REPORT)
    print(text)
    (args.result_dir / "analysis_report.md").write_text(text, encoding="utf-8")
    return 1 if check.failures else 0


if __name__ == "__main__":
    sys.exit(main())
