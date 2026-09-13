"""FP-SHORT-001 analysis: why do some records stop emitting?

Scores H1--H6. H1 and H2 are mandatory; H3--H6 are the pre-registered diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
TASK_ID = "FP-SHORT-001"
PRIMARY = ("expected_exact", "expected_finite")
ARMS = ("frozen_grid", "extended_grid")
SEALED_8X = (
    PROJECT / "results" / "FP-ITER8X-001" / "claude" / "formal" / "task_results.json"
)
REPORT: list[str] = []
FAILURES = 0

# Pre-registered in docs/research_tasks/FP-SHORT-001.md before the run.
H3_MARGIN_BAND = (1.0, 2.0)
H4_BLOCKING_STATES = 1
SHORT_MAX_STEPS = 5
H6_MIN_IMPROVEMENT = 0.10


def check(ok: bool, message: str) -> bool:
    global FAILURES
    if ok:
        REPORT.append(f"  PASS  {message}")
    else:
        FAILURES += 1
        REPORT.append(f"  FAIL  {message}")
    return bool(ok)


def hypothesis(ok: bool, message: str) -> bool:
    REPORT.append(f"  {'PASS' if ok else 'FALSIFIED'}  {message}")
    return bool(ok)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_of(bundle: dict):
    return [
        (r["mixing"], r["task_index"], route, r["routes"][route])
        for r in bundle["records"]
        for route in PRIMARY
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=PROJECT / "results" / "FP-SHORT-001" / "claude" / "formal",
    )
    parser.add_argument("--sealed-dir", type=Path, default=SEALED_8X)
    args = parser.parse_args()

    bundle = load(args.result_dir / "task_results.json")
    rows = rows_of(bundle)
    sealed = load(args.sealed_dir)
    sealed_by_key = {
        (float(r["mixing"]), int(r["task_index"])): r for r in sealed["records"]
    }

    REPORT.append(f"{TASK_ID} analysis: why do some records stop emitting?")
    REPORT.append("=" * 92)
    REPORT.append(f"  {len(rows)} route-records, horizon {bundle['max_steps']}, "
                  f"certification {bundle['cert_chains'] * 64:,} items")
    REPORT.append(f"  frozen grid   {bundle['frozen_grid']}")
    REPORT.append(f"  extended grid {bundle['extended_grid']}")

    # ---------------------------------------------------------------- H1
    REPORT.append("\n1. H1: is the instrumented path the frozen rule? (mandatory)")
    unfaithful = []
    steps_total = 0
    for m, t, route, block in rows:
        for s in block["frozen_grid"]["steps"]:
            steps_total += 1
            if s.get("faithful_to_sealed_rule") is not True:
                unfaithful.append((m, t, route, s["step"]))
    check(
        not unfaithful,
        f"the instrumented decision matches fs.improvement_for at all {steps_total} "
        f"steps ({len(unfaithful)} mismatches)",
    )
    # And the whole arm must reproduce the sealed FP-ITER8X-001 frozen arm.
    divergence = []
    for m, t, route, block in rows:
        ref = sealed_by_key.get((m, t))
        if ref is None:
            continue
        ref_steps = ref["routes"][route]["frozen"]["steps"]
        for i, s in enumerate(block["frozen_grid"]["steps"]):
            if i >= len(ref_steps):
                break
            r = ref_steps[i]
            if (
                s["update_emitted"] != r["update_emitted"]
                or s["eta_selected"] != r["eta_selected"]
                or s["e_q"] != r["e_q"]
            ):
                divergence.append((m, t, route, s["step"]))
    check(
        not divergence,
        f"the frozen-grid arm reproduces FP-ITER8X-001's sealed frozen arm exactly "
        f"({len(divergence)} divergences)",
    )
    for item in divergence[:5]:
        REPORT.append(f"        divergence: {item}")

    # ---------------------------------------------------------------- H2
    REPORT.append("\n2. H2: does the extension change any emitting decision? "
                  "(mandatory, and a theorem)")
    changed = []
    for m, t, route, block in rows:
        frozen = block["frozen_grid"]["steps"]
        extended = block["extended_grid"]["steps"]
        for i, s in enumerate(frozen):
            if i >= len(extended):
                break
            if not s["update_emitted"]:
                continue
            e = extended[i]
            if (
                not e["update_emitted"]
                or e["eta_selected"] != s["eta_selected"]
            ):
                changed.append((m, t, route, s["step"]))
    check(
        not changed,
        f"no emitting decision changes under the extended grid "
        f"({len(changed)} counterexamples)",
    )
    for item in changed[:5]:
        REPORT.append(f"        counterexample: {item}")

    # ------------------------------------------------------- 3. trajectories
    REPORT.append("\n3. Trajectory lengths, per arm")
    lengths: dict[str, dict[tuple, int]] = {arm: {} for arm in ARMS}
    for m, t, route, block in rows:
        for arm in ARMS:
            lengths[arm][(m, t, route)] = int(block[arm]["emitted_steps"])
    for arm in ARMS:
        vals = sorted(lengths[arm].values())
        REPORT.append(
            f"  {arm}: total emitted steps {sum(vals)}, median "
            f"{vals[len(vals) // 2]}, max {vals[-1]}, "
            f"short (<= {SHORT_MAX_STEPS}) {sum(1 for v in vals if v <= SHORT_MAX_STEPS)}"
        )
    extended_more = [
        key
        for key in lengths["frozen_grid"]
        if lengths["extended_grid"][key] > lengths["frozen_grid"][key]
    ]
    REPORT.append(f"  trajectories the extension lengthens: {len(extended_more)}")
    for key in extended_more[:8]:
        REPORT.append(
            f"        {key}: {lengths['frozen_grid'][key]} -> "
            f"{lengths['extended_grid'][key]} steps"
        )

    # ------------------------------------------------------- 4. the diagnosis
    REPORT.append("\n4. The diagnosis at each stopping step")
    short_keys = [
        key for key, v in lengths["frozen_grid"].items() if v <= SHORT_MAX_STEPS
    ]
    REPORT.append(
        f"  {len(short_keys)} short trajectories (<= {SHORT_MAX_STEPS} emitted steps "
        f"under the frozen grid)"
    )
    diagnosable = []
    REPORT.append(
        f"  {'record':>26} {'steps':>6} {'E_Q':>9} {'h':>9} {'E_Q/h':>7} "
        f"{'blocking':>9} {'smallest passing eta':>21}"
    )
    for m, t, route, block in rows:
        key = (m, t, route)
        if lengths["frozen_grid"][key] > SHORT_MAX_STEPS:
            continue
        steps = block["frozen_grid"]["steps"]
        last = steps[-1]
        diag = last.get("diagnosis")
        if not diag:
            continue
        passing = [
            p["eta"] for p in diag["per_eta"] if p["passes"]
        ]
        smallest = min(passing) if passing else None
        diagnosable.append(
            {
                "key": key,
                "steps": lengths["frozen_grid"][key],
                "e_q": diag["e_q"],
                "h": diag["gate_margin_h"],
                "ratio": diag["e_q_over_h"],
                "blocking": last["blocking_states"],
                "smallest_passing_eta": smallest,
            }
        )
        REPORT.append(
            f"  {f'{m}/{t} {route[:14]}':>26} {lengths['frozen_grid'][key]:>6} "
            f"{diag['e_q']:>9.5f} {diag['gate_margin_h']:>9.5f} "
            f"{(diag['e_q_over_h'] if diag['e_q_over_h'] is not None else float('nan')):>7.3f} "
            f"{last['blocking_states']:>9} "
            f"{(str(round(smallest, 5)) if smallest is not None else 'none passes'):>21}"
        )

    # ---------------------------------------------------------------- H3
    REPORT.append("\n5. H3: do the records stop narrowly?")
    ratios = [d["ratio"] for d in diagnosable if d["ratio"] is not None]
    within = sum(1 for r in ratios if H3_MARGIN_BAND[0] <= r <= H3_MARGIN_BAND[1])
    if ratios:
        REPORT.append(
            f"  E_Q/h over {len(ratios)} short trajectories: median "
            f"{sorted(ratios)[len(ratios) // 2]:.3f}, min {min(ratios):.3f}, "
            f"max {max(ratios):.3f}"
        )
    h3 = bool(ratios) and within * 2 > len(ratios)
    hypothesis(
        h3,
        f"H3 {'PASS' if h3 else 'FALSIFIED'}: {within}/{len(ratios)} stop with "
        f"E_Q/h in [{H3_MARGIN_BAND[0]:.0f}, {H3_MARGIN_BAND[1]:.0f}]",
    )

    # ---------------------------------------------------------------- H4
    REPORT.append("\n6. H4: does a single state block the update?")
    one = sum(1 for d in diagnosable if d["blocking"] == H4_BLOCKING_STATES)
    distribution: dict[int, int] = {}
    for d in diagnosable:
        distribution[d["blocking"]] = distribution.get(d["blocking"], 0) + 1
    REPORT.append(f"  blocking-state distribution over {len(diagnosable)} short "
                  f"trajectories: {dict(sorted(distribution.items()))}")
    h4 = bool(diagnosable) and one * 2 > len(diagnosable)
    hypothesis(
        h4,
        f"H4 {'PASS' if h4 else 'FALSIFIED'}: exactly one state blocks in {one}/"
        f"{len(diagnosable)} cases",
    )

    # ------------------------------------------------------------- H5/H6
    REPORT.append("\n7. H5/H6: is the eta grid the binding constraint?")
    h5 = len(extended_more) >= 1
    hypothesis(
        h5,
        f"H5 {'PASS' if h5 else 'FALSIFIED'}: the extended grid lengthens "
        f"{len(extended_more)} trajectory/trajectories",
    )
    frozen_total = sum(lengths["frozen_grid"].values())
    extended_total = sum(lengths["extended_grid"].values())
    gain = (
        (extended_total - frozen_total) / frozen_total if frozen_total else 0.0
    )
    REPORT.append(
        f"  total emitted steps: frozen {frozen_total}, extended {extended_total} "
        f"({gain:+.1%})"
    )
    h6 = gain >= H6_MIN_IMPROVEMENT
    hypothesis(
        h6,
        f"H6 {'PASS' if h6 else 'FALSIFIED'}: the extension raises total emitted "
        f"steps by {gain:+.1%} against the registered {H6_MIN_IMPROVEMENT:.0%}",
    )
    REPORT.append(
        "  the extended grid is a DIAGNOSTIC, not a proposed protocol: the line's "
        "frozen protocol specifies the grid, and adopting a new one would need its "
        "own task and justification."
    )

    # ------------------------------------------------------------- summary
    REPORT.append("\n" + "=" * 92)
    REPORT.append("SUMMARY")
    REPORT.append(f"  H1 faithful          : {'PASS' if not unfaithful else 'FAIL'}")
    REPORT.append(f"  H2 extension inert   : {'PASS' if not changed else 'FAIL'}")
    REPORT.append(f"  H3 narrow stop       : {'PASS' if h3 else 'FALSIFIED'}")
    REPORT.append(f"  H4 one state blocks  : {'PASS' if h4 else 'FALSIFIED'}")
    REPORT.append(f"  H5 grid is binding   : {'PASS' if h5 else 'FALSIFIED'}")
    REPORT.append(f"  H6 extension worth it: {'PASS' if h6 else 'FALSIFIED'}")
    REPORT.append("=" * 92)
    REPORT.append(
        "hypotheses were falsified, which per the task sheet is a RESULT, not a\n"
        "failed construction. RESULT below covers the construction checks only."
    )
    REPORT.append(
        "CONSTRUCTION: " + ("PASS" if FAILURES == 0 else f"FAIL ({FAILURES} failed)")
    )

    summary = {
        "task_id": TASK_ID,
        "short_trajectories": len(short_keys),
        "diagnosable": len(diagnosable),
        "H1_unfaithful": len(unfaithful),
        "H1_divergences_from_sealed": len(divergence),
        "H2_changed_decisions": len(changed),
        "H3": {
            "verdict": "PASS" if h3 else "FALSIFIED",
            "within_band": within,
            "total": len(ratios),
            "band": list(H3_MARGIN_BAND),
            "median_ratio": sorted(ratios)[len(ratios) // 2] if ratios else None,
        },
        "H4": {
            "verdict": "PASS" if h4 else "FALSIFIED",
            "one_state": one,
            "total": len(diagnosable),
            "distribution": {str(k): v for k, v in sorted(distribution.items())},
        },
        "H5": {
            "verdict": "PASS" if h5 else "FALSIFIED",
            "lengthened": len(extended_more),
        },
        "H6": {
            "verdict": "PASS" if h6 else "FALSIFIED",
            "frozen_total": frozen_total,
            "extended_total": extended_total,
            "gain": gain,
        },
        "trajectory_lengths": {
            arm: {f"{k[0]}/{k[1]}/{k[2]}": v for k, v in lengths[arm].items()}
            for arm in ARMS
        },
        "diagnosis_of_short": [
            {**d, "key": list(d["key"])} for d in diagnosable
        ],
    }
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.result_dir / "analysis_report.md").write_text(
        "\n".join(REPORT) + "\n", encoding="utf-8"
    )
    print("\n".join(REPORT))


if __name__ == "__main__":
    main()
