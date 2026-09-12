"""FP-ATTN-ITER6-001 analysis: the sixth certified step on the literal network.

Reads the network six-step bundle, the sealed FP-ITER5-001 network bundle (for the
inertness proof) and this line's numpy six-step bundle (for path agreement).

The point of the task is that the network path catches up, so the path comparison
is reported as counts AND as an explicit set comparison, and any decision or eta
flip is listed individually rather than summarised away.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parent
TASK_ID = "FP-ATTN-ITER6-001"
PRIMARY = ("expected_exact", "expected_finite")
NET6 = PROJECT / "results" / "FP-ATTN-ITER6-001" / "claude" / "network"
NET5 = PROJECT / "results" / "FP-ITER5-001" / "claude" / "network"
NP6 = PROJECT / "results" / "FP-ITER6-001" / "claude" / "numpy"
NP5 = PROJECT / "results" / "FP-ITER5-001" / "claude" / "numpy"
LEVELS = 6
REPORT: list[str] = []
FAILURES = 0


def check(ok: bool, message: str) -> bool:
    """A construction check. Failing one invalidates the run."""
    global FAILURES
    if ok:
        REPORT.append(f"  PASS  {message}")
    else:
        FAILURES += 1
        REPORT.append(f"  FAIL  {message}")
    return bool(ok)


def hypothesis(ok: bool, message: str) -> bool:
    """A scientific prediction. Falsifying one is a RESULT, not a failed check."""
    REPORT.append(f"  {'PASS' if ok else 'FALSIFIED'}  {message}")
    return bool(ok)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def blocks(bundle: dict):
    return [
        (r, route, r["routes"][route])
        for r in bundle["records"]
        for route in PRIMARY
    ]


def per_level(source_steps) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for level in range(1, LEVELS + 1):
        all_rows = [
            (r["mixing"], r["task_index"], route, s)
            for r, route, _, s in source_steps
            if s["step"] == level
        ]
        rows = [row for row in all_rows if row[3]["update_emitted"]]
        gains = [row[3]["oracle_audit"]["total_value_gain"] for row in rows]
        # The gap aggregate must be over EVERY row at the level, not only the
        # emitting ones: that is how the sealed headline figures were produced, and
        # an emitting-only maximum is not comparable to them.
        out[level] = {
            "emissions": len(rows),
            "mean": (sum(gains) / len(gains)) if gains else None,
            "min": min(gains) if gains else None,
            "max": max(gains) if gains else None,
            "emitters": {(row[0], row[1], row[2]) for row in rows},
            "gaps": [
                row[3]["q_hat_gap_vs_numpy"]
                for row in all_rows
                if "q_hat_gap_vs_numpy" in row[3]
            ],
            "gaps_emitted": [
                row[3]["q_hat_gap_vs_numpy"]
                for row in rows
                if "q_hat_gap_vs_numpy" in row[3]
            ],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-dir", type=Path, default=NET6)
    parser.add_argument("--sealed-five", type=Path, default=NET5)
    parser.add_argument("--numpy-dir", type=Path, default=NP6)
    args = parser.parse_args()

    bundle = load(args.network_dir / "task_results.json")
    sealed5 = load(args.sealed_five / "task_results.json")
    numpy6 = load(args.numpy_dir / "task_results.json")

    REPORT.append(f"{TASK_ID} analysis: a sixth certified step on the network path")
    REPORT.append("=" * 74)

    # ------------------------------------------------------------- 1. inputs
    REPORT.append("\n1. Frozen inputs")
    check(len(bundle["records"]) == 24, f"24 records (found {len(bundle['records'])})")
    check(
        int(bundle["max_steps"]) == LEVELS,
        f"network horizon {LEVELS} (found {bundle['max_steps']})",
    )
    atol = float(bundle["atol"])
    check(atol == 1e-4, f"ATOL is the frozen 1e-4 (found {atol})")
    # The comparison baseline must be the SIX-step numpy run. An earlier draft
    # checked `config["reference"]`, which the evaluator does not record -- that
    # check would have been vacuously true. The bundle itself is the evidence:
    # reference_comparisons can only reach step 6 if the reference bundle had a
    # sixth row, and reference_emitted_steps can only be 6 if it emitted six times.
    comparison_depths = [
        len(block["reference_comparisons"])
        for _, _, block in blocks(bundle)
    ]
    reference_six = [
        block["reference_emitted_steps"]
        for _, _, block in blocks(bundle)
        if block.get("reference_emitted_steps") is not None
    ]
    check(
        bool(comparison_depths) and max(comparison_depths) == LEVELS,
        f"the comparator reaches step {LEVELS}, so it is the six-step numpy run "
        f"(deepest comparison found: {max(comparison_depths, default=0)})",
    )
    check(
        max(reference_six, default=0) == LEVELS,
        f"the comparator itself emitted six steps "
        f"(max reference_emitted_steps: {max(reference_six, default=0)})",
    )
    REPORT.append(f"  torch {bundle.get('torch')}")

    steps = [
        (r, route, block, s)
        for r, route, block in blocks(bundle)
        for s in block["steps"]
    ]
    net_level = per_level(steps)
    np_steps = [
        (r, route, block, s)
        for r, route, block in blocks(numpy6)
        for s in block["steps"]
    ]
    np_level = per_level(np_steps)

    # --------------------------------------------------------- 2. provenance
    REPORT.append("\n2. Provenance: every Qhat came from the literal network")
    producers = {s.get("qhat_producer") for _, _, _, s in steps}
    check(
        producers == {"literal_attention_network"},
        f"every step records the literal network as its producer (found {producers})",
    )
    six = [s for _, _, _, s in steps if s["step"] == 6]
    check(
        bool(six) and all(
            s.get("qhat_producer") == "literal_attention_network" for s in six
        ),
        f"the sixth step in particular has {len(six)} rows, all network-produced",
    )

    # ------------------------------------------------------------- 3. H1
    REPORT.append("\n3. H1: the network horizon change is inert")
    sealed_by_key = {
        (float(r["mixing"]), int(r["task_index"])): r for r in sealed5["records"]
    }
    compared = expected = 0
    mismatches: list[tuple] = []
    gap_mismatches: list[tuple] = []
    for record, route, block in blocks(bundle):
        reference = sealed_by_key[
            (float(record["mixing"]), int(record["task_index"]))
        ]
        reference_steps = reference["routes"][route]["steps"]
        # The sealed five-step network bundle holds 117 rows across levels 1--5
        # (48 + 22 + 20 + 15 + 12). 105 is levels 1--4 only.
        expected += len(reference_steps)
        for index in range(min(len(reference_steps), len(block["steps"]))):
            compared += 1
            ours, theirs = block["steps"][index], reference_steps[index]
            if (
                ours["update_emitted"] != theirs["update_emitted"]
                or ours["eta_selected"] != theirs["eta_selected"]
                or ours["e_q"] != theirs["e_q"]
                or ours["ordered_reasons"] != theirs["ordered_reasons"]
            ):
                mismatches.append(
                    (record["mixing"], record["task_index"], route, index + 1)
                )
            # The recorded Qhat gaps are part of the sealed evidence: a rerun whose
            # float32 arithmetic differed would move them even if the decisions
            # happened to agree.
            if ours.get("q_hat_gap_vs_numpy") != theirs.get("q_hat_gap_vs_numpy"):
                gap_mismatches.append(
                    (record["mixing"], record["task_index"], route, index + 1)
                )
    check(compared == expected, f"every sealed row compared ({compared} == {expected})")
    check(compared == 117, f"117 sealed route-step rows in levels 1--5 (found {compared})")
    check(not mismatches, f"no decision/eta/E_Q/reason mismatch (found {len(mismatches)})")
    check(
        not gap_mismatches,
        f"every recorded Qhat gap is reproduced exactly "
        f"({len(gap_mismatches)} mismatches)",
    )
    for item in (mismatches + gap_mismatches)[:10]:
        REPORT.append(f"        mismatch: {item}")

    # --------------------------------------------------------- 4. per level
    REPORT.append("\n4. Per-level outcome, network path")
    REPORT.append(
        "  step  emissions   mean gain   minimum gain   max |dQ| vs numpy (all rows)"
    )
    for level in range(1, LEVELS + 1):
        info = net_level[level]
        mean = "n/a" if info["mean"] is None else f"{info['mean']:.6f}"
        low = "n/a" if info["min"] is None else f"{info['min']:.6f}"
        gap = max(info["gaps"]) if info["gaps"] else None
        gap_text = "n/a" if gap is None else f"{gap:.3e}"
        REPORT.append(
            f"  {level:>4}  {info['emissions']:>9}   {mean:>9}   {low:>12}   {gap_text:>16}"
        )
    REPORT.append(
        "  levels 1--5 of that column must reproduce the sealed FP-ITER5-001 "
        "headline gaps exactly; step 6 is the new number."
    )
    counts = [net_level[k]["emissions"] for k in range(1, LEVELS + 1)]
    np_counts = [np_level[k]["emissions"] for k in range(1, LEVELS + 1)]
    REPORT.append(f"  emissions by step, network : {counts}")
    REPORT.append(f"  emissions by step, numpy   : {np_counts}")

    # ------------------------------------------------------- 5. H2/H3/H4
    REPORT.append("\n5. H2/H3/H4: certifiable, valid, no violations")
    h2 = net_level[6]["emissions"] >= 1
    hypothesis(
        h2,
        f"H2 {'PASS' if h2 else 'FALSIFIED'}: {net_level[6]['emissions']} "
        f"route-records emit a sixth network step",
    )
    six_emitted = [s for _, _, _, s in steps if s["step"] == 6 and s["update_emitted"]]
    degrading = [
        s for s in six_emitted
        if not s["oracle_audit"].get("componentwise_nondegrading", False)
    ]
    nonpositive = [
        s for s in six_emitted if s["oracle_audit"]["total_value_gain"] <= 0.0
    ]
    h3 = bool(six_emitted) and not degrading and not nonpositive
    hypothesis(
        h3,
        f"H3 {'PASS' if h3 else 'FALSIFIED'}: all {len(six_emitted)} emitted sixth "
        f"steps non-degrading and strictly improving "
        f"({len(degrading)} degrading, {len(nonpositive)} non-positive)",
    )
    violations = [
        (r["mixing"], r["task_index"], route, s["step"])
        for r, route, _, s in steps
        if s["oracle_audit"].get("certificate_violation")
    ]
    check(
        not violations,
        f"H4 PASS: zero certificate violations across all six network steps "
        f"({len(violations)})",
    )
    for item in violations[:10]:
        REPORT.append(f"        violation: {item}")
    missing = [
        (r["mixing"], r["task_index"], route, s["step"])
        for r, route, _, s in steps
        if not s["update_emitted"] and not s["ordered_reasons"]
    ]
    check(not missing, f"every abstention carries a frozen reason ({len(missing)} missing)")
    nondeg_all = [
        (r["mixing"], r["task_index"], route, s["step"])
        for r, route, _, s in steps
        if s["update_emitted"]
        and min(s["oracle_audit"]["value_delta_vs_previous"]) < -1e-12
    ]
    check(
        not nondeg_all,
        f"no emitted step at any level degrades any state ({len(nondeg_all)})",
    )

    # ------------------------------------------------------------ 6. H5/H8
    REPORT.append("\n6. H5/H8: path agreement at step 6")
    set6_net = net_level[6]["emitters"]
    set6_np = np_level[6]["emitters"]
    lost = sorted(set6_np - set6_net)
    gained = sorted(set6_net - set6_np)
    retained = sorted(set6_np & set6_net)
    REPORT.append(
        f"  step-6 emitting sets: numpy {len(set6_np)}, network {len(set6_net)}; "
        f"shared {len(retained)}, numpy-only {len(lost)}, network-only {len(gained)}"
    )
    for item in lost:
        REPORT.append(f"        numpy-only  : mix={item[0]} task={item[1]} {item[2]}")
    for item in gained:
        REPORT.append(f"        network-only: mix={item[0]} task={item[1]} {item[2]}")
    h5 = set6_net == set6_np and len(set6_net) > 0
    hypothesis(
        h5,
        f"H5 {'PASS' if h5 else 'FALSIFIED'}: the two paths reach the same "
        f"step-6 emitting set ({len(set6_net)} vs {len(set6_np)})",
    )
    h8 = net_level[6]["emissions"] == np_level[6]["emissions"]
    hypothesis(
        h8,
        f"H8 {'PASS' if h8 else 'FALSIFIED'}: network step-6 count "
        f"{net_level[6]['emissions']} equals numpy "
        f"{np_level[6]['emissions']}",
    )
    for level in range(1, LEVELS + 1):
        same = net_level[level]["emissions"] == np_level[level]["emissions"]
        REPORT.append(
            f"  {'PASS' if same else 'MISMATCH'}  step {level}: network "
            f"{net_level[level]['emissions']} vs numpy {np_level[level]['emissions']}"
        )

    # ------------------------------------------------------------- 7. H6
    REPORT.append("\n7. H6: network-versus-numpy drift")
    gaps = [max(net_level[k]["gaps"]) if net_level[k]["gaps"] else None
            for k in range(1, LEVELS + 1)]
    REPORT.append(
        "  max |dQ| by step: "
        + ", ".join("n/a" if g is None else f"{g:.3e}" for g in gaps)
    )
    gap6 = gaps[5]
    h6 = gap6 is not None and gap6 <= atol
    hypothesis(
        h6,
        f"H6 {'PASS' if h6 else 'FALSIFIED'}: step-6 max gap "
        f"{'n/a' if gap6 is None else f'{gap6:.3e}'} <= {atol:.0e}",
    )
    for level in range(1, LEVELS + 1):
        g = gaps[level - 1]
        if g is not None and g > atol:
            REPORT.append(f"        step {level} exceeds ATOL: {g:.3e}")

    # ------------------------------------------------------------- 8. H7
    REPORT.append("\n8. H7: decision and eta flips against the numpy comparator")
    flips = [
        (r["mixing"], r["task_index"], route, s["step"])
        for r, route, _, s in steps
        if s.get("decision_flip_vs_numpy")
    ]
    eta_flips = [
        (r["mixing"], r["task_index"], route, s["step"])
        for r, route, _, s in steps
        if s.get("eta_flip_vs_numpy")
    ]
    six_flips = [f for f in flips if f[3] == 6]
    REPORT.append(f"  {len(steps)} step entries inspected")
    for item in flips[:10]:
        REPORT.append(f"        decision flip: mix={item[0]} task={item[1]} "
                      f"{item[2]} step={item[3]}")
    for item in eta_flips[:10]:
        REPORT.append(f"        eta flip: mix={item[0]} task={item[1]} "
                      f"{item[2]} step={item[3]}")
    h7 = not flips and not eta_flips
    hypothesis(
        h7,
        f"H7 {'PASS' if h7 else 'FALSIFIED'}: zero decision flips "
        f"({len(flips)}) and zero eta flips ({len(eta_flips)}), "
        f"of which {len(six_flips)} are at step 6",
    )

    # ------------------------------------------------------------- 9. H9
    REPORT.append("\n9. H9: mean-gain decay on the network path")
    mean6, mean5 = net_level[6]["mean"], net_level[5]["mean"]
    if mean6 is not None and mean5 is not None:
        h9 = mean6 < mean5
        hypothesis(
            h9,
            f"H9 {'PASS' if h9 else 'FALSIFIED'}: network mean6={mean6:.6f} "
            f"{'<' if h9 else '>='} mean5={mean5:.6f}",
        )
    else:
        h9 = None
        REPORT.append("  H9 not evaluable: no sixth-step emission")

    # ------------------------------------------- 10. step-5 cross-check too
    REPORT.append("\n10. The step-5 horizon also agrees (the alignment FP-ITER5-001 sealed)")
    net5_counts = [net_level[k]["emissions"] for k in range(1, LEVELS)]
    np5_bundle = load(NP5 / "task_results.json")
    np5_level = per_level(
        [(r, route, b, s) for r, route, b in blocks(np5_bundle) for s in b["steps"]]
    )
    np5_counts = [np5_level[k]["emissions"] for k in range(1, 6)]
    check(
        net5_counts == np5_counts,
        f"levels 1--5 agree between the paths ({net5_counts} vs {np5_counts})",
    )
    check(
        net5_counts == [22, 20, 15, 12, 12],
        f"levels 1--5 reproduce the sealed plateau ({net5_counts})",
    )

    REPORT.append("\n" + "=" * 74)
    REPORT.append("SUMMARY")
    REPORT.append(f"  emissions by step, network : {counts}")
    REPORT.append(f"  emissions by step, numpy   : {np_counts}")
    REPORT.append(
        "  mean gains by step, network: "
        f"{[None if net_level[k]['mean'] is None else round(net_level[k]['mean'], 6) for k in range(1, LEVELS + 1)]}"
    )
    REPORT.append(
        "  min gains by step, network : "
        f"{[None if net_level[k]['min'] is None else round(net_level[k]['min'], 6) for k in range(1, LEVELS + 1)]}"
    )
    REPORT.append(f"  H1 network horizon inert   : "
                  f"{'PASS' if compared == expected and not mismatches and not gap_mismatches else 'FAIL'}"
                  f" ({compared}/{expected} rows)")
    for name, value in (
        ("H2 sixth step certifiable", h2),
        ("H3 sixth step valid", h3),
        ("H5 path agreement step 6", h5),
        ("H7 no flips", h7),
        ("H8 count matches numpy", h8),
    ):
        REPORT.append(f"  {name:<27}: {'PASS' if value else 'FALSIFIED'}")
    REPORT.append(f"  H6 drift within ATOL       : "
                  f"{'PASS' if h6 else 'FALSIFIED'}")
    REPORT.append(
        f"  H9 mean gain decays        : "
        f"{'n/a' if h9 is None else ('PASS' if h9 else 'FALSIFIED')}"
    )
    REPORT.append("=" * 74)
    REPORT.append(
        "hypotheses were falsified, which per the task sheet is a RESULT, not a\n"
        "failed construction. RESULT below covers the construction checks only."
    )
    REPORT.append(
        "CONSTRUCTION: " + ("PASS" if FAILURES == 0 else f"FAIL ({FAILURES} failed)")
    )

    summary = {
        "task_id": TASK_ID,
        "torch": bundle.get("torch"),
        "emissions_by_step_network": counts,
        "emissions_by_step_numpy": np_counts,
        "mean_gain_by_step_network": [
            net_level[k]["mean"] for k in range(1, LEVELS + 1)
        ],
        "min_gain_by_step_network": [
            net_level[k]["min"] for k in range(1, LEVELS + 1)
        ],
        "max_gap_by_step": gaps,
        "H1_network_horizon_inert": {
            "compared": compared,
            "expected": expected,
            "mismatches": len(mismatches),
            "gap_mismatches": len(gap_mismatches),
        },
        "H2": "PASS" if h2 else "FALSIFIED",
        "H3": "PASS" if h3 else "FALSIFIED",
        "H4": "PASS" if not violations else "FAIL",
        "H5_path_agreement_step6": {
            "verdict": "PASS" if h5 else "FALSIFIED",
            "network": len(set6_net),
            "numpy": len(set6_np),
            "shared": len(retained),
            "numpy_only": [list(x) for x in lost],
            "network_only": [list(x) for x in gained],
        },
        "H6_drift": {"verdict": "PASS" if h6 else "FALSIFIED", "step6_max_gap": gap6},
        "H7_no_flips": {
            "verdict": "PASS" if h7 else "FALSIFIED",
            "decision_flips": len(flips),
            "eta_flips": len(eta_flips),
        },
        "H8_count_matches_numpy": "PASS" if h8 else "FALSIFIED",
        "H9_mean_decay": None if h9 is None else ("PASS" if h9 else "FALSIFIED"),
    }
    args.network_dir.mkdir(parents=True, exist_ok=True)
    (args.network_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.network_dir / "analysis_report.md").write_text(
        "\n".join(REPORT) + "\n", encoding="utf-8"
    )
    print("\n".join(REPORT))


if __name__ == "__main__":
    main()
