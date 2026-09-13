"""FP-GAP-001 analysis: does the certified iteration close the optimality gap?

Scores H1--H6 and reports H7. Every floor statement carries the policy-class caveat,
which the task sheet requires.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
TASK_ID = "FP-GAP-001"
PRIMARY = ("expected_exact", "expected_finite")
ARMS = ("frozen", "empirical_bernstein")
REPORT: list[str] = []
FAILURES = 0

# Pre-registered in docs/research_tasks/FP-GAP-001.md before the run.
H4_PLATEAU_THRESHOLD = 0.01
H4_MIN_STEPS = 6
H5_VANISHING_THRESHOLD = 1e-4
H6_FRACTION_CLOSED = 0.90
H7_FLOOR_TOLERANCE = 0.01


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


def trajectories(bundle: dict):
    for record in bundle["records"]:
        for route in PRIMARY:
            for arm in ARMS:
                yield (
                    record["mixing"],
                    record["task_index"],
                    route,
                    arm,
                    record["routes"][route][arm],
                    record,
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=PROJECT / "results" / "FP-GAP-001" / "claude" / "formal",
    )
    args = parser.parse_args()

    bundle = load(args.result_dir / "task_results.json")
    REPORT.append(f"{TASK_ID} analysis: does the certified iteration close the gap?")
    REPORT.append("=" * 92)
    REPORT.append(f"  {len(bundle['records'])} records, source horizon {bundle['horizon']}")
    REPORT.append("  measured over sealed data plus an exact v* solve; no new simulation")

    # ------------------------------------------------------------- H1
    REPORT.append("\n1. H1: the trajectory reconstruction closes (mandatory)")
    gaps = [
        (m, t, route, arm, traj["reconstruction_gap"])
        for m, t, route, arm, traj, _ in trajectories(bundle)
    ]
    worst = max(g[4] for g in gaps)
    check(
        worst <= 1e-12,
        f"all {len(gaps)} trajectories reconstruct to within {worst:.3e}",
    )
    for item in sorted(gaps, key=lambda g: -g[4])[:5]:
        if item[4] > 1e-12:
            REPORT.append(
                f"        worst: mix={item[0]} task={item[1]} {item[2]} {item[3]} "
                f"{item[4]:.3e}"
            )

    # ------------------------------------------------------------- H2
    REPORT.append("\n2. H2: v* is optimal, and dominates every trajectory")
    residuals = {r["bellman_residual_of_v_star"] for r in bundle["records"]}
    check(
        max(residuals) <= 1e-12,
        f"v* satisfies the Bellman optimality equation on every record "
        f"(worst residual {max(residuals):.3e})",
    )
    dominated = []
    for m, t, route, arm, traj, record in trajectories(bundle):
        import numpy as np

        v_star = np.asarray(record["v_star"], dtype=np.float64)
        v0 = np.asarray(traj["start_policy_value"], dtype=np.float64)
        if float(np.min(v_star - v0)) < -1e-9:
            dominated.append((m, t, route, arm))
    check(
        not dominated,
        f"v* dominates v^{{pi_0}} on every record ({len(dominated)} exceptions)",
    )

    # ------------------------------------------------------------- H3
    REPORT.append("\n3. H3: is the gap monotone?")
    rises = []
    for m, t, route, arm, traj, _ in trajectories(bundle):
        seq = traj["gap_trajectory"]
        for k in range(1, len(seq)):
            if seq[k] > seq[k - 1] + 1e-12:
                rises.append((m, t, route, arm, k, seq[k - 1], seq[k]))
    check(
        not rises,
        f"the gap is non-increasing on every trajectory ({len(rises)} rises)",
    )
    for item in rises[:5]:
        REPORT.append(
            f"        rise: mix={item[0]} task={item[1]} {item[2]} {item[3]} "
            f"step {item[4]}: {item[5]:.6e} -> {item[6]:.6e}"
        )

    # ------------------------------------------------------- 4. the numbers
    REPORT.append("\n4. The gap trajectory, per arm")
    stats: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    for m, t, route, arm, traj, _ in trajectories(bundle):
        seq = traj["gap_trajectory"]
        gap0, gapf = seq[0], seq[-1]
        closed = (gap0 - gapf) / gap0 if gap0 > 0 else 0.0
        last_gain = seq[-2] - seq[-1] if len(seq) > 1 else 0.0
        ratio = last_gain / gapf if gapf > 0 else 0.0
        floor_step = None
        for k in range(len(seq)):
            if seq[k] <= gapf * (1.0 + H7_FLOOR_TOLERANCE):
                floor_step = k
                break
        plateau_movement = None
        plateau_share = None
        if len(seq) - 1 >= H4_MIN_STEPS:
            idx = int((2 * (len(seq) - 1)) / 3)
            # REGISTERED form: movement over the last third as a fraction of the
            # FINAL gap. It explodes for a geometrically converging process, because
            # the denominator is tiny -- that is a mis-specification, and it is kept
            # only so the registered verdict can be reported as falsified.
            plateau_movement = (seq[idx] - seq[-1]) / gapf if gapf > 0 else 0.0
            # CORRECTED form: the same movement as a share of the INITIAL
            # suboptimality, which is the decision-relevant scale.
            plateau_share = (seq[idx] - seq[-1]) / gap0 if gap0 > 0 else 0.0
        stats[arm].append(
            {
                "key": (m, t, route),
                "gap0": gap0,
                "gap_final": gapf,
                "fraction_closed": closed,
                "last_gain_over_gap": ratio,
                "effective_stop_step": floor_step,
                "steps": len(seq) - 1,
                "plateau_movement": plateau_movement,
                "plateau_share_of_initial": plateau_share,
            }
        )
    for arm in ARMS:
        rows = stats[arm]
        moved = [r for r in rows if r["gap0"] > 0]
        closed = [r["fraction_closed"] for r in moved]
        mean_closed = sum(closed) / len(closed) if closed else 0.0
        REPORT.append(
            f"  {arm}: {len(moved)} trajectories with a positive initial gap; "
            f"fraction closed mean {mean_closed:.4%}, "
            f"min {min(closed):.4%}, max {max(closed):.4%}"
        )
        still = sum(1 for r in moved if r["gap_final"] > 0)
        REPORT.append(
            f"      trajectories whose final gap is still positive: {still}/{len(moved)}"
        )
    REPORT.append(
        "\n  CAVEAT, required by the task sheet wherever a floor is reported: "
        + bundle["policy_class_caveat"]
    )

    # --------------------------------- 4b. the same numbers, conditioned on length
    # POST-HOC BREAKDOWN, and it exists because the registered form of H5 was
    # MIS-SPECIFIED. "The final per-step gain" pooled every trajectory, including
    # short ones that stop after a handful of steps, where the last gain is
    # naturally a large fraction of what remains. The hypothesis was about the tail
    # of LONG trajectories, which is where "is this still doing work" has content.
    # The registered verdict stands below; this is the breakdown that says what it
    # actually measured.
    REPORT.append("\n4b. The same numbers, split by trajectory length (POST-HOC)")
    REPORT.append(
        f"  {'emitted steps':>14} {'n':>4} {'mean closed':>12} "
        f"{'median last gain / gap':>24} {'median plateau movement':>24}"
    )
    bands = ((1, 5), (6, 11), (12, 23), (24, 32))
    length_breakdown = {}
    for lo, hi in bands:
        rows = [
            r
            for arm in ARMS
            for r in stats[arm]
            if lo <= r["steps"] <= hi and r["gap0"] > 0
        ]
        if not rows:
            continue
        closed = sum(r["fraction_closed"] for r in rows) / len(rows)
        ratios = sorted(r["last_gain_over_gap"] for r in rows)
        moves = sorted(
            r["plateau_movement"] for r in rows if r["plateau_movement"] is not None
        )
        length_breakdown[f"{lo}-{hi}"] = {
            "n": len(rows),
            "mean_fraction_closed": closed,
            "median_last_gain_over_gap": ratios[len(ratios) // 2],
            "median_plateau_movement": moves[len(moves) // 2] if moves else None,
        }
        REPORT.append(
            f"  {f'{lo}-{hi}':>14} {len(rows):>4} {closed:>11.2%} "
            f"{ratios[len(ratios) // 2]:>24.3e} "
            f"{(f'{moves[len(moves) // 2]:.3e}' if moves else 'n/a'):>24}"
        )
    long_rows = [
        r
        for arm in ARMS
        for r in stats[arm]
        if r["steps"] >= 24 and r["gap0"] > 0
    ]
    if long_rows:
        long_ratios = sorted(r["last_gain_over_gap"] for r in long_rows)
        long_closed = sum(r["fraction_closed"] for r in long_rows) / len(long_rows)
        long_shares = sorted(
            r["plateau_share_of_initial"]
            for r in long_rows
            if r["plateau_share_of_initial"] is not None
        )
        REPORT.append(
            f"  the LONG trajectories (>= 24 emitted steps, n={len(long_rows)}), which "
            f"are the ones the hypothesis was about: median final gain / remaining "
            f"gap {long_ratios[len(long_ratios) // 2]:.3e}, mean fraction closed "
            f"{long_closed:.2%}"
        )
        if long_shares:
            REPORT.append(
                f"  CORRECTED H4 metric for those trajectories: the last third moves "
                f"the gap by a median of {long_shares[len(long_shares) // 2]:.4%} of "
                f"the INITIAL suboptimality"
            )
        REPORT.append(
            "  so the registered H5 verdict is a statement about the pooled set, "
            "not about the tail; both are reported and neither replaces the other."
        )
        REPORT.append(
            "  WHY H5 WAS THE WRONG TEST: a process converging geometrically has a "
            "CONSTANT ratio of per-step gain to remaining gap, not a vanishing one. "
            "A vanishing ratio would mean faster-than-geometric convergence."
        )
        REPORT.append(
            "  LABELLING, corrected after a cross-session check: the quantity in the "
            "last column above is the gain relative to the REMAINING gap, not a decay "
            "factor. For a geometric process with gap_k = rho * gap_{k-1} it equals "
            "(1-rho)/rho, so the three forms of the same measurement are: "
            "gain/remaining-gap ~0.33, decay factor rho = 1/(1+0.33) ~ 0.75, and "
            "fraction of the CURRENT gap closed (1-rho) ~ 0.25. A first version of "
            "this report called the first of those 'the decay factor', which it is "
            "not. See cross_check_fp_gap_001.md in this directory."
        )
        # Show the decay factor is stable rather than drifting, on one long
        # trajectory, so the geometric reading is visible rather than asserted.
        sample = None
        for m, t, route, arm, traj, _ in trajectories(bundle):
            if len(traj["gap_trajectory"]) - 1 >= 24 and traj["gap_trajectory"][0] > 0:
                sample = (m, t, route, arm, traj["gap_trajectory"])
                break
        if sample:
            seq = sample[4]
            REPORT.append(
                f"  decay factor on one long trajectory "
                f"(mix={sample[0]} task={sample[1]} {sample[2]} {sample[3]}):"
            )
            REPORT.append(
                "    step  gap  gain/gap : "
                + "  ".join(
                    f"{k}:{seq[k - 1] - seq[k]:.3f}/{seq[k]:.3f}="
                    f"{(seq[k - 1] - seq[k]) / seq[k]:.2f}"
                    for k in range(1, len(seq))
                    if seq[k] > 0 and k % 4 == 0
                )
            )
    # Trajectories that reach the optimum exactly are worth naming rather than
    # leaving inside a maximum.
    perfect = [
        (arm, r["key"], r["steps"])
        for arm in ARMS
        for r in stats[arm]
        if r["gap0"] > 0 and r["gap_final"] <= 1e-12
    ]
    REPORT.append(
        f"  trajectories that close the gap to zero: {len(perfect)}"
        + (f" {perfect[:4]}" if perfect else "")
    )
    zero_progress = [
        (arm, r["key"], r["steps"])
        for arm in ARMS
        for r in stats[arm]
        if r["gap0"] > 0 and r["fraction_closed"] <= 0.0
    ]
    REPORT.append(
        f"  trajectories that close nothing at all (never emitted): "
        f"{len(zero_progress)}"
    )

    # ------------------------------------------------------------- H4
    REPORT.append("\n5. H4: does the gap reach a floor before the horizon?")
    eligible = {
        arm: [r for r in stats[arm] if r["plateau_movement"] is not None]
        for arm in ARMS
    }
    plateaued = {
        arm: [r for r in eligible[arm] if r["plateau_movement"] < H4_PLATEAU_THRESHOLD]
        for arm in ARMS
    }
    for arm in ARMS:
        n = len(eligible[arm])
        k = len(plateaued[arm])
        REPORT.append(
            f"  {arm}: {k}/{n} trajectories with >= {H4_MIN_STEPS} steps move the gap "
            f"by less than {H4_PLATEAU_THRESHOLD:.0%} over their last third"
        )
    total_e = sum(len(eligible[a]) for a in ARMS)
    total_p = sum(len(plateaued[a]) for a in ARMS)
    h4 = total_p * 2 > total_e
    hypothesis(
        h4,
        f"H4 {'PASS' if h4 else 'FALSIFIED'}: the majority plateau "
        f"({total_p}/{total_e})",
    )

    # ------------------------------------------------------------- H5
    REPORT.append("\n6. H5: is the final per-step gain a vanishing fraction "
                  "of the remaining gap?")
    for arm in ARMS:
        ratios = [
            r["last_gain_over_gap"]
            for r in stats[arm]
            if r["steps"] > 0 and r["gap_final"] > 0
        ]
        below = sum(1 for v in ratios if v < H5_VANISHING_THRESHOLD)
        REPORT.append(
            f"  {arm}: {below}/{len(ratios)} below {H5_VANISHING_THRESHOLD:.0e}; "
            f"median {sorted(ratios)[len(ratios) // 2]:.3e}"
            if ratios
            else f"  {arm}: no trajectory emits"
        )
    ratios_all = [
        r["last_gain_over_gap"]
        for arm in ARMS
        for r in stats[arm]
        if r["steps"] > 0 and r["gap_final"] > 0
    ]
    below_all = sum(1 for v in ratios_all if v < H5_VANISHING_THRESHOLD)
    h5 = below_all * 2 > len(ratios_all)
    hypothesis(
        h5,
        f"H5 {'PASS' if h5 else 'FALSIFIED'}: the majority of final gains are below "
        f"{H5_VANISHING_THRESHOLD:.0e} of the remaining gap "
        f"({below_all}/{len(ratios_all)})",
    )

    # ------------------------------------------------------------- H6
    REPORT.append("\n7. H6: what share of the initial suboptimality is closed?")
    for arm in ARMS:
        moved = [r for r in stats[arm] if r["gap0"] > 0]
        mean_closed = (
            sum(r["fraction_closed"] for r in moved) / len(moved) if moved else 0.0
        )
        h6_arm = mean_closed > H6_FRACTION_CLOSED
        hypothesis(
            h6_arm,
            f"H6 {arm}: mean fraction closed {mean_closed:.4%} "
            f"{'>' if h6_arm else '<='} {H6_FRACTION_CLOSED:.0%}",
        )
    moved_all = [r for arm in ARMS for r in stats[arm] if r["gap0"] > 0]
    mean_all = sum(r["fraction_closed"] for r in moved_all) / len(moved_all)
    h6 = mean_all > H6_FRACTION_CLOSED
    if not h6:
        REPORT.append(
            "  this is the consequential failure: the iteration STOPS SHORT of what "
            "its own policy class can reach, and the binding constraint is something "
            "this line has not yet identified. Remember the caveat: part of the "
            "residual belongs to the policy class."
        )

    # ------------------------------------------------------------- H7
    REPORT.append("\n8. H7: the effective stopping step, per trajectory")
    REPORT.append(
        f"  the first step at which the gap is within {H7_FLOOR_TOLERANCE:.0%} of its "
        f"final value — the number a practitioner would want, as distinct from the "
        f"step at which the certificate stops emitting"
    )
    for arm in ARMS:
        steps = [
            r["effective_stop_step"] for r in stats[arm] if r["steps"] > 0
            and r["effective_stop_step"] is not None
        ]
        emitted = [r["steps"] for r in stats[arm] if r["steps"] > 0]
        if steps:
            REPORT.append(
                f"  {arm}: effective stop median {sorted(steps)[len(steps) // 2]}, "
                f"max {max(steps)}; emitted steps median "
                f"{sorted(emitted)[len(emitted) // 2]}, max {max(emitted)}"
            )
    all_eff = [
        r["effective_stop_step"]
        for arm in ARMS
        for r in stats[arm]
        if r["steps"] > 0 and r["effective_stop_step"] is not None
    ]
    all_emit = [r["steps"] for arm in ARMS for r in stats[arm] if r["steps"] > 0]
    if all_eff and all_emit:
        REPORT.append(
            f"  pooled: effective stop median {sorted(all_eff)[len(all_eff) // 2]} "
            f"versus emitted median {sorted(all_emit)[len(all_emit) // 2]}"
        )

    # ------------------------------------------------------------- summary
    REPORT.append("\n" + "=" * 92)
    REPORT.append("SUMMARY")
    REPORT.append(f"  H1 reconstruction closes : {'PASS' if worst <= 1e-12 else 'FAIL'}")
    REPORT.append(f"  H2 v* verified           : {'PASS' if not dominated else 'FAIL'}")
    REPORT.append(f"  H3 gap monotone          : {'PASS' if not rises else 'FAIL'}")
    REPORT.append(f"  H4 plateau               : {'PASS' if h4 else 'FALSIFIED'} "
                  f"({total_p}/{total_e})")
    REPORT.append(f"  H5 vanishing tail gain   : {'PASS' if h5 else 'FALSIFIED'} "
                  f"({below_all}/{len(ratios_all)})")
    REPORT.append(f"  H6 fraction closed       : {'PASS' if h6 else 'FALSIFIED'} "
                  f"({mean_all:.2%})")
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
        "records": len(bundle["records"]),
        "worst_reconstruction_gap": worst,
        "worst_bellman_residual": max(residuals),
        "H3_rises": len(rises),
        "H4": {
            "verdict": "PASS" if h4 else "FALSIFIED",
            "plateaued": total_p,
            "eligible": total_e,
            "threshold": H4_PLATEAU_THRESHOLD,
        },
        "H5": {
            "verdict": "PASS" if h5 else "FALSIFIED",
            "below": below_all,
            "total": len(ratios_all),
            "threshold": H5_VANISHING_THRESHOLD,
        },
        "H6": {
            "verdict": "PASS" if h6 else "FALSIFIED",
            "mean_fraction_closed": mean_all,
            "threshold": H6_FRACTION_CLOSED,
        },
        "H7_effective_stop_median": (
            sorted(all_eff)[len(all_eff) // 2] if all_eff else None
        ),
        "length_breakdown_posthoc": length_breakdown,
        "long_trajectory_final_gain_over_gap_median": (
            sorted(r["last_gain_over_gap"] for r in long_rows)[len(long_rows) // 2]
            if long_rows
            else None
        ),
        "trajectories_closing_to_zero": len(perfect),
        "H7_emitted_median": (
            sorted(all_emit)[len(all_emit) // 2] if all_emit else None
        ),
        "policy_class_caveat": bundle["policy_class_caveat"],
        "per_arm": {
            arm: {
                "mean_fraction_closed": (
                    sum(r["fraction_closed"] for r in stats[arm] if r["gap0"] > 0)
                    / max(sum(1 for r in stats[arm] if r["gap0"] > 0), 1)
                ),
                "final_gap_positive": sum(
                    1 for r in stats[arm] if r["gap0"] > 0 and r["gap_final"] > 0
                ),
                "with_positive_gap": sum(1 for r in stats[arm] if r["gap0"] > 0),
            }
            for arm in ARMS
        },
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
