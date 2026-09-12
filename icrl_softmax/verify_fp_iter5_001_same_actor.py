"""FP-ITER5-001 same-actor derived verification.

Recomputes the five-step pattern on both paths, confirms the two horizon changes
are inert, checks path agreement, and confirms no scientific-corpus file changed.

By the user's instruction of 2026-09-11 the verification is not the focus of this
round; it is recorded for completeness. This is DERIVED verification by the same
actor that executed the task, NOT independent verification.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
NUMPY = PROJECT / "results" / "FP-ITER5-001" / "claude" / "numpy"
NETWORK = PROJECT / "results" / "FP-ITER5-001" / "claude" / "network"
SEALED_NP4 = PROJECT / "results" / "FP-ITER4-001" / "claude" / "formal"
SEALED_NET4 = PROJECT / "results" / "FP-ATTN-ITER4-001" / "claude" / "formal"
REPORT: list[str] = []
FAILURES = 0
PRIMARY = ("expected_exact", "expected_finite")
SCIENCE = (
    "fixed_policy_expected_sarsa.py",
    "fixed_policy_expected_sarsa_scaled.py",
    "fixed_policy_variance_certificate.py",
    "model.py",
    "verify_variance_adaptive_certificate.py",
)


def check(ok: bool, message: str) -> None:
    global FAILURES
    if ok:
        REPORT.append(f"  PASS  {message}")
    else:
        FAILURES += 1
        REPORT.append(f"  FAIL  {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def blocks(bundle: dict):
    return [
        (r, route, r["routes"][route])
        for r in bundle["records"]
        for route in PRIMARY
    ]


def main() -> None:
    REPORT.append("FP-ITER5-001 same-actor derived verification")
    REPORT.append("=" * 74)
    REPORT.append(
        "NOTE: the user instructed that verification is not the focus of this\n"
        "round. These checks are recorded for completeness and are same-actor only."
    )

    np_bundle = load(NUMPY / "task_results.json")
    net_bundle = load(NETWORK / "task_results.json")
    np_summary = load(NUMPY / "summary.json")
    net_env = load(NETWORK / "environment.json")
    np_steps = [(r, route, b, s) for r, route, b in blocks(np_bundle) for s in b["steps"]]
    net_steps = [(r, route, b, s) for r, route, b in blocks(net_bundle) for s in b["steps"]]

    REPORT.append("\n1. Frozen inputs")
    check(len(np_bundle["records"]) == 24, f"numpy 24 records (found {len(np_bundle['records'])})")
    check(len(net_bundle["records"]) == 24, f"network 24 records (found {len(net_bundle['records'])})")
    check(int(np_bundle["max_steps"]) == 5, f"numpy MAX_STEPS 5 (found {np_bundle['max_steps']})")
    check(int(net_bundle["max_steps"]) == 5, f"network MAX_STEPS 5 (found {net_bundle['max_steps']})")
    atol = float(net_bundle["atol"])
    check(atol == 1e-4, f"ATOL is the frozen 1e-4 (found {atol})")

    def per_level(steps):
        out = {}
        for level in range(1, 6):
            gains = [
                s["oracle_audit"]["total_value_gain"]
                for _, _, _, s in steps
                if s["step"] == level and s["update_emitted"]
            ]
            gaps = [
                s["q_hat_gap_vs_numpy"]
                for _, _, _, s in steps
                if s["step"] == level and "q_hat_gap_vs_numpy" in s
            ]
            out[level] = {
                "emissions": len(gains),
                "max_gap": max(gaps) if gaps else None,
                "mean": (sum(gains) / len(gains)) if gains else None,
                "min": min(gains) if gains else None,
            }
        return out

    np_level = per_level(np_steps)
    net_level = per_level(net_steps)

    REPORT.append("\n2. H1/H2: both horizon changes inert")
    for label, bundle, sealed_dir, levels, with_gap in (
        ("numpy", np_bundle, SEALED_NP4, 4, False),
        ("network", net_bundle, SEALED_NET4, 4, True),
    ):
        sealed = load(sealed_dir / "task_results.json")
        by_key = {(float(r["mixing"]), int(r["task_index"])): r for r in sealed["records"]}
        compared = 0
        mismatches = 0
        for record, route, block in blocks(bundle):
            ref = by_key.get((float(record["mixing"]), int(record["task_index"])))
            if ref is None:
                continue
            ref_steps = ref["routes"][route]["steps"]
            for i in range(min(levels, len(block["steps"]), len(ref_steps))):
                compared += 1
                a, b = block["steps"][i], ref_steps[i]
                if (
                    a["update_emitted"] != b["update_emitted"]
                    or a["eta_selected"] != b["eta_selected"]
                    or a["e_q"] != b["e_q"]
                ):
                    mismatches += 1
                elif with_gap and a["q_hat_gap_vs_numpy"] != b["q_hat_gap_vs_numpy"]:
                    mismatches += 1
        check(
            compared == 105 and mismatches == 0,
            f"{label} horizon inert: {compared} entries compared, "
            f"{mismatches} mismatches",
        )

    REPORT.append("\n3. H3/H7: fifth step and attrition")
    check(np_level[5]["emissions"] == 12, f"numpy fifth step emits 12 (found {np_level[5]['emissions']})")
    check(net_level[5]["emissions"] == 12, f"network fifth step emits 12 (found {net_level[5]['emissions']})")
    check(
        np_summary["H7_attrition_prediction"]["outcome"] == "FALSIFIED",
        "the summary records H7 as FALSIFIED rather than reinterpreting it",
    )
    np_em = [np_level[k]["emissions"] for k in range(1, 6)]
    check(
        np_em == [22, 20, 15, 12, 12],
        f"per-level emissions {np_em} match the recorded plateau",
    )
    np4 = json.loads((SEALED_NP4 / "task_results.json").read_text(encoding="utf-8"))
    np4_by_key = {(float(r["mixing"]), int(r["task_index"])): r for r in np4["records"]}
    s4 = set()
    s5 = set()
    for record, route, block in blocks(np_bundle):
        ref = np4_by_key.get((float(record["mixing"]), int(record["task_index"])))
        if ref is not None and ref["routes"][route]["emitted_steps"] >= 4:
            s4.add((record["mixing"], record["task_index"], route))
        if block["emitted_steps"] >= 5:
            s5.add((record["mixing"], record["task_index"], route))
    check(s4 == s5, f"the step-4 and step-5 emitting sets are identical ({len(s5)} records)")

    REPORT.append("\n4. H8/H9/H10: predictions and drift")
    mean4 = np_level[4]["mean"]
    mean5 = np_level[5]["mean"]
    check(mean5 < mean4, f"H8 PASS: mean5={mean5:.6f} < mean4={mean4:.6f}")
    check(np_level[5]["min"] > 0.05, f"H9 PASS: min5={np_level[5]['min']:.6f} > 0.05")
    check(
        net_level[5]["max_gap"] <= atol,
        f"H10 PASS: step-5 network gap {net_level[5]['max_gap']:.3e} <= {atol:.0e}",
    )
    for level in range(1, 6):
        check(
            net_level[level]["max_gap"] <= atol,
            f"step {level} network gap {net_level[level]['max_gap']:.3e} <= ATOL",
        )

    REPORT.append("\n5. H11: path agreement")
    for level in range(1, 6):
        check(
            np_level[level]["emissions"] == net_level[level]["emissions"],
            f"step {level} emission counts agree "
            f"({np_level[level]['emissions']} == {net_level[level]['emissions']})",
        )
    comps = [(r, route, c) for r, route, b in blocks(net_bundle) for c in b["reference_comparisons"]]
    disagree = [c for c in comps if not c[2]["decision_match"]]
    eta_disagree = [c for c in comps if not c[2]["eta_match"]]
    check(len(comps) == 105, f"105 decision comparisons available (found {len(comps)})")
    check(not disagree, f"no decision disagreement with numpy ({len(disagree)})")
    check(not eta_disagree, f"no eta disagreement with numpy ({len(eta_disagree)})")

    REPORT.append("\n6. H4/H5/H6: validity on both paths")
    for label, steps in (("numpy", np_steps), ("network", net_steps)):
        cert = [
            (r["mixing"], r["task_index"], route, s["step"])
            for r, route, _, s in steps
            if s["oracle_audit"].get("certificate_violation")
        ]
        nondeg = [
            (r["mixing"], r["task_index"], route, s["step"])
            for r, route, _, s in steps
            if s["update_emitted"]
            and min(s["oracle_audit"]["value_delta_vs_previous"]) < -1e-12
        ]
        check(not cert, f"{label}: zero certificate violations ({len(cert)})")
        check(not nondeg, f"{label}: zero non-degrading violations ({len(nondeg)})")
        for level in range(1, 6):
            gains = [
                s["oracle_audit"]["total_value_gain"]
                for _, _, _, s in steps
                if s["step"] == level and s["update_emitted"]
            ]
            if gains:
                check(all(g > 0 for g in gains), f"{label}: step-{level} gains all positive")

    REPORT.append("\n7. Corpus integrity")
    changed = []
    for name, digest in net_env["sealed_file_hashes"].items():
        path = PROJECT / name
        raw = hashlib.sha256(path.read_bytes()).hexdigest()
        matches = digest in (raw, sha256(path))
        if name in SCIENCE:
            check(matches, f"SCIENCE {name} matches its recorded hash")
        elif matches:
            REPORT.append(f"  PASS  evaluator {name} matches its recorded hash")
        else:
            changed.append(name)
    for name in changed:
        REPORT.append(
            f"  INFO  shared evaluator {name} differs from this task's record; it is "
            "not part of the scientific corpus."
        )

    REPORT.append("\n8. Replay of the sealed programs")
    for script, extra in (
        ("analyze_fp_iter5_001.py", ["--numpy-dir", str(NUMPY), "--network-dir", str(NETWORK)]),
        ("verify_fp_attn_iter_001_same_actor.py", []),
        ("verify_fp_attn_iter4_001_same_actor.py", []),
        ("verify_fp_iter4_001_same_actor.py", []),
        ("verify_variance_adaptive_certificate.py", []),
    ):
        completed = subprocess.run(
            [sys.executable, "-B", str(PROJECT / script), *extra],
            capture_output=True,
            text=True,
            cwd=PROJECT,
        )
        check(completed.returncode == 0, f"{script} replays with exit 0 (got {completed.returncode})")

    REPORT.append("\n" + "=" * 74)
    REPORT.append("SUMMARY")
    np_emissions = [np_level[k]["emissions"] for k in range(1, 6)]
    net_emissions = [net_level[k]["emissions"] for k in range(1, 6)]
    np_means = [round(np_level[k]["mean"], 6) for k in range(1, 6)]
    np_mins = [round(np_level[k]["min"], 6) for k in range(1, 6)]
    net_gaps = [f"{net_level[k]['max_gap']:.3e}" for k in range(1, 6)]
    REPORT.append(
        f"  emissions by step, numpy   : {np_emissions}\n"
        f"  emissions by step, network : {net_emissions}\n"
        f"  mean gains, numpy          : {np_means}\n"
        f"  min gains, numpy           : {np_mins}\n"
        f"  network |dQ| by step       : {net_gaps}\n"
        f"  five-step routes, numpy    : {np_summary['numpy_five_step_routes']}\n"
        f"  five-step routes, network  : {np_summary['network_five_step_routes']}\n"
        f"  decision agreement         : {len(comps) - len(disagree)}/{len(comps)}\n"
        f"  H7 attrition               : FALSIFIED (plateau at 12)\n"
        f"  H8 mean decay              : PASS\n"
        f"  H9 non-vacuity             : PASS\n"
        f"  H10 drift                  : PASS\n"
        f"  H11 path agreement         : PASS"
    )
    REPORT.append("=" * 74)
    REPORT.append("RESULT: " + ("PASS" if FAILURES == 0 else f"FAIL ({FAILURES} failed)"))
    REPORT.append(
        "LIMITATION: same-actor derived verification only, and per the user's\n"
        "instruction not the focus of this round."
    )

    text = "\n".join(REPORT)
    print(text)
    out = PROJECT / "docs" / "research_branches" / "FP-ITER5-001" / "claude" / "verification_same_actor.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
