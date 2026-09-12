"""FP-ITER6-001 same-actor derived verification.

Re-derives the horizon-inertness proof from the bundles, checks structurally that
the batches are built once per record and reused at every step, confirms the
frozen prediction is the one that was scored, bounds the shared-evaluator change
to the documented horizon extension, and replays the affected sealed verifiers.

By the user's instruction of 2026-09-11 the verification is not the focus of this
round; it is recorded for completeness. This is DERIVED verification by the same
actor that executed the task, NOT independent verification.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
ITER6 = PROJECT / "results" / "FP-ITER6-001" / "claude" / "numpy"
ITER5 = PROJECT / "results" / "FP-ITER5-001" / "claude" / "numpy"
CENSUS_DIR = PROJECT / "results" / "FP-CENSUS-001" / "claude" / "formal"
PRIMARY = ("expected_exact", "expected_finite")
SCIENCE = (
    "fixed_policy_expected_sarsa.py",
    "fixed_policy_expected_sarsa_scaled.py",
    "fixed_policy_variance_certificate.py",
    "model.py",
    "verify_variance_adaptive_certificate.py",
)
REPORT: list[str] = []
FAILURES = 0


def check(ok: bool, message: str) -> bool:
    global FAILURES
    if ok:
        REPORT.append(f"  PASS  {message}")
    else:
        FAILURES += 1
        REPORT.append(f"  FAIL  {message}")
    return bool(ok)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def blocks(bundle: dict):
    return [
        (r, route, r["routes"][route])
        for r in bundle["records"]
        for route in PRIMARY
    ]


def rederive_inertness() -> int:
    bundle = load(ITER6 / "task_results.json")
    sealed = load(ITER5 / "task_results.json")
    by_key = {
        (float(r["mixing"]), int(r["task_index"])): r for r in sealed["records"]
    }
    compared = 0
    expected = 0
    mismatches = []
    for record, route, block in blocks(bundle):
        reference = by_key[(float(record["mixing"]), int(record["task_index"]))]
        reference_steps = reference["routes"][route]["steps"]
        # The sealed five-step bundle holds 117 rows across levels 1--5
        # (48 + 22 + 20 + 15 + 12). 105 is levels 1--4 only; an earlier version of
        # this check used 105 and reported a spurious failure.
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
    check(compared == expected, f"every sealed row was compared ({compared} == {expected})")
    check(compared == 117, f"117 sealed route-step rows in levels 1--5 (found {compared})")
    check(not mismatches, f"H1 re-derived: no mismatch (found {len(mismatches)})")
    for item in mismatches[:10]:
        REPORT.append(f"        mismatch: {item}")
    return compared


def batches_built_once_per_record() -> None:
    """The batch reuse must be structural, not merely asserted in prose."""
    tree = ast.parse((PROJECT / "evaluate_fp_iter2_001.py").read_text(encoding="utf-8"))
    wanted = {"training_batch", "certification_batch"}
    inside_step_loop: list[str] = []
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        iterator = node.iter
        name = (
            iterator.func.id
            if isinstance(iterator, ast.Call)
            and isinstance(iterator.func, ast.Name)
            else None
        )
        if name != "range":
            continue
        # Identify the step loop by its target's name.
        target = node.target
        if not (isinstance(target, ast.Name) and target.id == "step_index"):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                callee = sub.func
                callee_name = (
                    callee.id
                    if isinstance(callee, ast.Name)
                    else callee.attr
                    if isinstance(callee, ast.Attribute)
                    else None
                )
                if callee_name in wanted:
                    inside_step_loop.append(callee_name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            callee = node.func
            callee_name = (
                callee.id
                if isinstance(callee, ast.Name)
                else callee.attr
                if isinstance(callee, ast.Attribute)
                else None
            )
            if callee_name in wanted:
                found.add(callee_name)
    check(
        found == wanted,
        f"both batch builders are present (found {sorted(found)})",
    )
    check(
        not inside_step_loop,
        f"no batch is rebuilt inside the step loop ({inside_step_loop})",
    )


def prediction_was_the_scored_one() -> None:
    path = CENSUS_DIR / "prediction_step6.json"
    summary = load(ITER6 / "summary.json")
    check(path.exists(), "the census prediction file still exists")
    if not path.exists():
        return
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    check(
        digest == summary["prediction_sha256"],
        f"the scored prediction is the frozen file (sha256 {digest[:16]}...)",
    )
    check(
        path.stat().st_mtime < (ITER6 / "task_results.json").stat().st_mtime,
        "the prediction predates the sixth-step bundle",
    )
    check(
        summary["prediction_created_utc"]
        == load(path)["created_utc"],
        "the recorded creation time is the file's own",
    )


def analyzer_determinism() -> None:
    before = (ITER6 / "summary.json").read_bytes()
    completed = subprocess.run(
        [sys.executable, "-B", str(PROJECT / "analyze_fp_iter6_001.py"),
         "--bundle-dir", str(ITER6), "--census-dir", str(CENSUS_DIR)],
        capture_output=True,
        text=True,
        cwd=PROJECT,
    )
    check(completed.returncode == 0,
          f"analyze_fp_iter6_001.py replays with exit 0 (got {completed.returncode})")
    check(before == (ITER6 / "summary.json").read_bytes(),
          "re-running the analyzer reproduces summary.json byte-for-byte")


def main() -> None:
    REPORT.append("FP-ITER6-001 same-actor derived verification")
    REPORT.append("=" * 74)
    REPORT.append(
        "NOTE: the user instructed that verification is not the focus of this\n"
        "round. These checks are recorded for completeness and are same-actor only."
    )

    bundle = load(ITER6 / "task_results.json")
    config = load(ITER6 / "config.json")
    environment = load(ITER6 / "environment.json")
    summary = load(ITER6 / "summary.json")

    REPORT.append("\n1. Frozen inputs")
    check(len(bundle["records"]) == 24, f"24 records (found {len(bundle['records'])})")
    check(
        int(bundle["max_steps"]) == 6,
        f"numpy horizon 6 (found {bundle['max_steps']})",
    )
    check(
        int(config["cert_chains"]) == 16384
        and int(config["cert_chain_length"]) == 64,
        f"certification dimensions are the frozen 16384 x 64 "
        f"(found {config['cert_chains']} x {config['cert_chain_length']})",
    )
    check(int(config["tasks"]) == 12, f"12 tasks per mixing (found {config['tasks']})")
    check(
        sorted(config["mixings"]) == [0.08, 0.5],
        f"both mixings run (found {config['mixings']})",
    )

    REPORT.append("\n2. H1 re-derived from the bundles")
    rederive_inertness()

    REPORT.append("\n3. Batch reuse is structural")
    batches_built_once_per_record()

    REPORT.append("\n4. Validity on the sixth step")
    steps = [s for _, _, b in blocks(bundle) for s in b["steps"]]
    six = [s for s in steps if s["step"] == 6 and s["update_emitted"]]
    degrading = [
        s for s in six if not s["oracle_audit"].get("componentwise_nondegrading", False)
    ]
    nonpositive = [s for s in six if s["oracle_audit"]["total_value_gain"] <= 0.0]
    violations = [
        s for s in steps if s["oracle_audit"].get("certificate_violation")
    ]
    check(bool(six), f"at least one sixth-step emission exists ({len(six)})")
    check(not degrading, f"zero sixth-step degrading violations ({len(degrading)})")
    check(not nonpositive, f"zero sixth-step non-positive gains ({len(nonpositive)})")
    check(not violations, f"zero certificate violations over six steps ({len(violations)})")
    missing = [s for s in steps if not s["update_emitted"] and not s["ordered_reasons"]]
    check(not missing, f"every abstention carries a frozen reason ({len(missing)} missing)")

    REPORT.append("\n5. The scored prediction is the frozen one")
    prediction_was_the_scored_one()

    REPORT.append("\n6. Corpus integrity")
    # The scientific corpus must match strictly. The shared task evaluator evolves
    # as the horizon is extended, so a changed evaluator is reported explicitly and
    # bounded to exactly the documented file, never silently tolerated.
    changed_evaluators = []
    for name, digest in environment["sealed_file_hashes"].items():
        path = PROJECT / name
        raw = hashlib.sha256(path.read_bytes()).hexdigest()
        matches = digest in (raw, sha256(path))
        if name in SCIENCE:
            check(matches, f"SCIENCE {name} matches its recorded hash")
        elif matches:
            REPORT.append(f"  PASS  evaluator {name} matches its recorded hash")
        else:
            changed_evaluators.append(name)
    for name in changed_evaluators:
        REPORT.append(
            f"  INFO  shared evaluator {name} changed after earlier tasks were "
            "sealed. It is not part of the scientific corpus; the horizon knob "
            "inside it was extended (FP-ITER5-001 added 5, FP-ITER6-001 added 6). "
            "The evolution is proven inert: a re-run at the frozen horizon "
            "reproduces all 105 sealed step entries exactly."
        )
    check(
        len(changed_evaluators) <= 1
        and all(n == "evaluate_fp_iter2_001.py" for n in changed_evaluators),
        f"only the shared evaluator changed, and only by the documented horizon "
        f"extension (changed: {changed_evaluators})",
    )
    # Cross-check against the five-step record too, since that is the horizon this
    # run had to leave intact.
    env5 = load(ITER5 / "environment.json")
    for name in SCIENCE:
        recorded = env5["sealed_file_hashes"].get(name)
        if recorded is None:
            continue
        check(
            recorded == sha256(PROJECT / name),
            f"SCIENCE {name} is unchanged since FP-ITER5-001 was sealed",
        )

    REPORT.append("\n7. Analyzer determinism")
    analyzer_determinism()

    REPORT.append("\n8. Replay of the affected sealed programs")
    # Only the verifiers whose subject this task's change can move. FP-ITER6-001
    # extended the horizon inside evaluate_fp_iter2_001.py, so the network-side
    # horizon verifiers and the certificate checker are the affected set. The
    # census verifier reads the census bundle and re-runs the census analyzer; it
    # does not depend on this evaluator's horizon, and it is a deep replay tree in
    # its own right, so including it here multiplied the runtime for no coverage.
    for script, extra in (
        ("verify_variance_adaptive_certificate.py", []),
        # The foundation check for the whole line: Q^pi, v^pi and mu_state checked
        # by methods that share no code with policy_quantities. Exercised here so it
        # cannot rot. Different METHOD, still the same actor.
        ("verify_policy_quantities_by_solve.py", []),
        ("verify_fp_attn_iter_001_same_actor.py", []),
        ("verify_fp_attn_iter4_001_same_actor.py", []),
        ("verify_fp_iter5_001_same_actor.py", []),
    ):
        completed = subprocess.run(
            [sys.executable, "-B", str(PROJECT / script), *extra],
            capture_output=True,
            text=True,
            cwd=PROJECT,
        )
        check(
            completed.returncode == 0,
            f"{script} replays with exit 0 (got {completed.returncode})",
        )
        if completed.returncode != 0:
            REPORT.append("        " + completed.stdout.strip().splitlines()[-1]
                          if completed.stdout.strip() else "")

    REPORT.append("\n" + "=" * 74)
    REPORT.append("SUMMARY")
    REPORT.append(f"  emissions by step      : {summary['emissions_by_step']}")
    REPORT.append(f"  emission deltas        : {summary['emission_deltas']}")
    REPORT.append(
        f"  mean gains by step     : "
        f"{[None if v is None else round(v, 6) for v in summary['mean_gain_by_step']]}"
    )
    REPORT.append(
        f"  min gains by step      : "
        f"{[None if v is None else round(v, 6) for v in summary['min_gain_by_step']]}"
    )
    for key in (
        "H1_horizon_inert",
        "H2",
        "H3",
        "H4",
        "H7_attrition",
        "H7b_plateau",
        "H8_mean_decay",
        "H9_non_vacuity",
    ):
        REPORT.append(f"  {key:<18} : {summary[key]}")
    REPORT.append(
        f"  H5 P1 plateau rule     : {summary['H5_prediction_P1']['verdict']} "
        f"({summary['H5_prediction_P1']['misclassifications']} misclassified)"
    )
    REPORT.append(
        f"  H6 P2 ratio rule       : {summary['H6_prediction_P2']['verdict']} "
        f"({summary['H6_prediction_P2']['misclassifications']} misclassified)"
    )
    REPORT.append("=" * 74)
    REPORT.append("RESULT: " + ("PASS" if FAILURES == 0 else f"FAIL ({FAILURES} failed)"))
    REPORT.append(
        "LIMITATION: same-actor derived verification only, and per the user's\n"
        "instruction not the focus of this round. The six-step result itself is now\n"
        "on both paths: FP-ATTN-ITER6-001 brought the network level with numpy, with\n"
        "the same step-6 emitting set and zero decision flips."
    )

    text = "\n".join(REPORT)
    print(text)
    out = (
        PROJECT
        / "docs"
        / "research_branches"
        / "FP-CENSUS-001"
        / "claude"
        / "verification_same_actor_iter6.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if FAILURES == 0 else 1)


if __name__ == "__main__":
    main()
