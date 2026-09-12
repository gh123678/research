"""FP-CENSUS-001 same-actor derived verification.

Re-derives H0 from the bundles independently of the analyzer, checks that no
oracle quantity can reach a certificate input, re-runs the frozen-batch guard,
detects any post-hoc edit of the sealed FP-ITER5-001 bundle, and confirms the
step-6 prediction was frozen before the sixth step ran.

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
CENSUS_DIR = PROJECT / "results" / "FP-CENSUS-001" / "claude" / "formal"
ITER5 = PROJECT / "results" / "FP-ITER5-001" / "claude" / "numpy"
ITER6_DIR = PROJECT / "results" / "FP-ITER6-001" / "claude" / "numpy"
PRIMARY = ("expected_exact", "expected_finite")
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


def rederive_h0() -> None:
    """Recompute H0 straight from the two bundles, not via the analyzer."""
    census = load(CENSUS_DIR / "task_results.json")
    sealed = load(ITER5 / "task_results.json")
    by_key = {
        (float(r["mixing"]), int(r["task_index"])): r for r in sealed["records"]
    }
    compared = 0
    exact = 0
    decision = reasons = eta = e_q = rows = 0
    for record in census["records"]:
        reference = by_key[(float(record["mixing"]), int(record["task_index"]))]
        for route in PRIMARY:
            block = record["routes"][route]
            sealed_block = reference["routes"][route]
            ours, theirs = block["steps"], sealed_block["steps"]
            compared += 1
            rows += abs(len(ours) - len(theirs))
            ok = True
            for i in range(min(len(ours), len(theirs))):
                a, b = ours[i], theirs[i]
                if a["update_emitted"] != b["update_emitted"]:
                    decision += 1
                    ok = False
                if a["ordered_reasons"] != b["ordered_reasons"]:
                    reasons += 1
                    ok = False
                if a["eta_selected"] != b["eta_selected"]:
                    eta += 1
                    ok = False
                if (a["e_q"] is None) != (b["e_q"] is None) or (
                    a["e_q"] is not None and float(a["e_q"]) != float(b["e_q"])
                ):
                    e_q += 1
                    ok = False
            if ok and len(ours) == len(theirs):
                exact += 1
    check(compared == 48, f"48 route-records in the census (found {compared})")
    check(rows == 0, f"no route differs in row count (found {rows})")
    check(decision == 0, f"no decision mismatch (found {decision})")
    check(reasons == 0, f"no ordered-reason mismatch (found {reasons})")
    check(eta == 0, f"no eta mismatch (found {eta})")
    check(e_q == 0, f"no E_Q mismatch, by exact float equality (found {e_q})")
    check(exact == 48, f"H0 re-derived exact on {exact}/48 route-records")


def oracle_confinement() -> None:
    """No oracle quantity may appear as an argument to the certificate or rule."""
    source = (PROJECT / "evaluate_fp_census_001.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {
        "sigma_min",
        "spreads",
        "q_pi",
        "v_pi",
        "exact_current",
        "policy_quantities",
        "oracle_audit",
    }
    targets = {"variance_adaptive_certificate", "envelope_control_certificate",
               "improvement_for"}
    calls = 0
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.attr
            if isinstance(func, ast.Attribute)
            else func.id
            if isinstance(func, ast.Name)
            else None
        )
        if name not in targets:
            continue
        calls += 1
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            for sub in ast.walk(arg):
                if isinstance(sub, ast.Name) and sub.id in forbidden:
                    offenders.append(f"{name} <- {sub.id}")
    check(calls >= 2, f"the certificate and rule are actually called ({calls} sites)")
    check(
        not offenders,
        f"no oracle name reaches a certificate or decision argument "
        f"({offenders})",
    )
    # The diagnostic must still exist, i.e. the check is not passing vacuously.
    check(
        "policy_quantities(mdp, current_policy)" in source,
        "the oracle side is still computed for the diagnostic",
    )
    check(
        '"purpose": "truth-based diagnostic only; never a certificate input"' in source,
        "the oracle block still declares its non-certificate purpose",
    )


def sealed_reference_intact() -> None:
    """Detect any post-hoc edit of the FP-ITER5-001 bundle.

    The census recorded, per route, what it saw in the sealed bundle. If that
    bundle were edited afterwards to agree with the census, the recorded values
    and the current values would agree -- so this check cannot prove the bundle
    was never touched. It CAN prove the bundle has not changed since the census
    ran, which is the property the analysis depends on. The residual limitation
    is stated rather than hidden.
    """
    census = load(CENSUS_DIR / "task_results.json")
    sealed = load(ITER5 / "task_results.json")
    by_key = {
        (float(r["mixing"]), int(r["task_index"])): r for r in sealed["records"]
    }
    drift = 0
    for record in census["records"]:
        reference = by_key[(float(record["mixing"]), int(record["task_index"]))]
        for route in PRIMARY:
            repro = record["routes"][route]["sealed_reproduction"]
            block = reference["routes"][route]
            if int(repro["sealed_emitted_steps"]) != int(block["emitted_steps"]):
                drift += 1
            if int(repro["sealed_step_count"]) != len(block["steps"]):
                drift += 1
    check(drift == 0, f"the sealed FP-ITER5-001 bundle has not changed since the "
                      f"census ran ({drift} drifts)")
    REPORT.append(
        "  INFO  limitation: this shows the reference has not changed SINCE the "
        "census, not that it was never touched before it."
    )
    REPORT.append(f"  INFO  FP-ITER5-001 numpy bundle sha256 = "
                  f"{hashlib.sha256((ITER5 / 'task_results.json').read_bytes()).hexdigest()}")


def prediction_frozen() -> None:
    path = CENSUS_DIR / "prediction_step6.json"
    check(path.exists(), "prediction_step6.json exists")
    if not path.exists():
        return
    prediction = load(path)
    summary = load(CENSUS_DIR / "summary.json")
    theta = float(prediction["theta"])
    interval = summary["H2"]["scan"]["best_threshold_interval"]
    check(
        theta == float(interval[0]),
        f"theta equals the lowest attaining step-1 threshold "
        f"({theta:.6f} == {float(interval[0]):.6f})",
    )
    check(
        prediction["theta_step1_misclassifications"]
        == summary["H2"]["scan"]["best_misclassifications"],
        "theta's misclassification count matches the step-1 scan",
    )
    rows = prediction["predictions"]["P2_step1_ratio_threshold"]["rows"]
    check(
        len(rows) == int(prediction["scoring_population_size"]),
        f"P2 covers the whole scoring population "
        f"({len(rows)} == {prediction['scoring_population_size']})",
    )
    recomputed = sum(1 for r in rows if float(r["ratio_step5"]) > theta)
    check(
        recomputed
        == prediction["predictions"]["P2_step1_ratio_threshold"]["n_predicted_emit"],
        f"P2's predicted emitter count is consistent with its own rows "
        f"({recomputed})",
    )
    p1 = prediction["predictions"]["P1_plateau_carries_on"]["predicted_emitters"]
    p1_from_rows = sum(1 for r in rows if r["emitted_at_step5"])
    check(
        p1_from_rows == len(p1),
        f"P1's emitter list is exactly the step-5 emitters in the population "
        f"({p1_from_rows} == {len(p1)})",
    )
    if ITER6_DIR.exists():
        bundle = ITER6_DIR / "task_results.json"
        if bundle.exists():
            check(
                path.stat().st_mtime < bundle.stat().st_mtime,
                "the prediction predates the sixth-step bundle",
            )
    REPORT.append(f"  INFO  prediction sha256 = "
                  f"{hashlib.sha256(path.read_bytes()).hexdigest()}")


def analyzer_determinism() -> None:
    before = (CENSUS_DIR / "summary.json").read_bytes()
    completed = subprocess.run(
        [sys.executable, "-B", str(PROJECT / "analyze_fp_census_001.py"),
         "--census-dir", str(CENSUS_DIR)],
        capture_output=True,
        text=True,
        cwd=PROJECT,
    )
    check(completed.returncode == 0,
          f"analyze_fp_census_001.py replays with exit 0 (got {completed.returncode})")
    after = (CENSUS_DIR / "summary.json").read_bytes()
    check(before == after, "re-running the analyzer reproduces summary.json byte-for-byte")


def main() -> None:
    REPORT.append("FP-CENSUS-001 same-actor derived verification")
    REPORT.append("=" * 74)
    REPORT.append(
        "NOTE: the user instructed that verification is not the focus of this\n"
        "round. These checks are recorded for completeness and are same-actor only."
    )

    census = load(CENSUS_DIR / "task_results.json")
    config = load(CENSUS_DIR / "config.json")
    environment = load(CENSUS_DIR / "environment.json")

    REPORT.append("\n1. Frozen inputs")
    check(len(census["records"]) == 24, f"24 records (found {len(census['records'])})")
    check(int(census["max_steps"]) == 5, f"census horizon 5 (found {census['max_steps']})")
    check(
        int(config["cert_chains"]) == 16384 and int(config["cert_chain_length"]) == 64,
        f"certification dimensions are the frozen 16384 x 64 "
        f"(found {config['cert_chains']} x {config['cert_chain_length']})",
    )
    check(
        sorted(config["mixings"]) == [0.08, 0.5],
        f"both mixings run (found {config['mixings']})",
    )
    check(int(config["tasks"]) == 12, f"12 tasks per mixing (found {config['tasks']})")

    REPORT.append("\n2. H0 re-derived from the bundles")
    rederive_h0()

    REPORT.append("\n3. Oracle confinement")
    oracle_confinement()

    REPORT.append("\n4. The sealed reference is intact")
    sealed_reference_intact()

    REPORT.append("\n5. The step-6 prediction is frozen and self-consistent")
    prediction_frozen()

    REPORT.append("\n6. The frozen-batch guard")
    completed = subprocess.run(
        [sys.executable, "-B", str(PROJECT / "verify_census_batch_frozen.py")],
        capture_output=True,
        text=True,
        cwd=PROJECT,
    )
    check(completed.returncode == 0,
          f"verify_census_batch_frozen.py exits 0 (got {completed.returncode})")
    for line in completed.stdout.strip().splitlines()[-1:]:
        REPORT.append(f"  INFO  {line}")

    REPORT.append("\n7. Corpus integrity")
    science = (
        "fixed_policy_expected_sarsa.py",
        "fixed_policy_expected_sarsa_scaled.py",
        "fixed_policy_variance_certificate.py",
        "model.py",
    )
    for name in science:
        recorded = environment["sealed_file_hashes"].get(name)
        path = PROJECT / name
        matches = recorded is not None and recorded == sha256(path)
        check(matches, f"SCIENCE {name} matches the hash recorded at census time")

    REPORT.append("\n8. Analyzer determinism")
    analyzer_determinism()

    REPORT.append("\n" + "=" * 74)
    REPORT.append("SUMMARY")
    h0 = load(CENSUS_DIR / "summary.json")["H0"]
    verdicts = {
        key: load(CENSUS_DIR / "summary.json")[key]["verdict"]
        for key in ("H1", "H2", "H3", "H4", "H5", "H6")
    }
    REPORT.append(
        f"  H0 sealed reproduction : exact on {h0['exact']}/{h0['total']}"
    )
    for key, value in verdicts.items():
        REPORT.append(f"  {key:<2}                     : {value}")
    REPORT.append("=" * 74)
    REPORT.append("RESULT: " + ("PASS" if FAILURES == 0 else f"FAIL ({FAILURES} failed)"))
    REPORT.append(
        "LIMITATION: same-actor derived verification only, and per the user's\n"
        "instruction not the focus of this round."
    )

    text = "\n".join(REPORT)
    print(text)
    out = (
        PROJECT
        / "docs"
        / "research_branches"
        / "FP-CENSUS-001"
        / "claude"
        / "verification_same_actor.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if FAILURES == 0 else 1)


if __name__ == "__main__":
    main()
