"""FP-ATTN-ITER6-001 same-actor derived verification.

Re-derives the network horizon-inertness proof from the bundles, proves
structurally that the network path's decisions are taken on the literal network's
`Qhat` and never on the numpy comparator, checks the path agreement at step 6,
bounds the shared-evaluator change, and replays the affected verifiers.

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
NET6 = PROJECT / "results" / "FP-ATTN-ITER6-001" / "claude" / "network"
NET5 = PROJECT / "results" / "FP-ITER5-001" / "claude" / "network"
NP6 = PROJECT / "results" / "FP-ITER6-001" / "claude" / "numpy"
PRIMARY = ("expected_exact", "expected_finite")
SCIENCE = (
    "fixed_policy_expected_sarsa.py",
    "fixed_policy_expected_sarsa_scaled.py",
    "fixed_policy_variance_certificate.py",
    "model.py",
    "verify_variance_adaptive_certificate.py",
)
# Task evaluators, not science: these evolve as horizons are extended. Both are
# allowed to differ from an older record, and the second one's change is
# documented and proven inert by FP-ITER6-001.
ALLOWED_EVALUATOR_CHANGES = {"evaluate_fp_attn_iter_001.py", "evaluate_fp_iter2_001.py"}
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
    bundle = load(NET6 / "task_results.json")
    sealed = load(NET5 / "task_results.json")
    by_key = {
        (float(r["mixing"]), int(r["task_index"])): r for r in sealed["records"]
    }
    compared = expected = 0
    mismatches: list[tuple] = []
    gap_mismatches: list[tuple] = []
    for record, route, block in blocks(bundle):
        reference = by_key[(float(record["mixing"]), int(record["task_index"]))]
        reference_steps = reference["routes"][route]["steps"]
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
            if ours.get("q_hat_gap_vs_numpy") != theirs.get("q_hat_gap_vs_numpy"):
                gap_mismatches.append(
                    (record["mixing"], record["task_index"], route, index + 1)
                )
    check(compared == expected, f"every sealed row compared ({compared} == {expected})")
    check(compared == 117, f"117 sealed rows in levels 1--5 (found {compared})")
    check(not mismatches, f"no decision/eta/E_Q/reason mismatch ({len(mismatches)})")
    check(
        not gap_mismatches,
        f"every recorded Qhat gap reproduced exactly ({len(gap_mismatches)} mismatches)",
    )
    return compared


def decisions_use_the_literal_network() -> None:
    """The recorded decision must be taken on the network's Qhat, not numpy's.

    The evaluator legitimately computes a numpy comparator too, so it is not enough
    to check that some call site uses the literal. The check is on the call sites
    whose RESULT is bound to ``certificate`` and ``decision`` -- the ones that
    actually drive the iteration.
    """
    tree = ast.parse(
        (PROJECT / "evaluate_fp_attn_iter_001.py").read_text(encoding="utf-8")
    )
    inspected = 0
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if targets not in (["certificate"], ["decision"]):
            continue
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        inspected += 1
        names = {sub.id for sub in ast.walk(value) if isinstance(sub, ast.Name)}
        if "q_hat_literal" not in names:
            offenders.append(f"{targets[0]} <- {sorted(names)}")
        if "q_hat_numpy" in names:
            offenders.append(f"{targets[0]} <- q_hat_numpy")
    check(inspected == 2, f"found both driving call sites ({inspected})")
    check(
        not offenders,
        f"the certificate and the decision are both taken on the literal "
        f"network's Qhat ({offenders})",
    )
    # And the numpy comparator must still exist, so the check is not vacuous.
    source = (PROJECT / "evaluate_fp_attn_iter_001.py").read_text(encoding="utf-8")
    check(
        "cert_numpy = vc.variance_adaptive_certificate" in source
        and "decision_numpy = fs.improvement_for" in source,
        "the numpy comparator is still computed, so the confinement check is not vacuous",
    )


def provenance_recorded() -> None:
    bundle = load(NET6 / "task_results.json")
    steps = [s for _, _, b in blocks(bundle) for s in b["steps"]]
    producers = {s.get("qhat_producer") for s in steps}
    check(
        producers == {"literal_attention_network"},
        f"every one of {len(steps)} steps records the literal network as producer "
        f"(found {producers})",
    )
    six = [s for s in steps if s["step"] == 6]
    check(bool(six), f"the sixth step exists ({len(six)} rows)")


def batches_built_once_per_record() -> None:
    """Batch reuse must be structural, not merely asserted in prose."""
    tree = ast.parse(
        (PROJECT / "evaluate_fp_attn_iter_001.py").read_text(encoding="utf-8")
    )
    wanted = {"training_batch", "certification_batch"}
    found: set[str] = set()
    inside_step_loop: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            callee = node.func
            name = (
                callee.id
                if isinstance(callee, ast.Name)
                else callee.attr
                if isinstance(callee, ast.Attribute)
                else None
            )
            if name in wanted:
                found.add(name)
        if isinstance(node, ast.For):
            target = node.target
            if not (isinstance(target, ast.Name) and target.id == "step_index"):
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    callee = sub.func
                    name = (
                        callee.id
                        if isinstance(callee, ast.Name)
                        else callee.attr
                        if isinstance(callee, ast.Attribute)
                        else None
                    )
                    if name in wanted:
                        inside_step_loop.append(name)
    check(found == wanted, f"both batch builders are present (found {sorted(found)})")
    check(not inside_step_loop, f"no batch rebuilt inside the step loop ({inside_step_loop})")


def validity() -> None:
    bundle = load(NET6 / "task_results.json")
    steps = [s for _, _, b in blocks(bundle) for s in b["steps"]]
    six = [s for s in steps if s["step"] == 6 and s["update_emitted"]]
    degrading = [
        s for s in six if not s["oracle_audit"].get("componentwise_nondegrading", False)
    ]
    nonpositive = [s for s in six if s["oracle_audit"]["total_value_gain"] <= 0.0]
    violations = [s for s in steps if s["oracle_audit"].get("certificate_violation")]
    nondeg_all = [
        s for s in steps
        if s["update_emitted"]
        and min(s["oracle_audit"]["value_delta_vs_previous"]) < -1e-12
    ]
    missing = [s for s in steps if not s["update_emitted"] and not s["ordered_reasons"]]
    check(bool(six), f"at least one sixth-step emission exists ({len(six)})")
    check(not degrading, f"zero sixth-step degrading violations ({len(degrading)})")
    check(not nonpositive, f"zero sixth-step non-positive gains ({len(nonpositive)})")
    check(not violations, f"zero certificate violations over six steps ({len(violations)})")
    check(not nondeg_all, f"no emitted step degrades any state ({len(nondeg_all)})")
    check(not missing, f"every abstention carries a frozen reason ({len(missing)} missing)")


def path_agreement() -> None:
    net = load(NET6 / "task_results.json")
    np6 = load(NP6 / "task_results.json")

    def level_sets(bundle):
        out = {}
        for level in range(1, 7):
            out[level] = {
                (r["mixing"], r["task_index"], route)
                for r, route, b in blocks(bundle)
                for s in b["steps"]
                if s["step"] == level and s["update_emitted"]
            }
        return out

    a, b = level_sets(net), level_sets(np6)
    for level in range(1, 7):
        same = a[level] == b[level]
        check(
            same,
            f"step {level} emitting sets agree ({len(a[level])} vs {len(b[level])})"
            + ("" if same else f": network-only {sorted(a[level] - b[level])}, "
                               f"numpy-only {sorted(b[level] - a[level])}"),
        )
    flips = [
        (r["mixing"], r["task_index"], route, s["step"])
        for r, route, b_ in blocks(net)
        for s in b_["steps"]
        if s.get("decision_flip_vs_numpy")
    ]
    eta_flips = [
        (r["mixing"], r["task_index"], route, s["step"])
        for r, route, b_ in blocks(net)
        for s in b_["steps"]
        if s.get("eta_flip_vs_numpy")
    ]
    check(not flips, f"zero decision flips against the comparator ({len(flips)})")
    check(not eta_flips, f"zero eta flips against the comparator ({len(eta_flips)})")
    atol = float(net["atol"])
    gaps = [
        (float(s["q_hat_gap_vs_numpy"]), r["mixing"], r["task_index"], route, s["step"])
        for r, route, b_ in blocks(net)
        for s in b_["steps"]
        if "q_hat_gap_vs_numpy" in s
    ]
    worst = max(gaps)[0] if gaps else None
    check(
        worst is not None and worst <= atol,
        f"the worst network-versus-numpy gap over all steps is "
        f"{'n/a' if worst is None else f'{worst:.3e}'} <= {atol:.0e}",
    )
    # The Qhat gap must be a real float32-vs-float64 difference, not a suspicious
    # exact zero that would suggest the network output was replaced by numpy.
    check(
        bool(gaps) and min(g[0] for g in gaps) > 0.0,
        f"every step shows a nonzero float32 gap, so no numpy substitution "
        f"(minimum {min((g[0] for g in gaps), default=float('nan')):.3e})",
    )


def corpus_integrity() -> None:
    environment = load(NET5 / "environment.json")
    changed = []
    for name, digest in environment["sealed_file_hashes"].items():
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
            f"  INFO  shared evaluator {name} changed after FP-ITER5-001 was "
            "sealed. It is not part of the scientific corpus; the horizon knob "
            "inside it was extended (FP-ITER6-001 added 6). The evolution is "
            "proven inert: a re-run at the frozen horizon reproduces all 117 "
            "sealed step entries exactly."
        )
    # The network evaluator itself is recorded by no sealed bundle, so it is
    # reported rather than bounded.
    net_eval = PROJECT / "evaluate_fp_attn_iter_001.py"
    REPORT.append(
        f"  INFO  evaluate_fp_attn_iter_001.py is recorded by no sealed bundle; "
        f"its current sha256 is {sha256(net_eval)}"
    )
    check(
        set(changed) <= ALLOWED_EVALUATOR_CHANGES,
        f"only documented task evaluators changed (changed: {changed})",
    )


def analyzer_determinism() -> None:
    before = (NET6 / "summary.json").read_bytes()
    completed = subprocess.run(
        [sys.executable, "-B", str(PROJECT / "analyze_fp_attn_iter6_001.py"),
         "--network-dir", str(NET6), "--sealed-five", str(NET5), "--numpy-dir", str(NP6)],
        capture_output=True,
        text=True,
        cwd=PROJECT,
    )
    check(completed.returncode == 0,
          f"analyze_fp_attn_iter6_001.py replays with exit 0 (got {completed.returncode})")
    check(before == (NET6 / "summary.json").read_bytes(),
          "re-running the analyzer reproduces summary.json byte-for-byte")


def main() -> None:
    REPORT.append("FP-ATTN-ITER6-001 same-actor derived verification")
    REPORT.append("=" * 74)
    REPORT.append(
        "NOTE: the user instructed that verification is not the focus of this\n"
        "round. These checks are recorded for completeness and are same-actor only."
    )

    bundle = load(NET6 / "task_results.json")
    config = load(NET6 / "config.json")
    summary = load(NET6 / "summary.json")

    REPORT.append("\n1. Frozen inputs")
    check(len(bundle["records"]) == 24, f"24 records (found {len(bundle['records'])})")
    check(int(bundle["max_steps"]) == 6, f"network horizon 6 (found {bundle['max_steps']})")
    check(float(bundle["atol"]) == 1e-4, f"ATOL is 1e-4 (found {bundle['atol']})")
    check(
        int(config["cert_chains"]) == 16384 and int(config["cert_chain_length"]) == 64,
        f"certification dimensions are 16384 x 64 "
        f"(found {config['cert_chains']} x {config['cert_chain_length']})",
    )
    check(int(config["tasks"]) == 12, f"12 tasks per mixing (found {config['tasks']})")
    REPORT.append(f"  INFO  torch {bundle.get('torch')}")

    REPORT.append("\n2. H1 re-derived from the bundles")
    rederive_inertness()

    REPORT.append("\n3. Provenance: the decision is taken on the network's Qhat")
    decisions_use_the_literal_network()
    provenance_recorded()

    REPORT.append("\n4. Batch reuse is structural")
    batches_built_once_per_record()

    REPORT.append("\n5. Validity over six network steps")
    validity()

    REPORT.append("\n6. Path agreement, level by level")
    path_agreement()

    REPORT.append("\n7. Corpus integrity")
    corpus_integrity()

    REPORT.append("\n8. Analyzer determinism")
    analyzer_determinism()

    REPORT.append("\n9. Replay of the affected sealed programs")
    # Only the verifiers whose subject this task actually touches: the network
    # evaluator's change can move the network tasks and the shared certificate, and
    # nothing else. The census and numpy sixth-step verifiers exercise the numpy
    # path, which this task does not change, and they are themselves replayed by
    # FP-ITER6-001's verifier -- including them here would nest the replay tree and
    # multiply the runtime for no extra coverage.
    for script, extra in (
        ("verify_variance_adaptive_certificate.py", []),
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

    REPORT.append("\n" + "=" * 74)
    REPORT.append("SUMMARY")
    REPORT.append(f"  emissions by step, network : {summary['emissions_by_step_network']}")
    REPORT.append(f"  emissions by step, numpy   : {summary['emissions_by_step_numpy']}")
    REPORT.append(
        "  mean gains, network        : "
        f"{[None if v is None else round(v, 6) for v in summary['mean_gain_by_step_network']]}"
    )
    REPORT.append(
        "  min gains, network         : "
        f"{[None if v is None else round(v, 6) for v in summary['min_gain_by_step_network']]}"
    )
    REPORT.append(
        "  max |dQ| by step           : "
        f"{[None if v is None else f'{v:.3e}' for v in summary['max_gap_by_step']]}"
    )
    REPORT.append(f"  H1 network horizon inert   : {summary['H1_network_horizon_inert']}")
    for key in ("H2", "H3", "H4", "H8_count_matches_numpy", "H9_mean_decay"):
        REPORT.append(f"  {key:<27}: {summary[key]}")
    for key in ("H5_path_agreement_step6", "H6_drift", "H7_no_flips"):
        REPORT.append(f"  {key:<27}: {summary[key]['verdict']}")
    REPORT.append("=" * 74)
    REPORT.append("RESULT: " + ("PASS" if FAILURES == 0 else f"FAIL ({FAILURES} failed)"))
    REPORT.append(
        "LIMITATION: same-actor derived verification only, and per the user's\n"
        "instruction not the focus of this round."
    )

    text = "\n".join(REPORT)
    print(text)
    out = (
        PROJECT / "docs" / "research_branches" / "FP-ATTN-ITER6-001" / "claude"
        / "verification_same_actor.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if FAILURES == 0 else 1)


if __name__ == "__main__":
    main()
