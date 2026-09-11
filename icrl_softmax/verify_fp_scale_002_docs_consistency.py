"""Verify that the committed documentation matches the sealed artifacts.

Checks every headline number quoted in ACTIVE_WORKSPACE.md, the FP-SCALE-002
task sheet and the route journal against the sealed formal bundle, so that
"reports are written" is backed by a mechanical cross-check rather than prose.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent / "icrl_softmax"
FORMAL = PROJECT / "results" / "FP-SCALE-002" / "claude" / "formal"
summary = json.loads((FORMAL / "summary.json").read_text(encoding="utf-8"))
bundle = json.loads((FORMAL / "task_results.json").read_text(encoding="utf-8"))

PASS = 0
FAIL = 0


def claim(label: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label} {detail}")


# ---- ground truth recomputed from the sealed bundle ------------------------
PRIMARY = ("variance_adaptive_exact", "variance_adaptive_finite")
CONTROL = "envelope_control_exact"
attempted = emitted = nondegrading = strict = violations = control_emitted = 0
ratios = []
blank = []
for record in bundle["records"]:
    control_e = record["routes"][CONTROL]["e_q"]
    if record["routes"][CONTROL]["update_emitted"]:
        control_emitted += 1
    for name in PRIMARY:
        entry = record["routes"][name]
        audit = entry["oracle_audit"]
        attempted += 1
        if entry["e_q"] is not None and control_e is not None and entry["e_q"] > 0:
            ratios.append(control_e / entry["e_q"])
        if audit["certificate_violation"]:
            violations += 1
        if entry["update_emitted"]:
            emitted += 1
            if audit["componentwise_nondegrading"]:
                nondegrading += 1
            if audit["total_value_gain"] > 0:
                strict += 1

print("Ground truth from the sealed bundle")
print(f"  attempted={attempted} emitted={emitted} nondeg={nondegrading} "
      f"strict={strict} violations={violations} control_emitted={control_emitted}")
print(f"  ratio range {min(ratios):.3f}..{max(ratios):.3f}")
print()

print("Cross-checking committed documentation")
docs = {
    "ACTIVE_WORKSPACE.md": (PROJECT / "ACTIVE_WORKSPACE.md").read_text(encoding="utf-8"),
    "task sheet": (PROJECT / "docs/research_tasks/FP-SCALE-002.md").read_text(encoding="utf-8"),
    "journal": (PROJECT / "docs/research_branches/FP-SCALE-002/claude/first_result.md").read_text(
        encoding="utf-8"
    ),
    "robustness": (
        PROJECT / "docs/research_branches/FP-SCALE-002/claude/robustness_check.md"
    ).read_text(encoding="utf-8"),
    "verification": (
        PROJECT / "docs/research_branches/FP-SCALE-002/claude/verification_same_actor.md"
    ).read_text(encoding="utf-8"),
}

claim("summary agrees: attempted", summary["primary_route_records"] == attempted)
claim("summary agrees: emissions", summary["primary_emissions"] == emitted)
claim("summary agrees: non-degrading", summary["componentwise_nondegrading"] == nondegrading)
claim("summary agrees: strict", summary["strict_improvements"] == strict)
claim("summary agrees: violations", summary["certificate_violations"] == violations)
claim("summary agrees: control emissions", summary["control_emissions"] == control_emitted)
claim(
    "summary agrees: ratio range",
    abs(summary["min_control_over_adaptive_ratio"] - min(ratios)) < 1e-9
    and abs(summary["max_control_over_adaptive_ratio"] - max(ratios)) < 1e-9,
)

emission_rate = emitted / attempted
flat = {name: re.sub(r"\s+", " ", text) for name, text in docs.items()}
claim(
    "flattening preserved the ACTIVE_WORKSPACE limitation wording",
    "same-actor derived verification" in flat["ACTIVE_WORKSPACE.md"]
    and "reciprocal verification" in flat["ACTIVE_WORKSPACE.md"],
)
for name, text in flat.items():
    if name == "robustness":
        continue
    claim(
        f"{name}: states the 22/48 emission count",
        "22 / 48" in text or "22/48" in text,
        "(emission count missing)",
    )
    claim(f"{name}: states 45.8%", "45.8" in text, "(rate missing)")
    claim(
        f"{name}: states zero certificate violations",
        "certificate violations" in text
        and ("**0**" in text or "violations : 0" in text or "0 certificate violations" in text),
    )

claim(
    "ACTIVE_WORKSPACE: states the same-actor limitation",
    "same-actor derived verification" in flat["ACTIVE_WORKSPACE.md"]
    and "reciprocal verification" in flat["ACTIVE_WORKSPACE.md"],
)
claim(
    "journal: records the superseded FP-SCALE-001 formal run",
    "superseded" in (PROJECT / "docs/research_tasks/FP-SCALE-001.md").read_text(encoding="utf-8"),
)
claim("journal: records the falsified eta expectation", "falsified" in flat["journal"])
claim(
    "verification: states the limitation",
    "not reciprocal verification" in flat["verification"],
)
claim(
    "robustness: states it does not amend the formal run",
    "does not amend" in flat["robustness"] or "does **not** amend" in flat["robustness"],
)

# The property that matters: no document may claim MORE emissions than the
# sealed count of 22. Rather than trying to parse every ratio, look for the
# explicit overstatement form "<n> / 48 emissions" / "<n>/48 emissions" with
# n > emitted, and separately assert that every document quotes the true pair.
for name, text in docs.items():
    flat_text = re.sub(r"\s+", " ", text)
    overstated = [
        n
        for n in range(emitted + 1, attempted + 1)
        if f"{n} / 48 emission" in flat_text or f"{n}/48 emission" in flat_text
    ]
    claim(
        f"{name}: never claims more than {emitted} emissions",
        not overstated,
        f"found {overstated}",
    )
    if name == "robustness":
        continue
    claim(
        f"{name}: quotes the true emission pair {emitted} of {attempted}",
        f"{emitted} / {attempted}" in flat_text or f"{emitted}/{attempted}" in flat_text,
    )

print()
print(f"{PASS} passed, {FAIL} failed")
print("DOCS CONSISTENT WITH SEALED ARTIFACTS" if FAIL == 0 else "DOCS INCONSISTENT")

