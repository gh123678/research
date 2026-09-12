"""FP-ATTN-001 same-actor derived verification.

Recomputes the comparison independently of the evaluator and analyzer, checks
the no-mask/no-gate claim executably, verifies that no sealed file changed, and
replays the sealed programs.

This is DERIVED verification by the same actor that executed the task. It is NOT
independent verification: no second actor reconstructed this route.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import fixed_policy_expected_sarsa_scaled as fs  # noqa: E402
from model import (  # noqa: E402
    EndToEndFiniteSoftmaxExpectedSARSA,
    EndToEndMaskedSoftmaxExpectedSARSA,
)

FORMAL = PROJECT / "results" / "FP-ATTN-001" / "claude" / "formal"
SEALED_ENV = (
    PROJECT / "results" / "FP-SCALE-002" / "claude" / "formal" / "environment.json"
)
REPORT: list[str] = []
FAILURES = 0
PAIRS = ("expected_exact", "expected_finite")


def check(ok: bool, message: str) -> None:
    global FAILURES
    if ok:
        REPORT.append(f"  PASS  {message}")
    else:
        FAILURES += 1
        REPORT.append(f"  FAIL  {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> None:
    REPORT.append("FP-ATTN-001 same-actor derived verification")
    REPORT.append("=" * 66)

    bundle = json.loads((FORMAL / "task_results.json").read_text(encoding="utf-8"))
    summary = json.loads((FORMAL / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((FORMAL / "config.json").read_text(encoding="utf-8"))
    environment = json.loads((FORMAL / "environment.json").read_text(encoding="utf-8"))

    REPORT.append("\n1. Frozen inputs")
    check(len(bundle["records"]) == 24, f"24 records (found {len(bundle['records'])})")
    check(float(bundle["atol"]) == 1e-4, "ATOL is the frozen 1e-4")
    check(config["tasks"] == 12, "12 tasks per mixing")
    check(config["mixings"] == [0.08, 0.5], "both frozen mixing settings")

    REPORT.append("\n2. Independently recomputed comparison statistics")
    entries = [
        (r, route, r["routes"][route]) for r in bundle["records"] for route in PAIRS
    ]
    gaps = [e["q_hat_gap_inf"] for _, _, e in entries]
    flips = [x for x in entries if x[2]["decision_flip"]]
    eta_flips = [x for x in entries if x[2]["eta_flip"]]
    numpy_emitted = sum(1 for _, _, e in entries if e["numpy_update_emitted"])
    literal_emitted = sum(1 for _, _, e in entries if e["literal_update_emitted"])
    check(len(entries) == 48, f"48 route-records compared (found {len(entries)})")
    check(
        abs(max(gaps) - summary["max_q_hat_gap_inf"]) < 1e-15,
        f"max |dQ| {max(gaps):.3e} matches summary",
    )
    check(len(flips) == summary["decision_flips"] == 0, "zero decision flips")
    check(len(eta_flips) == summary["eta_flips"] == 0, "zero selected-eta flips")
    check(
        numpy_emitted == summary["numpy_emissions"] == 22,
        f"numpy emissions {numpy_emitted} matches summary and the sealed 22/48",
    )
    check(
        literal_emitted == summary["literal_emissions"] == 22,
        f"literal emissions {literal_emitted} equals the numpy count",
    )
    check(
        summary["sealed_regeneration_failures"] == 0,
        "sealed regeneration validated for every route-record",
    )
    check(
        summary["max_sealed_e_q_gap"] == 0.0,
        f"sealed E_Q reproduced exactly (max gap {summary['max_sealed_e_q_gap']})",
    )
    check(
        summary["max_sealed_lb_gap"] == 0.0,
        f"sealed lower bounds reproduced exactly (max gap {summary['max_sealed_lb_gap']})",
    )
    check(
        all(e["literal_finite"] for _, _, e in entries),
        "every literal Qhat is finite",
    )
    check(
        all(e["literal_within_divergence_guard"] for _, _, e in entries),
        "no literal run tripped the divergence guard",
    )
    layer0 = summary["max_layer0_diagnostic_gap"]
    check(
        layer0 and max(layer0.values()) == 0.0,
        f"layer-0 diagnostics match exactly (max {max(layer0.values()) if layer0 else None})",
    )

    REPORT.append("\n3. H5: the literal finite route is gate-free (executable)")
    finite_net = EndToEndFiniteSoftmaxExpectedSARSA(
        gamma=fs.GAMMA, alpha=fs.ALPHA, zeta=fs.ZETA, xi=fs.XI, tau=fs.TAU
    )
    masked_net = EndToEndMaskedSoftmaxExpectedSARSA(gamma=fs.GAMMA, alpha=fs.ALPHA)
    rng = np.random.default_rng(4242)
    states = torch.as_tensor(rng.integers(0, fs.N_STATES, 512), dtype=torch.long)
    actions = torch.as_tensor(rng.integers(0, fs.N_ACTIONS, 512), dtype=torch.long)
    rewards = torch.as_tensor(rng.normal(size=512), dtype=torch.float32)
    next_states = torch.as_tensor(rng.integers(0, fs.N_STATES, 512), dtype=torch.long)
    policy = torch.full((fs.N_STATES, fs.N_ACTIONS), 1.0 / fs.N_ACTIONS)
    q0 = torch.zeros((fs.N_STATES, fs.N_ACTIONS), dtype=torch.float32)

    with torch.no_grad():
        _, finite_diag = finite_net(q0, states, actions, rewards, next_states, policy)
        _, masked_diag = masked_net(q0, states, actions, rewards, next_states, policy)

    check(
        "visited" not in finite_diag,
        "the finite route exposes no visited-query gate",
    )
    check(
        "visited" in masked_diag,
        "the masked exact route does carry a visited gate and null token",
    )
    finite_write = finite_diag["write_attention"]
    check(
        bool(torch.all(finite_write > 0.0)),
        "every finite-route write weight is strictly positive (full support, no -inf mask)",
    )
    check(
        bool(torch.allclose(finite_write.sum(dim=0), torch.ones_like(finite_write.sum(dim=0)))),
        "finite-route write attention is row-normalised",
    )
    masked_write = masked_diag["write_attention"]
    check(
        bool((masked_write == 0.0).any()),
        "the masked route's write attention does contain exact zeros (its mask)",
    )
    finite_source = (PROJECT / "evaluate_fp_attn_001.py").read_text(encoding="utf-8")
    check(
        "float('-inf')" not in finite_source.replace(" ", ""),
        "this evaluator introduces no -inf mask of its own",
    )

    REPORT.append("\n4. H6: no sealed file changed")
    sealed_env = json.loads(SEALED_ENV.read_text(encoding="utf-8"))
    recorded = environment["sealed_file_hashes"]
    for name, digest in recorded.items():
        current = sha256(PROJECT / name)
        check(current == digest, f"{name} byte-identical to its hash at this run")
    for name, key in (
        ("fixed_policy_expected_sarsa.py", "baseline_module_sha256"),
        ("fixed_policy_variance_certificate.py", "certificate_module_sha256"),
    ):
        forms = {
            "raw": hashlib.sha256((PROJECT / name).read_bytes()).hexdigest(),
            "lf": sha256(PROJECT / name),
        }
        check(
            sealed_env[key] in forms.values(),
            f"{name} still matches its FP-SCALE-002 sealed hash (form: "
            f"{[k for k, v in forms.items() if v == sealed_env[key]] or 'NONE'})",
        )

    REPORT.append("\n5. Replay of the sealed programs")
    for script, extra in (
        ("analyze_fp_attn_001.py", ["--result-dir", str(FORMAL)]),
        ("verify_variance_adaptive_certificate.py", []),
        ("verify_fp_scale_002_same_actor.py", []),
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

    REPORT.append("\n" + "=" * 66)
    REPORT.append("SUMMARY")
    REPORT.append(
        f"  route-records compared        : {len(entries)}\n"
        f"  max |literal Q - numpy Q|_inf : {max(gaps):.3e}  (ATOL {bundle['atol']:.0e})\n"
        f"  numpy emissions               : {numpy_emitted} / 48\n"
        f"  literal emissions             : {literal_emitted} / 48\n"
        f"  decision flips                : {len(flips)}\n"
        f"  selected-eta flips            : {len(eta_flips)}\n"
        f"  sealed regeneration failures  : {summary['sealed_regeneration_failures']}"
    )
    REPORT.append("=" * 66)
    REPORT.append("RESULT: " + ("PASS" if FAILURES == 0 else f"FAIL ({FAILURES} failed)"))
    REPORT.append(
        "LIMITATION: same-actor derived verification only. No second actor\n"
        "reconstructed this route; this is not reciprocal verification."
    )

    text = "\n".join(REPORT)
    print(text)
    out = (
        PROJECT
        / "docs"
        / "research_branches"
        / "FP-ATTN-001"
        / "claude"
        / "verification_same_actor.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
