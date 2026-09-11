"""Independent FP-KERN-002 cross-route reconciliation: GPT sealed bundle vs Claude sealed bundle.

Written fresh for Claude's reciprocal verification of the GPT route. Does not
import either route's analyzer. Compares the sealed GPT formal bundle under
the codex worktree against the sealed Claude formal bundle and the frozen
common input, field by field, with exact float equality.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

float_diffs: Counter[str] = Counter()
max_float_ulp: dict[str, int] = {}
max_float_rel: dict[str, float] = {}
source_order_diffs = 0
source_zero_count_only_diffs = 0
source_set_mismatches = 0

ROOT = Path(r"C:\Users\Admin\Desktop\research\icrl_softmax")
INPUT_DIR = ROOT / "results" / "FP-KERN-002" / "input"
GPT_FORMAL = (
    ROOT
    / "results"
    / "FP-KERN-002"
    / "codex_worktree"
    / "icrl_softmax"
    / "results"
    / "FP-KERN-002"
    / "codex"
    / "formal"
)
CLAUDE_FORMAL = ROOT / "results" / "FP-KERN-002" / "claude"
PREDECESSOR_SUMMARY = (
    ROOT
    / "results"
    / "FP-KERN-001"
    / "codex_worktree"
    / "icrl_softmax"
    / "results"
    / "FP-KERN-001"
    / "codex"
    / "summary.json"
)
FROZEN_HASHES = {
    "config.json": "78aa1bcb5bd2529ab7346412a818ec95e2058deb07dff6436777424e074fb33a",
    "task_results.json": "9f3e777e819fb64625bc2c119bdc5ad462a62277b2ff337b04d2f360253c5da1",
    "source_manifest.json": "670648f7a2761d919f58b25881db45dc6d9d49d4fec25e307c6fe72bf8b31966",
}
ROUTES = (
    "oracle_q_nearest2",
    "oracle_generator_cluster",
    "observable_balanced_cluster",
)

failures: list[str] = []
checks = 0


def check(condition: bool, label: str) -> None:
    global checks
    checks += 1
    if not condition:
        failures.append(label)


def strict_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant in {path.name}: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path.name}: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_float(observed: Any, expected: Any, path: str, bucket: str) -> None:
    if observed is None or expected is None:
        check(observed is None and expected is None, f"null mismatch at {path}")
        return
    if observed == expected:
        return
    ulps = abs(
        np.float64(observed).view(np.int64) - np.float64(expected).view(np.int64)
    )
    relative = abs(observed - expected) / max(abs(observed), abs(expected), 1e-300)
    float_diffs[bucket] += 1
    max_float_ulp[bucket] = max(max_float_ulp.get(bucket, 0), int(ulps))
    max_float_rel[bucket] = max(max_float_rel.get(bucket, 0.0), float(relative))
    check(
        relative <= 1e-12,
        f"float mismatch beyond 1e-12 relative at {path}: {observed!r} != {expected!r}",
    )


def compare_estimates(observed: Any, expected: Any, path: str, bucket: str) -> None:
    for s in range(6):
        for a in range(4):
            compare_float(observed[s][a], expected[s][a], f"{path}[{s}][{a}]", bucket)


def compare_exact(observed: Any, expected: Any, path: str) -> None:
    if isinstance(expected, float) or isinstance(observed, float):
        check(
            isinstance(observed, float)
            and isinstance(expected, float)
            and observed == expected,
            f"float mismatch at {path}: {observed!r} != {expected!r}",
        )
        return
    if type(observed) is not type(expected):
        check(False, f"type mismatch at {path}")
        return
    if isinstance(expected, dict):
        check(set(observed) == set(expected), f"key mismatch at {path}")
        for key in expected:
            if key in observed:
                compare_exact(observed[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        check(len(observed) == len(expected), f"length mismatch at {path}")
        for index, (left, right) in enumerate(zip(observed, expected)):
            compare_exact(left, right, f"{path}[{index}]")
        return
    check(observed == expected, f"value mismatch at {path}: {observed!r} != {expected!r}")


def main() -> None:
    # 1. Frozen input identity.
    for name, expected in FROZEN_HASHES.items():
        check(sha256(INPUT_DIR / name) == expected, f"input hash mismatch: {name}")

    # 2. GPT bundle strict-JSON parse and internal artifact hashes.
    gpt_summary = strict_load(GPT_FORMAL / "summary.json")
    gpt_analysis = strict_load(GPT_FORMAL / "analysis.json")
    gpt_baseline = strict_load(GPT_FORMAL / "baseline_reproduction.json")
    gpt_records = strict_load(GPT_FORMAL / "diagnostic_records.json")
    gpt_environment = strict_load(GPT_FORMAL / "environment.json")
    gpt_hashes = strict_load(GPT_FORMAL / "artifact_hashes.json")
    gpt_manifest = strict_load(GPT_FORMAL / "source_manifest.json")
    check(isinstance(gpt_records, list) and len(gpt_records) == 480, "gpt record count")
    for name, expected in gpt_hashes.items():
        check(sha256(GPT_FORMAL / name) == expected, f"gpt artifact hash mismatch: {name}")
    check(
        gpt_hashes["source_manifest.json"] == FROZEN_HASHES["source_manifest.json"],
        "gpt bundled manifest hash != frozen input manifest hash",
    )
    check(gpt_manifest["task_id"] == "FP-KERN-002", "gpt manifest task id")
    check(
        gpt_manifest["common_execution_start_commit"]
        == "ffdf26b029efde08ea794454a7b5da890108c355",
        "gpt manifest common-start commit",
    )
    check(
        gpt_environment["script_sha256"]
        == sha256(
            ROOT
            / "results"
            / "FP-KERN-002"
            / "codex_worktree"
            / "icrl_softmax"
            / "analyze_kernel_reuse_diagnostics.py"
        ),
        "gpt environment script hash",
    )

    # 3. Frozen input records.
    source_records = strict_load(INPUT_DIR / "task_results.json")
    check(len(source_records) == 480, "input record count")

    # 4. Claude sealed bundle.
    claude_summary = strict_load(CLAUDE_FORMAL / "summary.json")
    claude_analysis = strict_load(CLAUDE_FORMAL / "analysis.json")
    claude_baseline = strict_load(CLAUDE_FORMAL / "baseline_reproduction.json")
    claude_bundle = strict_load(CLAUDE_FORMAL / "diagnostic_records.json")
    claude_records = claude_bundle["records"]
    check(len(claude_records) == 480, "claude record count")

    # 5. Per-record reconciliation.
    identity_fields = (
        "record_index",
        "environment_family",
        "trajectory_length",
        "mixing",
        "gap_bonus",
        "task_index",
    )
    abstentions: dict[str, dict[str, int]] = {route: {} for route in ROUTES}
    for gpt_rec, claude_rec, source in zip(gpt_records, claude_records, source_records):
        index = gpt_rec["record_index"]
        for field in identity_fields:
            check(
                gpt_rec[field] == claude_rec[field] == source[field],
                f"record {index} identity field {field}",
            )
        check(gpt_rec["task_id"] == "FP-KERN-002", f"record {index} task id")
        check(gpt_rec["task_version"] == "1.0", f"record {index} task version")
        check(gpt_rec["source_task_id"] == "FP-KERN-001", f"record {index} source task")
        family = gpt_rec["environment_family"]
        for route in ROUTES:
            gpt_route = gpt_rec["routes"][route]
            claude_route = claude_rec["routes"][route]
            if family == "current_unstructured" and route == "oracle_generator_cluster":
                check(
                    gpt_route == {"status": "not_applicable_family"}
                    and claude_route == {"status": "not_applicable_family"},
                    f"record {index} generator not_applicable serialization",
                )
                abstentions[route]["not_applicable_family"] = (
                    abstentions[route].get("not_applicable_family", 0) + 1
                )
                continue
            compare_estimates(
                gpt_route["estimate"],
                claude_route["estimate"],
                f"record {index} {route} estimate",
                f"{route}.estimate",
            )
            compare_exact(
                gpt_route["reason"],
                claude_route["reason"],
                f"record {index} {route} reason",
            )
            # Source sets: GPT declares only positive-count members; Claude
            # declares all selected group members. Zero-count members contribute
            # exactly zero to numerator and denominator, so the difference is
            # serialization-only. Verify every divergence is zero-count-only.
            for s in range(6):
                for a in range(4):
                    gpt_sources = gpt_route["source_states"][s][a]
                    claude_sources = claude_route["source_states"][s][a]
                    if gpt_sources == claude_sources:
                        continue
                    if sorted(gpt_sources) == sorted(claude_sources):
                        globals()["source_order_diffs"] += 1
                        continue
                    gpt_set = set(gpt_sources)
                    claude_set = set(claude_sources)
                    extra = gpt_set.symmetric_difference(claude_set)
                    zero_only = bool(extra) and all(
                        source["observable_inputs"]["target_counts"][x][a] == 0
                        for x in extra
                    )
                    subset_relation = gpt_set < claude_set or claude_set < gpt_set
                    if zero_only and subset_relation:
                        globals()["source_zero_count_only_diffs"] += 1
                    else:
                        globals()["source_set_mismatches"] += 1
                        check(
                            False,
                            f"record {index} {route} source set mismatch {s},{a}: "
                            f"{gpt_sources} != {claude_sources}",
                        )
            compare_estimates(
                gpt_route["denominator"],
                claude_route["denominator"],
                f"record {index} {route} denominator",
                f"{route}.denominator",
            )
            for row in gpt_route["reason"]:
                for reason in row:
                    abstentions[route][reason] = abstentions[route].get(reason, 0) + 1
            if route == "oracle_q_nearest2":
                check(
                    gpt_route["oracle_use"] == "true_q_peer_ranking_only",
                    f"record {index} oracle-q use label",
                )
                for state in range(6):
                    for action in range(4):
                        sources = gpt_route["source_states"][state][action]
                        check(
                            len(sources) <= 3,
                            f"record {index} oracle-q peer cap at {state},{action}",
                        )
                        target_count = source["observable_inputs"]["target_counts"][state][action]
                        peers = [s for s in sources if s != state]
                        check(len(peers) <= 2, f"record {index} oracle-q <=2 peers")
                        check(
                            all(
                                source["observable_inputs"]["target_counts"][p][action] > 0
                                for p in peers
                            ),
                            f"record {index} oracle-q peers positive-count",
                        )
                        if target_count > 0:
                            check(
                                sources[0] == state,
                                f"record {index} oracle-q self-inclusion",
                            )
                        else:
                            check(
                                state not in sources,
                                f"record {index} oracle-q zero-count excludes self",
                            )
                        true_q = source["oracle_audit"]["true_q"]
                        keys = [
                            (abs(true_q[state][action] - true_q[p][action]), p)
                            for p in peers
                        ]
                        check(
                            keys == sorted(keys),
                            f"record {index} oracle-q frozen peer order",
                        )
            if route == "oracle_generator_cluster":
                check(
                    gpt_route["oracle_use"] == "generator_cluster_labels_only",
                    f"record {index} generator use label",
                )
                labels = source["oracle_audit"]["generator"]["cluster_by_state"]
                expected_groups = [
                    [s for s in range(6) if labels[s] == value]
                    for value in sorted(set(labels))
                ]
                check(
                    gpt_route["groups"] == expected_groups,
                    f"record {index} generator groups match labels",
                )
                for state in range(6):
                    for action in range(4):
                        sources = gpt_route["source_states"][state][action]
                        check(
                            all(labels[s] == labels[state] for s in sources),
                            f"record {index} generator sources same label",
                        )
                        check(
                            all(
                                source["observable_inputs"]["target_counts"][s][action] > 0
                                for s in sources
                            ),
                            f"record {index} generator sources positive-count",
                        )
            if route == "observable_balanced_cluster":
                check(
                    gpt_route["observable_input_keys"]
                    == sorted(source["observable_inputs"].keys()),
                    f"record {index} observable input keys boundary",
                )
                gpt_parts = gpt_route["partition_by_action"]
                claude_parts = claude_route["partition_by_action"]
                check(len(gpt_parts) == len(claude_parts) == 4, f"record {index} partitions")
                for action, (gpt_part, claude_part) in enumerate(
                    zip(gpt_parts, claude_parts)
                ):
                    check(
                        gpt_part["status"] == claude_part["status"],
                        f"record {index} action {action} partition status",
                    )
                    check(
                        gpt_part["groups"] == claude_part["groups"],
                        f"record {index} action {action} partition groups",
                    )
                    check(
                        gpt_part["score"] == claude_part["score"],
                        f"record {index} action {action} partition score",
                    )
                    check(
                        gpt_part["missing_distance_replacement"] == 1.0,
                        f"record {index} action {action} missing-distance rule",
                    )
                    gpt_scores = [c["score"] for c in gpt_part["candidate_scores"]]
                    check(
                        len(gpt_scores) == 10,
                        f"record {index} action {action} ten partitions",
                    )
                    check(
                        sorted(gpt_scores) == sorted(claude_part["scores"]),
                        f"record {index} action {action} candidate score set",
                    )
                    check(
                        gpt_scores == claude_part["scores"],
                        f"record {index} action {action} candidate score order",
                    )
                    gpt_groups = [c["groups"] for c in gpt_part["candidate_scores"]]
                    check(
                        all(g[0][0] == 0 for g in gpt_groups),
                        f"record {index} action {action} state-zero-first serialization",
                    )
                    if gpt_part["status"] == "ok":
                        winners = [
                            s for s in gpt_scores if s == min(gpt_scores)
                        ]
                        check(
                            len(winners) == 1,
                            f"record {index} action {action} unique minimum",
                        )
                # Distances: GPT serializes distance_by_action; recompute
                # leave-one-action-out distances independently from the input.
                obs = source["observable_inputs"]
                sig_q = obs["signature_q"]
                sig_c = obs["signature_counts"]
                vb = obs["value_bound"]
                for action in range(4):
                    for s1 in range(6):
                        for s2 in range(s1 + 1, 6):
                            common = [
                                a
                                for a in range(4)
                                if a != action
                                and sig_c[s1][a] > 0
                                and sig_c[s2][a] > 0
                            ]
                            observed = gpt_route["distance_by_action"][action][s1][s2]
                            if len(common) < 2:
                                check(
                                    observed is None,
                                    f"record {index} distance unavailable {action},{s1},{s2}",
                                )
                            else:
                                scaled = np.array(
                                    [
                                        (sig_q[s1][a] - sig_q[s2][a]) / (2.0 * vb)
                                        for a in common
                                    ],
                                    dtype=np.float64,
                                )
                                expected = float(np.sqrt(np.mean(np.square(scaled))))
                                check(
                                    observed == expected,
                                    f"record {index} distance {action},{s1},{s2}: "
                                    f"{observed!r} != {expected!r}",
                                )
                            check(
                                gpt_route["distance_by_action"][action][s2][s1] == observed,
                                f"record {index} distance symmetry {action},{s1},{s2}",
                            )
            # Estimate reconstructs from declared sources, sums, counts.
            sums = source["observable_inputs"]["target_sums"]
            counts = source["observable_inputs"]["target_counts"]
            for state in range(6):
                for action in range(4):
                    reason = gpt_route["reason"][state][action]
                    estimate = gpt_route["estimate"][state][action]
                    sources = gpt_route["source_states"][state][action]
                    denominator = sum(counts[s][action] for s in sources)
                    if reason == "ok":
                        check(denominator > 0, f"record {index} {route} ok denominator")
                        numpy_estimate = float(
                            np.sum([sums[s][action] for s in sources], dtype=np.float64)
                            / np.sum([counts[s][action] for s in sources], dtype=np.float64)
                        )
                        check(
                            estimate == numpy_estimate,
                            f"record {index} {route} estimate numpy-order reconstruction "
                            f"{state},{action}",
                        )
                        if counts[state][action] == 0:
                            check(
                                state not in sources,
                                f"record {index} {route} zero-count other-state-only",
                            )
                    else:
                        check(
                            estimate is None,
                            f"record {index} {route} abstained estimate null",
                        )

    expected_abstentions = {
        "oracle_q_nearest2": {"ok": 11520},
        "oracle_generator_cluster": {
            "not_applicable_family": 240,
            "ok": 5745,
            "target_source_unavailable": 15,
        },
        "observable_balanced_cluster": {"ok": 11499, "target_source_unavailable": 21},
    }
    check(abstentions == expected_abstentions, f"abstention tallies: {abstentions}")

    # 6. Summary reconciliation (schema mapping route_* <-> primary_*).
    check(gpt_summary["classification"] == claude_summary["classification"] == "NO_BORROWING_EVIDENCE", "classification")
    check(
        gpt_summary["gates"]
        == {
            "peer_headroom": False,
            "generator_structure_useful": False,
            "observable_structure_useful": False,
            "observable_current_family_pass": False,
        },
        "gpt gates",
    )
    check(gpt_summary["gates"] == {
        **claude_summary["hidden_family_gates"],
        "observable_current_family_pass": claude_summary["observable_current_family_pass"],
    }, "gates cross-route")
    check(gpt_summary["records"] == claude_summary["records"] == 480, "summary records")
    check(
        gpt_summary["predecessor_classification"] == "NOT_SUPPORTED",
        "predecessor classification in summary",
    )

    def normalize_screen(screen: dict[str, Any]) -> Any:
        if isinstance(screen, dict):
            return {
                ("route_mean" if key == "primary_mean" else
                 "route_count" if key == "primary_count" else
                 "route_rate" if key == "primary_rate" else
                 "route" if key == "route" else key): normalize_screen(value)
                for key, value in screen.items()
            }
        if isinstance(screen, list):
            return [normalize_screen(item) for item in screen]
        return screen

    for family in ("current_unstructured", "hidden_cluster"):
        for route in ROUTES:
            gpt_screen = gpt_summary["family_screens"][family][route]
            claude_screen = claude_summary["family_screens"][family][route]
            if isinstance(gpt_screen, dict) and gpt_screen.get("status") == "not_applicable_family":
                check(
                    claude_screen == "not_applicable_family"
                    or claude_screen == {"status": "not_applicable_family"},
                    f"{family} {route} not-applicable screen serialization",
                )
                continue
            gpt_screen = {k: v for k, v in gpt_screen.items() if k != "route"}
            compare_exact(
                normalize_screen(claude_screen),
                gpt_screen,
                f"summary {family} {route}",
            )

    # 7. Cluster diagnostics reconciliation.
    gpt_cluster = gpt_summary["cluster_diagnostics"]
    claude_secondary = claude_analysis["secondary_hidden_diagnostics"]
    compare_exact(
        gpt_cluster["adjusted_rand_index"],
        claude_secondary["adjusted_rand_index"]["summary"],
        "adjusted rand index",
    )
    check(
        gpt_cluster["partition_tie_record_actions"]
        == claude_secondary["excluded_partition_ties"]
        == 0,
        "partition tie count",
    )
    check(
        gpt_cluster["emitted_record_actions"]
        == claude_secondary["emitted_record_actions"]
        == 960,
        "emitted record/actions",
    )
    gpt_precision = gpt_cluster["same_cluster_peer_precision"]
    claude_precision = claude_secondary["peer_precision"]
    check(
        gpt_precision["matches"] == claude_precision["matches"] == 5656,
        "peer precision matches",
    )
    check(
        gpt_precision["assignments"] == claude_precision["assignments"] == 11520,
        "peer precision assignments",
    )
    check(
        gpt_precision["rate"] == claude_precision["micro_average"],
        "peer precision rate",
    )
    for gpt_bin, claude_bin in (("zero", "0"), ("1-4", "1-4")):
        gpt_recovery = gpt_cluster["oracle_benefit_recovery"][gpt_bin]
        claude_recovery = claude_analysis["oracle_benefit_recovery"][claude_bin]
        check(
            gpt_recovery["records"] == claude_recovery["contributing_records"],
            f"recovery {gpt_bin} records",
        )
        check(
            gpt_recovery["observable_mean_improvement"]
            == claude_recovery["mean_observable_improvement"],
            f"recovery {gpt_bin} observable mean",
        )
        check(
            gpt_recovery["generator_mean_improvement"]
            == claude_recovery["mean_generator_improvement"],
            f"recovery {gpt_bin} generator mean",
        )
        check(
            gpt_recovery["recovery_ratio"] is None
            and claude_recovery["recovery"] is None,
            f"recovery {gpt_bin} unavailable per frozen rule",
        )
    compare_exact(
        claude_analysis["abstention_counts"],
        expected_abstentions,
        "claude abstention counts vs recomputed",
    )
    check(
        gpt_analysis["decision_rule"] == gpt_summary["gates"],
        "gpt analysis decision rule mirrors gates",
    )
    check(
        gpt_analysis["classification"] == gpt_summary["classification"],
        "gpt analysis classification",
    )
    check(gpt_analysis["invalid_input"] is False, "gpt invalid_input flag")

    # 8. Stage 0 reconciliation.
    check(gpt_baseline["status"] == "PASS", "gpt baseline status")
    check(gpt_baseline["records"] == 480, "gpt baseline records")
    check(
        gpt_baseline["route_reconstruction_mismatches"] == 0,
        "gpt baseline route mismatches",
    )
    check(
        gpt_baseline["classification"] == claude_baseline["replayed_classification"]
        == "NOT_SUPPORTED",
        "baseline classification",
    )
    check(
        gpt_baseline["frozen_summary_file_sha256"] == sha256(PREDECESSOR_SUMMARY),
        "frozen predecessor summary file hash on disk",
    )
    canonical_bytes = (
        PREDECESSOR_SUMMARY.read_bytes().replace(b"\r\n", b"\n").rstrip(b"\n")
    )
    check(
        hashlib.sha256(canonical_bytes).hexdigest()
        == gpt_baseline["canonical_summary_sha256"]
        == "e6327594ccca29e02eeb718849be09d7445a56f1ccbc2d53ee8129d79d2da9d0",
        "canonical CRLF-normalized predecessor summary hash",
    )
    for family in ("current_unstructured", "hidden_cluster"):
        compare_exact(
            gpt_baseline["family_screens"][family],
            claude_baseline["family_screens"][family],
            f"baseline family screen {family}",
        )
    check(
        gpt_baseline["analysis"]["reconstruction_mismatches"] == 0,
        "gpt baseline analysis mismatches",
    )
    check(
        gpt_baseline["analysis"]["decision_rule"]
        == {"hidden_cluster_pass": False, "current_unstructured_pass": False},
        "gpt baseline decision rule",
    )

    print(f"checks={checks} failures={len(failures)}")
    print(
        f"source_order_only_diffs={source_order_diffs} "
        f"source_zero_count_only_diffs={source_zero_count_only_diffs} "
        f"source_set_mismatches={source_set_mismatches}"
    )
    print(f"float_diff_buckets={dict(float_diffs)}")
    print(f"max_float_ulp={max_float_ulp}")
    print(f"max_float_rel={max_float_rel}")
    for failure in failures[:50]:
        print(f"FAIL {failure}")
    if failures:
        raise SystemExit(1)
    print("PASS GPT formal bundle reconciles exactly with the Claude sealed route and frozen input")


if __name__ == "__main__":
    main()
