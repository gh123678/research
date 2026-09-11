"""FP-KERN-002 Claude route: independent post-hoc formal-output check.

Re-derives every formal diagnostic quantity from the frozen common input
without importing the analyzer, then compares against the sealed formal
bundle bit-for-bit (estimates, peers, partitions, sources) or at absolute
tolerance 1e-12 (aggregate metrics and intervals). Read-only: it never
writes to the input or result directories.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

INPUT_DIR = Path(
    r"C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-KERN-002\input"
)
RESULT_DIR = Path(
    r"C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-KERN-002\claude"
)
N_STATES = 6
N_ACTIONS = 4
MISSING_DISTANCE = 1.0
ATOL = 1e-12

failures: list[str] = []
checks = 0


def note(ok: bool, label: str) -> None:
    global checks
    checks += 1
    if not ok:
        failures.append(label)


def close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= ATOL


def strict_load(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda c: (_ for _ in ()).throw(
            ValueError(f"non-strict constant {c} in {path.name}")
        ),
    )


def loao_distances(sig_q, sig_counts, value_bound):
    n_states, n_actions = sig_q.shape
    distances = np.full((n_actions, n_states, n_states), np.nan)
    for action in range(n_actions):
        for state in range(n_states):
            for other in range(n_states):
                if state == other:
                    distances[action, state, other] = 0.0
                    continue
                common = (sig_counts[state] > 0) & (sig_counts[other] > 0)
                common[action] = False
                if int(np.sum(common)) < 2:
                    continue
                scaled = (
                    sig_q[state, common] - sig_q[other, common]
                ) / (2.0 * value_bound)
                distance = float(np.sqrt(np.mean(scaled * scaled)))
                if np.isfinite(distance):
                    distances[action, state, other] = distance
    return distances


def median_positive(values):
    array = np.asarray(list(values), dtype=np.float64)
    selected = np.sort(array[np.isfinite(array) & (array > 0.0)])
    if selected.size == 0:
        return None
    middle = selected.size // 2
    if selected.size % 2:
        return float(selected[middle])
    return float((selected[middle - 1] + selected[middle]) / 2.0)


def all_partitions():
    partitions = []
    others = list(range(1, N_STATES))
    for left in range(len(others)):
        for right in range(left + 1, len(others)):
            first = tuple(sorted((0, others[left], others[right])))
            second = tuple(s for s in range(N_STATES) if s not in first)
            partitions.append((first, second))
    return partitions


def partition_score(distance_matrix, partition):
    replaced = np.where(
        np.isfinite(distance_matrix), distance_matrix, MISSING_DISTANCE
    )
    within = []
    for group in partition:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                within.append(float(replaced[group[i], group[j]]) ** 2)
    return float(np.mean(np.sort(np.asarray(within, dtype=np.float64))))


def pool(target_sums, target_counts, action, sources):
    numerator = 0.0
    denominator = 0.0
    for source in sorted(int(s) for s in sources):
        numerator += float(target_sums[source, action])
        denominator += float(target_counts[source, action])
    if denominator <= 0.0:
        return None, "target_source_unavailable", denominator
    estimate = numerator / denominator
    if not np.isfinite(estimate):
        return None, "pool_denominator_invalid", denominator
    return estimate, "ok", denominator


def mean_ci(values):
    if not values:
        return {"n": 0, "mean": None, "lower": None, "upper": None}
    array = np.asarray(list(values), dtype=np.float64)
    mean = float(np.mean(array))
    if array.size < 2:
        return {"n": int(array.size), "mean": mean, "lower": None, "upper": None}
    sem = float(np.std(array, ddof=1) / np.sqrt(array.size))
    critical = float(student_t.ppf(0.975, df=array.size - 1))
    return {
        "n": int(array.size),
        "mean": mean,
        "lower": mean - critical * sem,
        "upper": mean + critical * sem,
    }


def rmse(errors):
    return float(np.sqrt(np.mean(np.square(np.asarray(errors, dtype=np.float64)))))


def adjusted_rand(labels_a, labels_b):
    n = len(labels_a)
    contingency: dict[tuple[int, int], int] = {}
    a_counts: dict[int, int] = {}
    b_counts: dict[int, int] = {}
    for x, y in zip(labels_a, labels_b):
        contingency[(x, y)] = contingency.get((x, y), 0) + 1
        a_counts[x] = a_counts.get(x, 0) + 1
        b_counts[y] = b_counts.get(y, 0) + 1

    def c2(v):
        return v * (v - 1) / 2.0

    sum_ij = sum(c2(v) for v in contingency.values())
    sum_a = sum(c2(v) for v in a_counts.values())
    sum_b = sum(c2(v) for v in b_counts.values())
    total = c2(n)
    expected = sum_a * sum_b / total
    maximum = 0.5 * (sum_a + sum_b)
    return (sum_ij - expected) / (maximum - expected)


def estimate_array(route_output):
    return np.array(
        [
            [np.nan if v is None else float(v) for v in row]
            for row in route_output["estimate"]
        ],
        dtype=np.float64,
    )


def serialized_estimate(record, route):
    return estimate_array(record["kernel_generalization"]["routes"][route])


def main() -> int:
    records_in = strict_load(INPUT_DIR / "task_results.json")
    bundle = strict_load(RESULT_DIR / "diagnostic_records.json")
    summary = strict_load(RESULT_DIR / "summary.json")
    analysis = strict_load(RESULT_DIR / "analysis.json")
    out_records = bundle["records"]

    # 1. Matrix identity between frozen input and formal output records.
    note(bundle["record_count"] == 480 and len(out_records) == 480, "count")
    by_index = {int(r["record_index"]): r for r in records_in}
    note(len(by_index) == 480, "input unique indices")
    families = {"current_unstructured": 0, "hidden_cluster": 0}
    cells = set()
    for out in out_records:
        src = by_index[int(out["record_index"])]
        same_identity = all(
            out[k] == src[k]
            for k in (
                "environment_family",
                "trajectory_length",
                "mixing",
                "gap_bonus",
                "task_index",
            )
        )
        note(same_identity, f"identity record {out['record_index']}")
        families[out["environment_family"]] += 1
        cells.add(
            (
                out["environment_family"],
                out["task_index"],
                out["trajectory_length"],
                out["mixing"],
                out["gap_bonus"],
            )
        )
    note(families == {"current_unstructured": 240, "hidden_cluster": 240}, "families")
    note(len(cells) == 2 * 15 * 4 * 2 * 2, "matrix cells")

    partitions = all_partitions()
    note(len(partitions) == 10, "ten partitions")
    note(all(0 in p[0] for p in partitions), "state zero first group")

    # 2. Exact per-record route reconstruction.
    exact_mismatch = 0
    invariant_breach = 0
    for out in out_records:
        src = by_index[int(out["record_index"])]
        obs = src["observable_inputs"]
        sig_q = np.asarray(obs["signature_q"], dtype=np.float64)
        sig_counts = np.asarray(obs["signature_counts"], dtype=np.int64)
        target_sums = np.asarray(obs["target_sums"], dtype=np.float64)
        target_counts = np.asarray(obs["target_counts"], dtype=np.int64)
        value_bound = float(obs["value_bound"])
        true_q = np.asarray(src["oracle_audit"]["true_q"], dtype=np.float64)
        hidden = src["environment_family"] == "hidden_cluster"
        distances = loao_distances(sig_q, sig_counts, value_bound)

        # observable route: partitions, sources, pooled estimates
        obs_out = out["routes"]["observable_balanced_cluster"]
        selections = []
        for action in range(N_ACTIONS):
            scores = [partition_score(distances[action], p) for p in partitions]
            minimum = min(scores)
            winners = [i for i, s in enumerate(scores) if s == minimum]
            serial = obs_out["partition_by_action"][action]
            if serial["scores"] != scores:
                exact_mismatch += 1
            if len(winners) != 1:
                if serial["status"] != "partition_tie":
                    exact_mismatch += 1
                selections.append(None)
            else:
                first, second = partitions[winners[0]]
                if serial["status"] != "ok" or serial["groups"] != [
                    list(first),
                    list(second),
                ]:
                    exact_mismatch += 1
                if not close(serial["score"], minimum):
                    exact_mismatch += 1
                selections.append((set(first), set(second)))
        for state in range(N_STATES):
            for action in range(N_ACTIONS):
                reason = obs_out["reason"][state][action]
                estimate = obs_out["estimate"][state][action]
                sources = obs_out["source_states"][state][action]
                selection = selections[action]
                if selection is None:
                    if reason != "partition_tie" or estimate is not None:
                        exact_mismatch += 1
                    continue
                group = next(g for g in selection if state in g)
                expected_sources = sorted(
                    group if target_counts[state, action] > 0 else group - {state}
                )
                exp_est, exp_reason, exp_den = pool(
                    target_sums, target_counts, action, expected_sources
                )
                if (
                    sources != expected_sources
                    or reason != exp_reason
                    or estimate != exp_est
                    or not close(obs_out["denominator"][state][action], exp_den)
                ):
                    exact_mismatch += 1

        # oracle q route: peers, sources, pooled estimates
        oracle_out = out["routes"]["oracle_q_nearest2"]
        for state in range(N_STATES):
            for action in range(N_ACTIONS):
                candidates = [
                    other
                    for other in range(N_STATES)
                    if other != state and target_counts[other, action] > 0
                ]
                candidates.sort(
                    key=lambda o: (
                        abs(float(true_q[state, action]) - float(true_q[o, action])),
                        o,
                    )
                )
                peers = candidates[:2]
                sources = sorted(
                    peers + ([state] if target_counts[state, action] > 0 else [])
                )
                exp_est, exp_reason, exp_den = pool(
                    target_sums, target_counts, action, sources
                )
                if (
                    oracle_out["peers"][state][action] != peers
                    or oracle_out["source_states"][state][action] != sources
                    or oracle_out["reason"][state][action] != exp_reason
                    or oracle_out["estimate"][state][action] != exp_est
                ):
                    exact_mismatch += 1

        # generator cluster route
        gen_out = out["routes"]["oracle_generator_cluster"]
        if not hidden:
            if gen_out.get("status") != "not_applicable_family" or "estimate" in gen_out:
                exact_mismatch += 1
        else:
            labels = [int(x) for x in src["oracle_audit"]["generator"]["cluster_by_state"]]
            groups: dict[int, list[int]] = {}
            for state, label in enumerate(labels):
                groups.setdefault(label, []).append(state)
            for state in range(N_STATES):
                cluster = sorted(groups[labels[state]])
                for action in range(N_ACTIONS):
                    sources = (
                        list(cluster)
                        if target_counts[state, action] > 0
                        else [s for s in cluster if s != state]
                    )
                    exp_est, exp_reason, _ = pool(
                        target_sums, target_counts, action, sources
                    )
                    if (
                        gen_out["source_states"][state][action] != sources
                        or gen_out["reason"][state][action] != exp_reason
                        or gen_out["estimate"][state][action] != exp_est
                    ):
                        exact_mismatch += 1

        # pool invariants across all routes
        for route_name in ("oracle_q_nearest2", "observable_balanced_cluster") + (
            ("oracle_generator_cluster",) if hidden else ()
        ):
            route = out["routes"][route_name]
            for state in range(N_STATES):
                for action in range(N_ACTIONS):
                    reason = route["reason"][state][action]
                    estimate = route["estimate"][state][action]
                    sources = route["source_states"][state][action]
                    if reason != "ok" and estimate is not None:
                        invariant_breach += 1
                    if reason == "ok":
                        if target_counts[state, action] == 0 and state in sources:
                            invariant_breach += 1
                        if target_counts[state, action] > 0 and state not in sources:
                            invariant_breach += 1
    note(exact_mismatch == 0, f"exact route reconstruction ({exact_mismatch} mismatches)")
    note(invariant_breach == 0, f"pool invariants ({invariant_breach} breaches)")

    # 3. Predecessor eligibility and zero-count coverage denominators.
    eligibility = {}
    for src in records_in:
        obs = src["observable_inputs"]
        sig_q = np.asarray(obs["signature_q"], dtype=np.float64)
        sig_counts = np.asarray(obs["signature_counts"], dtype=np.int64)
        target_counts = np.asarray(obs["target_counts"], dtype=np.int64)
        distances = loao_distances(sig_q, sig_counts, float(obs["value_bound"]))
        eligible = np.zeros((N_STATES, N_ACTIONS), dtype=bool)
        for action in range(N_ACTIONS):
            upper = [
                float(distances[action, i, j])
                for i in range(N_STATES)
                for j in range(i + 1, N_STATES)
                if np.isfinite(distances[action, i, j])
            ]
            bandwidth = median_positive(upper)
            for state in range(N_STATES):
                finite_neighbors = any(
                    other != state and np.isfinite(distances[action, state, other])
                    for other in range(N_STATES)
                )
                if target_counts[state, action] == 0 and not finite_neighbors:
                    continue
                if bandwidth is None or bandwidth <= 0.0:
                    continue
                eligible[state, action] = True
        eligibility[int(src["record_index"])] = eligible

    # 4. Full five-item screen reconstruction per family per route.
    def screen(family, route_name):
        family_records = [
            r for r in records_in if r["environment_family"] == family
        ]
        family_outputs = [
            o for o in out_records if o["environment_family"] == family
        ]
        zero_eligible = zero_covered = 0
        zero_imp, zero_base, zero_prim = [], [], []
        sparse_imp, sparse_base, sparse_prim = [], [], []
        top_imp, top_prim, top_base = [], [], []
        zero_empty = sparse_empty = top_empty = 0
        p_false = b_false = comparisons = 0
        for src, out in zip(
            sorted(family_records, key=lambda r: r["record_index"]),
            sorted(family_outputs, key=lambda o: o["record_index"]),
        ):
            counts = np.asarray(
                src["observable_inputs"]["target_counts"], dtype=np.int64
            )
            true_q = np.asarray(src["oracle_audit"]["true_q"], dtype=np.float64)
            primary = estimate_array(out["routes"][route_name])
            action_pool = serialized_estimate(src, "action_only_pool")
            local = serialized_estimate(src, "local_unpooled")
            elig = eligibility[int(src["record_index"])]
            zero_mask = counts == 0
            zero_eligible += int(np.sum(zero_mask & elig))
            zero_covered += int(np.sum(zero_mask & elig & np.isfinite(primary)))
            zero_common = zero_mask & np.isfinite(primary) & np.isfinite(action_pool)
            if np.any(zero_common):
                pr = rmse(primary[zero_common] - true_q[zero_common])
                br = rmse(action_pool[zero_common] - true_q[zero_common])
                zero_imp.append(br - pr)
                zero_base.append(br)
                zero_prim.append(pr)
            else:
                zero_empty += 1
            sparse_common = (
                (counts >= 1) & (counts <= 4) & np.isfinite(primary) & np.isfinite(local)
            )
            if np.any(sparse_common):
                pr = rmse(primary[sparse_common] - true_q[sparse_common])
                br = rmse(local[sparse_common] - true_q[sparse_common])
                sparse_imp.append(br - pr)
                sparse_base.append(br)
                sparse_prim.append(pr)
            else:
                sparse_empty += 1
            sparse_states = np.any(counts <= 4, axis=1)
            complete = (
                np.all(np.isfinite(primary), axis=1)
                & np.all(np.isfinite(action_pool), axis=1)
                & sparse_states
            )
            if np.any(complete):
                truth = np.argmax(true_q[complete], axis=1)
                pa = float(np.mean(np.argmax(primary[complete], axis=1) == truth))
                ba = float(np.mean(np.argmax(action_pool[complete], axis=1) == truth))
                top_imp.append(pa - ba)
                top_prim.append(pa)
                top_base.append(ba)
            else:
                top_empty += 1
            for state in range(N_STATES):
                for left in range(N_ACTIONS):
                    for right in range(left + 1, N_ACTIONS):
                        if not all(
                            np.isfinite(x)
                            for x in (
                                primary[state, left],
                                primary[state, right],
                                action_pool[state, left],
                                action_pool[state, right],
                            )
                        ):
                            continue
                        td = float(true_q[state, left] - true_q[state, right])
                        pd = float(primary[state, left] - primary[state, right])
                        bd = float(
                            action_pool[state, left] - action_pool[state, right]
                        )
                        p_false += int(pd > 0.0 and td <= 0.0)
                        b_false += int(bd > 0.0 and td <= 0.0)
                        comparisons += 1
        zero_ci = mean_ci(zero_imp)
        sparse_ci = mean_ci(sparse_imp)
        top_ci = mean_ci(top_imp)
        zero_rel = float(np.mean(zero_imp)) / float(np.mean(zero_base))
        sparse_rel = float(np.mean(sparse_imp)) / float(np.mean(sparse_base))
        coverage = zero_covered / zero_eligible if zero_eligible else None
        criteria = {
            "zero_eligible_coverage_at_least_50pct": coverage is not None
            and coverage >= 0.50,
            "zero_rmse_improves_10pct_and_ci_positive": zero_rel >= 0.10
            and zero_ci["lower"] is not None
            and zero_ci["lower"] > 0.0,
            "sparse_rmse_improves_10pct_and_ci_positive": sparse_rel >= 0.10
            and sparse_ci["lower"] is not None
            and sparse_ci["lower"] > 0.0,
            "top_action_improves_5pp_and_ci_positive": top_ci["mean"] is not None
            and top_ci["mean"] >= 0.05
            and top_ci["lower"] is not None
            and top_ci["lower"] > 0.0,
            "false_improvement_within_1pp": (
                p_false / comparisons <= b_false / comparisons + 0.01
            ),
        }
        return {
            "records": len(family_records),
            "zero_coverage": {
                "eligible": zero_eligible,
                "covered": zero_covered,
                "rate": coverage,
            },
            "zero_rmse": {
                "primary_mean": float(np.mean(zero_prim)),
                "baseline_mean": float(np.mean(zero_base)),
                "relative_improvement": zero_rel,
                "paired_improvement": zero_ci,
                "empty_records": zero_empty,
            },
            "sparse_rmse": {
                "primary_mean": float(np.mean(sparse_prim)),
                "baseline_mean": float(np.mean(sparse_base)),
                "relative_improvement": sparse_rel,
                "paired_improvement": sparse_ci,
                "empty_records": sparse_empty,
            },
            "top_action": {
                "primary_mean": float(np.mean(top_prim)),
                "baseline_mean": float(np.mean(top_base)),
                "paired_improvement": top_ci,
                "empty_records": top_empty,
            },
            "false_improvement": {
                "comparisons": comparisons,
                "primary_count": p_false,
                "baseline_count": b_false,
                "primary_rate": p_false / comparisons,
                "baseline_rate": b_false / comparisons,
            },
            "criteria": criteria,
            "screen_pass": all(criteria.values()),
        }

    def compare_screen(family, route_name):
        expected = summary["family_screens"][family][route_name]
        actual = screen(family, route_name)
        diffs = []

        def walk(e, a, path):
            if isinstance(e, dict):
                for key in e:
                    walk(e[key], a[key], f"{path}.{key}")
            elif isinstance(e, bool):
                if e != a:
                    diffs.append(path)
            elif isinstance(e, (int, float)) and e is not None:
                if not close(float(e), float(a)):
                    diffs.append(f"{path}: {e} vs {a}")
            elif e is None:
                if a is not None:
                    diffs.append(path)

        walk(expected, actual, f"{family}.{route_name}")
        note(not diffs, f"screen {family}/{route_name}: {diffs[:3]}")
        return actual["screen_pass"]

    gates = {}
    gates["peer_headroom"] = compare_screen("hidden_cluster", "oracle_q_nearest2")
    gates["generator_structure_useful"] = compare_screen(
        "hidden_cluster", "oracle_generator_cluster"
    )
    gates["observable_structure_useful"] = compare_screen(
        "hidden_cluster", "observable_balanced_cluster"
    )
    observable_current = compare_screen(
        "current_unstructured", "observable_balanced_cluster"
    )
    compare_screen("current_unstructured", "oracle_q_nearest2")
    note(
        summary["family_screens"]["current_unstructured"]["oracle_generator_cluster"]
        == "not_applicable_family",
        "generator not_applicable on current family",
    )

    # 5. Ordered classification.
    if gates["observable_structure_useful"] and observable_current:
        expected_class = "GENERAL_PROMISING"
    elif gates["observable_structure_useful"]:
        expected_class = "STRUCTURE_CONDITIONAL_PROMISING"
    elif not gates["observable_structure_useful"] and gates[
        "generator_structure_useful"
    ]:
        expected_class = "REPRESENTATION_GAP"
    elif (
        not gates["generator_structure_useful"]
        and not gates["observable_structure_useful"]
        and gates["peer_headroom"]
    ):
        expected_class = "GENERATOR_STRUCTURE_MISALIGNED"
    else:
        expected_class = "NO_BORROWING_EVIDENCE"
    note(expected_class == summary["classification"], "ordered classification")
    note(expected_class == analysis["classification"], "analysis classification")
    note(
        summary["hidden_family_gates"]
        == {
            "peer_headroom": gates["peer_headroom"],
            "generator_structure_useful": gates["generator_structure_useful"],
            "observable_structure_useful": gates["observable_structure_useful"],
        },
        "hidden gates",
    )

    # 6. Secondary hidden-family diagnostics from serialized partitions.
    ari_values = []
    peer_matches = peer_total = ties = emitted = 0
    for out in out_records:
        if out["environment_family"] != "hidden_cluster":
            continue
        src = by_index[int(out["record_index"])]
        labels = [int(x) for x in src["oracle_audit"]["generator"]["cluster_by_state"]]
        obs_out = out["routes"]["observable_balanced_cluster"]
        for action in range(N_ACTIONS):
            selection = obs_out["partition_by_action"][action]
            if selection["status"] != "ok":
                ties += 1
                continue
            emitted += 1
            groups = selection["groups"]
            partition_labels = [0] * N_STATES
            for state in groups[1]:
                partition_labels[state] = 1
            ari_values.append(adjusted_rand(partition_labels, labels))
            for group in groups:
                for state in group:
                    for other in group:
                        if other == state:
                            continue
                        peer_total += 1
                        peer_matches += int(labels[state] == labels[other])
    ari_ci = mean_ci(ari_values)
    secondary = analysis["secondary_hidden_diagnostics"]
    note(emitted == secondary["emitted_record_actions"], "emitted count")
    note(ties == secondary["excluded_partition_ties"], "tie count")
    note(close(ari_ci["mean"], secondary["adjusted_rand_index"]["summary"]["mean"]), "ARI mean")
    note(close(ari_ci["lower"], secondary["adjusted_rand_index"]["summary"]["lower"]), "ARI lower")
    note(close(ari_ci["upper"], secondary["adjusted_rand_index"]["summary"]["upper"]), "ARI upper")
    note(peer_matches == secondary["peer_precision"]["matches"], "peer matches")
    note(peer_total == secondary["peer_precision"]["assignments"], "peer assignments")
    note(
        close(
            peer_matches / peer_total,
            secondary["peer_precision"]["micro_average"],
        ),
        "peer micro average",
    )

    # 7. Abstention counts.
    abstentions = {name: {} for name in analysis["abstention_counts"]}
    for out in out_records:
        for name in abstentions:
            route = out["routes"][name]
            if route.get("status") == "not_applicable_family":
                abstentions[name]["not_applicable_family"] = (
                    abstentions[name].get("not_applicable_family", 0) + 1
                )
                continue
            for row in route["reason"]:
                for reason in row:
                    abstentions[name][reason] = abstentions[name].get(reason, 0) + 1
    note(abstentions == analysis["abstention_counts"], "abstention counts")

    print(f"independent formal-output checks: {checks} groups evaluated")
    if failures:
        print(f"FAIL {len(failures)} check groups:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        "PASS all independent reconstruction, screen, secondary, abstention, "
        "and classification checks"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
