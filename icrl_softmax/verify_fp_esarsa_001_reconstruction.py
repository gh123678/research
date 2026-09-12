"""FP-ESARSA-001 post-hoc independent reconstruction of the sealed corpus.

Why this exists: FP-ESARSA-001 closed `VERIFIED` by explicit user exemption of
the independent acceptance, and its task sheet states plainly that its evidence
is single-route and was never independently reconstructed. FP-SCALE-001 and
FP-SCALE-002 inherited its certificate and interface, so the project's
foundation rests on a result that had only one route.

What this can and cannot do, stated before any number is reported:

  CAN, from the sealed bundle alone:
    - reproduce the frozen seed schedule and recompute every record's
      generator identity from source, including the truth-based occupancy and
      action-gap statistics;
    - recompute the certificate formula, the mixture radii from the frozen
      verified inversion at the recorded pair counts, the contraction premise,
      and the improvement decision rule.

  CANNOT:
    - recompute the residual means, and therefore `E_Q` from first principles,
      because the held-out trajectories were not serialized in the bundle. The
      residual means are therefore only subject to internal consistency checks,
      never to first-principles recomputation.

So this is a partial reconstruction, and the residual-mean gap is disclosed
rather than papered over.

Read-only: writes nothing, runs no experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent / "icrl_softmax"
import sys  # noqa: E402

sys.path.insert(0, str(PROJECT))

import fixed_policy_expected_sarsa as es  # noqa: E402
from evaluate_fixed_policy_q_routes import make_mdp, make_policy  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402
from evaluate_fixed_policy_expected_sarsa import empirical_route_kernel  # noqa: E402
from time_uniform_mixture_certificate import (  # noqa: E402
    MixtureInversionError,
    build_mixture_grid,
    solve_mixture_boundary,
)

BUNDLE = PROJECT / "results" / "FP-ESARSA-001" / "claude" / "task_results.json"
SUMMARY = PROJECT / "results" / "FP-ESARSA-001" / "claude" / "summary.json"
CONFIG = PROJECT / "results" / "FP-ESARSA-001" / "claude" / "config.json"

RECORDS = json.loads(BUNDLE.read_text(encoding="utf-8"))
SUMMARY_DATA = json.loads(SUMMARY.read_text(encoding="utf-8"))
CONFIG_DATA = json.loads(CONFIG.read_text(encoding="utf-8"))

ROUTES = ("expected_exact", "expected_finite", "sampled_exact")
PRIMARY = ("expected_exact", "expected_finite")
PASSED = 0
FAILED = 0
NOTES: list[str] = []


def check(ok: bool, message: str) -> None:
    global PASSED, FAILED
    if ok:
        PASSED += 1
    else:
        FAILED += 1
        print(f"  FAIL  {message}")


# --------------------------------------------------------------------------
# 1. Reproduce the frozen seed schedule and generator identities from source.
# --------------------------------------------------------------------------
print("FP-ESARSA-001 independent reconstruction (partial, see module docstring)")
print()
print("1. Generator identity reproduced from the frozen seed schedule")

N_STATES = int(CONFIG_DATA["n_states"])
N_ACTIONS = int(CONFIG_DATA["n_actions"])
PI_MIN = float(CONFIG_DATA["pi_min"])
GAMMA = float(CONFIG_DATA["gamma"])
SEED = int(CONFIG_DATA["seed"])
MIXING = tuple(CONFIG_DATA["mixing"])
GAPS = tuple(CONFIG_DATA["gap_bonuses"])
LENGTHS = tuple(sorted(CONFIG_DATA["trajectory_lengths"]))
# FROZEN_TASKS is 30 in the frozen evaluator; records are 2*2*4*30 = 480.
TASKS = len(RECORDS) // (len(MIXING) * len(GAPS) * len(LENGTHS))

seed_sequence = np.random.SeedSequence(SEED)
total_cells = 1 * 1 * len(MIXING) * len(GAPS) * 30
child_seeds = iter(seed_sequence.spawn(total_cells))

expected_identities: list[dict] = []
for mixing in MIXING:
    for gap_bonus in GAPS:
        cell_seeds = [next(child_seeds) for _ in range(30)]
        for task_index in range(TASKS):
            child_seed = cell_seeds[task_index]
            rng = np.random.default_rng(child_seed)
            mdp = make_mdp(N_STATES, N_ACTIONS, GAMMA, mixing, gap_bonus, rng)
            policy = make_policy(N_STATES, N_ACTIONS, PI_MIN, rng)
            exact = policy_quantities(mdp, policy)
            sorted_q = np.sort(np.asarray(exact["q_pi"], dtype=np.float64), axis=1)
            action_gaps = sorted_q[:, -1] - sorted_q[:, -2]
            expected_identities.append(
                {
                    "mixing": mixing,
                    "gap_bonus": gap_bonus,
                    "task_index": task_index,
                    "seed_entropy": int(child_seed.entropy),
                    "spawn_key": list(child_seed.spawn_key),
                    "true_pair_occupancy_min": float(np.min(exact["mu_pair"])),
                    "true_action_gap_min": float(np.min(action_gaps)),
                    "true_action_gap_mean": float(np.mean(action_gaps)),
                    "q_pi": np.asarray(exact["q_pi"], dtype=np.float64),
                }
            )

# Index the sealed records the same way.
sealed_index: dict[tuple, dict] = {}
for record in RECORDS:
    key = (
        float(record["mixing"]),
        float(record["gap_bonus"]),
        int(record["task_index"]),
        int(record["trajectory_length"]),
    )
    sealed_index[key] = record

identity_mismatch = 0
expected_meta = {
    (e["mixing"], e["gap_bonus"], e["task_index"]): e for e in expected_identities
}
for key, record in sealed_index.items():
    mixing, gap_bonus, task_index, _length = key
    expected = expected_meta.get((mixing, gap_bonus, task_index))
    if expected is None:
        identity_mismatch += 1
        continue
    identity = record["oracle_audit"]["generator_identity"]
    for field in ("seed_entropy", "spawn_key"):
        if identity[field] != expected[field]:
            identity_mismatch += 1
    for field in (
        "true_pair_occupancy_min",
        "true_action_gap_min",
        "true_action_gap_mean",
    ):
        if abs(float(identity[field]) - expected[field]) > 1e-12:
            identity_mismatch += 1

check(len(RECORDS) == 480, f"sealed bundle holds 480 records (found {len(RECORDS)})")
check(
    len(sealed_index) == 480,
    f"all 480 (mixing, gap, task, length) keys are distinct (found {len(sealed_index)})",
)
check(
    identity_mismatch == 0,
    f"every generator identity reproduces from source ({identity_mismatch} mismatches)",
)
print(f"  -> {PASSED} identity checks passed, {identity_mismatch} mismatches")

# --------------------------------------------------------------------------
# 2. Certificate formula, from the serialized pieces.
# --------------------------------------------------------------------------
print()
print("2. Certificate formula recomputed from the serialized pieces")
formula_mismatch = 0
epsilon_mismatch = 0
groups_mismatch = 0
for record in RECORDS:
    for route in ROUTES:
        entry = record["routes"][route]
        cert = entry["certificate"]
        if cert["e_q"] is None:
            continue
        means = np.asarray(cert["residual_means"], dtype=np.float64)
        radii = np.asarray(cert["radii"], dtype=np.float64)
        epsilon = float(np.max(np.abs(means) + radii))
        if abs(epsilon - float(cert["epsilon_res"])) > 1e-12:
            epsilon_mismatch += 1
        recomputed = epsilon / (1.0 - GAMMA)
        if abs(recomputed - float(cert["e_q"])) > 1e-9:
            formula_mismatch += 1
        if int(cert["n_groups"]) != N_STATES * N_ACTIONS:
            groups_mismatch += 1

check(epsilon_mismatch == 0, f"epsilon_res = max_x(|Ybar_x| + r_x) everywhere ({epsilon_mismatch})")
check(formula_mismatch == 0, f"E_Q = epsilon_res/(1-gamma) everywhere ({formula_mismatch})")
check(groups_mismatch == 0, f"n_groups = |S||A| = {N_STATES * N_ACTIONS} everywhere ({groups_mismatch})")

# --------------------------------------------------------------------------
# 3. Mixture radii: invert the frozen formula to recover the held-out count.
# --------------------------------------------------------------------------
# The bundle does not serialize held-out per-pair counts, only training counts,
# so the radius cannot be recomputed forward. It can still be tested: for every
# sealed radius, find the integer held-out count whose frozen-formula radius
# equals it, and require that such a count exists for every non-zero radius.
# A radius that matches no integer count did not come from the frozen formula.
print()
print("3. Mixture radii inverted against the frozen verified formula")

DELTA = float(CONFIG_DATA["certificate_delta"])
value_bound = float(CONFIG_DATA["reward_bound"]) / (1.0 - GAMMA)
PAIR_GRID = build_mixture_grid(n_groups=N_STATES * N_ACTIONS, delta=DELTA)

_CALIBRATION_MAX = 65536
_radius_to_count: dict[int, int] = {}
_calibration_unbracketed = 0
for _count in range(1, _CALIBRATION_MAX + 1):
    try:
        _solution = solve_mixture_boundary(
            _count, PAIR_GRID, n_groups=N_STATES * N_ACTIONS, delta=DELTA
        )
    except MixtureInversionError:
        _calibration_unbracketed += 1
        continue
    _radius = 2.0 * value_bound * float(_solution["boundary"]) / _count
    _radius_to_count.setdefault(round(_radius, 12), _count)


def count_for_radius(radius: float) -> int | None:
    """Count whose frozen-formula radius equals ``radius``, if any."""
    return _radius_to_count.get(round(radius, 12))


radius_checked = 0
radius_unmatched = 0
counts_seen: set[int] = set()
for record in RECORDS:
    for route in ROUTES:
        radii = np.asarray(record["routes"][route]["certificate"]["radii"], dtype=np.float64)
        for radius in radii:
            if radius <= 0.0:
                continue
            radius_checked += 1
            matched = count_for_radius(float(radius))
            if matched is None:
                radius_unmatched += 1
            else:
                counts_seen.add(matched)

check(
    radius_unmatched == 0,
    f"every sealed radius equals the frozen formula at some integer held-out "
    f"count ({radius_unmatched} of {radius_checked} matched nothing)",
)
NOTES.append(
    f"radii inverted: {radius_checked} non-zero radii checked against the frozen "
    f"formula over held-out counts 1..{_CALIBRATION_MAX}"
    f" ({_calibration_unbracketed} counts unbracketed in that range); "
    f"implied held-out counts range "
    f"{min(counts_seen) if counts_seen else 'n/a'}..{max(counts_seen) if counts_seen else 'n/a'}"
)

# --------------------------------------------------------------------------
# 4. Contraction premise and improvement decision rule, via sealed definitions.
# --------------------------------------------------------------------------
print()
print("4. Contraction premise and improvement decision rule")
premise_mismatch = 0
premise_flag_mismatch = 0
emitted_records = 0
emitted_nondecreasing = 0
ethical_issue = 0
for record in RECORDS:
    for route in ROUTES:
        entry = record["routes"][route]
        contraction = entry["contraction"]
        sharpness = contraction["sharpness"]
        counts = np.asarray(entry["train_pair_counts"], dtype=np.float64)
        # Use the SEALED kernel and premise definitions, not a re-derivation.
        kernel = empirical_route_kernel(counts, sharpness)
        diagonal_min = float(np.diag(kernel).min())
        if abs(diagonal_min - float(contraction["diagonal_min"])) > 1e-12:
            premise_mismatch += 1
        if bool(es.contraction_premise_satisfied(kernel, GAMMA)) != bool(
            contraction["premise_satisfied"]
        ):
            premise_flag_mismatch += 1

        improvement = entry["improvement"]
        if improvement["status"] == "safe_update_emitted":
            emitted_records += 1
            policy_plus = np.asarray(improvement["policy_plus"], dtype=np.float64)
            if not np.allclose(policy_plus.sum(axis=1), 1.0, atol=1e-9):
                ethical_issue += 1
            if not np.all(policy_plus > 0.0):
                ethical_issue += 1
            oracle = record["oracle_audit"]["routes"][route]
            if not oracle["oracle_value_decrease"]:
                emitted_nondecreasing += 1

check(
    premise_mismatch == 0,
    f"contraction diagonal_min reproduces from the sealed kernel "
    f"({premise_mismatch} mismatches)",
)
check(
    premise_flag_mismatch == 0,
    f"contraction premise flag reproduces from the sealed predicate "
    f"({premise_flag_mismatch} mismatches)",
)
NOTES.append(f"emissions found in the sealed corpus: {emitted_records}")
check(ethical_issue == 0, f"every emitted policy is positive and row-normalised ({ethical_issue} bad)")

# --------------------------------------------------------------------------
# 5. Claimed totals against the sealed summary.
# --------------------------------------------------------------------------
print()
print("5. Claimed totals recomputed from the record level")
attempted = 0
emitted = 0
certificates = 0
violations = 0
value_decreases = 0
residual_violations = 0
for record in RECORDS:
    for route in PRIMARY:
        entry = record["routes"][route]
        oracle = record["oracle_audit"]["routes"][route]
        attempted += 1
        if entry["certificate"]["e_q"] is not None:
            certificates += 1
        if entry["improvement"]["status"] == "safe_update_emitted":
            emitted += 1
        if oracle["certificate_violation"]:
            violations += 1
        if oracle["oracle_value_decrease"]:
            value_decreases += 1
        if oracle["residual_event_violation"]:
            residual_violations += 1

declared = SUMMARY_DATA["routes"]
for route in PRIMARY:
    block = declared[route]
    check(
        block["records"] == 480,
        f"{route}: summary declares 480 records (found {block['records']})",
    )
    check(
        block["safe_update_emitted"] == 0,
        f"{route}: summary declares 0 safe updates (found {block['safe_update_emitted']})",
    )
    check(
        block["oracle_certificate_violations"] == 0
        and block["oracle_value_decreases"] == 0,
        f"{route}: summary declares 0 oracle violations",
    )

check(emitted == 0, f"record level confirms zero safe updates across primary routes (found {emitted})")
check(violations == 0, f"record level confirms zero certificate violations (found {violations})")
check(value_decreases == 0, f"record level confirms zero oracle value decreases (found {value_decreases})")
check(
    residual_violations == 0,
    f"record level confirms zero residual-event violations (found {residual_violations})",
)
check(
    certificates == 2 * SUMMARY_DATA["routes"]["expected_exact"]["certificate_emitted"],
    f"certificate counts agree between record level ({certificates}) and summary "
    f"({2 * SUMMARY_DATA['routes']['expected_exact']['certificate_emitted']})",
)

# --------------------------------------------------------------------------
print()
print("=" * 70)
for note in NOTES:
    print(f"NOTE  {note}")
print(f"{PASSED} checks passed, {FAILED} failed")
print(
    "PARTIAL RECONSTRUCTION PASS"
    if FAILED == 0
    else "PARTIAL RECONSTRUCTION FAIL"
)
print()
print("DISCLOSURE: the held-out trajectories were not serialized in the sealed")
print("bundle, so residual means and therefore E_Q could not be recomputed from")
print("first principles. Fabricating that claim would be worse than reporting the")
print("gap. What was reconstructed: the seed schedule, every generator identity")
print("including truth-based statistics, the certificate formula, the mixture")
print("radii by inversion against the frozen formula, the contraction diagonal")
print("and premise flag, the emitted-policy validity, and every claimed total.")
print()
print("This is a same-actor, post-hoc, partial reconstruction. It is NOT the")
print("independent second-route acceptance that FP-ESARSA-001 never received,")
print("and it does not convert that task into a reciprocally verified one.")
