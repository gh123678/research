"""FP-CERTFIX-001: per-pair sample extraction and the per-step batch plan.

EXTRACTION PROTOCOLS

* ``first_visit_batch`` -- the VALID one (lemma A' of the derivation, replacing
  the falsified lemma A). For every ``(state, action)`` pair, keep ONE item per
  independent chain: the chain's FIRST visit to the pair. The retained count
  ``N_x`` is then a function of the chains' pre-visit trajectories, hence
  independent of the retained values; conditional on ``N_x = n`` the sample is
  iid ``P_x^{⊗n}``, and Maurer-Pontil applies per ``n`` with no union cost over
  the random ``N``.
* ``first_n_batch`` -- RETAINED ONLY to reproduce the 2026-09-13 morning
  artifacts. Its guarantee rested on lemma A, which the independent review
  (``docs/research_branches/FP-CERTFIX-001/claude/review_of_derivation.md``)
  proved FALSE with an exact counterexample. Do not use it for new claims.
* ``step_seed_parts`` is the pre-registered per-step seed schedule that makes
  the step-``k`` certification batch independent of the history
  ``(train, B_1, ..., B_{k-1})`` (theorem 2, derivation section 6).
"""

from __future__ import annotations

from typing import Any

import numpy as np

N_STATES = 4
N_ACTIONS = 3


def first_visit_batch(
    batch: dict[str, Any],
    chain_length: int,
    *,
    n_states: int = N_STATES,
    n_actions: int = N_ACTIONS,
    max_chains: int | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Keep one item per chain per pair: the chain's FIRST visit to the pair.

    The batch is chain-concatenated with fixed ``chain_length`` (as produced by
    ``vectorised_batch``). ``max_chains`` restricts to the FIRST that many
    chains (an independent subset -- used so a "4x data" arm can share the same
    drawn batch with the 1x arms). Returns ``(reduced_batch, counts)`` where
    ``counts[x]`` is the number of retained chains visiting pair ``x``.
    """
    flat = np.asarray(batch["states"], dtype=np.int64) * int(n_actions) + np.asarray(
        batch["actions"], dtype=np.int64
    )
    total = flat.size
    if total % int(chain_length) != 0:
        raise ValueError("batch length is not a multiple of chain_length")
    n_chains = total // int(chain_length)
    if max_chains is not None:
        n_chains = min(n_chains, int(max_chains))
    grid = flat[: n_chains * int(chain_length)].reshape(n_chains, int(chain_length))
    keep: list[int] = []
    counts = np.zeros(int(n_states) * int(n_actions), dtype=np.int64)
    for pair in range(int(n_states) * int(n_actions)):
        mask = grid == pair
        has = mask.any(axis=1)
        counts[pair] = int(has.sum())
        if counts[pair]:
            first_pos = mask[has].argmax(axis=1)
            chain_ids = np.flatnonzero(has)
            keep.extend((chain_ids * int(chain_length) + first_pos).tolist())
    keep_arr = np.sort(np.asarray(keep, dtype=np.int64))
    reduced = {key: np.asarray(value)[keep_arr] for key, value in batch.items()}
    return reduced, counts


def first_n_batch(
    batch: dict[str, Any], n_per_pair: int, *, n_states: int = N_STATES, n_actions: int = N_ACTIONS
) -> tuple[dict[str, np.ndarray] | None, np.ndarray]:
    """DEPRECATED (lemma A falsified): first ``n_per_pair`` visits per pair.

    Retained ONLY to reproduce the 2026-09-13 morning sealed artifacts. The
    fixed-count-under-``{N_x >= n}`` claim is false (the count is a function of
    the residual sequence); use ``first_visit_batch`` for any new claim.
    """
    n = int(n_per_pair)
    flat = np.asarray(batch["states"], dtype=np.int64) * int(n_actions) + np.asarray(
        batch["actions"], dtype=np.int64
    )
    counts = np.bincount(flat, minlength=int(n_states) * int(n_actions))
    if int(counts.min()) < n:
        return None, counts.astype(np.int64)
    keep = np.empty(int(n_states) * int(n_actions) * n, dtype=np.int64)
    pos = 0
    for pair in range(int(n_states) * int(n_actions)):
        members = np.flatnonzero(flat == pair)
        keep[pos : pos + n] = members[:n]
        pos += n
    keep.sort()
    reduced = {key: np.asarray(value)[keep] for key, value in batch.items()}
    return reduced, counts.astype(np.int64)


def step_seed_parts(
    base_seed: int, task_salt: int, mixing: float, task_index: int, step: int
) -> list[Any]:
    """Pre-registered seed schedule: one independent stream per (record, step).

    The schedule is fixed before any run and indexed by ``step``, so the step-k
    batch is drawn independently of everything the history used (theorem 2).
    """
    return [
        int(base_seed),
        int(task_salt),
        int(round(float(mixing) * 100)),
        int(task_index),
        int(step),
    ]
