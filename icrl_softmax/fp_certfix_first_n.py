"""FP-CERTFIX-001: first-n per-pair extraction and the per-step batch plan.

Implements the data-side of lemma A in
``docs/derivations/FP-CERTFIX-001-certificate-rederivation.md``:

* ``first_n_batch`` keeps, for every ``(state, action)`` pair, the FIRST
  ``n_per_pair`` visits in chain-concatenated time order, and refuses (raises)
  if any pair was visited fewer times. Never split by the realised count, never
  subsample from the realised pool -- both are outside lemma A.
* ``step_seed_parts`` is the pre-registered per-step seed schedule that makes
  the step-``k`` certification batch independent of the history
  ``(train, B_1, ..., B_{k-1})`` (theorem 2, derivation section 6).
"""

from __future__ import annotations

from typing import Any

import numpy as np

N_STATES = 4
N_ACTIONS = 3


def first_n_batch(
    batch: dict[str, Any], n_per_pair: int, *, n_states: int = N_STATES, n_actions: int = N_ACTIONS
) -> tuple[dict[str, np.ndarray] | None, np.ndarray]:
    """Keep the first ``n_per_pair`` visits of every pair, in visit order.

    Returns ``(reduced_batch, counts)`` where every pair has exactly
    ``n_per_pair`` items, or ``(None, counts)`` if any pair fell short -- the
    caller then abstains with ``heldout_pair_support_missing``.
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
