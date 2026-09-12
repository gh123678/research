"""FP-SAMPLE-001: how much certification data does a flip cost?

FP-TIGHT-001 measured that `4x` certification data cuts `E_Q` by `-64%` -- more than
any sound constant repair, and more than the *unsound* envelope-deletion ceiling of
`-45%`. But it measured that on a three-record subset containing no abstainer, so
the only question that matters went untested: **does more data revive records?**

This task answers it with a ladder: `1x, 2x, 4x, 8x` certification data, all `48`
route-records at step 1, so the abstainers are in the population and the flip
clause is exercised.

Why a vectorised sampler is used, and why that is legitimate here
-----------------------------------------------------------------

The sealed generator loops per chain in Python, so `8x` (`8.4M` items) would take
hours per record. That cost is not intrinsic -- it is an artifact of the loop. This
module samples the same law with vectorised draws:

    per step: draw C uniforms, invert the policy CDF for every chain at once,
              then draw C uniforms and invert the transition CDF for the chosen
              (state, action) of every chain at once.

`Generator.choice` with an explicit ``p`` does **not** consume the random stream the
way raw uniform draws do -- that is exactly what the FP-CENSUS-001 batch guard
caught -- so this sampler does NOT reproduce the sealed batch, and it is not used
anywhere that requires the sealed one.

It is legitimate here because the sample-size question needs a *fresh independent*
certification sample, not the sealed one, and because **both rungs of every
comparison use the same sampler**. The ladder is internally consistent. As a
sanity check the `1x` rung is compared against the sealed `1x` arm from FP-TIGHT-001
to confirm the sampler is unbiased rather than merely fast; a discrepancy there
would invalidate the ladder, so it is a mandatory check rather than a remark.

Memory: items are held as int32/int64 mix and generated in chunks, so the `8x`
rung stays inside a few hundred MB.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent / "icrl_softmax"
sys.path.insert(0, str(PROJECT))

import fixed_policy_expected_sarsa_scaled as fs  # noqa: E402

N_STATES = fs.N_STATES
N_ACTIONS = fs.N_ACTIONS
CERT_CHAINS = 16384
CERT_CHAIN_LENGTH = 64
CHUNK_CHAINS = 32768


def vectorised_batch(
    mdp: Any,
    policy: np.ndarray,
    mu_state: np.ndarray,
    seed_parts: list[Any],
    chains: int,
    chain_length: int = CERT_CHAIN_LENGTH,
) -> dict[str, np.ndarray]:
    """A fresh certification sample of ``chains x chain_length`` items.

    Draws in chunks to bound memory. Valid for the sample-size ladder, where a new
    independent sample is required; NOT a reproduction of the sealed batch.
    """
    transition = np.asarray(mdp["P"], dtype=np.float64)
    reward = np.asarray(mdp["R"], dtype=np.float64)
    rng = np.random.default_rng(seed_parts)
    policy_cdf = np.cumsum(policy, axis=1)
    policy_cdf[:, -1] = 1.0

    total = chains * chain_length
    states = np.empty(total, dtype=np.int32)
    actions = np.empty(total, dtype=np.int32)
    rewards = np.empty(total, dtype=np.float64)
    next_states = np.empty(total, dtype=np.int32)

    done = 0
    while done < chains:
        size = min(CHUNK_CHAINS, chains - done)
        current = rng.choice(N_STATES, size=size, p=mu_state).astype(np.int32)
        for step in range(chain_length):
            uniform = rng.random(size)
            chosen = (uniform[:, None] > policy_cdf[current]).sum(axis=1)
            np.clip(chosen, 0, N_ACTIONS - 1, out=chosen)
            following_uniform = rng.random(size)
            # Per-chain transition row: cumsum over the last axis of P[state, action].
            rows = transition[current, chosen]
            following = (following_uniform[:, None] > np.cumsum(rows, axis=1)).sum(
                axis=1
            )
            np.clip(following, 0, N_STATES - 1, out=following)
            following = following.astype(np.int32)
            base = (done + np.arange(size)) * chain_length + step
            states[base] = current
            actions[base] = chosen.astype(np.int32)
            rewards[base] = reward[current, chosen, following]
            next_states[base] = following
            current = following
        done += size
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "next_actions": actions,
    }


def main() -> None:
    """Self-test: the vectorised sampler is unbiased against the sealed generator.

    Compares the per-pair item distribution and the empirical residual second
    moment at ``1x`` against the sealed per-chain generator. A sampler that were
    fast but wrong would shift the certificate, so this is a gate.
    """
    import fixed_policy_variance_certificate as vc
    from evaluate_fixed_policy_q_routes import policy_quantities
    from evaluate_fp_tight_001 import certification_batch as sealed_batch

    print("vectorised sampler vs the sealed per-chain generator, 1x")
    print("=" * 78)
    worst = 0.0
    for mixing, task_index in ((0.08, 0), (0.08, 5), (0.5, 0), (0.5, 4)):
        mdp, policy, _ = fs.build_task(task_index=task_index, mixing=mixing)
        exact = policy_quantities(mdp, policy)
        mu_state = np.asarray(exact["mu_state"], dtype=np.float64)
        seed_parts = [fs.SEED, 5881, int(round(mixing * 100)), task_index]
        sealed = sealed_batch(mdp, policy, mu_state, seed_parts)
        fast = vectorised_batch(mdp, policy, mu_state, seed_parts, CERT_CHAINS)

        q_hat = np.asarray(
            fs.run_route(
                "expected_exact",
                policy,
                fs.training_batch(
                    mdp, policy, mu_state,
                    np.random.default_rng([fs.SEED, int(round(mixing * 100)), task_index]),
                ),
            )["q_hat"],
            dtype=np.float64,
        ).reshape(N_STATES, N_ACTIONS)

        flat_s = sealed["states"].astype(np.int64) * N_ACTIONS + sealed["actions"]
        flat_f = fast["states"].astype(np.int64) * N_ACTIONS + fast["actions"]
        counts_s = np.bincount(flat_s, minlength=12) / flat_s.size
        counts_f = np.bincount(flat_f, minlength=12) / flat_f.size
        tv = 0.5 * float(np.sum(np.abs(counts_s - counts_f)))

        res_s = vc.residuals_for(q_hat, policy, sealed)
        res_f = vc.residuals_for(q_hat, policy, fast)
        m2_s = float(np.mean(res_s**2))
        m2_f = float(np.mean(res_f**2))
        rel = abs(m2_f - m2_s) / max(m2_s, 1e-12)
        worst = max(worst, tv, rel)
        print(
            f"  mixing={mixing} task={task_index:>2}  pair-share TV distance "
            f"{tv:.4f}   E[Y^2] sealed {m2_s:.5f} vs fast {m2_f:.5f} "
            f"(rel {rel:+.1%})"
        )
    print("=" * 78)
    print(f"worst discrepancy {worst:.4f}; a sampler that were wrong would shift E_Q")
    print("and invalidate the whole ladder, so this gate matters.")
    return 0 if worst < 0.02 else 1


if __name__ == "__main__":
    raise SystemExit(main())
