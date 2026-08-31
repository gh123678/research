"""Generate the paper's figures from saved experiment JSON (no re-running needed).

Produces:
  figures/fig1_main_result.png   — 3-seed mean closed-loop curve (random tabular MDPs)
  figures/fig2_tauscan_dense.png — temperature phase transition, dense grid world
  figures/fig3_tauscan_sparse.png— temperature phase transition, sparse grid world

Data sources (all already saved):
  results/eval_main400_s{0,1,2}/returns.json   (hard-max greedy bootstrap)
  results/tauscan_gw_dense/tau_scan.json
  results/tauscan_gw_sparse/tau_scan.json
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = r"C:\Users\Admin\Desktop\research\icrl_softmax"
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fig1():
    seeds = [0, 1, 2]
    trans, teach = [], []
    oracle = []
    random = []
    steps = None
    for s in seeds:
        d = load(os.path.join(BASE, "results", f"eval_main400_s{s}", "returns.json"))
        if steps is None:
            steps = np.array(d["update_steps"], dtype=float)
        trans.append(np.array(d["ret_transformer"], dtype=float))
        teach.append(np.array(d["ret_teacher"], dtype=float))
        oracle.append(d["oracle"])
        random.append(d["random"])
    trans = np.mean(trans, axis=0)
    teach = np.mean(teach, axis=0)
    oracle = float(np.mean(oracle))
    random = float(np.mean(random))

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(steps, trans, "o-", color="tab:blue", label="C3-v3 layer (3-seed mean)", linewidth=1.6, markersize=4)
    ax.plot(steps, teach, "s--", color="tab:green", label="numpy semi-gradient teacher", linewidth=1.6, markersize=4)
    ax.axhline(oracle, color="tab:red", linestyle=":", linewidth=1.4, label="oracle (value iteration)")
    ax.axhline(random, color="tab:gray", linestyle="-.", linewidth=1.4, label="random")
    ax.set_xlabel("in-context update step t")
    ax.set_ylabel("discounted return")
    ax.set_ylim(-0.1, 0.6)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "fig1-main-result.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"OK {out}  (final transformer={trans[-1]:.4f}, teacher={teach[-1]:.4f}, oracle={oracle:.4f}, random={random:.4f})")


def tauscan(json_path, out_name, label_suffix):
    d = load(json_path)
    taus = np.array(d["taus"], dtype=float)
    finals = np.array(d["att_td_lin_final"], dtype=float)
    qlearn = d["qlearn_final"]
    ceiling = d["lin_ceiling"]
    random = d["random"]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.semilogx(taus, finals, "o-", color="tab:blue", label="att_td_lin (score-scaled softmax)", linewidth=1.6, markersize=4)
    ax.axhline(qlearn, color="tab:green", linestyle="--", linewidth=1.4, label="qlearn (semi-gradient)")
    ax.axhline(ceiling, color="tab:orange", linestyle=":", linewidth=1.4, label="RBF linear ceiling")
    ax.axhline(random, color="tab:gray", linestyle="-.", linewidth=1.4, label="random")
    ax.set_xlabel(r"temperature $\tau$ (log)")
    ax.set_ylabel("discounted return")
    ax.set_xticks(taus)
    ax.set_xticklabels([f"{t:g}" for t in taus])
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, out_name)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"OK {out}  ({label_suffix}: best tau={taus[int(np.argmax(finals))]:g} -> {finals.max():.4f}, qlearn={qlearn:.4f})")


def main():
    fig1()
    tauscan(
        os.path.join(BASE, "results", "tauscan_gw_dense", "tau_scan.json"),
        "fig2-tauscan-dense.png",
        "dense",
    )
    tauscan(
        os.path.join(BASE, "results", "tauscan_gw_sparse", "tau_scan.json"),
        "fig3-tauscan-sparse.png",
        "sparse",
    )


if __name__ == "__main__":
    main()
