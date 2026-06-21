#!/usr/bin/env python3
"""Generate Track 1 figures from the crossover sweep CSV.

Produces:
  - figures/track1_speedup_heatmap_q0.png/svg     — speedup heatmap at Q=0
  - figures/track1_speedup_heatmap_q100.png/svg   — at Q=100
  - figures/track1_qbreak_lineplot.png/svg        — Q_break vs k for each n
  - figures/track1_per_epoch_n1M.png/svg          — wall time vs k at n=1M
"""

import sys
import os
sys.path.insert(0, '/Users/shubham/Library/Python/3.9/lib/python/site-packages')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

REPO = "/Users/shubham/Repos/deltasort"
RESULTS = os.path.join(REPO, "rust", "results")
FIGURES = os.path.join(REPO, "figures")
os.makedirs(FIGURES, exist_ok=True)

sweep = pd.read_csv(os.path.join(RESULTS, "track1_sweep.csv"))
qbreak = pd.read_csv(os.path.join(RESULTS, "track1_qbreak.csv"))

# Best BTree per row = min of btree_nth_us (where defined) and btree_mat_us
def best_btree(row):
    v = row["btree_mat_us"]
    if pd.notna(row.get("btree_nth_us")) and not math.isnan(row["btree_nth_us"]):
        v = min(v, row["btree_nth_us"])
    return v

sweep["btree_best_us"] = sweep.apply(best_btree, axis=1)
sweep["speedup_vs_btree"] = sweep["btree_best_us"] / sweep["delta_us"]
sweep["speedup_vs_full"]  = sweep["full_us"]  / sweep["delta_us"]
sweep["speedup_vs_esm"]   = sweep["esm_us"]   / sweep["delta_us"]

# ------------ Heatmap: speedup vs BTree for selected Q ------------
def heatmap_speedup(q_target, fname):
    sub = sweep[sweep["q"] == q_target].copy()
    if sub.empty:
        print(f"No data for Q={q_target}")
        return
    pivot = sub.pivot(index="n", columns="k", values="speedup_vs_btree")
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(np.log10(pivot.values), aspect="auto", cmap="RdBu_r",
                   vmin=-1.5, vmax=1.5, origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{k}" for k in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{n:,}" for n in pivot.index])
    ax.set_xlabel("k  (number of dirty indices per epoch)")
    ax.set_ylabel("n  (collection size)")
    ax.set_title(f"DeltaSort + sorted Vec speedup over best BTreeMap variant\nQ = {q_target} indexed reads per epoch")
    # write speedup factors in cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if pd.notna(v):
                color = "white" if abs(np.log10(v)) > 0.7 else "black"
                ax.text(j, i, f"{v:.1f}x", ha="center", va="center", color=color, fontsize=9)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("log10(speedup)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, fname + ".png"), dpi=150)
    plt.savefig(os.path.join(FIGURES, fname + ".svg"))
    print(f"Wrote {fname}.png/.svg")
    plt.close()

heatmap_speedup(0,   "track1_speedup_heatmap_q0")
heatmap_speedup(100, "track1_speedup_heatmap_q100")

# ------------ Line plot: Q_break vs k for each n ------------
fig, ax = plt.subplots(figsize=(7, 4.5))
for n_val in sorted(qbreak["n"].unique()):
    sub = qbreak[qbreak["n"] == n_val]
    qb = sub["q_break_delta"].apply(lambda v: float(v) if str(v) != "NA" else np.nan)
    ax.plot(sub["k"], qb, marker="o", label=f"n = {n_val:,}")
ax.set_xscale("log")
ax.set_xlabel("k  (number of dirty indices per epoch)")
ax.set_ylabel("Q_break (reads per epoch for sorted Vec to win)")
ax.set_title("Q_break: smallest Q at which sorted Vec + DeltaSort beats best BTreeMap variant")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "track1_qbreak_lineplot.png"), dpi=150)
plt.savefig(os.path.join(FIGURES, "track1_qbreak_lineplot.svg"))
print("Wrote track1_qbreak_lineplot.png/.svg")
plt.close()

# ------------ Per-epoch wall time at n=1M ------------
sub = sweep[(sweep["n"] == 1_000_000) & (sweep["q"] == 0)].sort_values("k")
if not sub.empty:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sub["k"], sub["full_us"],  marker="o", label="sorted Vec + full re-sort")
    ax.plot(sub["k"], sub["esm_us"],   marker="s", label="sorted Vec + ESM")
    ax.plot(sub["k"], sub["delta_us"], marker="^", label="sorted Vec + DeltaSort")
    ax.plot(sub["k"], sub["btree_mat_us"], marker="d", label="BTreeMap + materialize")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k  (dirty indices per epoch)")
    ax.set_ylabel("wall time per epoch (µs, log)")
    ax.set_title("Per-epoch wall time at n = 1,000,000  (Q = 0 reads per epoch)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "track1_per_epoch_n1M.png"), dpi=150)
    plt.savefig(os.path.join(FIGURES, "track1_per_epoch_n1M.svg"))
    print("Wrote track1_per_epoch_n1M.png/.svg")
    plt.close()

# ------------ Per-epoch wall time at n=1M, Q=100 ------------
sub = sweep[(sweep["n"] == 1_000_000) & (sweep["q"] == 100)].sort_values("k")
if not sub.empty:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sub["k"], sub["full_us"],  marker="o", label="sorted Vec + full re-sort")
    ax.plot(sub["k"], sub["esm_us"],   marker="s", label="sorted Vec + ESM")
    ax.plot(sub["k"], sub["delta_us"], marker="^", label="sorted Vec + DeltaSort")
    ax.plot(sub["k"], sub["btree_mat_us"], marker="d", label="BTreeMap + materialize")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel("wall time per epoch (µs, log)")
    ax.set_title("Per-epoch wall time at n = 1,000,000  (Q = 100 reads per epoch)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "track1_per_epoch_n1M_q100.png"), dpi=150)
    plt.savefig(os.path.join(FIGURES, "track1_per_epoch_n1M_q100.svg"))
    print("Wrote track1_per_epoch_n1M_q100.png/.svg")
    plt.close()

# Print headline summary
print("\n=== Headline speedups (sorted Vec + DeltaSort vs best BTreeMap, Q=0) ===")
print(sweep[sweep["q"] == 0][["n","k","speedup_vs_btree","speedup_vs_full","speedup_vs_esm"]].to_string(index=False))
