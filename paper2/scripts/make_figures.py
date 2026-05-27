#!/usr/bin/env python3
"""Generate paper2 figures from rust/results/validated.csv."""
import csv
import os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CSV = os.path.join(ROOT, "rust", "results", "validated.csv")
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))
os.makedirs(FIG, exist_ok=True)

def load():
    rows = list(csv.DictReader(open(CSV)))
    groups = defaultdict(dict)
    for r in rows:
        key = (r["benchmark"], r["distribution"], int(r["n"]), int(r["k"]))
        groups[key][r["strategy"]] = r
    return groups

def to_us(ns_str):
    return float(ns_str) / 1000.0

GROUPS = load()

# --------------------------------------------------------------------
# Figure 1: Headline — DS vs ESM speedup at k/n=0.1% across all 7 wins
# --------------------------------------------------------------------

WIN_APPS = [
    ("depth",    "Game depth sort"),
    ("qos",      "Network WFQ"),
    ("lob",      "Limit order book"),
    ("tracking", "Radar tracking"),
    ("hft",      "HFT alpha basket"),
    ("audio",    "Audio voices"),
    ("mht",      "MHT (12-field cmp)"),
]

# Per app: pick the cell closest to k/n=0.1% (or smallest k), uniform dist
fig, ax = plt.subplots(figsize=(7.2, 3.6))
speedups_esm = []
speedups_full = []
labels = []
for bench, label in WIN_APPS:
    cells = [(k, v) for k, v in GROUPS.items() if k[0] == bench and k[1] == "uniform"]
    cells.sort(key=lambda x: (-x[0][2], x[0][3]))
    chosen = cells[0]
    ds = float(chosen[1]["DeltaSort"]["p50"])
    esm = float(chosen[1]["ESM"]["p50"])
    full = float(chosen[1]["Full"]["p50"])
    speedups_esm.append(esm / ds)
    speedups_full.append(full / ds)
    labels.append(f"{label}\n($n{{=}}{chosen[0][2]}$, $k{{=}}{chosen[0][3]}$)")

x = np.arange(len(labels))
w = 0.38
ax.bar(x - w/2, speedups_esm, w, label="vs ESM (incremental)", color="#3b7dd8")
ax.bar(x + w/2, speedups_full, w, label="vs Full sort", color="#d97a3b")
ax.set_yscale("log")
ax.set_ylabel("Speedup (DS $p50$ baseline)")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.2, rotation=30, ha="right")
ax.legend(loc="upper right", fontsize=8)
ax.grid(axis="y", alpha=0.3, which="both")
ax.axhline(1.0, color="k", linewidth=0.5)
for i, (e, f) in enumerate(zip(speedups_esm, speedups_full)):
    ax.text(i - w/2, e * 1.15, f"{e:.0f}×", ha="center", fontsize=7)
    ax.text(i + w/2, f * 1.15, f"{f:.0f}×", ha="center", fontsize=7)
ax.set_title("DeltaSort speedup at low $k$, large $n$")
ax.set_ylim(0.7, ax.get_ylim()[1] * 1.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "headline.pdf"))
plt.savefig(os.path.join(FIG, "headline.png"), dpi=150)
print("wrote headline.pdf/.png")

# --------------------------------------------------------------------
# Figure 2: DS / ESM ratio vs k/n for representative apps
# --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 3.2))
COLORS = plt.cm.tab10.colors
selected = [("depth", 10000, "Depth"),
            ("lob",    5000, "LOB"),
            ("hft",    5000, "HFT"),
            ("audio",  1024, "Audio"),
            ("mht",    5000, "MHT"),
            ("tracking", 10000, "Tracking")]
for i, (bench, n, label) in enumerate(selected):
    pts = [(int(k[3]), float(v["DeltaSort"]["p50"]), float(v["ESM"]["p50"]))
           for k, v in GROUPS.items() if k[0]==bench and k[1]=="uniform" and k[2]==n]
    pts.sort()
    if not pts: continue
    xs = [p[0]/n*100 for p in pts]
    ys = [p[2]/p[1] for p in pts]
    ax.plot(xs, ys, "o-", label=f"{label} (n={n})", color=COLORS[i])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("$k/n$ (%)")
ax.set_ylabel("ESM time / DS time (×)")
ax.axhline(1.0, color="k", linewidth=0.5, linestyle="--")
ax.text(0.015, 1.7, "DS wins ↑", fontsize=8, color="gray")
ax.text(0.015, 0.55, "ESM wins ↓", fontsize=8, color="gray")
ax.grid(alpha=0.3, which="both")
ax.legend(loc="lower left", fontsize=8, ncol=2)
ax.set_title("Crossover: DS advantage shrinks as $k/n$ grows")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "crossover.pdf"))
plt.savefig(os.path.join(FIG, "crossover.png"), dpi=150)
print("wrote crossover.pdf/.png")

# --------------------------------------------------------------------
# Figure 3: Negative result — heap dominates DS on A*
# --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.8, 3.0))
ns_set = sorted({k[2] for k in GROUPS if k[0]=="astar"})
ks_to_plot = [5, 50, 500]
for kk in ks_to_plot:
    xs, ys_ds, ys_hp = [], [], []
    for n in ns_set:
        key = ("astar", "uniform", n, kk)
        if key not in GROUPS: continue
        if "HeapPosMap" not in GROUPS[key]: continue
        xs.append(n)
        ys_ds.append(float(GROUPS[key]["DeltaSort"]["p50"]))
        ys_hp.append(float(GROUPS[key]["HeapPosMap"]["p50"]))
    ax.plot(xs, ys_ds, "o--", label=f"DS, k={kk}")
    ax.plot(xs, ys_hp, "s-",  label=f"Heap+PosMap, k={kk}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Frontier size $n$")
ax.set_ylabel("$p50$ latency (ns)")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3, which="both")
ax.set_title("A* / Dijkstra frontier: heap beats DS by 4–600$\\times$")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "negative.pdf"))
plt.savefig(os.path.join(FIG, "negative.png"), dpi=150)
print("wrote negative.pdf/.png")

# --------------------------------------------------------------------
# Print the headline summary table for the paper
# --------------------------------------------------------------------
print("\n=== Headline numbers (for paper text) ===")
for bench, label in WIN_APPS:
    # Best speedup vs ESM across all (n, k, dist) cells
    best = (0, None, None, None)
    for k, v in GROUPS.items():
        if k[0] != bench: continue
        if "ESM" not in v or "DeltaSort" not in v: continue
        r = float(v["ESM"]["p50"]) / float(v["DeltaSort"]["p50"])
        if r > best[0]:
            best = (r, k[1], k[2], k[3])
    print(f"  {label:<25} best vs ESM: {best[0]:>6.1f}x  ({best[1]}, n={best[2]}, k={best[3]})")

# --------------------------------------------------------------------
# Figure 4: JUCE C++ production validation
# --------------------------------------------------------------------
JUCE_CSV = os.path.join(ROOT, "rust", "results", "juce_midi_cpp.csv")
juce_rows = list(csv.DictReader(open(JUCE_CSV)))
juce_grp = defaultdict(dict)
for r in juce_rows:
    key = (r["op"], int(r["n"]), int(r["k"]))
    juce_grp[key][r["strategy"]] = r

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
op_colors = {"stock": "#a83232", "unstable": "#d97a3b", "delta": "#2d8a3e"}
op_markers = {"stock": "s", "unstable": "^", "delta": "o"}
for ax, op, title in zip(axes, ["drag", "quantize"],
                          ["Drag (contiguous selection)",
                           "Quantize (scattered selection)"]):
    pts_by_n = defaultdict(list)
    for (o, n, k), s in juce_grp.items():
        if o != op: continue
        if "stock" not in s or "delta" not in s: continue
        pts_by_n[n].append((k / n,
                            float(s["stock"]["p50"]) / float(s["delta"]["p50"]),
                            float(s["unstable"]["p50"]) / float(s["delta"]["p50"])))
    n_colors = {1000: "#bbbbbb", 5000: "#888888",
                20000: "#444444", 100000: "#000000"}
    for n in sorted(pts_by_n.keys()):
        pts_by_n[n].sort()
        xs = [p[0] * 100 for p in pts_by_n[n]]
        st = [p[1] for p in pts_by_n[n]]
        un = [p[2] for p in pts_by_n[n]]
        ax.plot(xs, st, "o-",  color=n_colors[n], label=f"JUCE-stock/DS, n={n}", markersize=4)
        ax.plot(xs, un, "s--", color=n_colors[n], alpha=0.7, markersize=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$k/n$ (%)")
    ax.axhline(1.0, color="k", linewidth=0.5, linestyle=":")
    ax.grid(alpha=0.3, which="both")
    ax.set_title(title)
axes[0].set_ylabel("Speedup over DeltaSort (×)")
axes[0].legend(loc="lower left", fontsize=6.5, ncol=1, frameon=False)
axes[1].text(0.15, 1.8, "DS wins ↑", fontsize=8, color="gray")
fig.suptitle("Real JUCE \\texttt{MidiMessageSequence} on Apple M3 Pro: "
             "JUCE's \\texttt{std::stable\\_sort} (solid) and \\texttt{std::sort} (dashed) vs DeltaSort", fontsize=9)
plt.tight_layout(rect=(0, 0, 1, 0.94))
plt.savefig(os.path.join(FIG, "juce.pdf"))
plt.savefig(os.path.join(FIG, "juce.png"), dpi=150)
print("wrote juce.pdf/.png")
