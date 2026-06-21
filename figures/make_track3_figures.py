#!/usr/bin/env python3
"""Generate Track 3 (MIDI editing) figures from results CSV."""

import sys
sys.path.insert(0, '/Users/shubham/Library/Python/3.9/lib/python/site-packages')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS = '/Users/shubham/Repos/deltasort/rust/results'
OUT = '/Users/shubham/Repos/deltasort/figures'
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(f'{RESULTS}/track3_summary.csv')
df['speedup_vs_btree'] = df.btree_us / df.delta_us
df['speedup_vs_full'] = df.full_us / df.delta_us

ops = ['single_drag', 'selection_drag', 'velocity_quantize', 'bulk_paste']
op_labels = {
    'single_drag': 'single drag\n(k=1)',
    'selection_drag': 'selection drag\n(k=2..200)',
    'velocity_quantize': 'velocity quantize\n(k=10..500, key unchanged)',
    'bulk_paste': 'bulk paste\n(k=50..1000, random)',
}
ns = sorted(df.n.unique())
n_label = lambda v: {1_000: '1K', 10_000: '10K', 100_000: '100K', 1_000_000: '1M'}.get(v, str(v))

# ----------------------------------------------------------------------------
# Figure 1: per-op time grouped by op, log-scale, color by strategy.
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
for ax, op in zip(axes, ops):
    sub = df[df.op == op].sort_values('n')
    x = np.arange(len(sub))
    w = 0.25
    ax.bar(x - w, sub.full_us, w, label='full sort', color='#888')
    ax.bar(x, sub.delta_us, w, label='DeltaSort', color='#d62728')
    ax.bar(x + w, sub.btree_us, w, label='BTreeSet', color='#1f77b4')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([n_label(n) for n in sub.n])
    ax.set_title(op_labels[op], fontsize=10)
    ax.set_xlabel('n (events)')
    ax.grid(True, alpha=0.3, axis='y')
axes[0].set_ylabel('mean time per edit (µs, log)')
axes[0].legend(loc='upper left', fontsize=9)
fig.suptitle('Per-edit time across MIDI edit operations', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/track3_per_op_time.png', dpi=140)
plt.savefig(f'{OUT}/track3_per_op_time.svg')
plt.close()
print('Saved track3_per_op_time.{png,svg}')

# ----------------------------------------------------------------------------
# Figure 2: speedup heatmap (op x n) for DeltaSort vs BTreeSet
# ----------------------------------------------------------------------------
piv = df.pivot(index='op', columns='n', values='speedup_vs_btree').reindex(ops)
fig, ax = plt.subplots(figsize=(7, 4))
import matplotlib.colors as mcolors
cmap = plt.cm.get_cmap('RdBu_r')
norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1.5)
matrix = np.log10(piv.values)
im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        val = piv.values[i, j]
        color = 'white' if abs(matrix[i, j]) > 0.5 else 'black'
        ax.text(j, i, f'{val:.1f}x', ha='center', va='center', color=color, fontsize=10)
ax.set_xticks(range(piv.shape[1]))
ax.set_xticklabels([n_label(n) for n in piv.columns])
ax.set_yticks(range(piv.shape[0]))
ax.set_yticklabels([o.replace('_', ' ') for o in piv.index])
ax.set_xlabel('n (events)')
ax.set_title('DeltaSort speedup over BTreeSet\n(>1 = DeltaSort faster)')
plt.colorbar(im, ax=ax, label='log10(speedup)')
plt.tight_layout()
plt.savefig(f'{OUT}/track3_speedup_vs_btree_heatmap.png', dpi=140)
plt.savefig(f'{OUT}/track3_speedup_vs_btree_heatmap.svg')
plt.close()
print('Saved track3_speedup_vs_btree_heatmap.{png,svg}')

# ----------------------------------------------------------------------------
# Figure 3: simulated edit-session aggregate.
# We model a realistic 1000-edit session with mix:
#  - 50% single_drag, 30% selection_drag, 15% velocity_quantize, 5% bulk_paste
# Compute total session time for each (n, strategy).
# ----------------------------------------------------------------------------
mix = {'single_drag': 0.50, 'selection_drag': 0.30, 'velocity_quantize': 0.15, 'bulk_paste': 0.05}
session_size = 1000

sess_rows = []
for n in ns:
    sub = df[df.n == n].set_index('op')
    full = sum(sub.loc[op, 'full_us'] * mix[op] for op in mix) * session_size
    delta = sum(sub.loc[op, 'delta_us'] * mix[op] for op in mix) * session_size
    btree = sum(sub.loc[op, 'btree_us'] * mix[op] for op in mix) * session_size
    sess_rows.append({'n': n, 'full': full / 1e6, 'delta': delta / 1e6, 'btree': btree / 1e6})
sess = pd.DataFrame(sess_rows)

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(sess))
w = 0.25
ax.bar(x - w, sess.full, w, label='full sort', color='#888')
ax.bar(x, sess.delta, w, label='DeltaSort', color='#d62728')
ax.bar(x + w, sess.btree, w, label='BTreeSet', color='#1f77b4')
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels([n_label(n) for n in sess.n])
ax.set_xlabel('n (events in track)')
ax.set_ylabel(f'total session time (s, log) — {session_size}-edit session')
ax.set_title('Aggregate 1000-edit session time\n(mix: 50% drag, 30% selection, 15% quantize, 5% paste)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{OUT}/track3_session_time.png', dpi=140)
plt.savefig(f'{OUT}/track3_session_time.svg')
plt.close()
print('Saved track3_session_time.{png,svg}')

# ----------------------------------------------------------------------------
# Headline summary
# ----------------------------------------------------------------------------
print('\n=== Track 3 headline ===')
for _, row in df.iterrows():
    print(f"{row.op:>18}  n={int(row.n):>8}  k={row.mean_k:>6.1f}  "
          f"delta_vs_full={row.full_us/row.delta_us:>7.1f}x  "
          f"delta_vs_btree={row.btree_us/row.delta_us:>6.2f}x")

print('\n=== 1000-edit session summary ===')
for _, row in sess.iterrows():
    print(f"n={int(row.n):>8}:  full={row.full:>8.2f}s   delta={row.delta:>6.3f}s   "
          f"btree={row.btree:>6.3f}s   speedup_vs_btree={row.btree/row.delta:.2f}x")
