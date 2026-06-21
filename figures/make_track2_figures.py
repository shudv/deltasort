#!/usr/bin/env python3
"""Generate Track 2 (HPC particle) figures from results CSVs."""

import sys
sys.path.insert(0, '/Users/shubham/Library/Python/3.9/lib/python/site-packages')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

RESULTS = '/Users/shubham/Repos/deltasort/rust/results'
OUT = '/Users/shubham/Repos/deltasort/figures'
os.makedirs(OUT, exist_ok=True)

summary = pd.read_csv(f'{RESULTS}/track2_summary.csv')
trace = pd.read_csv(f'{RESULTS}/track2_trace.csv')

# ----------------------------------------------------------------------------
# Figure 1: per-step sort time over simulation for n=1M, uniform IC,
# three regimes side-by-side.
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, regime in zip(axes, ['cold', 'warm', 'hot']):
    df = trace[(trace.n == 1_000_000) & (trace.ic == 'uniform') & (trace.regime == regime)]
    if df.empty:
        df = trace[(trace.n == 100_000) & (trace.ic == 'uniform') & (trace.regime == regime)]
    ax.plot(df.step, df.full_us / 1000.0, label='full sort', color='#888', alpha=0.7)
    ax.plot(df.step, df.delta_us / 1000.0, label='DeltaSort', color='#d62728', linewidth=1.4)
    ax.plot(df.step, df.esm_us / 1000.0, label='ESM', color='#1f77b4', linewidth=1.4)
    mean_k = df.k.mean()
    n_val = df.n.iloc[0]
    ax.set_title(f'{regime} (mean k/n = {100*mean_k/n_val:.1f}%)')
    ax.set_xlabel('time step')
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel('sort time (ms)')
axes[0].legend(loc='upper right', fontsize=9)
fig.suptitle(f'Per-step sort time, n=1M particles, uniform IC', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/track2_per_step_n1M.png', dpi=140)
plt.savefig(f'{OUT}/track2_per_step_n1M.svg')
plt.close()
print('Saved track2_per_step_n1M.{png,svg}')

# ----------------------------------------------------------------------------
# Figure 2: k/n histogram per regime, n=1M, uniform IC
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
for ax, regime in zip(axes, ['cold', 'warm', 'hot']):
    df = trace[(trace.n == 1_000_000) & (trace.ic == 'uniform') & (trace.regime == regime)]
    if df.empty:
        df = trace[(trace.n == 100_000) & (trace.ic == 'uniform') & (trace.regime == regime)]
    n_val = df.n.iloc[0]
    ax.hist(100 * df.k / n_val, bins=30, color='#d62728', alpha=0.7)
    ax.set_title(f'{regime}: k/n distribution')
    ax.set_xlabel('k/n (%)')
    ax.set_ylabel('count of steps')
    ax.axvline(100 * df.k.mean() / n_val, color='k', linestyle='--', linewidth=1,
               label=f'mean = {100*df.k.mean()/n_val:.1f}%')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
fig.suptitle('Per-step k/n distributions across regimes (n=1M, uniform)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/track2_k_distribution.png', dpi=140)
plt.savefig(f'{OUT}/track2_k_distribution.svg')
plt.close()
print('Saved track2_k_distribution.{png,svg}')

# ----------------------------------------------------------------------------
# Figure 3: aggregate strategy comparison bar chart
# ----------------------------------------------------------------------------
agg = summary[summary.ic == 'uniform'].copy()
agg['n_label'] = agg.n.map({10_000: '10K', 100_000: '100K', 1_000_000: '1M'})
agg['key'] = agg.regime + '\n' + agg.n_label

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(agg))
w = 0.25
ax.bar(x - w, agg.mean_full_us / 1000, w, label='full sort', color='#888')
ax.bar(x, agg.mean_delta_us / 1000, w, label='DeltaSort', color='#d62728')
ax.bar(x + w, agg.mean_esm_us / 1000, w, label='ESM', color='#1f77b4')
ax.set_xticks(x)
ax.set_xticklabels(agg.key, fontsize=9)
ax.set_ylabel('mean per-step sort time (ms, log)')
ax.set_yscale('log')
ax.set_title('Per-step sort time per regime (uniform IC)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{OUT}/track2_strategy_bars.png', dpi=140)
plt.savefig(f'{OUT}/track2_strategy_bars.svg')
plt.close()
print('Saved track2_strategy_bars.{png,svg}')

# ----------------------------------------------------------------------------
# Figure 4: sort time vs k scatter from trace (shows crossover)
# ----------------------------------------------------------------------------
df = trace[(trace.n == 1_000_000) & (trace.ic == 'uniform')].copy()
if df.empty:
    df = trace[(trace.n == 100_000) & (trace.ic == 'uniform')].copy()
n_val = df.n.iloc[0]
df['kpct'] = 100.0 * df.k / n_val

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(df.kpct, df.full_us / 1000, label='full sort', s=10, color='#888', alpha=0.5)
ax.scatter(df.kpct, df.delta_us / 1000, label='DeltaSort', s=10, color='#d62728', alpha=0.7)
ax.scatter(df.kpct, df.esm_us / 1000, label='ESM', s=10, color='#1f77b4', alpha=0.7)
ax.set_xlabel('k/n (%)')
ax.set_ylabel('per-step sort time (ms)')
ax.set_title(f'Per-step sort time vs k/n  (n={n_val:,}, uniform IC, all regimes pooled)')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/track2_time_vs_k.png', dpi=140)
plt.savefig(f'{OUT}/track2_time_vs_k.svg')
plt.close()
print('Saved track2_time_vs_k.{png,svg}')

# ----------------------------------------------------------------------------
# Print headline summary
# ----------------------------------------------------------------------------
print('\n=== Track 2 headline ===')
for _, row in summary.iterrows():
    spd_vs_full = row.mean_full_us / row.mean_delta_us
    spd_vs_esm = row.mean_esm_us / row.mean_delta_us
    print(f"n={int(row.n):>8} {row.ic:>8} {row.regime:>5}  k/n={100*row.mean_k/row.n:>5.1f}%  "
          f"delta_vs_full={spd_vs_full:>5.1f}x  delta_vs_esm={spd_vs_esm:>5.2f}x")
