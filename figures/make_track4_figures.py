#!/usr/bin/env python3
"""Generate Track 4 (Bevy 3D rendering phase sort) figures from Criterion data."""

import sys
sys.path.insert(0, '/Users/shubham/Library/Python/3.9/lib/python/site-packages')

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

CRIT_ROOT = '/Users/shubham/Repos/bevy-deltasort/target/criterion'
OUT = '/Users/shubham/Repos/deltasort/figures'
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Load criterion estimates
# ---------------------------------------------------------------------------
def load(group, method, k):
    path = f'{CRIT_ROOT}/{group}/{method}/{k}/new/estimates.json'
    try:
        with open(path) as f:
            d = json.load(f)
        return d['mean']['point_estimate'] / 1000.0  # ns -> µs
    except FileNotFoundError:
        return None

ks = [0, 1, 10, 50, 100, 500, 1000]

vec_full  = [load('phase_sort_vec_100k', 'full_sort',        k) for k in ks]
vec_delta = [load('phase_sort_vec_100k', 'deltasort_serial', k) for k in ks]
imap_std  = [load('phase_sort_indexmap_100k', 'sort_by_key',       k) for k in ks]
imap_delt = [load('phase_sort_indexmap_100k', 'deltasort_indexmap', k) for k in ks]

# Filter to only k values with complete data
ks_vec  = [(k, vf, vd) for k, vf, vd in zip(ks, vec_full, vec_delta) if vf and vd]
ks_imap = [(k, ms, md) for k, ms, md in zip(ks, imap_std, imap_delt)
           if ms is not None and md is not None]

# ---------------------------------------------------------------------------
# Figure 1: side-by-side sort latency  (Vec | IndexMap), log-Y
# n=100K, k in {0, 1, 10, 50, 100, 500, 1000}
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Vec subplot
kv, vf, vd = zip(*ks_vec)
ax1.plot(kv, vf, 'o--', color='#888',    label='full sort', linewidth=1.5)
ax1.plot(kv, vd, 's-',  color='#d62728', label='DeltaSort', linewidth=1.8)
ax1.set_xlabel('k  (# dirty / changed draw calls)')
ax1.set_ylabel('sort latency (µs, log)')
ax1.set_yscale('log')
ax1.set_title('Vec<PhaseItem> — transparent phase sort\n(n = 100 K, single-threaded)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

# Add speedup annotations
for k_v, vf_v, vd_v in zip(kv, vf, vd):
    spd = vf_v / vd_v
    ax1.annotate(f'{spd:.0f}×', (k_v, vd_v), textcoords='offset points',
                 xytext=(5, -3), fontsize=8, color='#d62728')

# IndexMap subplot
ki, ms, md = zip(*ks_imap)
ax2.plot(ki, ms, 'o--', color='#1f77b4', label='sort_by_key (Bevy default)', linewidth=1.5)
ax2.plot(ki, md, 's-',  color='#d62728', label='DeltaSort over IndexMap',   linewidth=1.8)
ax2.set_xlabel('k  (# dirty draw calls)')
ax2.set_yscale('log')
ax2.set_title("IndexMap<u64,PhaseItem> \u2014 Bevy's actual phase data\n(n = 100 K, single-threaded)")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

# Mark crossover for IndexMap
# k=50 is where DeltaSort loses; annotate
for k_i, ms_i, md_i in zip(ki, ms, md):
    spd = ms_i / md_i
    color = '#d62728' if spd >= 1.0 else '#555'
    label = f'{spd:.1f}×' if spd >= 1.0 else f'{spd:.2f}×\n(slow)'
    ax2.annotate(label, (k_i, md_i), textcoords='offset points',
                 xytext=(5, -3), fontsize=8, color=color)

fig.suptitle('Bevy 3D transparent-phase sort benchmark  (n = 100 K draw calls)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT}/track4_bevy_phase_sort.png', dpi=140)
plt.savefig(f'{OUT}/track4_bevy_phase_sort.svg')
plt.close()
print('Saved track4_bevy_phase_sort.{png,svg}')

# ---------------------------------------------------------------------------
# Figure 2: sort fraction of frame budget (honest framing)
# If the scene runs at 60 FPS, frame budget = 16.67 ms.
# What fraction does the sort consume for each k?
# ---------------------------------------------------------------------------
frame_budget_us = 16_667.0  # 60 FPS

fig, ax = plt.subplots(figsize=(8, 4.5))

kv_arr = np.array(kv)
vf_arr = np.array(vf)
vd_arr = np.array(vd)
ki_arr = np.array(ki)
ms_arr = np.array(ms)
md_arr = np.array(md)

ax.plot(kv_arr, 100 * vf_arr  / frame_budget_us, 'o--', color='#888',    label='Vec full sort',      linewidth=1.5)
ax.plot(kv_arr, 100 * vd_arr  / frame_budget_us, 's-',  color='#d62728', label='Vec DeltaSort',      linewidth=1.8)
ax.plot(ki_arr, 100 * ms_arr  / frame_budget_us, 'D--', color='#1f77b4', label='IndexMap sort_by_key', linewidth=1.5, alpha=0.8)
ax.plot(ki_arr, 100 * md_arr  / frame_budget_us, '^-',  color='#ff7f0e', label='IndexMap DeltaSort', linewidth=1.8, alpha=0.8)

ax.axhline(10, color='k', linestyle=':', linewidth=1, label='10% frame budget')
ax.axhline(1,  color='gray', linestyle=':', linewidth=1, label='1% frame budget')

ax.set_xlabel('k  (# dirty draw calls)')
ax.set_ylabel('sort fraction of 60 FPS frame budget (%)')
ax.set_title('Sort cost as fraction of 16.67 ms frame at 60 FPS (n = 100 K)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/track4_bevy_frame_fraction.png', dpi=140)
plt.savefig(f'{OUT}/track4_bevy_frame_fraction.svg')
plt.close()
print('Saved track4_bevy_frame_fraction.{png,svg}')

# ---------------------------------------------------------------------------
# Print headline summary
# ---------------------------------------------------------------------------
print('\n=== Track 4 headline (Vec, n=100K) ===')
print(f'{"k":>6}  {"full_sort(µs)":>14}  {"delta(µs)":>12}  {"speedup":>10}  {"frame%_full":>12}  {"frame%_delta":>12}')
for k_v, vf_v, vd_v in zip(kv, vf, vd):
    spd = vf_v / vd_v
    pf = 100 * vf_v / frame_budget_us
    pd = 100 * vd_v / frame_budget_us
    print(f'{k_v:>6}  {vf_v:>14.1f}  {vd_v:>12.1f}  {spd:>10.1f}×  {pf:>11.2f}%  {pd:>11.2f}%')

print('\n=== Track 4 headline (IndexMap, n=100K) ===')
print(f'{"k":>6}  {"sort_by_key(µs)":>16}  {"delta(µs)":>12}  {"speedup":>10}')
for k_i, ms_i, md_i in zip(ki, ms, md):
    spd = ms_i / md_i
    print(f'{k_i:>6}  {ms_i:>16.1f}  {md_i:>12.1f}  {spd:>10.2f}×')

print('\nKey finding: Vec+DeltaSort beats full_sort 4–14× for k=1..1K.')
print('IndexMap+DeltaSort beats sort_by_key only for k≤10;')
print('at k=50+ DeltaSort LOSES (IndexMap dirty-set recovery overhead dominates).')
print('Sort is <1% of 16.7ms frame budget for k≤10 with DeltaSort.')
print('Honest conclusion: GPU-bound 3D rendering is NOT a material application.')
