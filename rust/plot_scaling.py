#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Data from benchmark (n=1M)
k_1m = [1000, 5000, 10000, 20000, 50000, 100000]
serial_1m = [699.8, 1944.5, 3055.5, 5206.0, 9655.1, 18493.7]
parallel_1m = [713.7, 1496.8, 2302.7, 4198.5, 6266.5, 7671.7]

# Data from benchmark (n=500K)
k_500k = [500, 2000, 5000, 10000, 25000, 50000]
serial_500k = [297.5, 790.9, 1160.6, 1818.5, 3633.4, 7809.8]
parallel_500k = [293.2, 818.6, 1131.1, 1274.1, 2230.4, 2966.3]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot n=1M
ax1 = axes[0]
ax1.plot(k_1m, serial_1m, 'o-', label='Serial', linewidth=2, markersize=8, color='#d62728')
ax1.plot(k_1m, parallel_1m, 's-', label='Parallel', linewidth=2, markersize=8, color='#2ca02c')
# Reference lines
k_ref = np.array(k_1m)
ax1.plot(k_ref, k_ref * serial_1m[0] / k_1m[0], '--', alpha=0.4, color='gray', label='O(k) reference')
ax1.plot(k_ref, np.sqrt(k_ref) * serial_1m[0] / np.sqrt(k_1m[0]), ':', alpha=0.4, color='gray', label='O(sqrt(k)) reference')
ax1.set_xlabel('k (number of updates)', fontsize=12)
ax1.set_ylabel('Time (microseconds)', fontsize=12)
ax1.set_title('n = 1M', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')
ax1.set_yscale('log')

# Plot n=500K
ax2 = axes[1]
ax2.plot(k_500k, serial_500k, 'o-', label='Serial', linewidth=2, markersize=8, color='#d62728')
ax2.plot(k_500k, parallel_500k, 's-', label='Parallel', linewidth=2, markersize=8, color='#2ca02c')
# Reference lines
k_ref = np.array(k_500k)
ax2.plot(k_ref, k_ref * serial_500k[0] / k_500k[0], '--', alpha=0.4, color='gray', label='O(k) reference')
ax2.plot(k_ref, np.sqrt(k_ref) * serial_500k[0] / np.sqrt(k_500k[0]), ':', alpha=0.4, color='gray', label='O(sqrt(k)) reference')
ax2.set_xlabel('k (number of updates)', fontsize=12)
ax2.set_ylabel('Time (microseconds)', fontsize=12)
ax2.set_title('n = 500K', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')
ax2.set_yscale('log')

plt.suptitle('DeltaSort Runtime vs k: Serial vs Parallel (12 threads)', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('parallel_scaling.png', dpi=150, bbox_inches='tight')
print('Saved to parallel_scaling.png')

# Slope analysis
print('\nSlope analysis (log-log, higher = steeper growth):')
print('  Reference: 1.0 = O(k) linear, 0.5 = O(sqrt(k))')
print()
for name, k, t in [('Serial n=1M', k_1m, serial_1m), 
                   ('Parallel n=1M', k_1m, parallel_1m),
                   ('Serial n=500K', k_500k, serial_500k), 
                   ('Parallel n=500K', k_500k, parallel_500k)]:
    log_k = np.log10(k)
    log_t = np.log10(t)
    slope = np.polyfit(log_k, log_t, 1)[0]
    print(f'  {name:20s}: slope = {slope:.2f}')
