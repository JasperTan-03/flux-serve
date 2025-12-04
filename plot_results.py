import matplotlib.pyplot as plt
import numpy as np

# Your Data
batch_sizes = [1, 2, 4, 8, 16, 32]
latency = [0.0250, 0.0331, 0.0577, 0.1130, 0.2132, 0.4013]
throughput = [39.96, 60.46, 69.29, 70.79, 75.06, 79.75]
memory = [16.02, 16.08, 16.20, 16.44, 16.92, 17.89]

# --- Plot 1: Throughput vs Latency (The "Free Lunch" Curve) ---
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Batch Size (Log Scale)', fontsize=12)
ax1.set_ylabel('Throughput (req/s)', color=color, fontsize=12)
ax1.plot(batch_sizes, throughput, marker='o', color=color, linewidth=3, label='Throughput')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xscale('log')
ax1.set_xticks(batch_sizes)
ax1.set_xticklabels(batch_sizes)
ax1.grid(True, linestyle='--', alpha=0.5)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Latency per Step (s)', color=color, fontsize=12)  
ax2.plot(batch_sizes, latency, marker='x', linestyle='--', color=color, linewidth=2, label='Latency')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Characterization: LLaDA is Compute Bound (Sub-linear Scaling)', fontsize=14)
plt.tight_layout()
plt.savefig('results/throughput_latency.png', dpi=300)
print("Saved throughput_latency.png")

# --- Plot 2: Memory Stability (The "Anti-vLLM" Curve) ---
plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, memory, marker='s', color='green', linewidth=3)

# Add a fake "Standard LLM" line for comparison (Conceptual)
# Standard LLMs grow linearly with batch size due to KV Cache
fake_ar_memory = [16 + (b * 0.5) for b in batch_sizes] # Fake slope
plt.plot(batch_sizes, fake_ar_memory, linestyle=':', color='gray', label='Standard LLM (Conceptual)')

plt.title('Characterization: LLaDA Memory is Static', fontsize=14)
plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('GPU Memory (GB)', fontsize=12)
plt.ylim(0, 40) # Show full A100 capacity to emphasize how small the growth is
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('results/memory_stability.png', dpi=300)
print("Saved memory_stability.png")