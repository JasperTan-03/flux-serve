# FluxServe: Characterizing and Scheduling Compute-Bound Diffusion LLMs

A trace-driven simulator for evaluating dynamic batching strategies for diffusion-based large language models (dLLMs) like LLaDA-8B. This project demonstrates that compute-bound dLLMs benefit from dynamic batching, achieving up to **52% latency reduction** at low request rates compared to static timeout-based schedulers.

## Overview

Diffusion LLMs (dLLMs) are compute-bound with flat memory usage, creating opportunities for aggressive batching. Unlike autoregressive models, dLLMs process fixed-length sequences through multiple diffusion steps, enabling predictable batch execution times. This simulator:

- **Profiles** real dLLM performance (LLaDA-8B) across batch sizes
- **Compares** static timeout-based vs. dynamic greedy batching schedulers
- **Evaluates** performance under various workloads (steady-state, bursts, sensitivity)

### Key Findings

- **Dynamic batching** reduces latency by 27-52% at low-to-medium RPS (1-4 req/s)
- **Static schedulers** are hard to tune—no single timeout works across all load levels
- **Sub-linear scaling**: Doubling batch size increases latency by only ~1.4x (not 2x)
- **Memory is static**: No KV cache growth, enabling larger batches without memory concerns

## Setup

### Prerequisites
- TACC Lonestar6 (or similar HPC system with CUDA)
- Conda/Mamba

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd flux-serve
   ```

2. **Load TACC modules** (for CUDA compatibility)
   ```bash
   module load cuda/12.2
   module load python3
   ```

3. **Create conda environment** (use `$WORK` or `$SCRATCH`—`$HOME` has limited quota)
   ```bash
   cd $WORK  # or $SCRATCH
   conda env create -f environment.yml
   conda activate llada_env
   ```

4. **Set HuggingFace cache** (to avoid home directory quota issues)
   ```bash
   export HF_HOME=/work/$USER/.cache/huggingface
   # Or add to your ~/.bashrc for persistence
   ```

## Quick Start

### Run the Main Simulation

Compare static vs. dynamic schedulers across multiple request rates:

```bash
# Full comparison (default: 30s duration, RPS 1-10)
python -m simulator.main

# Quick mode (5s duration, fewer RPS points)
python -m simulator.main --quick

# Custom parameters
python -m simulator.main --duration 60 --rps-max 15 --workers 8
```

### Generate Visualizations

After running the simulator, generate plots:

```bash
# Default (saves to results/)
python -m simulator.plotter

# Custom output directory
python -m simulator.plotter --output-dir /path/to/custom/folder
```

**Output files:**
- `scheduler_comparison.png` - Latency comparison across RPS
- `tail_latency_comparison.png` - P50/P99 latency analysis
- `batch_size_analysis.png` - Batch size distribution
- `profiling_data.png` - Measured latency vs. batch size
- `improvement_summary.png` - Performance improvements

## Experiments

### 1. Burst Stress Test

Evaluates scheduler behavior under sudden traffic spikes:

```bash
python -m experiments.burst
```

**Workload pattern:**
- **Calm** (0-20s): 1 RPS
- **Burst** (20-40s): 8 RPS  
- **Recovery** (40-60s): 1 RPS

**Output:** `results/burst_timeline.png`, `results/burst_phase_comparison.png`

**Result:** Dynamic batching adapts better to traffic spikes, reducing average latency by ~8% compared to static schedulers.

### 2. Timeout Sensitivity Analysis

Demonstrates that static batching requires careful timeout tuning:

```bash
python -m experiments.sensitivity

# Custom duration
python -m experiments.sensitivity --duration 30
```

Tests multiple timeout values (τ=0.1s, 1.0s, 5.0s) across RPS 1-10.

**Output:** `results/timeout_sensitivity.png`, `results/timeout_winners.png`, `results/timeout_regret.png`

**Key insight:** Dynamic batching has the lowest total regret (0.68s) across all load levels, while static schedulers require different timeouts for different RPS ranges.

## Profiling Scripts

To re-profile the model and update latency measurements:

```bash
# Profile per-step latency across batch sizes
python profile_throughput.py

# Profile memory usage over diffusion steps
python profile_llada.py

# Validate simulator accuracy against real GPU
python validate_real_system.py
```

**Note:** These scripts require GPU access and will download the LLaDA-8B model (~16GB) on first run.

## Project Structure

```
flux-serve/
├── simulator/          # Core simulation engine
│   ├── config.py      # Profiling data and constants
│   ├── scheduler.py   # Static and dynamic schedulers
│   ├── simulator.py   # Event-driven simulation
│   └── plotter.py     # Visualization generation
├── experiments/       # Advanced experiments
│   ├── burst.py       # Burst workload test
│   └── sensitivity.py # Timeout sensitivity sweep
├── results/           # Generated plots and data
├── profile_*.py       # Model profiling scripts
└── validate_real_system.py  # Real GPU validation
```
