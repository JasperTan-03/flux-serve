# flux-serve
FluxServe: Characterizing and Scheduling Compute-Bound Diffusion LLMs

## Set up
1. Clone Repo

2. Load TACC modules for CUDA compatibility
```bash
module load cuda/12.2
module load python3
```

3. Create the environment (Use $WORK or $SCRATCH as $HOME is small on TACC)
```bash
cd $WORK
conda env create -f environment.yml
```

4. Activate
```bash
conda activate llada_env
```

## To run Simulator

```python
# Run full comparison
python -m simulator.main

# Quick mode (5s duration, fewer RPS points)
python -m simulator.main --quick

# Custom parameters
python -m simulator.main --duration 60 --rps-max 15 --workers 8

# Default (saves to results/)
python -m simulator.plotter

# Custom output directory
python -m simulator.plotter --output-dir /path/to/custom/folder
```

## Advanced Experiments

### Burst Stress Test
Demonstrates how schedulers react to sudden traffic spikes:
- **Calm period** (0-20s): 1 RPS
- **Burst period** (20-40s): 8 RPS  
- **Recovery period** (40-60s): 1 RPS

```bash
python -m experiments.burst
```

**Output**: `results/burst_timeline.png`, `results/burst_phase_comparison.png`

### Timeout Sensitivity Sweep
Proves that static batching is hard to tune - no single timeout works for all loads:

```bash
python -m experiments.sensitivity

# With custom duration
python -m experiments.sensitivity --duration 30
```

**Output**: `results/timeout_sensitivity.png`, `results/timeout_winners.png`, `results/timeout_regret.png`
