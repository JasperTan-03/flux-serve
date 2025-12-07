"""
Burst Stress Test Experiment for dLLM Scheduling.

This experiment demonstrates how different schedulers react to sudden
traffic spikes - a critical real-world scenario for ML serving systems.

Systems Insight:
    Real production traffic is bursty, not uniform. A scheduler must:
    1. Maintain low latency during calm periods (no unnecessary waiting)
    2. Handle bursts gracefully (not collapse under load)
    3. Recover quickly after load subsides
    
    Static batching fails (1) because it waits for target batch/timeout
    even when traffic is low. Dynamic batching succeeds because it
    immediately processes whatever is available.

Workload Pattern:
    - 0s-20s:  Calm period (1 RPS) - Tests low-load efficiency
    - 20s-40s: Burst period (8 RPS) - Tests saturation behavior  
    - 40s-60s: Recovery period (1 RPS) - Tests queue draining

Expected Results:
    - Static: High latency in calm periods (waiting for timeout),
              similar to Dynamic during burst, slow recovery
    - Dynamic: Low latency in calm periods (immediate processing),
               fast recovery after burst
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import random
import math

import matplotlib.pyplot as plt
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.workload import Request
from simulator.scheduler import StaticBatchScheduler, DynamicBatchScheduler
from simulator.simulator import Simulator
from copy import deepcopy


# =============================================================================
# Burst Workload Generation
# =============================================================================

@dataclass
class BurstPhase:
    """Defines a phase in the burst workload."""
    start_time: float
    end_time: float
    rps: float
    name: str


def generate_phased_workload(
    phases: List[BurstPhase],
    seed: Optional[int] = None
) -> List[Request]:
    """
    Generate a workload with distinct traffic phases.
    
    This creates a realistic traffic pattern where load varies over time,
    allowing us to observe scheduler behavior under changing conditions.
    
    Args:
        phases: List of traffic phases with their RPS rates
        seed: Random seed for reproducibility
        
    Returns:
        List of Request objects sorted by arrival time
    """
    rng = random.Random(seed)
    requests: List[Request] = []
    request_id = 0
    
    for phase in phases:
        if phase.rps <= 0:
            continue
            
        current_time = phase.start_time
        mean_interarrival = 1.0 / phase.rps
        
        while current_time < phase.end_time:
            # Exponential inter-arrival (Poisson process)
            interarrival = -math.log(1 - rng.random()) * mean_interarrival
            current_time += interarrival
            
            if current_time < phase.end_time:
                requests.append(Request(
                    request_id=request_id,
                    arrival_time=current_time
                ))
                request_id += 1
    
    # Sort by arrival time (phases may overlap in theory)
    requests.sort(key=lambda r: r.arrival_time)
    return requests


def create_burst_workload(seed: int = 42) -> Tuple[List[Request], List[BurstPhase]]:
    """
    Create the standard burst test workload.
    
    Pattern: Calm → Burst → Recovery
    - 0s-20s:  1 RPS (Calm)
    - 20s-40s: 8 RPS (Burst) 
    - 40s-60s: 1 RPS (Recovery)
    
    Returns:
        Tuple of (requests, phases) for simulation and visualization
    """
    phases = [
        BurstPhase(start_time=0.0, end_time=20.0, rps=1.0, name="Calm"),
        BurstPhase(start_time=20.0, end_time=40.0, rps=8.0, name="Burst"),
        BurstPhase(start_time=40.0, end_time=60.0, rps=1.0, name="Recovery"),
    ]
    
    requests = generate_phased_workload(phases, seed=seed)
    return requests, phases


# =============================================================================
# Simulation Runner
# =============================================================================

def run_burst_experiment(
    requests: List[Request],
    seed: int = 42
) -> dict:
    """
    Run burst experiment with both schedulers.
    
    Args:
        requests: Pre-generated workload trace
        seed: Random seed (for any stochastic scheduler behavior)
        
    Returns:
        Dict with scheduler name -> list of completed requests
    """
    results = {}
    
    # Static Scheduler (with typical timeout)
    static_scheduler = StaticBatchScheduler(
        target_batch_size=8,
        timeout=5.0  # 5 second timeout - will cause waiting in calm periods
    )
    
    # Dynamic Scheduler (our proposal)
    dynamic_scheduler = DynamicBatchScheduler()
    
    schedulers = [
        ("Static (timeout=5s)", static_scheduler),
        ("Dynamic", dynamic_scheduler),
    ]
    
    for name, scheduler in schedulers:
        # Deep copy requests so each simulation has fresh state
        sim_requests = [
            Request(
                request_id=r.request_id,
                arrival_time=r.arrival_time
            )
            for r in requests
        ]
        
        sim = Simulator(scheduler=scheduler, requests=sim_requests)
        sim.run()
        
        results[name] = sim.completed_requests
        
        # Print summary
        latencies = [r.latency for r in sim.completed_requests if r.latency]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        print(f"{name}: {len(sim.completed_requests)} requests, "
              f"avg latency = {avg_lat:.3f}s")
    
    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_burst_timeline(
    results: dict,
    phases: List[BurstPhase],
    output_path: str = "results/burst_timeline.png"
) -> None:
    """
    Create a timeline visualization of request latencies.
    
    This plot shows how each scheduler performs throughout the
    burst workload, clearly illustrating the difference in behavior
    during calm, burst, and recovery periods.
    
    Args:
        results: Dict mapping scheduler name to completed requests
        phases: List of traffic phases for background shading
        output_path: Where to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Color scheme
    colors = {
        "Static (timeout=5s)": "#E69F00",  # Orange
        "Dynamic": "#0072B2",              # Blue
    }
    
    phase_colors = {
        "Calm": "#E8F5E9",      # Light green
        "Burst": "#FFEBEE",     # Light red
        "Recovery": "#E3F2FD",  # Light blue
    }
    
    # Draw phase backgrounds
    for phase in phases:
        ax.axvspan(
            phase.start_time, phase.end_time,
            alpha=0.3,
            color=phase_colors.get(phase.name, "#F5F5F5"),
            label=f"{phase.name} ({phase.rps} RPS)"
        )
        
        # Add phase label
        mid_time = (phase.start_time + phase.end_time) / 2
        ax.text(
            mid_time, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 10,
            phase.name,
            ha='center', va='bottom',
            fontsize=11, fontweight='bold',
            alpha=0.7
        )
    
    # Plot latency for each scheduler
    for name, completed_requests in results.items():
        arrivals = [r.arrival_time for r in completed_requests]
        latencies = [r.latency for r in completed_requests if r.latency]
        
        ax.scatter(
            arrivals[:len(latencies)],
            latencies,
            c=colors.get(name, "gray"),
            alpha=0.6,
            s=30,
            label=name,
            edgecolors='none'
        )
        
        # Add trend line (rolling average)
        if len(latencies) > 10:
            window = 5
            rolling_avg = np.convolve(latencies, np.ones(window)/window, mode='valid')
            rolling_times = arrivals[window-1:len(rolling_avg)+window-1]
            ax.plot(
                rolling_times[:len(rolling_avg)],
                rolling_avg,
                color=colors.get(name, "gray"),
                linewidth=2,
                alpha=0.8
            )
    
    # Styling
    ax.set_xlabel('Request Arrival Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Request Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Burst Stress Test: How Schedulers React to Traffic Spikes',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    # Add phase annotations at top
    for phase in phases:
        mid_time = (phase.start_time + phase.end_time) / 2
        y_pos = ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 8
        ax.annotate(
            f'{phase.name}\n({phase.rps} RPS)',
            xy=(mid_time, y_pos),
            ha='center', va='top',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='gray', alpha=0.8)
        )
    
    ax.set_xlim(0, 60)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Add insight annotation
    ax.annotate(
        'Static waits for timeout\nduring calm periods',
        xy=(10, 5.0),
        fontsize=9,
        ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', 
                 edgecolor='#E69F00', alpha=0.9)
    )
    
    ax.annotate(
        'Dynamic processes\nimmediately',
        xy=(10, 1.0),
        fontsize=9,
        ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', 
                 edgecolor='#0072B2', alpha=0.9)
    )
    
    plt.tight_layout()
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_phase_comparison(
    results: dict,
    phases: List[BurstPhase],
    output_path: str = "results/burst_phase_comparison.png"
) -> None:
    """
    Create a bar chart comparing average latency per phase.
    
    This provides a clear quantitative comparison of scheduler
    performance across different traffic conditions.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scheduler_names = list(results.keys())
    phase_names = [p.name for p in phases]
    
    # Calculate average latency per phase for each scheduler
    data = {name: [] for name in scheduler_names}
    
    for name, completed_requests in results.items():
        for phase in phases:
            phase_requests = [
                r for r in completed_requests
                if r.arrival_time >= phase.start_time 
                and r.arrival_time < phase.end_time
                and r.latency is not None
            ]
            avg_lat = (sum(r.latency for r in phase_requests) / len(phase_requests)
                      if phase_requests else 0)
            data[name].append(avg_lat)
    
    # Bar positions
    x = np.arange(len(phase_names))
    width = 0.35
    
    colors = ["#E69F00", "#0072B2"]
    
    for idx, name in enumerate(scheduler_names):
        offset = width * (idx - 0.5)
        bars = ax.bar(x + offset, data[name], width, 
                     label=name, color=colors[idx], alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, data[name]):
            ax.annotate(
                f'{val:.2f}s',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9, fontweight='bold'
            )
    
    ax.set_xlabel('Traffic Phase', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Average Latency by Traffic Phase',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p.name}\n({p.rps} RPS)" for p in phases])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Run the burst stress test experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Burst Stress Test: How schedulers handle traffic spikes"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="results",
        help="Directory to save plots (default: results)"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print(" Burst Stress Test Experiment")
    print("=" * 60)
    print("\nWorkload Pattern:")
    print("  0s-20s:  Calm (1 RPS)")
    print("  20s-40s: Burst (8 RPS)")
    print("  40s-60s: Recovery (1 RPS)")
    print()
    
    # Generate workload
    print("Generating burst workload...")
    requests, phases = create_burst_workload(seed=args.seed)
    print(f"Generated {len(requests)} total requests")
    print()
    
    # Run experiments
    print("Running simulations...")
    results = run_burst_experiment(requests, seed=args.seed)
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_burst_timeline(
        results, phases,
        str(output_dir / "burst_timeline.png")
    )
    plot_phase_comparison(
        results, phases,
        str(output_dir / "burst_phase_comparison.png")
    )
    
    print("\n" + "=" * 60)
    print(" Experiment Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

