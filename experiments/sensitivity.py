"""
Timeout Sensitivity Sweep Experiment for dLLM Scheduling.

This experiment proves that static batching is fundamentally hard to tune,
while dynamic batching works well across all load conditions.

Systems Insight:
    Static batching requires choosing a timeout parameter that balances:
    - Too short: Sends small batches, wastes compute efficiency
    - Too long: Requests wait unnecessarily, increases latency
    
    The optimal timeout depends on the arrival rate:
    - Low load: Short timeout is better (don't wait)
    - High load: Timeout doesn't matter (queue always has requests)
    
    This creates a "no free lunch" situation where any fixed timeout
    will be suboptimal for some workloads. Dynamic batching sidesteps
    this by adapting automatically - it just processes whatever is
    available immediately.

Experiment Design:
    Sweep RPS from 1 to 10 and compare:
    - Static (timeout=0.1s): Fast dispatch, small batches at low load
    - Static (timeout=1.0s): Medium tradeoff
    - Static (timeout=5.0s): Waits for batches, good at high load
    - Dynamic: No timeout needed, adapts to load

Expected Results:
    - No single static configuration wins at all RPS levels
    - Dynamic consistently performs well across the board
    - Static with short timeout wins at low RPS
    - Static with long timeout might win at high RPS (if it helps batching)
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.scheduler import StaticBatchScheduler, DynamicBatchScheduler, Scheduler
from simulator.simulator import run_simulation, SimulationMetrics


# =============================================================================
# Configuration
# =============================================================================

# Static scheduler configurations to test
STATIC_CONFIGS = [
    {"timeout": 0.1, "label": "Static (τ=0.1s)", "color": "#FFA726"},
    {"timeout": 1.0, "label": "Static (τ=1.0s)", "color": "#EF5350"},
    {"timeout": 5.0, "label": "Static (τ=5.0s)", "color": "#AB47BC"},
]

# Dynamic scheduler (our proposal)
DYNAMIC_CONFIG = {
    "label": "Dynamic (Proposed)",
    "color": "#0072B2",
}

# RPS range to test
DEFAULT_RPS_RANGE = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


# =============================================================================
# Experiment Runner
# =============================================================================

def run_sensitivity_sweep(
    rps_range: List[float],
    duration: float = 30.0,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Dict[float, SimulationMetrics]]:
    """
    Run the timeout sensitivity sweep experiment.
    
    Tests multiple static scheduler configurations against the dynamic
    scheduler across a range of load levels.
    
    Args:
        rps_range: List of RPS values to test
        duration: Simulation duration per experiment
        seed: Base random seed
        verbose: Print progress updates
        
    Returns:
        Nested dict: {scheduler_label: {rps: metrics}}
    """
    results: Dict[str, Dict[float, SimulationMetrics]] = {}
    
    # Build list of all schedulers to test
    schedulers: List[tuple] = []
    
    # Add static configurations
    for config in STATIC_CONFIGS:
        scheduler = StaticBatchScheduler(
            target_batch_size=8,
            timeout=config["timeout"]
        )
        schedulers.append((config["label"], scheduler))
    
    # Add dynamic scheduler
    dynamic = DynamicBatchScheduler()
    schedulers.append((DYNAMIC_CONFIG["label"], dynamic))
    
    total_experiments = len(schedulers) * len(rps_range)
    completed = 0
    
    for label, scheduler in schedulers:
        results[label] = {}
        
        for rps in rps_range:
            # Run simulation
            metrics = run_simulation(
                scheduler=scheduler,
                rps=rps,
                duration=duration,
                seed=seed + int(rps * 100)  # Consistent seed per RPS
            )
            results[label][rps] = metrics
            
            completed += 1
            if verbose:
                print(f"  [{completed}/{total_experiments}] {label} @ {rps} RPS: "
                      f"avg_lat={metrics.avg_latency:.3f}s")
    
    return results


# =============================================================================
# Analysis
# =============================================================================

def find_best_config_per_rps(
    results: Dict[str, Dict[float, SimulationMetrics]],
    rps_range: List[float]
) -> Dict[float, str]:
    """
    Determine which scheduler configuration wins at each RPS level.
    
    Returns:
        Dict mapping RPS to the winning scheduler label
    """
    winners = {}
    
    for rps in rps_range:
        best_label = None
        best_latency = float('inf')
        
        for label, rps_results in results.items():
            lat = rps_results[rps].avg_latency
            if lat < best_latency:
                best_latency = lat
                best_label = label
        
        winners[rps] = best_label
    
    return winners


def compute_regret(
    results: Dict[str, Dict[float, SimulationMetrics]],
    rps_range: List[float]
) -> Dict[str, float]:
    """
    Compute total "regret" for each scheduler.
    
    Regret = sum of (scheduler_latency - best_latency) across all RPS.
    Lower regret = more robust across all loads.
    
    Returns:
        Dict mapping scheduler label to total regret
    """
    regret = {label: 0.0 for label in results.keys()}
    
    for rps in rps_range:
        # Find best latency at this RPS
        best_lat = min(results[label][rps].avg_latency for label in results)
        
        # Compute regret for each scheduler
        for label in results:
            regret[label] += results[label][rps].avg_latency - best_lat
    
    return regret


# =============================================================================
# Visualization
# =============================================================================

def plot_sensitivity_curves(
    results: Dict[str, Dict[float, SimulationMetrics]],
    rps_range: List[float],
    output_path: str = "results/timeout_sensitivity.png"
) -> None:
    """
    Plot latency vs RPS for all scheduler configurations.
    
    This is the main visualization showing that no single timeout
    works well for all loads, while dynamic adapts automatically.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Color mapping
    colors = {}
    for config in STATIC_CONFIGS:
        colors[config["label"]] = config["color"]
    colors[DYNAMIC_CONFIG["label"]] = DYNAMIC_CONFIG["color"]
    
    # Line styles for static (dashed) vs dynamic (solid)
    linestyles = {}
    for config in STATIC_CONFIGS:
        linestyles[config["label"]] = "--"
    linestyles[DYNAMIC_CONFIG["label"]] = "-"
    
    # Plot each configuration
    for label in results:
        latencies = [results[label][rps].avg_latency for rps in rps_range]
        
        ax.plot(
            rps_range,
            latencies,
            color=colors.get(label, "gray"),
            linestyle=linestyles.get(label, "-"),
            linewidth=2.5,
            marker='o',
            markersize=8,
            label=label
        )
    
    # Styling
    ax.set_xlabel('Input Load (Requests Per Second)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Request Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Timeout Sensitivity: Static Batching is Hard to Tune',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min(rps_range) - 0.5, max(rps_range) + 0.5)
    ax.set_ylim(bottom=0)
    
    # Add insight annotation
    ax.annotate(
        'Dynamic adapts to all loads\nwithout tuning',
        xy=(3, results[DYNAMIC_CONFIG["label"]][3.0].avg_latency),
        xytext=(5, results[DYNAMIC_CONFIG["label"]][1.0].avg_latency * 0.5),
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', 
                 edgecolor='#0072B2', alpha=0.9)
    )
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_winner_heatmap(
    results: Dict[str, Dict[float, SimulationMetrics]],
    rps_range: List[float],
    output_path: str = "results/timeout_winners.png"
) -> None:
    """
    Visualize which scheduler wins at each RPS level.
    
    This clearly shows that different static configurations win
    at different loads, while dynamic is consistently competitive.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    
    winners = find_best_config_per_rps(results, rps_range)
    
    # Color mapping
    colors = {}
    for config in STATIC_CONFIGS:
        colors[config["label"]] = config["color"]
    colors[DYNAMIC_CONFIG["label"]] = DYNAMIC_CONFIG["color"]
    
    # Create bars showing winner at each RPS
    bar_colors = [colors.get(winners[rps], "gray") for rps in rps_range]
    bars = ax.bar(range(len(rps_range)), [1] * len(rps_range), 
                  color=bar_colors, alpha=0.8, edgecolor='white', linewidth=2)
    
    # Add winner labels
    for idx, rps in enumerate(rps_range):
        winner = winners[rps]
        short_name = winner.split("(")[0].strip()
        ax.text(
            idx, 0.5, short_name,
            ha='center', va='center',
            fontsize=9, fontweight='bold',
            rotation=45
        )
    
    ax.set_xticks(range(len(rps_range)))
    ax.set_xticklabels([f'{rps:.0f}' for rps in rps_range])
    ax.set_xlabel('RPS', fontsize=12, fontweight='bold')
    ax.set_ylabel('')
    ax.set_title(
        'Which Scheduler Wins at Each Load Level?',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_yticks([])
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[label], label=label, alpha=0.8)
        for label in results.keys()
    ]
    ax.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_regret_comparison(
    results: Dict[str, Dict[float, SimulationMetrics]],
    rps_range: List[float],
    output_path: str = "results/timeout_regret.png"
) -> None:
    """
    Plot total regret for each configuration.
    
    Regret measures how much worse each scheduler is compared to
    the best at each RPS level. Lower = more robust.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    regret = compute_regret(results, rps_range)
    
    # Sort by regret (lowest first)
    sorted_labels = sorted(regret.keys(), key=lambda x: regret[x])
    sorted_regret = [regret[label] for label in sorted_labels]
    
    # Color mapping
    colors = {}
    for config in STATIC_CONFIGS:
        colors[config["label"]] = config["color"]
    colors[DYNAMIC_CONFIG["label"]] = DYNAMIC_CONFIG["color"]
    
    bar_colors = [colors.get(label, "gray") for label in sorted_labels]
    
    bars = ax.barh(range(len(sorted_labels)), sorted_regret, 
                   color=bar_colors, alpha=0.8)
    
    # Add value labels
    for bar, val in zip(bars, sorted_regret):
        ax.text(
            val + 0.1, bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}s',
            va='center', fontsize=10, fontweight='bold'
        )
    
    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels)
    ax.set_xlabel('Total Regret (lower = more robust)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Scheduler Robustness: Cumulative Regret Across All Loads',
        fontsize=14,
        fontweight='bold'
    )
    ax.grid(True, alpha=0.3, axis='x')
    
    # Highlight winner
    ax.annotate(
        '← Most Robust',
        xy=(sorted_regret[0], 0),
        xytext=(sorted_regret[0] + 2, 0),
        fontsize=10,
        va='center',
        color='green',
        fontweight='bold'
    )
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_summary_table(
    results: Dict[str, Dict[float, SimulationMetrics]],
    rps_range: List[float]
) -> None:
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print(" Average Latency (seconds) by RPS")
    print("=" * 80)
    
    # Header
    header = f"{'Scheduler':<25} | " + " | ".join(f"{rps:>5.0f}" for rps in rps_range)
    print(header)
    print("-" * len(header))
    
    # Rows
    for label in results:
        latencies = [results[label][rps].avg_latency for rps in rps_range]
        row = f"{label:<25} | " + " | ".join(f"{lat:>5.2f}" for lat in latencies)
        print(row)
    
    # Regret summary
    print("\n" + "-" * 40)
    print("Total Regret (lower = more robust):")
    regret = compute_regret(results, rps_range)
    for label, reg in sorted(regret.items(), key=lambda x: x[1]):
        print(f"  {label}: {reg:.3f}s")


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Run the timeout sensitivity sweep experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Timeout Sensitivity Sweep: Proving static batching is hard to tune"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="results",
        help="Directory to save plots (default: results)"
    )
    parser.add_argument(
        "--duration", "-d", type=float, default=30.0,
        help="Simulation duration per experiment (default: 30.0)"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--rps-max", type=float, default=10.0,
        help="Maximum RPS to test (default: 10.0)"
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    # Build RPS range
    rps_range = [float(i) for i in range(1, int(args.rps_max) + 1)]
    
    print("=" * 60)
    print(" Timeout Sensitivity Sweep Experiment")
    print("=" * 60)
    print("\nConfigurations being tested:")
    for config in STATIC_CONFIGS:
        print(f"  - {config['label']}")
    print(f"  - {DYNAMIC_CONFIG['label']}")
    print(f"\nRPS Range: {rps_range}")
    print(f"Duration: {args.duration}s per experiment")
    print()
    
    # Run experiments
    print("Running sensitivity sweep...")
    results = run_sensitivity_sweep(
        rps_range=rps_range,
        duration=args.duration,
        seed=args.seed,
        verbose=True
    )
    
    # Print summary
    print_summary_table(results, rps_range)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_sensitivity_curves(
        results, rps_range,
        str(output_dir / "timeout_sensitivity.png")
    )
    plot_winner_heatmap(
        results, rps_range,
        str(output_dir / "timeout_winners.png")
    )
    plot_regret_comparison(
        results, rps_range,
        str(output_dir / "timeout_regret.png")
    )
    
    print("\n" + "=" * 60)
    print(" Experiment Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

