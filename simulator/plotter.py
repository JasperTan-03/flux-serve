"""
Visualization module for the dLLM Trace-Driven Simulator.

This module generates publication-quality plots comparing scheduler
performance. The key visualization shows how average latency varies
with load for different scheduling policies.

Expected Visualization Story:
    - At low load: Both schedulers perform similarly
    - As load increases: Dynamic batching maintains lower latency
    - At saturation: The gap widens, showing the benefit of greedy batching
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

from simulator.main import run_experiments
from simulator.simulator import SimulationMetrics
from simulator.config import PROFILING_DATA, MAX_BATCH_SIZE, NUM_STEPS


# Style configuration for publication-quality plots
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette (colorblind-friendly)
COLORS = {
    'static': '#E69F00',    # Orange
    'dynamic': '#0072B2',   # Blue
    'accent': '#CC79A7',    # Pink
    'grid': '#CCCCCC',      # Light gray
}


def plot_latency_comparison(
    results: Dict[str, Dict[float, SimulationMetrics]],
    rps_range: List[float],
    output_path: str = "scheduler_comparison.png",
    show: bool = False
) -> None:
    """
    Generate the main comparison plot: Latency vs Load.
    
    This is the key figure demonstrating that dynamic batching
    achieves lower latency than static batching for dLLMs.
    
    Args:
        results: Nested dict from compare_schedulers
        rps_range: RPS values tested
        output_path: Where to save the figure
        show: Whether to display interactively
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scheduler_names = list(results.keys())
    colors = [COLORS['static'], COLORS['dynamic']]
    markers = ['o', 's']
    
    for idx, name in enumerate(scheduler_names):
        latencies = [results[name][rps].avg_latency for rps in rps_range]
        
        # Determine label
        if 'Static' in name:
            label = 'Static Batching (Baseline)'
        else:
            label = 'Dynamic Batching (Proposed)'
        
        ax.plot(
            rps_range, latencies,
            color=colors[idx],
            marker=markers[idx],
            markersize=8,
            linewidth=2.5,
            label=label
        )
    
    # Styling
    ax.set_xlabel('Input Load (Requests Per Second)', fontweight='bold')
    ax.set_ylabel('Average Request Latency (seconds)', fontweight='bold')
    ax.set_title(
        'dLLM Scheduling: Dynamic Batching Outperforms Static Batching',
        fontweight='bold',
        pad=15
    )
    
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, color=COLORS['grid'])
    ax.set_xlim(min(rps_range) - 0.5, max(rps_range) + 0.5)
    ax.set_ylim(bottom=0)
    
    # Add annotation explaining the insight
    ax.annotate(
        'Dynamic batching exploits\nsub-linear latency scaling',
        xy=(7, results[scheduler_names[1]][7.0].avg_latency),
        xytext=(8.5, results[scheduler_names[0]][5.0].avg_latency),
        fontsize=9,
        arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
    )
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_tail_latency_comparison(
    results: Dict[str, Dict[float, SimulationMetrics]],
    rps_range: List[float],
    output_path: str = "tail_latency_comparison.png",
    show: bool = False
) -> None:
    """
    Plot P99 latency comparison.
    
    Tail latency is crucial for SLO compliance. This plot shows
    that dynamic batching also improves worst-case latency.
    
    Args:
        results: Nested dict from compare_schedulers
        rps_range: RPS values tested
        output_path: Where to save the figure
        show: Whether to display interactively
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scheduler_names = list(results.keys())
    colors = [COLORS['static'], COLORS['dynamic']]
    
    for idx, name in enumerate(scheduler_names):
        p99_latencies = [results[name][rps].p99_latency for rps in rps_range]
        
        if 'Static' in name:
            label = 'Static Batching (P99)'
        else:
            label = 'Dynamic Batching (P99)'
        
        ax.plot(
            rps_range, p99_latencies,
            color=colors[idx],
            marker='o',
            markersize=8,
            linewidth=2.5,
            label=label
        )
    
    ax.set_xlabel('Input Load (Requests Per Second)', fontweight='bold')
    ax.set_ylabel('P99 Latency (seconds)', fontweight='bold')
    ax.set_title(
        'Tail Latency Comparison: Dynamic Batching Reduces P99',
        fontweight='bold',
        pad=15
    )
    
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, color=COLORS['grid'])
    ax.set_xlim(min(rps_range) - 0.5, max(rps_range) + 0.5)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_batch_size_distribution(
    results: Dict[str, Dict[float, SimulationMetrics]],
    rps_range: List[float],
    output_path: str = "batch_size_analysis.png",
    show: bool = False
) -> None:
    """
    Plot average batch sizes used by each scheduler.
    
    This helps explain WHY dynamic batching works: it forms
    larger batches on average, exploiting sub-linear scaling.
    
    Args:
        results: Nested dict from compare_schedulers
        rps_range: RPS values tested
        output_path: Where to save the figure
        show: Whether to display interactively
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scheduler_names = list(results.keys())
    colors = [COLORS['static'], COLORS['dynamic']]
    
    x = np.arange(len(rps_range))
    width = 0.35
    
    for idx, name in enumerate(scheduler_names):
        batch_sizes = [results[name][rps].avg_batch_size for rps in rps_range]
        
        if 'Static' in name:
            label = 'Static Batching'
        else:
            label = 'Dynamic Batching'
        
        offset = width * (idx - 0.5)
        ax.bar(x + offset, batch_sizes, width, label=label, color=colors[idx], alpha=0.8)
    
    ax.set_xlabel('Input Load (Requests Per Second)', fontweight='bold')
    ax.set_ylabel('Average Batch Size', fontweight='bold')
    ax.set_title(
        'Batch Size Analysis: Dynamic Forms Larger Batches',
        fontweight='bold',
        pad=15
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f'{rps:.0f}' for rps in rps_range])
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])
    ax.set_ylim(bottom=0, top=MAX_BATCH_SIZE + 2)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_profiling_data(
    output_path: str = "profiling_data.png",
    show: bool = False
) -> None:
    """
    Visualize the profiling data showing sub-linear scaling.
    
    This plot demonstrates the key insight: latency grows
    sub-linearly with batch size, meaning larger batches
    are more efficient per-request.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    batch_sizes = list(PROFILING_DATA.keys())
    latencies = [PROFILING_DATA[bs] for bs in batch_sizes]
    
    # Plot 1: Raw latency vs batch size
    ax1.plot(batch_sizes, latencies, 'o-', color=COLORS['dynamic'], 
             markersize=10, linewidth=2.5, label='Measured')
    
    # Show linear scaling for comparison
    linear_latencies = [latencies[0] * bs for bs in batch_sizes]
    ax1.plot(batch_sizes, linear_latencies, '--', color='gray', 
             linewidth=1.5, alpha=0.7, label='Linear scaling (if no batching benefit)')
    
    ax1.set_xlabel('Batch Size', fontweight='bold')
    ax1.set_ylabel('Per-Step Latency (seconds)', fontweight='bold')
    ax1.set_title('LLaDA-8B: Sub-Linear Latency Scaling', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, color=COLORS['grid'])
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    
    # Plot 2: Throughput (requests/second)
    throughputs = [bs / (PROFILING_DATA[bs] * NUM_STEPS) for bs in batch_sizes]
    ax2.bar(range(len(batch_sizes)), throughputs, color=COLORS['dynamic'], alpha=0.8)
    ax2.set_xticks(range(len(batch_sizes)))
    ax2.set_xticklabels(batch_sizes)
    ax2.set_xlabel('Batch Size', fontweight='bold')
    ax2.set_ylabel('Throughput (requests/second)', fontweight='bold')
    ax2.set_title('Throughput Increases with Batch Size', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])
    
    # Add annotation
    ax2.annotate(
        f'{throughputs[-1]/throughputs[0]:.1f}x\nhigher!',
        xy=(len(batch_sizes)-1, throughputs[-1]),
        xytext=(len(batch_sizes)-2, throughputs[-1] * 0.7),
        fontsize=11,
        fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
    )
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_improvement_heatmap(
    results: Dict[str, Dict[float, SimulationMetrics]],
    rps_range: List[float],
    output_path: str = "improvement_summary.png",
    show: bool = False
) -> None:
    """
    Create a summary visualization showing improvement percentages.
    """
    scheduler_names = list(results.keys())
    static_name = scheduler_names[0]
    dynamic_name = scheduler_names[1]
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    improvements = []
    for rps in rps_range:
        baseline_lat = results[static_name][rps].avg_latency
        proposed_lat = results[dynamic_name][rps].avg_latency
        improvement = (baseline_lat - proposed_lat) / baseline_lat * 100
        improvements.append(improvement)
    
    colors = [COLORS['dynamic'] if imp > 0 else COLORS['static'] for imp in improvements]
    bars = ax.bar(range(len(rps_range)), improvements, color=colors, alpha=0.8)
    
    # Add value labels on bars
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.annotate(f'{imp:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
    
    ax.set_xticks(range(len(rps_range)))
    ax.set_xticklabels([f'{rps:.0f}' for rps in rps_range])
    ax.set_xlabel('Input Load (RPS)', fontweight='bold')
    ax.set_ylabel('Latency Improvement (%)', fontweight='bold')
    ax.set_title(
        'Dynamic Batching: Latency Improvement Over Static Baseline',
        fontweight='bold',
        pad=15
    )
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def generate_all_plots(
    results: Optional[Dict[str, Dict[float, SimulationMetrics]]] = None,
    rps_range: Optional[List[float]] = None,
    output_dir: str = "results",
    show: bool = False
) -> None:
    """
    Generate all visualization plots.
    
    Args:
        results: Pre-computed results (if None, runs experiments)
        rps_range: RPS values to test
        output_dir: Directory to save plots
        show: Whether to display plots interactively
    """
    if rps_range is None:
        rps_range = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    
    if results is None:
        print("Running experiments for plotting...")
        results = run_experiments(rps_range, duration=30.0, seed=42, verbose=False)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating plots...")
    
    # Main comparison plot (required output)
    plot_latency_comparison(
        results, rps_range,
        str(output_path / "scheduler_comparison.png"),
        show
    )
    
    # Additional analysis plots
    plot_tail_latency_comparison(
        results, rps_range,
        str(output_path / "tail_latency_comparison.png"),
        show
    )
    
    plot_batch_size_distribution(
        results, rps_range,
        str(output_path / "batch_size_analysis.png"),
        show
    )
    
    plot_profiling_data(
        str(output_path / "profiling_data.png"),
        show
    )
    
    plot_improvement_heatmap(
        results, rps_range,
        str(output_path / "improvement_summary.png"),
        show
    )
    
    print("\nAll plots generated successfully!")


def main() -> None:
    """Main entry point for plotting with CLI arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate visualization plots for dLLM scheduler comparison"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="results",
        help="Directory to save plots (default: results)"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display plots interactively"
    )
    parser.add_argument(
        "--duration", "-d", type=float, default=30.0,
        help="Simulation duration in seconds (default: 30.0)"
    )
    
    args = parser.parse_args()
    
    generate_all_plots(output_dir=args.output_dir, show=args.show)


if __name__ == "__main__":
    main()

