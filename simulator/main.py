"""
Main entry point for the dLLM Trace-Driven Simulator.

This script runs comparative experiments between scheduling policies
and produces summary statistics. The goal is to demonstrate that
dynamic/greedy batching outperforms static batching for dLLMs.

Experiment Design:
    - Sweep RPS from 1 to 10 (covering under-loaded to saturated)
    - Compare Static (baseline) vs Dynamic (proposed) scheduler
    - Report average latency at each load level
    
Expected Results:
    - At low RPS: Similar performance (both have slack)
    - At high RPS: Dynamic wins (better GPU utilization)
    - Dynamic should show lower tail latencies (no waiting for batch)
"""

import sys
from typing import List, Tuple, Dict

from simulator.scheduler import StaticBatchScheduler, DynamicBatchScheduler, Scheduler
from simulator.simulator import run_simulation, SimulationMetrics, compare_schedulers
from simulator.config import DEFAULT_TARGET_BATCH, DEFAULT_TIMEOUT


def print_header(title: str) -> None:
    """Print a formatted section header."""
    width = 70
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_table(
    results: Dict[str, Dict[float, SimulationMetrics]],
    rps_range: List[float]
) -> None:
    """
    Print a comparison table of scheduler results.
    
    Args:
        results: Nested dict from compare_schedulers
        rps_range: RPS values (column headers)
    """
    scheduler_names = list(results.keys())
    
    # Table header
    header = f"{'Scheduler':<45} | " + " | ".join(f"{rps:>6.1f}" for rps in rps_range)
    print(header)
    print("-" * len(header))
    
    # Average latency row for each scheduler
    for name in scheduler_names:
        latencies = [results[name][rps].avg_latency for rps in rps_range]
        row = f"{name:<45} | " + " | ".join(f"{lat:>6.3f}" for lat in latencies)
        print(row)


def print_detailed_results(
    results: Dict[str, Dict[float, SimulationMetrics]],
    rps_range: List[float]
) -> None:
    """Print detailed metrics for each configuration."""
    for rps in rps_range:
        print_header(f"RPS = {rps}")
        
        for name, rps_results in results.items():
            metrics = rps_results[rps]
            print(f"\n  {name}:")
            print(f"    Requests:      {metrics.total_requests}")
            print(f"    Avg Latency:   {metrics.avg_latency:.4f}s")
            print(f"    P50 Latency:   {metrics.p50_latency:.4f}s")
            print(f"    P99 Latency:   {metrics.p99_latency:.4f}s")
            print(f"    Avg Queue:     {metrics.avg_queue_time:.4f}s")
            print(f"    Avg Batch:     {metrics.avg_batch_size:.2f}")
            print(f"    Throughput:    {metrics.throughput:.2f} req/s")


def calculate_improvement(
    results: Dict[str, Dict[float, SimulationMetrics]],
    baseline_name: str,
    proposed_name: str,
    rps_range: List[float]
) -> None:
    """Print improvement of proposed over baseline."""
    print_header("Improvement Summary (Proposed vs Baseline)")
    
    for rps in rps_range:
        baseline = results[baseline_name][rps]
        proposed = results[proposed_name][rps]
        
        latency_improvement = (baseline.avg_latency - proposed.avg_latency) / baseline.avg_latency * 100
        p99_improvement = (baseline.p99_latency - proposed.p99_latency) / baseline.p99_latency * 100
        
        print(f"  RPS {rps:>5.1f}: Latency -{latency_improvement:>5.1f}%, P99 -{p99_improvement:>5.1f}%")


def run_experiments(
    rps_range: List[float],
    duration: float = 30.0,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Dict[float, SimulationMetrics]]:
    """
    Run the full experiment suite.
    
    Args:
        rps_range: List of RPS values to test
        duration: Simulation duration per experiment
        seed: Random seed for reproducibility
        verbose: Whether to print detailed output
        
    Returns:
        Results dictionary
    """
    # Create schedulers
    static_scheduler = StaticBatchScheduler(
        target_batch_size=DEFAULT_TARGET_BATCH,
        timeout=DEFAULT_TIMEOUT
    )
    dynamic_scheduler = DynamicBatchScheduler()
    
    schedulers = [static_scheduler, dynamic_scheduler]
    
    if verbose:
        print_header("dLLM Serving Simulator")
        print("\nComparing scheduling policies for Diffusion LLMs (LLaDA-8B)")
        print("\nKey insight: dLLMs are compute-bound with flat memory,")
        print("enabling aggressive batching without memory constraints.")
        print(f"\nSimulation: {duration}s duration, seed={seed}")
        print(f"RPS range: {rps_range}")
    
    # Run comparisons
    if verbose:
        print("\nRunning simulations...")
    
    results = compare_schedulers(
        schedulers=schedulers,
        rps_range=rps_range,
        duration=duration,
        seed=seed
    )
    
    return results


def main() -> None:
    """Main entry point."""
    # Configuration
    rps_range = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    duration = 30.0  # 30 seconds per experiment
    seed = 42
    
    # Run experiments
    results = run_experiments(rps_range, duration, seed, verbose=True)
    
    # Get scheduler names for printing
    scheduler_names = list(results.keys())
    baseline_name = scheduler_names[0]  # Static
    proposed_name = scheduler_names[1]  # Dynamic
    
    # Print summary table
    print_header("Average Latency (seconds) by RPS")
    print_table(results, rps_range)
    
    # Print detailed results
    print_detailed_results(results, rps_range)
    
    # Print improvement summary
    calculate_improvement(results, baseline_name, proposed_name, rps_range)
    
    print("\n" + "=" * 70)
    print(" Simulation Complete!")
    print(" Run 'python -m simulator.plotter' to generate visualizations.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

