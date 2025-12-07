"""
dLLM Trace-Driven Simulator

A discrete-event simulator for comparing scheduling policies in
Diffusion LLM (dLLM) serving systems.

Key Components:
    - config: Profiling data and simulation constants
    - workload: Synthetic workload generation (Poisson process)
    - scheduler: Batching policies (Static vs Dynamic)
    - simulator: Event-driven simulation engine
    - plotter: Visualization utilities

Usage:
    # Run full comparison
    python -m simulator.main
    
    # Generate plots
    python -m simulator.plotter
    
    # Run tests
    pytest tests/
"""

from simulator.config import (
    PROFILING_DATA,
    MAX_BATCH_SIZE,
    NUM_STEPS,
    get_batch_latency,
    get_total_batch_time,
)

from simulator.workload import (
    Request,
    WorkloadGenerator,
)

from simulator.scheduler import (
    Scheduler,
    StaticBatchScheduler,
    DynamicBatchScheduler,
    RequestQueue,
    Batch,
    create_scheduler,
)

from simulator.simulator import (
    Simulator,
    SimulationMetrics,
    run_simulation,
    compare_schedulers,
)

__version__ = "0.1.0"
__author__ = "Systems Research Team"

__all__ = [
    # Config
    "PROFILING_DATA",
    "MAX_BATCH_SIZE",
    "NUM_STEPS",
    "get_batch_latency",
    "get_total_batch_time",
    # Workload
    "Request",
    "WorkloadGenerator",
    # Scheduler
    "Scheduler",
    "StaticBatchScheduler",
    "DynamicBatchScheduler",
    "RequestQueue",
    "Batch",
    "create_scheduler",
    # Simulator
    "Simulator",
    "SimulationMetrics",
    "run_simulation",
    "compare_schedulers",
]

