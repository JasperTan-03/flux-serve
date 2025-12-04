"""
Core simulation engine for the dLLM Trace-Driven Simulator.

This module implements a discrete-event simulation of a serving system.
The simulation tracks:
    - Global clock (simulated time)
    - Request queue (pending requests)
    - Worker state (busy/free)
    - Per-request metrics (arrival, start, end times)

Simulation Approach:
    We use an event-driven approach where the clock advances to the next
    significant event (request arrival or batch completion). This is more
    efficient than time-stepping for sparse workloads.

Key Insight for dLLMs:
    Unlike AR LLMs where batch execution time varies with sequence length,
    dLLMs have fixed execution time based solely on batch size and the
    constant number of diffusion steps. This makes simulation straightforward.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import heapq
from enum import Enum, auto

from simulator.workload import Request, WorkloadGenerator
from simulator.scheduler import Scheduler, RequestQueue, Batch
from simulator.config import get_total_batch_time, NUM_STEPS


class EventType(Enum):
    """Types of events in the simulation."""
    REQUEST_ARRIVAL = auto()
    BATCH_COMPLETE = auto()


@dataclass(order=True)
class Event:
    """
    A simulation event with a timestamp.
    
    Events are ordered by time for the priority queue.
    """
    time: float
    event_type: EventType = field(compare=False)
    data: Optional[object] = field(default=None, compare=False)


@dataclass
class SimulationMetrics:
    """
    Aggregated metrics from a simulation run.
    
    Attributes:
        total_requests: Number of requests processed
        avg_latency: Average end-to-end latency (seconds)
        avg_queue_time: Average time waiting in queue (seconds)
        p50_latency: 50th percentile latency
        p99_latency: 99th percentile latency
        throughput: Requests processed per second
        avg_batch_size: Average batch size used
        total_batches: Number of batches executed
    """
    total_requests: int
    avg_latency: float
    avg_queue_time: float
    p50_latency: float
    p99_latency: float
    throughput: float
    avg_batch_size: float
    total_batches: int
    
    def __str__(self) -> str:
        return (
            f"Requests: {self.total_requests}, "
            f"Avg Latency: {self.avg_latency:.3f}s, "
            f"P99 Latency: {self.p99_latency:.3f}s, "
            f"Throughput: {self.throughput:.2f} req/s"
        )


class Simulator:
    """
    Discrete-event simulator for dLLM serving.
    
    This simulator models a single-worker serving system where:
    - Requests arrive according to a workload trace
    - A scheduler decides how to batch requests
    - The worker processes batches using profiled latencies
    
    The simulation uses an event queue to efficiently advance time
    to significant events rather than stepping through every millisecond.
    
    Attributes:
        scheduler: Batching policy to evaluate
        requests: Pre-generated workload trace
        clock: Current simulation time
        queue: Pending requests waiting for processing
        completed_requests: Requests that have finished processing
        batch_sizes: Record of batch sizes used (for analysis)
    """
    
    def __init__(
        self,
        scheduler: Scheduler,
        requests: List[Request]
    ) -> None:
        """
        Initialize the simulator.
        
        Args:
            scheduler: Scheduler policy to use for batching
            requests: List of requests with arrival times
        """
        self.scheduler = scheduler
        self.requests = sorted(requests, key=lambda r: r.arrival_time)
        
        # Simulation state
        self.clock: float = 0.0
        self.queue = RequestQueue()
        self.completed_requests: List[Request] = []
        self.batch_sizes: List[int] = []
        
        # Worker state
        self._worker_busy: bool = False
        self._worker_free_time: float = 0.0
        
        # Event queue (min-heap by time)
        self._events: List[Event] = []
    
    def run(self) -> SimulationMetrics:
        """
        Execute the full simulation and return metrics.
        
        Uses a simplified event-driven approach:
        1. Process arrivals in order
        2. When worker is free and queue has requests, try to batch
        3. Execute batch and advance clock
        
        Returns:
            SimulationMetrics with aggregate statistics
        """
        # Convert requests to arrival events
        arrivals = [(r.arrival_time, r) for r in self.requests]
        arrivals.sort(key=lambda x: x[0])
        arrival_idx = 0
        
        self.clock = 0.0
        worker_free_at = 0.0
        
        while arrival_idx < len(arrivals) or not self.queue.is_empty:
            # Add all requests that have arrived by current time
            while arrival_idx < len(arrivals) and arrivals[arrival_idx][0] <= self.clock:
                self.queue.enqueue(arrivals[arrival_idx][1])
                arrival_idx += 1
            
            # If worker is busy, advance to when it's free
            if self.clock < worker_free_at:
                self.clock = worker_free_at
                continue
            
            # Try to form and execute a batch
            if not self.queue.is_empty:
                batch = self.scheduler.get_next_batch(self.queue, self.clock)
                
                if batch is not None:
                    # Execute batch
                    execution_time = get_total_batch_time(batch.batch_size)
                    
                    for request in batch.requests:
                        request.start_time = self.clock
                        request.end_time = self.clock + execution_time
                        self.completed_requests.append(request)
                    
                    self.batch_sizes.append(batch.batch_size)
                    worker_free_at = self.clock + execution_time
                    self.clock = worker_free_at
                    continue
            
            # No batch formed - advance time to next interesting point
            next_times = []
            
            # Next arrival
            if arrival_idx < len(arrivals):
                next_times.append(arrivals[arrival_idx][0])
            
            # For static scheduler with requests waiting, advance past timeout
            if not self.queue.is_empty and hasattr(self.scheduler, 'timeout'):
                # Get scheduler's batch start time if tracking
                batch_start = getattr(self.scheduler, '_batch_start_time', None)
                if batch_start is not None:
                    timeout_time = batch_start + self.scheduler.timeout
                    # Only advance if timeout is in the future
                    if timeout_time > self.clock:
                        next_times.append(timeout_time)
                    else:
                        # Timeout already passed but no batch - advance slightly
                        next_times.append(self.clock + 0.001)
            
            if next_times:
                new_clock = max(min(next_times), self.clock + 0.0001)  # Always advance
                self.clock = new_clock
            else:
                break
        
        return self._compute_metrics()
    
    def _initialize_events(self) -> None:
        """Schedule all request arrivals as events."""
        self._events = []
        for request in self.requests:
            event = Event(
                time=request.arrival_time,
                event_type=EventType.REQUEST_ARRIVAL,
                data=request
            )
            heapq.heappush(self._events, event)
    
    def _process_pending_events(self) -> None:
        """Process all events that have occurred by current time."""
        while self._events and self._events[0].time <= self.clock:
            event = heapq.heappop(self._events)
            self._handle_event(event)
    
    def _handle_event(self, event: Event) -> None:
        """Handle a single simulation event."""
        if event.event_type == EventType.REQUEST_ARRIVAL:
            request = event.data
            if isinstance(request, Request):
                self.queue.enqueue(request)
        elif event.event_type == EventType.BATCH_COMPLETE:
            self._worker_busy = False
    
    def _try_schedule_batch(self) -> None:
        """Attempt to form and execute a batch."""
        batch = self.scheduler.get_next_batch(self.queue, self.clock)
        
        if batch is None:
            # Scheduler decided not to batch yet
            # Advance clock to next arrival to re-evaluate
            if self._events:
                next_event = heapq.heappop(self._events)
                self.clock = next_event.time
                self._handle_event(next_event)
            return
        
        # Execute the batch
        self._execute_batch(batch)
    
    def _execute_batch(self, batch: Batch) -> None:
        """
        Simulate batch execution.
        
        This is where we use the profiled latency data. The total
        execution time is determined by batch size and NUM_STEPS.
        
        Args:
            batch: Batch of requests to process
        """
        batch_size = batch.batch_size
        execution_time = get_total_batch_time(batch_size)
        
        # Record batch start time for all requests
        start_time = self.clock
        for request in batch.requests:
            request.start_time = start_time
        
        # Advance clock by execution time
        end_time = start_time + execution_time
        
        # Record batch end time for all requests
        for request in batch.requests:
            request.end_time = end_time
            self.completed_requests.append(request)
        
        # Update worker state
        self._worker_busy = True
        self._worker_free_time = end_time
        self.clock = end_time
        
        # Record batch size for analysis
        self.batch_sizes.append(batch_size)
        
        # Mark worker as free after batch completes
        self._worker_busy = False
    
    def _compute_metrics(self) -> SimulationMetrics:
        """Compute aggregate metrics from completed requests."""
        if not self.completed_requests:
            return SimulationMetrics(
                total_requests=0,
                avg_latency=0.0,
                avg_queue_time=0.0,
                p50_latency=0.0,
                p99_latency=0.0,
                throughput=0.0,
                avg_batch_size=0.0,
                total_batches=0
            )
        
        latencies = [r.latency for r in self.completed_requests if r.latency is not None]
        queue_times = [r.queue_time for r in self.completed_requests if r.queue_time is not None]
        
        # Sort for percentiles
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        # Compute percentiles
        p50_idx = int(n * 0.50)
        p99_idx = int(n * 0.99)
        
        # Total simulation time
        first_arrival = min(r.arrival_time for r in self.completed_requests)
        last_completion = max(r.end_time for r in self.completed_requests if r.end_time is not None)
        total_time = last_completion - first_arrival if last_completion > first_arrival else 1.0
        
        return SimulationMetrics(
            total_requests=len(self.completed_requests),
            avg_latency=sum(latencies) / len(latencies) if latencies else 0.0,
            avg_queue_time=sum(queue_times) / len(queue_times) if queue_times else 0.0,
            p50_latency=sorted_latencies[p50_idx] if sorted_latencies else 0.0,
            p99_latency=sorted_latencies[min(p99_idx, n - 1)] if sorted_latencies else 0.0,
            throughput=len(self.completed_requests) / total_time,
            avg_batch_size=sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0.0,
            total_batches=len(self.batch_sizes)
        )


def run_simulation(
    scheduler: Scheduler,
    rps: float,
    duration: float = 30.0,
    seed: Optional[int] = None
) -> SimulationMetrics:
    """
    Convenience function to run a complete simulation.
    
    Args:
        scheduler: Scheduler policy to evaluate
        rps: Request rate (requests per second)
        duration: Simulation duration in seconds
        seed: Random seed for reproducibility
        
    Returns:
        SimulationMetrics from the simulation
    """
    # Generate workload
    workload_gen = WorkloadGenerator(rps=rps, duration=duration, seed=seed)
    requests = workload_gen.generate()
    
    # Run simulation
    sim = Simulator(scheduler=scheduler, requests=requests)
    return sim.run()


def _run_single_experiment(args: tuple) -> tuple:
    """
    Worker function for parallel experiment execution.
    
    Args:
        args: Tuple of (scheduler_type, scheduler_kwargs, rps, duration, seed)
        
    Returns:
        Tuple of (scheduler_name, rps, metrics)
    """
    scheduler_type, scheduler_kwargs, scheduler_name, rps, duration, seed = args
    
    # Recreate scheduler in worker process (can't pickle scheduler objects easily)
    from simulator.scheduler import create_scheduler
    scheduler = create_scheduler(scheduler_type, **scheduler_kwargs)
    
    metrics = run_simulation(
        scheduler=scheduler,
        rps=rps,
        duration=duration,
        seed=seed
    )
    return (scheduler_name, rps, metrics)


def compare_schedulers(
    schedulers: List[Scheduler],
    rps_range: List[float],
    duration: float = 30.0,
    seed: int = 42,
    parallel: bool = True,
    max_workers: Optional[int] = None
) -> Dict[str, Dict[float, SimulationMetrics]]:
    """
    Compare multiple schedulers across different load levels.
    
    Uses multiprocessing to run experiments in parallel for significant
    speedup on multi-core systems.
    
    Args:
        schedulers: List of schedulers to compare
        rps_range: List of RPS values to test
        duration: Simulation duration per experiment
        seed: Base random seed (incremented for each RPS level)
        parallel: Whether to use parallel execution (default: True)
        max_workers: Max parallel workers (default: CPU count)
        
    Returns:
        Nested dict: {scheduler_name: {rps: metrics}}
    """
    # Build list of experiment configurations
    experiments = []
    for scheduler in schedulers:
        # Extract scheduler type and kwargs for recreation in worker
        if 'Static' in scheduler.name:
            sched_type = 'static'
            sched_kwargs = {
                'target_batch_size': getattr(scheduler, 'target_batch_size', 8),
                'timeout': getattr(scheduler, 'timeout', 0.1),
                'max_batch_size': getattr(scheduler, 'max_batch_size', 32),
            }
        else:
            sched_type = 'dynamic'
            sched_kwargs = {
                'max_batch_size': getattr(scheduler, 'max_batch_size', 32),
                'min_batch_size': getattr(scheduler, 'min_batch_size', 1),
            }
        
        for rps in rps_range:
            experiments.append((
                sched_type,
                sched_kwargs,
                scheduler.name,
                rps,
                duration,
                seed + int(rps * 100)
            ))
    
    results: Dict[str, Dict[float, SimulationMetrics]] = {}
    for scheduler in schedulers:
        results[scheduler.name] = {}
    
    if parallel and len(experiments) > 1:
        # Parallel execution using ProcessPoolExecutor
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import os
        
        n_workers = max_workers or min(os.cpu_count() or 4, len(experiments))
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_run_single_experiment, exp): exp 
                      for exp in experiments}
            
            for future in as_completed(futures):
                scheduler_name, rps, metrics = future.result()
                results[scheduler_name][rps] = metrics
    else:
        # Sequential execution (for debugging or single experiment)
        for exp in experiments:
            scheduler_name, rps, metrics = _run_single_experiment(exp)
            results[scheduler_name][rps] = metrics
    
    return results


def compare_schedulers_sequential(
    schedulers: List[Scheduler],
    rps_range: List[float],
    duration: float = 30.0,
    seed: int = 42
) -> Dict[str, Dict[float, SimulationMetrics]]:
    """
    Sequential version of compare_schedulers (for debugging).
    
    Args:
        schedulers: List of schedulers to compare
        rps_range: List of RPS values to test
        duration: Simulation duration per experiment
        seed: Base random seed (incremented for each RPS level)
        
    Returns:
        Nested dict: {scheduler_name: {rps: metrics}}
    """
    return compare_schedulers(
        schedulers=schedulers,
        rps_range=rps_range,
        duration=duration,
        seed=seed,
        parallel=False
    )

