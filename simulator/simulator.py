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
        
        The simulation proceeds by:
        1. Scheduling all request arrivals as events
        2. Processing events in time order
        3. When worker is free and queue non-empty, form and execute batch
        4. Continue until all requests are processed
        
        Returns:
            SimulationMetrics with aggregate statistics
        """
        self._initialize_events()
        
        while self._events or not self.queue.is_empty:
            # Process all events up to current time or next event
            self._process_pending_events()
            
            # Try to schedule work if worker is free
            if not self._worker_busy and not self.queue.is_empty:
                self._try_schedule_batch()
            
            # If worker is busy and events remain, advance to next event
            if self._events and self._worker_busy:
                next_event = heapq.heappop(self._events)
                self.clock = next_event.time
                self._handle_event(next_event)
            elif self._worker_busy:
                # No more arrival events, just wait for batch to complete
                self.clock = self._worker_free_time
                self._worker_busy = False
        
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


def compare_schedulers(
    schedulers: List[Scheduler],
    rps_range: List[float],
    duration: float = 30.0,
    seed: int = 42
) -> Dict[str, Dict[float, SimulationMetrics]]:
    """
    Compare multiple schedulers across different load levels.
    
    Args:
        schedulers: List of schedulers to compare
        rps_range: List of RPS values to test
        duration: Simulation duration per experiment
        seed: Base random seed (incremented for each RPS level)
        
    Returns:
        Nested dict: {scheduler_name: {rps: metrics}}
    """
    results: Dict[str, Dict[float, SimulationMetrics]] = {}
    
    for scheduler in schedulers:
        results[scheduler.name] = {}
        
        for rps in rps_range:
            # Use consistent seed per RPS level across schedulers
            metrics = run_simulation(
                scheduler=scheduler,
                rps=rps,
                duration=duration,
                seed=seed + int(rps * 100)
            )
            results[scheduler.name][rps] = metrics
    
    return results

