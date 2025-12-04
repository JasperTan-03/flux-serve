"""
Workload generation module for the dLLM Trace-Driven Simulator.

This module generates synthetic request arrival patterns using a Poisson
process, which models real-world request arrivals where requests arrive
independently at a constant average rate.

The Poisson process is characterized by:
    - Inter-arrival times following an exponential distribution
    - Memoryless property (past arrivals don't affect future ones)
    - Rate parameter λ (requests per second)
"""

from dataclasses import dataclass, field
from typing import List, Optional
import random
import math


@dataclass
class Request:
    """
    Represents a single inference request in the serving system.
    
    Attributes:
        request_id: Unique identifier for this request
        arrival_time: When the request entered the system (simulation time)
        start_time: When processing began (set by simulator)
        end_time: When processing completed (set by simulator)
    """
    request_id: int
    arrival_time: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    @property
    def latency(self) -> Optional[float]:
        """Total request latency (end-to-end time in system)."""
        if self.end_time is not None and self.arrival_time is not None:
            return self.end_time - self.arrival_time
        return None
    
    @property
    def queue_time(self) -> Optional[float]:
        """Time spent waiting in queue before processing."""
        if self.start_time is not None and self.arrival_time is not None:
            return self.start_time - self.arrival_time
        return None
    
    @property
    def processing_time(self) -> Optional[float]:
        """Actual processing time (execution on GPU)."""
        if self.end_time is not None and self.start_time is not None:
            return self.end_time - self.start_time
        return None


@dataclass
class WorkloadGenerator:
    """
    Generates synthetic workloads using a Poisson arrival process.
    
    The Poisson process is ideal for modeling request arrivals because:
    1. Requests arrive independently (user actions are independent)
    2. The rate is constant over the simulation period
    3. It's well-understood and widely used in queueing theory
    
    Attributes:
        rps: Request rate (Requests Per Second / λ parameter)
        duration: Total simulation duration in seconds
        seed: Random seed for reproducibility
        
    Example:
        >>> gen = WorkloadGenerator(rps=5.0, duration=10.0, seed=42)
        >>> requests = gen.generate()
        >>> len(requests)  # Approximately 50 requests
        48
    """
    rps: float
    duration: float
    seed: Optional[int] = None
    _rng: random.Random = field(default_factory=random.Random, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize the random number generator with seed if provided."""
        if self.seed is not None:
            self._rng.seed(self.seed)
    
    def generate(self) -> List[Request]:
        """
        Generate a list of requests with Poisson-distributed arrival times.
        
        Uses the inverse transform method: inter-arrival times are sampled
        from an exponential distribution with rate λ (rps).
        
        Inter-arrival time: T ~ Exponential(λ) = -ln(U) / λ
        where U ~ Uniform(0, 1)
        
        Returns:
            List of Request objects sorted by arrival time
        """
        requests: List[Request] = []
        current_time = 0.0
        request_id = 0
        
        # Handle edge case of zero RPS
        if self.rps <= 0:
            return requests
        
        # Mean inter-arrival time
        mean_interarrival = 1.0 / self.rps
        
        while current_time < self.duration:
            # Sample inter-arrival time from exponential distribution
            # Using inverse transform: -ln(U) / λ
            u = self._rng.random()
            # Avoid log(0) by using 1-u or clamping
            interarrival = -math.log(1 - u) * mean_interarrival
            
            current_time += interarrival
            
            if current_time < self.duration:
                requests.append(Request(
                    request_id=request_id,
                    arrival_time=current_time
                ))
                request_id += 1
        
        return requests
    
    def expected_requests(self) -> float:
        """
        Calculate the expected number of requests for this workload.
        
        For a Poisson process: E[N] = λ * T
        where λ is the rate and T is the duration.
        
        Returns:
            Expected number of requests (may be fractional)
        """
        return self.rps * self.duration


def generate_bursty_workload(
    base_rps: float,
    burst_rps: float,
    duration: float,
    burst_fraction: float = 0.2,
    seed: Optional[int] = None
) -> List[Request]:
    """
    Generate a workload with periodic bursts (for stress testing).
    
    This creates a more realistic workload pattern where traffic
    occasionally spikes, testing the scheduler's ability to handle
    load variations.
    
    Args:
        base_rps: Normal request rate
        burst_rps: Request rate during burst periods
        duration: Total simulation duration
        burst_fraction: Fraction of time in burst mode (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        List of Request objects with mixed arrival patterns
    """
    rng = random.Random(seed)
    requests: List[Request] = []
    request_id = 0
    current_time = 0.0
    
    # Alternate between normal and burst periods
    period_length = duration * 0.1  # 10% of total duration per period
    
    while current_time < duration:
        # Determine if we're in a burst period
        period_idx = int(current_time / period_length)
        in_burst = (period_idx % 5 == 0) if burst_fraction > 0 else False
        
        current_rps = burst_rps if in_burst else base_rps
        
        if current_rps <= 0:
            current_time += period_length
            continue
            
        mean_interarrival = 1.0 / current_rps
        interarrival = -math.log(1 - rng.random()) * mean_interarrival
        current_time += interarrival
        
        if current_time < duration:
            requests.append(Request(
                request_id=request_id,
                arrival_time=current_time
            ))
            request_id += 1
    
    return requests

