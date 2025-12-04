"""
Scheduler module for the dLLM Trace-Driven Simulator.

This module implements scheduling policies for batching inference requests.
The key insight driving our design is that dLLMs are compute-bound with
flat memory usage, which fundamentally changes optimal scheduling strategy.

Key Differences from Autoregressive LLM Scheduling:
    - AR LLMs: KV cache grows, memory is the bottleneck, must be conservative
    - dLLMs: No KV cache, memory is flat, can batch aggressively
    
Schedulers Implemented:
    1. StaticBatchScheduler (Baseline): Waits for target batch or timeout
    2. DynamicBatchScheduler (Proposed): Greedily maximizes GPU utilization
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from collections import deque

from simulator.workload import Request
from simulator.config import MAX_BATCH_SIZE, DEFAULT_TIMEOUT, DEFAULT_TARGET_BATCH


@dataclass
class Batch:
    """
    Represents a batch of requests to be processed together.
    
    All requests in a batch share the same start and end times,
    since dLLMs process the entire batch through diffusion steps together.
    
    Attributes:
        requests: List of requests in this batch
        batch_size: Number of requests (cached for convenience)
    """
    requests: List[Request]
    
    @property
    def batch_size(self) -> int:
        return len(self.requests)
    
    def __len__(self) -> int:
        return len(self.requests)


class RequestQueue:
    """
    FIFO queue for pending inference requests.
    
    This queue holds requests that have arrived but haven't been
    scheduled for processing yet. The scheduler pulls from this queue
    according to its batching policy.
    """
    
    def __init__(self) -> None:
        self._queue: deque[Request] = deque()
    
    def enqueue(self, request: Request) -> None:
        """Add a request to the back of the queue."""
        self._queue.append(request)
    
    def dequeue(self, count: int = 1) -> List[Request]:
        """
        Remove and return up to 'count' requests from the front.
        
        Args:
            count: Maximum number of requests to dequeue
            
        Returns:
            List of dequeued requests (may be fewer than count if queue is small)
        """
        result: List[Request] = []
        for _ in range(min(count, len(self._queue))):
            result.append(self._queue.popleft())
        return result
    
    def peek(self, count: int = 1) -> List[Request]:
        """View up to 'count' requests without removing them."""
        return list(self._queue)[:count]
    
    def __len__(self) -> int:
        return len(self._queue)
    
    @property
    def is_empty(self) -> bool:
        return len(self._queue) == 0


class Scheduler(ABC):
    """
    Abstract base class for batch schedulers.
    
    A scheduler decides WHEN to form a batch and WHICH requests to include.
    Different policies optimize for different objectives:
    - Latency: Minimize time requests spend in the system
    - Throughput: Maximize requests processed per second
    - Fairness: Ensure no request waits too long
    """
    
    @abstractmethod
    def get_next_batch(
        self,
        queue: RequestQueue,
        current_time: float
    ) -> Optional[Batch]:
        """
        Determine the next batch to process.
        
        Args:
            queue: The request queue to pull from
            current_time: Current simulation time (for timeout logic)
            
        Returns:
            Batch of requests to process, or None if no batch should run
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this scheduler."""
        pass


class StaticBatchScheduler(Scheduler):
    """
    Baseline scheduler using static batching with timeout.
    
    This represents the naive approach common in serving systems:
    - Wait until target_batch_size requests accumulate
    - OR timeout expires (to bound latency for small batches)
    
    Problems with this approach for dLLMs:
    1. Wastes GPU cycles waiting for full batch
    2. Doesn't exploit sub-linear latency scaling
    3. Under low load, requests wait unnecessarily
    
    Attributes:
        target_batch_size: Desired batch size before processing
        timeout: Maximum wait time before processing partial batch
        max_batch_size: Hard limit on batch size
        _last_arrival_time: Time of oldest request in queue (for timeout)
    """
    
    def __init__(
        self,
        target_batch_size: int = DEFAULT_TARGET_BATCH,
        timeout: float = DEFAULT_TIMEOUT,
        max_batch_size: int = MAX_BATCH_SIZE
    ) -> None:
        self.target_batch_size = target_batch_size
        self.timeout = timeout
        self.max_batch_size = max_batch_size
        self._batch_start_time: Optional[float] = None
    
    def get_next_batch(
        self,
        queue: RequestQueue,
        current_time: float
    ) -> Optional[Batch]:
        """
        Get next batch using static batching policy.
        
        Forms a batch when either:
        1. Queue has >= target_batch_size requests
        2. Timeout has expired since first request arrived
        
        Args:
            queue: Request queue to pull from
            current_time: Current simulation time
            
        Returns:
            Batch if conditions met, None otherwise
        """
        if queue.is_empty:
            self._batch_start_time = None
            return None
        
        # Track when we started accumulating this batch
        if self._batch_start_time is None:
            # Use arrival time of first request in queue
            first_request = queue.peek(1)[0]
            self._batch_start_time = first_request.arrival_time
        
        queue_size = len(queue)
        time_waiting = current_time - self._batch_start_time
        
        # Check if we should form a batch
        should_batch = (
            queue_size >= self.target_batch_size or  # Full batch ready
            time_waiting >= self.timeout             # Timeout expired
        )
        
        if not should_batch:
            return None
        
        # Form batch (up to max size)
        batch_size = min(queue_size, self.max_batch_size)
        requests = queue.dequeue(batch_size)
        
        # Reset timer for next batch
        self._batch_start_time = None
        
        return Batch(requests=requests)
    
    @property
    def name(self) -> str:
        return f"Static(target={self.target_batch_size}, timeout={self.timeout}s)"


class DynamicBatchScheduler(Scheduler):
    """
    Compute-aware dynamic scheduler (Our Proposal).
    
    This scheduler exploits the key property of dLLMs: sub-linear latency
    scaling means we get "free compute" by batching more aggressively.
    
    Strategy: Greedily pull ALL available requests (up to max) whenever
    the worker becomes free. Don't wait for a "full" batch.
    
    Why this works for dLLMs:
    1. No KV cache = flat memory = can always fit max batch
    2. Sub-linear latency = processing 2 requests barely costs more than 1
    3. Immediate processing = lower queue times
    
    Example (using profiling data):
    - 2 requests arrive, worker is free
    - Static: Wait for 8 more requests (target=8)
    - Dynamic: Process immediately!
      - Batch 2 latency: 0.0331s × 20 = 0.662s total
      - vs Batch 1 × 2: 0.025s × 20 × 2 = 1.0s total
      - 34% faster by batching NOW
    
    Attributes:
        max_batch_size: Maximum requests to pull per batch
        min_batch_size: Minimum requests needed to form batch (default: 1)
    """
    
    def __init__(
        self,
        max_batch_size: int = MAX_BATCH_SIZE,
        min_batch_size: int = 1
    ) -> None:
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
    
    def get_next_batch(
        self,
        queue: RequestQueue,
        current_time: float
    ) -> Optional[Batch]:
        """
        Get next batch using dynamic/greedy policy.
        
        Immediately pulls min(queue_length, max_batch_size) requests
        whenever called (i.e., when worker is free).
        
        This maximizes GPU utilization by:
        1. Never waiting when work is available
        2. Batching as much as possible to exploit sub-linear scaling
        
        Args:
            queue: Request queue to pull from
            current_time: Current simulation time (unused, but kept for interface)
            
        Returns:
            Batch of all available requests (up to max), or None if queue empty
        """
        if len(queue) < self.min_batch_size:
            return None
        
        # Greedily take everything available (up to max)
        batch_size = min(len(queue), self.max_batch_size)
        requests = queue.dequeue(batch_size)
        
        return Batch(requests=requests)
    
    @property
    def name(self) -> str:
        return f"Dynamic(max={self.max_batch_size})"


# =============================================================================
# Factory function for easy scheduler creation
# =============================================================================

def create_scheduler(
    scheduler_type: str,
    **kwargs
) -> Scheduler:
    """
    Factory function to create schedulers by name.
    
    Args:
        scheduler_type: "static" or "dynamic"
        **kwargs: Arguments passed to scheduler constructor
        
    Returns:
        Configured scheduler instance
        
    Raises:
        ValueError: If scheduler_type is unknown
    """
    schedulers = {
        "static": StaticBatchScheduler,
        "dynamic": DynamicBatchScheduler,
    }
    
    if scheduler_type.lower() not in schedulers:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. "
            f"Available: {list(schedulers.keys())}"
        )
    
    return schedulers[scheduler_type.lower()](**kwargs)

