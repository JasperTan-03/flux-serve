"""
Configuration module for the dLLM Trace-Driven Simulator.

This module stores profiling data from LLaDA-8B and provides utilities
for interpolating batch execution latencies. The key insight is that
dLLMs are compute-bound with flat memory usage, enabling aggressive batching.

Profiling Results:
    - Memory is flat (no KV cache growth)
    - Throughput scales sub-linearly with batch size
    - This creates "free compute" opportunities for smart batching
"""

from typing import Dict, Tuple
import bisect


# =============================================================================
# Profiling Data: LLaDA-8B Latency vs Batch Size
# =============================================================================
# These measurements show sub-linear scaling: doubling batch size does NOT
# double latency, indicating compute-bound behavior we can exploit.

PROFILING_DATA: Dict[int, float] = {
    1: 0.0250,   # Batch 1:  25.0ms per step
    2: 0.0331,   # Batch 2:  33.1ms per step (1.32x latency for 2x batch)
    4: 0.0577,   # Batch 4:  57.7ms per step (1.74x latency for 2x batch)
    8: 0.1130,   # Batch 8:  113.0ms per step (1.96x latency for 2x batch)
    16: 0.2132,  # Batch 16: 213.2ms per step (1.89x latency for 2x batch)
    32: 0.4013,  # Batch 32: 401.3ms per step (1.88x latency for 2x batch)
}

# Sorted batch sizes for interpolation
_BATCH_SIZES = sorted(PROFILING_DATA.keys())
_LATENCIES = [PROFILING_DATA[bs] for bs in _BATCH_SIZES]


# =============================================================================
# Simulation Constants
# =============================================================================

MAX_BATCH_SIZE: int = 32
"""Maximum batch size the GPU can handle."""

NUM_STEPS: int = 20
"""Fixed number of diffusion steps per request (dLLMs don't vary this)."""

DEFAULT_TIMEOUT: float = 1.0
"""Default timeout (seconds) for static batching scheduler."""

DEFAULT_TARGET_BATCH: int = 8
"""Default target batch size for static batching scheduler."""


# =============================================================================
# Latency Interpolation
# =============================================================================

def get_batch_latency(batch_size: int) -> float:
    """
    Get the per-step latency for a given batch size using linear interpolation.
    
    This function interpolates between profiled data points to estimate
    latency for batch sizes not explicitly measured. This is valid because
    latency scaling is approximately linear between measured points.
    
    Args:
        batch_size: Number of requests in the batch (1 to MAX_BATCH_SIZE)
        
    Returns:
        Per-step latency in seconds
        
    Raises:
        ValueError: If batch_size is less than 1 or greater than MAX_BATCH_SIZE
        
    Example:
        >>> get_batch_latency(1)
        0.025
        >>> get_batch_latency(3)  # Interpolated between 2 and 4
        0.0454
    """
    if batch_size < 1:
        raise ValueError(f"Batch size must be >= 1, got {batch_size}")
    if batch_size > MAX_BATCH_SIZE:
        raise ValueError(f"Batch size must be <= {MAX_BATCH_SIZE}, got {batch_size}")
    
    # Exact match - return profiled value
    if batch_size in PROFILING_DATA:
        return PROFILING_DATA[batch_size]
    
    # Find surrounding points for interpolation
    idx = bisect.bisect_left(_BATCH_SIZES, batch_size)
    
    # Edge cases (shouldn't happen given our data, but handle gracefully)
    if idx == 0:
        return _LATENCIES[0]
    if idx >= len(_BATCH_SIZES):
        return _LATENCIES[-1]
    
    # Linear interpolation between surrounding points
    bs_low, bs_high = _BATCH_SIZES[idx - 1], _BATCH_SIZES[idx]
    lat_low, lat_high = _LATENCIES[idx - 1], _LATENCIES[idx]
    
    # Interpolation factor
    t = (batch_size - bs_low) / (bs_high - bs_low)
    interpolated_latency = lat_low + t * (lat_high - lat_low)
    
    return interpolated_latency


def get_total_batch_time(batch_size: int) -> float:
    """
    Get total execution time for a batch (latency per step * NUM_STEPS).
    
    For dLLMs, total time = per_step_latency * num_steps, since all
    requests in a batch proceed through diffusion steps together.
    
    Args:
        batch_size: Number of requests in the batch
        
    Returns:
        Total batch execution time in seconds
    """
    return get_batch_latency(batch_size) * NUM_STEPS


def compute_throughput(batch_size: int) -> float:
    """
    Compute throughput (requests/second) for a given batch size.
    
    This metric reveals the sub-linear scaling advantage:
    - Batch 1: 1 / (0.025 * 20) = 2.0 req/s
    - Batch 32: 32 / (0.4013 * 20) = 3.99 req/s (2x throughput!)
    
    Args:
        batch_size: Number of requests in the batch
        
    Returns:
        Throughput in requests per second
    """
    total_time = get_total_batch_time(batch_size)
    return batch_size / total_time

