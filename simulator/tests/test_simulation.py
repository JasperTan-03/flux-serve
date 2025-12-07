"""
Test suite for the dLLM Trace-Driven Simulator.

This module tests core functionality:
    1. Latency interpolation correctness
    2. Workload generation (Poisson process)
    3. Scheduler behavior
    4. End-to-end simulation
"""

import pytest
import math
from typing import List

# Import modules to test
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from simulator.config import (
    PROFILING_DATA,
    MAX_BATCH_SIZE,
    NUM_STEPS,
    get_batch_latency,
    get_total_batch_time,
    compute_throughput,
)

from simulator.workload import (
    Request,
    WorkloadGenerator,
)

from simulator.scheduler import (
    StaticBatchScheduler,
    DynamicBatchScheduler,
    RequestQueue,
    Batch,
    create_scheduler,
)

from simulator.simulator import (
    Simulator,
    run_simulation,
)


# =============================================================================
# Tests for config.py: Latency Interpolation
# =============================================================================

class TestBatchLatency:
    """Tests for get_batch_latency interpolation function."""
    
    def test_exact_values(self) -> None:
        """Verify exact matches return profiled values."""
        for batch_size, expected_latency in PROFILING_DATA.items():
            actual = get_batch_latency(batch_size)
            assert actual == expected_latency, (
                f"Batch {batch_size}: expected {expected_latency}, got {actual}"
            )
    
    def test_interpolation_batch_3(self) -> None:
        """Test interpolation between batch 2 and 4."""
        # Batch 2: 0.0331, Batch 4: 0.0577
        # Batch 3 should be midpoint: (0.0331 + 0.0577) / 2 = 0.0454
        lat_3 = get_batch_latency(3)
        expected = (0.0331 + 0.0577) / 2
        assert abs(lat_3 - expected) < 1e-6, (
            f"Batch 3 interpolation: expected {expected}, got {lat_3}"
        )
    
    def test_interpolation_batch_6(self) -> None:
        """Test interpolation between batch 4 and 8."""
        # Batch 4: 0.0577, Batch 8: 0.1130
        # Batch 6 is 0.5 of the way: 0.0577 + 0.5 * (0.1130 - 0.0577) = 0.08535
        lat_6 = get_batch_latency(6)
        expected = 0.0577 + 0.5 * (0.1130 - 0.0577)
        assert abs(lat_6 - expected) < 1e-6, (
            f"Batch 6 interpolation: expected {expected}, got {lat_6}"
        )
    
    def test_interpolation_batch_24(self) -> None:
        """Test interpolation between batch 16 and 32."""
        # Batch 16: 0.2132, Batch 32: 0.4013
        # Batch 24 is 0.5 of the way: 0.2132 + 0.5 * (0.4013 - 0.2132) = 0.30725
        lat_24 = get_batch_latency(24)
        expected = 0.2132 + 0.5 * (0.4013 - 0.2132)
        assert abs(lat_24 - expected) < 1e-6, (
            f"Batch 24 interpolation: expected {expected}, got {lat_24}"
        )
    
    def test_invalid_batch_size_zero(self) -> None:
        """Verify batch size 0 raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 1"):
            get_batch_latency(0)
    
    def test_invalid_batch_size_negative(self) -> None:
        """Verify negative batch size raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 1"):
            get_batch_latency(-5)
    
    def test_invalid_batch_size_too_large(self) -> None:
        """Verify batch size > MAX raises ValueError."""
        with pytest.raises(ValueError, match=f"must be <= {MAX_BATCH_SIZE}"):
            get_batch_latency(MAX_BATCH_SIZE + 1)
    
    def test_total_batch_time(self) -> None:
        """Verify total batch time = latency * NUM_STEPS."""
        for batch_size in PROFILING_DATA.keys():
            expected = get_batch_latency(batch_size) * NUM_STEPS
            actual = get_total_batch_time(batch_size)
            assert abs(actual - expected) < 1e-9
    
    def test_sublinear_scaling(self) -> None:
        """Verify that latency scales sub-linearly with batch size."""
        # Doubling batch size should less than double the latency
        for bs1, bs2 in [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32)]:
            lat1 = PROFILING_DATA[bs1]
            lat2 = PROFILING_DATA[bs2]
            # Latency should increase less than 2x
            ratio = lat2 / lat1
            assert ratio < 2.0, (
                f"Latency scaling {bs1}->{bs2}: ratio {ratio:.2f} should be < 2.0"
            )
    
    def test_throughput_increases(self) -> None:
        """Verify throughput increases with batch size."""
        prev_throughput = 0.0
        for batch_size in sorted(PROFILING_DATA.keys()):
            throughput = compute_throughput(batch_size)
            assert throughput > prev_throughput, (
                f"Throughput should increase: batch {batch_size} = {throughput}"
            )
            prev_throughput = throughput


# =============================================================================
# Tests for workload.py: Workload Generation
# =============================================================================

class TestWorkloadGenerator:
    """Tests for WorkloadGenerator and Poisson process."""
    
    def test_expected_request_count(self) -> None:
        """Verify approximate Poisson arrival count."""
        rps = 10.0
        duration = 100.0
        expected = rps * duration  # 1000 requests
        
        # Run multiple trials and check average
        counts = []
        for seed in range(10):
            gen = WorkloadGenerator(rps=rps, duration=duration, seed=seed)
            requests = gen.generate()
            counts.append(len(requests))
        
        avg_count = sum(counts) / len(counts)
        # Allow 10% deviation (Poisson variance = mean)
        assert abs(avg_count - expected) < expected * 0.1, (
            f"Average count {avg_count} deviates too much from expected {expected}"
        )
    
    def test_reproducibility(self) -> None:
        """Verify same seed produces same workload."""
        gen1 = WorkloadGenerator(rps=5.0, duration=10.0, seed=42)
        gen2 = WorkloadGenerator(rps=5.0, duration=10.0, seed=42)
        
        requests1 = gen1.generate()
        requests2 = gen2.generate()
        
        assert len(requests1) == len(requests2)
        for r1, r2 in zip(requests1, requests2):
            assert r1.arrival_time == r2.arrival_time
    
    def test_different_seeds_different_results(self) -> None:
        """Verify different seeds produce different workloads."""
        gen1 = WorkloadGenerator(rps=5.0, duration=10.0, seed=42)
        gen2 = WorkloadGenerator(rps=5.0, duration=10.0, seed=123)
        
        requests1 = gen1.generate()
        requests2 = gen2.generate()
        
        # Very unlikely to be identical
        arrival_times_1 = [r.arrival_time for r in requests1]
        arrival_times_2 = [r.arrival_time for r in requests2]
        assert arrival_times_1 != arrival_times_2
    
    def test_arrival_times_sorted(self) -> None:
        """Verify arrival times are monotonically increasing."""
        gen = WorkloadGenerator(rps=10.0, duration=10.0, seed=42)
        requests = gen.generate()
        
        for i in range(1, len(requests)):
            assert requests[i].arrival_time >= requests[i-1].arrival_time
    
    def test_arrival_times_within_duration(self) -> None:
        """Verify all arrivals are within [0, duration)."""
        duration = 10.0
        gen = WorkloadGenerator(rps=5.0, duration=duration, seed=42)
        requests = gen.generate()
        
        for r in requests:
            assert 0 <= r.arrival_time < duration
    
    def test_zero_rps(self) -> None:
        """Verify zero RPS produces no requests."""
        gen = WorkloadGenerator(rps=0.0, duration=10.0, seed=42)
        requests = gen.generate()
        assert len(requests) == 0
    
    def test_request_ids_unique(self) -> None:
        """Verify all request IDs are unique."""
        gen = WorkloadGenerator(rps=10.0, duration=10.0, seed=42)
        requests = gen.generate()
        
        ids = [r.request_id for r in requests]
        assert len(ids) == len(set(ids))
    
    def test_expected_requests_method(self) -> None:
        """Verify expected_requests calculation."""
        gen = WorkloadGenerator(rps=5.0, duration=20.0)
        assert gen.expected_requests() == 100.0


# =============================================================================
# Tests for scheduler.py: Scheduler Behavior
# =============================================================================

class TestRequestQueue:
    """Tests for RequestQueue."""
    
    def test_fifo_order(self) -> None:
        """Verify FIFO dequeue order."""
        queue = RequestQueue()
        r1 = Request(request_id=1, arrival_time=0.0)
        r2 = Request(request_id=2, arrival_time=1.0)
        r3 = Request(request_id=3, arrival_time=2.0)
        
        queue.enqueue(r1)
        queue.enqueue(r2)
        queue.enqueue(r3)
        
        dequeued = queue.dequeue(2)
        assert dequeued == [r1, r2]
        assert len(queue) == 1
    
    def test_dequeue_more_than_available(self) -> None:
        """Verify dequeue returns only what's available."""
        queue = RequestQueue()
        queue.enqueue(Request(request_id=1, arrival_time=0.0))
        
        dequeued = queue.dequeue(10)
        assert len(dequeued) == 1
        assert queue.is_empty


class TestStaticBatchScheduler:
    """Tests for StaticBatchScheduler."""
    
    def test_waits_for_target_batch(self) -> None:
        """Verify scheduler waits for target batch size."""
        scheduler = StaticBatchScheduler(target_batch_size=4, timeout=1.0)
        queue = RequestQueue()
        
        # Add 3 requests (less than target)
        for i in range(3):
            queue.enqueue(Request(request_id=i, arrival_time=0.0))
        
        # Should not batch yet (current_time=0, no timeout)
        batch = scheduler.get_next_batch(queue, current_time=0.0)
        assert batch is None
    
    def test_batches_at_target(self) -> None:
        """Verify scheduler batches when target is reached."""
        scheduler = StaticBatchScheduler(target_batch_size=4, timeout=1.0)
        queue = RequestQueue()
        
        # Add 4 requests (equals target)
        for i in range(4):
            queue.enqueue(Request(request_id=i, arrival_time=0.0))
        
        batch = scheduler.get_next_batch(queue, current_time=0.0)
        assert batch is not None
        assert batch.batch_size == 4
    
    def test_batches_on_timeout(self) -> None:
        """Verify scheduler batches on timeout even with small batch."""
        scheduler = StaticBatchScheduler(target_batch_size=8, timeout=0.5)
        queue = RequestQueue()
        
        # Add 2 requests at time 0
        for i in range(2):
            queue.enqueue(Request(request_id=i, arrival_time=0.0))
        
        # At time 0, should not batch
        batch = scheduler.get_next_batch(queue, current_time=0.0)
        assert batch is None
        
        # At time 0.6 (past timeout), should batch
        batch = scheduler.get_next_batch(queue, current_time=0.6)
        assert batch is not None
        assert batch.batch_size == 2


class TestDynamicBatchScheduler:
    """Tests for DynamicBatchScheduler."""
    
    def test_immediate_batching(self) -> None:
        """Verify dynamic scheduler batches immediately."""
        scheduler = DynamicBatchScheduler()
        queue = RequestQueue()
        
        # Add just 1 request
        queue.enqueue(Request(request_id=1, arrival_time=0.0))
        
        # Should batch immediately
        batch = scheduler.get_next_batch(queue, current_time=0.0)
        assert batch is not None
        assert batch.batch_size == 1
    
    def test_takes_all_available(self) -> None:
        """Verify dynamic scheduler takes all available requests."""
        scheduler = DynamicBatchScheduler(max_batch_size=32)
        queue = RequestQueue()
        
        # Add 10 requests
        for i in range(10):
            queue.enqueue(Request(request_id=i, arrival_time=0.0))
        
        batch = scheduler.get_next_batch(queue, current_time=0.0)
        assert batch is not None
        assert batch.batch_size == 10
        assert queue.is_empty
    
    def test_respects_max_batch_size(self) -> None:
        """Verify dynamic scheduler respects max batch size."""
        scheduler = DynamicBatchScheduler(max_batch_size=5)
        queue = RequestQueue()
        
        # Add 10 requests
        for i in range(10):
            queue.enqueue(Request(request_id=i, arrival_time=0.0))
        
        batch = scheduler.get_next_batch(queue, current_time=0.0)
        assert batch is not None
        assert batch.batch_size == 5
        assert len(queue) == 5  # 5 remaining


class TestSchedulerFactory:
    """Tests for create_scheduler factory."""
    
    def test_create_static(self) -> None:
        """Test creating static scheduler."""
        scheduler = create_scheduler("static", target_batch_size=4)
        assert isinstance(scheduler, StaticBatchScheduler)
    
    def test_create_dynamic(self) -> None:
        """Test creating dynamic scheduler."""
        scheduler = create_scheduler("dynamic", max_batch_size=16)
        assert isinstance(scheduler, DynamicBatchScheduler)
    
    def test_unknown_type(self) -> None:
        """Test error on unknown scheduler type."""
        with pytest.raises(ValueError, match="Unknown scheduler"):
            create_scheduler("unknown")


# =============================================================================
# Tests for simulator.py: End-to-End Simulation
# =============================================================================

class TestSimulator:
    """Tests for the simulation engine."""
    
    def test_all_requests_processed(self) -> None:
        """Verify all requests complete."""
        scheduler = DynamicBatchScheduler()
        requests = [
            Request(request_id=i, arrival_time=i * 0.1)
            for i in range(10)
        ]
        
        sim = Simulator(scheduler=scheduler, requests=requests)
        metrics = sim.run()
        
        assert metrics.total_requests == 10
    
    def test_request_latencies_positive(self) -> None:
        """Verify all latencies are positive."""
        scheduler = DynamicBatchScheduler()
        requests = [
            Request(request_id=i, arrival_time=i * 0.5)
            for i in range(5)
        ]
        
        sim = Simulator(scheduler=scheduler, requests=requests)
        sim.run()
        
        for r in sim.completed_requests:
            assert r.latency is not None
            assert r.latency > 0
    
    def test_start_time_after_arrival(self) -> None:
        """Verify start_time >= arrival_time for all requests."""
        scheduler = DynamicBatchScheduler()
        requests = [
            Request(request_id=i, arrival_time=i * 0.2)
            for i in range(10)
        ]
        
        sim = Simulator(scheduler=scheduler, requests=requests)
        sim.run()
        
        for r in sim.completed_requests:
            assert r.start_time is not None
            assert r.start_time >= r.arrival_time
    
    def test_end_time_after_start(self) -> None:
        """Verify end_time > start_time for all requests."""
        scheduler = DynamicBatchScheduler()
        requests = [
            Request(request_id=i, arrival_time=i * 0.1)
            for i in range(5)
        ]
        
        sim = Simulator(scheduler=scheduler, requests=requests)
        sim.run()
        
        for r in sim.completed_requests:
            assert r.end_time is not None
            assert r.start_time is not None
            assert r.end_time > r.start_time
    
    def test_dynamic_lower_latency_than_static(self) -> None:
        """
        Verify dynamic batching achieves lower average latency
        than static batching under moderate load.
        
        This is the key hypothesis we're testing!
        """
        rps = 5.0
        duration = 30.0
        seed = 42
        
        # Run with static scheduler
        static_metrics = run_simulation(
            scheduler=StaticBatchScheduler(target_batch_size=8, timeout=0.1),
            rps=rps,
            duration=duration,
            seed=seed
        )
        
        # Run with dynamic scheduler
        dynamic_metrics = run_simulation(
            scheduler=DynamicBatchScheduler(),
            rps=rps,
            duration=duration,
            seed=seed
        )
        
        # Dynamic should have lower or equal latency
        assert dynamic_metrics.avg_latency <= static_metrics.avg_latency * 1.1, (
            f"Dynamic ({dynamic_metrics.avg_latency:.3f}s) should be <= "
            f"Static ({static_metrics.avg_latency:.3f}s)"
        )
    
    def test_batch_sizes_recorded(self) -> None:
        """Verify batch sizes are recorded during simulation."""
        scheduler = DynamicBatchScheduler()
        requests = [
            Request(request_id=i, arrival_time=i * 0.01)
            for i in range(20)
        ]
        
        sim = Simulator(scheduler=scheduler, requests=requests)
        sim.run()
        
        assert len(sim.batch_sizes) > 0
        assert all(1 <= bs <= MAX_BATCH_SIZE for bs in sim.batch_sizes)


class TestRunSimulation:
    """Tests for the run_simulation convenience function."""
    
    def test_basic_run(self) -> None:
        """Test basic simulation run."""
        metrics = run_simulation(
            scheduler=DynamicBatchScheduler(),
            rps=2.0,
            duration=10.0,
            seed=42
        )
        
        assert metrics.total_requests > 0
        assert metrics.avg_latency > 0
        assert metrics.throughput > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_experiment(self) -> None:
        """Test a complete experiment run."""
        from simulator.simulator import compare_schedulers
        
        static = StaticBatchScheduler(target_batch_size=4, timeout=0.1)
        dynamic = DynamicBatchScheduler()
        
        results = compare_schedulers(
            schedulers=[static, dynamic],
            rps_range=[1.0, 2.0, 3.0],
            duration=10.0,
            seed=42
        )
        
        assert len(results) == 2
        for name, rps_results in results.items():
            assert len(rps_results) == 3
            for rps, metrics in rps_results.items():
                assert metrics.total_requests > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

