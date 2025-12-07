"""
Advanced experiments for the dLLM Trace-Driven Simulator.

This module contains specialized experiments that demonstrate
key systems insights about dLLM scheduling:

    - burst.py: How schedulers react to sudden traffic spikes
    - sensitivity.py: Why static batching is hard to tune

These experiments strengthen the research narrative by showing
that dynamic scheduling is not just faster, but also more robust.
"""

__all__ = ["burst", "sensitivity"]

