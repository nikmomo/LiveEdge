"""Simulation module for LiveEdge.

This module provides streaming simulation for testing adaptive sampling systems.
"""

from liveedge.simulation.environment import (
    SimulationConfig,
    SimulationResult,
    StepResult,
    StreamingSimulator,
)
from liveedge.simulation.runner import (
    ExperimentResult,
    ExperimentRunner,
)

__all__ = [
    # Environment
    "SimulationConfig",
    "StepResult",
    "SimulationResult",
    "StreamingSimulator",
    # Runner
    "ExperimentResult",
    "ExperimentRunner",
]
