"""LiveEdge: Energy-efficient livestock behavior monitoring with adaptive sampling.

This package provides tools for:
- Preprocessing and feature extraction from accelerometer data
- Behavior classification using traditional ML and deep learning models
- Adaptive sampling policies for energy optimization
- Energy consumption modeling for edge devices
- End-to-end simulation and evaluation
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("liveedge")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
