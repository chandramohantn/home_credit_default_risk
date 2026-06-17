"""
Visualization module for data profiling.

This package provides visual tools to interpret data profiling results,
including missing values, target distribution, and correlations.

The __init__.py file simplifies the package's public API by exposing core 
visualizers directly at the package level, allowing for cleaner imports 
like 'from visualization import TargetVisualizer'.
"""

from .missing_values import MissingValueVisualizer
from .target_analysis import TargetVisualizer
from .correlations import CorrelationVisualizer
