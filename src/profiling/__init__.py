"""
Profiling module for data quality analysis.

This package provides tools for analyzing dataset statistics, missing values,
duplicates, constant features, cardinality, and target distribution.

The __init__.py file simplifies the package's public API by exposing core 
classes directly at the package level, allowing for cleaner imports like 
'from profiling import DataProfiler'.
"""

from .profiler import DataProfiler
from .report import ReportGenerator
from .schema import ProfilingResult, DatasetStats
from .statistics import DatasetStatistics
from .quality import (
    FeatureInspector,
    MissingValueAnalyzer,
    DuplicateAnalyzer,
    ConstantFeatureAnalyzer,
    CardinalityAnalyzer,
    LeakageAnalyzer
)
from .target import TargetAnalyzer
