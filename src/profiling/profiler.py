"""Main orchestration module for the profiling framework."""

import pandas as pd
from typing import Optional
from .schema import ProfilingResult
from .statistics import DatasetStatistics
from .quality import (
    MissingValueAnalyzer,
    DuplicateAnalyzer,
    ConstantFeatureAnalyzer,
    CardinalityAnalyzer,
    LeakageAnalyzer
)
from .target import TargetAnalyzer

class DataProfiler:
    """Orchestrates multiple analysis components to profile a dataset."""

    def __init__(self, target_column: Optional[str] = None):
        """Initializes the profiler.

        Args:
            target_column (str, optional): The name of the target column for 
                specialized analysis (leakage, target distribution). 
                Defaults to None.
        """
        self.target_column = target_column

    def run(self, df: pd.DataFrame) -> ProfilingResult:
        """Executes all profiling analysis steps on the given DataFrame.

        Args:
            df (pd.DataFrame): The dataset to profile.

        Returns:
            ProfilingResult: A comprehensive dataclass containing results 
                from all analyzers.
        """
        dataset_stats = DatasetStatistics.summarize(df)
        missing_values = MissingValueAnalyzer.analyze(df)
        duplicates = DuplicateAnalyzer.analyze(df)
        constant_features = ConstantFeatureAnalyzer.analyze(df)
        cardinality = CardinalityAnalyzer.analyze(df)
        
        target_analysis = None
        if self.target_column:
            target_analysis = TargetAnalyzer.analyze(df, self.target_column)
            
        leakage = LeakageAnalyzer.analyze(df, self.target_column)

        return ProfilingResult(
            dataset_stats=dataset_stats,
            missing_values=missing_values,
            duplicates=duplicates,
            constant_features=constant_features,
            cardinality=cardinality,
            target_analysis=target_analysis,
            leakage=leakage
        )
