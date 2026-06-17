"""Module for analyzing the target variable distribution and class imbalance."""

import pandas as pd
from .schema import TargetAnalysisReport

class TargetAnalyzer:
    """Analyzes the target variable for class distribution and imbalance."""

    @staticmethod
    def analyze(df: pd.DataFrame, target_column: str) -> TargetAnalysisReport:
        """Calculates target counts, imbalance ratio, and baseline accuracy.

        Args:
            df (pd.DataFrame): The DataFrame to analyze.
            target_column (str): The name of the target column.

        Returns:
            TargetAnalysisReport: Dataclass containing target analysis metrics.

        Raises:
            ValueError: If the target column is not found in the DataFrame.
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        counts = df[target_column].value_counts()
        positive_count = int(counts.get(1, 0))
        negative_count = int(counts.get(0, 0))
        
        imbalance_ratio = negative_count / positive_count if positive_count > 0 else float('inf')
        baseline_accuracy = (max(positive_count, negative_count) / len(df)) * 100

        return TargetAnalysisReport(
            positive_count=positive_count,
            negative_count=negative_count,
            imbalance_ratio=imbalance_ratio,
            baseline_accuracy=baseline_accuracy
        )
