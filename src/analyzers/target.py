"""Module for advanced target variable analysis in EDA."""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TargetEDAResult:
    """Dataclass for target analysis results."""
    counts: Dict[Any, int]
    percentages: Dict[Any, float]
    baseline_accuracy: float
    baseline_precision: float
    baseline_recall: float
    baseline_f1: float

class TargetAnalyzer:
    """Analyzes the target variable distribution and baseline metrics."""

    @staticmethod
    def analyze(df: pd.DataFrame, target_column: str) -> TargetEDAResult:
        """Calculates target distribution and majority class baseline metrics.

        Args:
            df (pd.DataFrame): The dataset containing the target.
            target_column (str): Name of the target column.

        Returns:
            TargetEDAResult: Calculated distribution and baseline metrics.
        """
        counts = df[target_column].value_counts().to_dict()
        total = len(df)
        percentages = {k: (v / total) * 100 for k, v in counts.items()}
        
        # Majority class baseline metrics (predicting majority for everyone)
        majority_class = max(counts, key=counts.get)
        majority_count = counts[majority_class]
        
        baseline_accuracy = (majority_count / total) * 100
        
        # For minority class (assuming 1 is positive if binary)
        # If we predict 0 for everyone:
        # TP = 0, FP = 0, TN = majority_count, FN = minority_count
        baseline_precision = 0.0
        baseline_recall = 0.0
        baseline_f1 = 0.0
        
        return TargetEDAResult(
            counts=counts,
            percentages=percentages,
            baseline_accuracy=baseline_accuracy,
            baseline_precision=baseline_precision,
            baseline_recall=baseline_recall,
            baseline_f1=baseline_f1
        )
