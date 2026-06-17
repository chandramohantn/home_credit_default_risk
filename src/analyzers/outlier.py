"""Module for outlier analysis in EDA."""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class OutlierStats:
    """Statistics for outliers in a feature."""
    iqr_outliers_count: int
    zscore_outliers_count: int
    outlier_indices: List[int]

class OutlierAnalyzer:
    """Detects outliers using IQR and Z-score methods."""

    @staticmethod
    def analyze(df: pd.DataFrame, columns: List[str] = None, z_threshold: float = 3.0) -> Dict[str, OutlierStats]:
        """Identifies outliers in numerical features.

        Args:
            df (pd.DataFrame): The dataset.
            columns (List[str], optional): Columns to analyze.
            z_threshold (float): Z-score threshold for outliers.

        Returns:
            Dict[str, OutlierStats]: Outlier statistics per feature.
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        results = {}
        for col in columns:
            series = df[col].dropna()
            
            # IQR Method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = series[(series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))]
            
            # Z-score Method
            z_scores = (series - series.mean()) / series.std()
            z_outliers = series[np.abs(z_scores) > z_threshold]
            
            # Combined unique indices
            outlier_indices = list(set(iqr_outliers.index) | set(z_outliers.index))
            
            results[col] = OutlierStats(
                iqr_outliers_count=len(iqr_outliers),
                zscore_outliers_count=len(z_outliers),
                outlier_indices=outlier_indices
            )
        return results
