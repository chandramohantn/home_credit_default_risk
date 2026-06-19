"""Data leakage detection validation module.

This module provides the LeakageValidator, which scans the dataset prior to training 
to detect potential variables that introduce target or data leakage.

Data leakage occurs when target information is implicitly encoded in training features, 
which artificially inflates validation performance but leads to model failure in production.
The validator checks for high cardinality columns that serve as primary keys (like ID cols) 
and features with suspiciously high mathematical correlation with the target.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any

class LeakageValidator:
    """Validator class to scan features for data leakage indicators.

    Attributes:
        target_column (str): The column name representing the target label.
        cardinality_threshold (float): Cardinality ratio threshold above which integer/string 
            columns are marked as primary key ID-like columns. Defaults to 0.95.
        correlation_threshold (float): Pearson correlation threshold above which 
            numerical columns are marked as suspicious. Defaults to 0.95.
    """
    
    def __init__(self, 
                 target_column: str, 
                 cardinality_threshold: float = 0.95, 
                 correlation_threshold: float = 0.95):
        """Initializes the LeakageValidator.

        Args:
            target_column (str): Target label column name.
            cardinality_threshold (float): Cardinality warning limit. Defaults to 0.95.
            correlation_threshold (float): Target correlation warning limit. Defaults to 0.95.
        """
        self.target_column = target_column
        self.cardinality_threshold = cardinality_threshold
        self.correlation_threshold = correlation_threshold

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyzes features to flag potential data leakage occurrences.

        Identifies:
        - Columns that act like primary keys (high ratio of unique values to rows).
        - Columns that are near-constant (offering no predictive power).
        - Numerical columns with extremely high absolute correlation with target (>= 0.95).

        Args:
            df (pd.DataFrame): Training DataFrame (including target column).

        Returns:
            dict: Leakage validation findings containing keys:
                - 'id_like_columns' (list of str)
                - 'highly_correlated_columns' (dict: col -> correlation_coefficient)
                - 'near_constant_columns' (list of str)
        """
        report = {
            "id_like_columns": [],
            "highly_correlated_columns": {},
            "near_constant_columns": []
        }
        
        # 1. Verification of Target Variable Presence
        if self.target_column not in df.columns:
            return report

        total_rows = len(df)
        if total_rows == 0:
            return report

        target_series = df[self.target_column]

        # 2. Iterate columns to analyze statistical anomalies
        for col in df.columns:
            if col == self.target_column:
                continue

            series = df[col]
            
            # Near-constant columns (low variance)
            # Why: Columns with single values contain no information and can cause issues for some models
            if series.nunique(dropna=True) <= 1:
                report["near_constant_columns"].append(col)
                continue

            # ID-like columns (high cardinality)
            # Why: Primary keys (like SK_ID_CURR) can lead to overfitting if tree models split on them
            if pd.api.types.is_integer_dtype(series) or pd.api.types.is_object_dtype(series):
                unique_ratio = series.nunique(dropna=True) / total_rows
                if unique_ratio > self.cardinality_threshold:
                    report["id_like_columns"].append(col)

            # Highly correlated columns
            # Why: Extreme correlation (e.g. >0.95) usually indicates that the feature is a proxy for target,
            # or recorded retrospectively after the default event occurred.
            if pd.api.types.is_numeric_dtype(series):
                try:
                    # Ignore null values during correlation calculation to prevent NaN results
                    valid_mask = series.notnull() & target_series.notnull()
                    if valid_mask.sum() > 1:
                        correlation = float(series[valid_mask].corr(target_series[valid_mask]))
                        if abs(correlation) >= self.correlation_threshold:
                            report["highly_correlated_columns"][col] = correlation
                except Exception:
                    pass  # Pass if calculation raises an exception (e.g. standard deviation is 0)

        return report
