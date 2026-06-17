"""Module for missing value pattern analysis in EDA."""

import pandas as pd
import numpy as np
from typing import List

class MissingPatternAnalyzer:
    """Analyzes missingness patterns and correlations."""

    @staticmethod
    def calculate_missing_correlations(df: pd.DataFrame) -> pd.DataFrame:
        """Calculates correlations between missingness of different features.

        Args:
            df (pd.DataFrame): The dataset.

        Returns:
            pd.DataFrame: Correlation matrix of null indicators.
        """
        null_df = df.isnull().astype(int)
        # Filter out columns with no missing values (std=0)
        null_df = null_df.loc[:, null_df.std() > 0]
        return null_df.corr()

    @staticmethod
    def analyze_missing_vs_target(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Analyzes if missingness is related to the target variable.

        Args:
            df (pd.DataFrame): The dataset.
            target_column (str): The target column.

        Returns:
            pd.DataFrame: Target rate for missing vs non-missing entries.
        """
        results = []
        for col in df.columns:
            if col == target_column:
                continue
            
            is_null = df[col].isnull()
            if is_null.any() and not is_null.all():
                target_rate_null = df.loc[is_null, target_column].mean()
                target_rate_not_null = df.loc[~is_null, target_column].mean()
                results.append({
                    "feature": col,
                    "target_rate_missing": target_rate_null,
                    "target_rate_present": target_rate_not_null,
                    "diff": target_rate_null - target_rate_not_null
                })
        
        return pd.DataFrame(results).sort_values("diff", ascending=False)
