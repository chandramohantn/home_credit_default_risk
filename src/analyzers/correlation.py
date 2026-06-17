"""Module for correlation and multicollinearity analysis in EDA."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from statsmodels.stats.outliers_influence import variance_inflation_factor

class CorrelationAnalyzer:
    """Analyzes correlations and multicollinearity."""

    @staticmethod
    def analyze_correlations(df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
        """Calculates correlation matrix for numeric features.

        Args:
            df (pd.DataFrame): The dataset.
            method (str): Correlation method ('pearson', 'spearman').

        Returns:
            pd.DataFrame: Correlation matrix.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        return numeric_df.corr(method=method)

    @staticmethod
    def calculate_vif(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """Calculates Variance Inflation Factor (VIF).

        Args:
            df (pd.DataFrame): The dataset.
            columns (List[str], optional): Columns to analyze.

        Returns:
            pd.DataFrame: VIF scores.
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        X = df[columns].dropna()
        # Add constant for VIF
        X_const = X.assign(const=1)
        
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_const.columns
        vif_data["VIF"] = [
            variance_inflation_factor(X_const.values, i) 
            for i in range(len(X_const.columns))
        ]
        return vif_data[vif_data["feature"] != "const"]
