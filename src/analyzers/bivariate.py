"""Module for bivariate analysis in EDA."""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.preprocessing import LabelEncoder

@dataclass
class NumericBivariateStats:
    """Statistics for a numerical feature vs target."""
    mean_diff: float
    median_diff: float
    cohen_d: float
    mutual_info: float

@dataclass
class CategoricalBivariateStats:
    """Statistics for a categorical feature vs target."""
    target_rate_per_category: Dict[Any, float]
    chi2_score: float
    mutual_info: float

class BivariateAnalyzer:
    """Analyzes relationships between features and the target."""

    @staticmethod
    def analyze_numeric(df: pd.DataFrame, target_column: str, columns: List[str] = None) -> Dict[str, NumericBivariateStats]:
        """Analyzes numerical features vs target.

        Args:
            df (pd.DataFrame): The dataset.
            target_column (str): The target column name.
            columns (List[str], optional): Columns to analyze.

        Returns:
            Dict[str, NumericBivariateStats]: Analysis results.
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in columns:
                columns.remove(target_column)

        results = {}
        target = df[target_column]
        
        for col in columns:
            data = df[[col, target_column]].dropna()
            group0 = data[data[target_column] == 0][col]
            group1 = data[data[target_column] == 1][col]
            
            if len(group0) == 0 or len(group1) == 0:
                continue

            mean_diff = float(group1.mean() - group0.mean())
            median_diff = float(group1.median() - group0.median())
            
            # Cohen's D
            n1, n2 = len(group0), len(group1)
            var1, var2 = group0.var(), group1.var()
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            cohen_d = mean_diff / pooled_std if pooled_std != 0 else 0.0
            
            # Mutual Information (using a sample if too large for speed)
            sample_df = data.sample(min(20000, len(data)))
            mi = float(mutual_info_classif(sample_df[[col]], sample_df[target_column])[0])
            
            results[col] = NumericBivariateStats(
                mean_diff=mean_diff,
                median_diff=median_diff,
                cohen_d=cohen_d,
                mutual_info=mi
            )
        return results

    @staticmethod
    def analyze_categorical(df: pd.DataFrame, target_column: str, columns: List[str] = None) -> Dict[str, CategoricalBivariateStats]:
        """Analyzes categorical features vs target.

        Args:
            df (pd.DataFrame): The dataset.
            target_column (str): The target column name.
            columns (List[str], optional): Columns to analyze.

        Returns:
            Dict[str, CategoricalBivariateStats]: Analysis results.
        """
        if columns is None:
            columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

        results = {}
        for col in columns:
            data = df[[col, target_column]].dropna()
            
            # Target rate per category
            target_rates = data.groupby(col)[target_column].mean().to_dict()
            
            # Label encode for MI and Chi2
            le = LabelEncoder()
            X = le.fit_transform(data[col].astype(str)).reshape(-1, 1)
            y = data[target_column]
            
            # Mutual Info
            sample_indices = np.random.choice(len(X), min(20000, len(X)), replace=False)
            mi = float(mutual_info_classif(X[sample_indices], y.iloc[sample_indices], discrete_features=True)[0])
            
            # Chi2
            c2_score, _ = chi2(X, y)
            
            results[col] = CategoricalBivariateStats(
                target_rate_per_category=target_rates,
                chi2_score=float(c2_score[0]),
                mutual_info=mi
            )
        return results
