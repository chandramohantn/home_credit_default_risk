"""Module for univariate analysis in EDA."""

import pandas as pd
import numpy as np
from typing import List, Dict

from .schema import CategoricalUnivariateStats, NumericalUnivariateStats

class UnivariateAnalyzer:
    """Analyzes features individually."""

    @staticmethod
    def analyze_numeric(df: pd.DataFrame, columns: List[str] = None) -> Dict[str, NumericalUnivariateStats]:
        """Calculates statistics for numerical columns.

        Args:
            df (pd.DataFrame): The dataset.
            columns (List[str], optional): Columns to analyze. If None, all 
                numeric columns are used.

        Returns:
            Dict[str, NumericalUnivariateStats]: Mapping of feature names 
                to their statistics.
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        results = {}
        for col in columns:
            series = df[col].dropna()
            results[col] = NumericalUnivariateStats(
                mean=float(series.mean()),
                median=float(series.median()),
                std=float(series.std()),
                min=float(series.min()),
                max=float(series.max()),
                skew=float(series.skew()),
                kurtosis=float(series.kurtosis())
            )
        return results

    @staticmethod
    def analyze_categorical(
        df: pd.DataFrame, 
        columns: List[str] = None, 
        rare_threshold: float = 1.0
    ) -> Dict[str, CategoricalUnivariateStats]:
        """Calculates statistics for categorical columns.

        Args:
            df (pd.DataFrame): The dataset.
            columns (List[str], optional): Columns to analyze.
            rare_threshold (float): Percentage threshold for rare categories.

        Returns:
            Dict[str, CategoricalUnivariateStats]: Mapping of feature names 
                to their statistics.
        """
        if columns is None:
            columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        results = {}
        total = len(df)
        for col in columns:
            counts = df[col].value_counts()
            cardinality = len(counts)
            top_cats = counts.head(10).to_dict()
            
            rare_cats = counts[counts / total * 100 < rare_threshold].index.tolist()
            
            results[col] = CategoricalUnivariateStats(
                cardinality=cardinality,
                top_categories=top_cats,
                rare_categories=rare_cats,
                rare_threshold=rare_threshold
            )
        return results
