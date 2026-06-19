"""Rare category handling transformer.

This module provides the RareCategoryTransformer, which groups low-frequency/sparse 
categorical levels into a standard grouping level (e.g., 'OTHER').

In credit risk datasets, high-cardinality features (like occupation type or organization 
type) often contain rare labels with very few samples (e.g., 2 or 3 records out of 300,000). 
Grouping these levels prevents overfitting, reduces encoding size (specifically one-hot size), 
and makes the models robust against unseen categories during deployment.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Any, Optional
from src.preprocessing.transformers.base import BaseTransformer

class RareCategoryTransformer(BaseTransformer):
    """Groups rare labels inside categorical columns into a designated fallback value.

    Supports thresholds based on minimum percentage frequency, exact count, 
    or keeping only the top K categories.

    Attributes:
        threshold_pct (float): Percentage threshold below which categories are grouped.
        threshold_count (int, optional): Minimum exact occurrences required to keep a category.
        top_k (int, optional): Max number of categories to retain per column.
        fill_value (str): Label used for grouped rare categories (e.g., 'OTHER').
        exclude_cols (list of str): List of columns to skip.
        frequent_categories_ (dict): Stored list of frequent levels per column learned in fit.
        feature_names_in_ (list of str): Input feature names seen in fit.
        feature_names_out_ (list of str): Output feature names (matches input).
    """
    
    def __init__(self,
                 threshold_pct: float = 0.01,
                 threshold_count: Optional[int] = None,
                 top_k: Optional[int] = None,
                 fill_value: str = "OTHER",
                 exclude_cols: Optional[List[str]] = None):
        """Initializes the RareCategoryTransformer.

        Args:
            threshold_pct (float): Level frequency ratio threshold (e.g. 0.01 matches < 1% occurrence).
            threshold_count (int, optional): Minimum counts threshold. Takes precedence over threshold_pct if specified.
            top_k (int, optional): Keep only the top K categories. Takes precedence if specified.
            fill_value (str): Label assigned to rare classes. Defaults to "OTHER".
            exclude_cols (list of str, optional): List of columns to ignore.
        """
        super().__init__()
        self.threshold_pct = threshold_pct
        self.threshold_count = threshold_count
        self.top_k = top_k
        self.fill_value = fill_value
        self.exclude_cols = exclude_cols or []

        # Fitted parameters
        self.frequent_categories_: Dict[str, List[Any]] = {}
        self.feature_names_in_: List[str] = []
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Identifies frequent categories in categorical columns.

        Args:
            X (pd.DataFrame): Input DataFrame to learn frequencies from.
            y (pd.Series, optional): Target series. Defaults to None.

        Returns:
            RareCategoryTransformer: The fitted instance of the transformer.

        Raises:
            TypeError: If X is not a pandas DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        self.feature_names_in_ = list(X.columns)
        self.frequent_categories_ = {}

        total_rows = len(X)

        for col in X.columns:
            if col in self.exclude_cols:
                continue

            # Why: Apply only on string or category columns
            is_cat = isinstance(X[col].dtype, pd.CategoricalDtype) or X[col].dtype == "object"
            if not is_cat:
                continue

            # Compute levels frequencies
            value_counts = X[col].value_counts(dropna=True)
            freqs = value_counts / total_rows

            frequent = []
            
            # Select frequent levels based on strategy ordering: top_k > count > pct
            if self.top_k is not None:
                frequent = list(value_counts.head(self.top_k).index)
            elif self.threshold_count is not None:
                frequent = list(value_counts[value_counts >= self.threshold_count].index)
            else:
                frequent = list(freqs[freqs >= self.threshold_pct].index)

            self.frequent_categories_[col] = frequent

        self.feature_names_out_ = self.feature_names_in_
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replaces low frequency categories with the fill_value.

        Args:
            X (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame with rare levels grouped.

        Raises:
            TypeError: If X is not a pandas DataFrame.
        """
        super().transform(X)
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        df = X.copy()

        for col in self.frequent_categories_:
            if col in df.columns:
                frequent = self.frequent_categories_[col]
                
                # Check for category dtype to add 'fill_value' category safely without causing NaN conversion
                is_categorical_dtype = isinstance(df[col].dtype, pd.CategoricalDtype)
                
                if is_categorical_dtype:
                    categories = list(df[col].dtype.categories)
                    if self.fill_value not in categories:
                        df[col] = df[col].cat.add_categories([self.fill_value])

                # Replace values not in frequent list with fill_value.
                # Why: We only replace non-null rare categories so that we don't interfere with 
                # missing value imputation steps.
                mask = ~df[col].isin(frequent) & df[col].notnull()
                df.loc[mask, col] = self.fill_value

                # Remove unused categories to free up memory and prevent categorical schema mismatch
                if is_categorical_dtype:
                    df[col] = df[col].cat.remove_unused_categories()

        return df

    def get_feature_names_out(self, input_features=None):
        """Returns the output feature names list.

        Args:
            input_features (list of str, optional): The input feature names. Defaults to None.

        Returns:
            list of str: Column names.
        """
        return self.feature_names_out_
