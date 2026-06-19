"""Numerical feature scaling transformer.

This module provides the FeatureScaler, which applies scaling transformations 
(StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer) to numerical features.

Scaling is essential for algorithms sensitive to value scale (like Logistic Regression 
and Neural Networks) to ensure stable optimization and prevent features with larger 
absolute values (like income) from dominating features with smaller scales (like age).
Binary indicator columns are automatically skipped to retain their interpretability.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from src.preprocessing.transformers.base import BaseTransformer

class FeatureScaler(BaseTransformer):
    """Scales numerical features inside a pandas DataFrame.

    Prevents scaling binary indicator columns or non-numeric variables.

    Attributes:
        default_strategy (str): Default scaling method ('standard', 'minmax', 'robust', 'quantile').
        column_strategies (dict): specific strategy overrides per column.
        exclude_cols (list of str): List of columns to bypass scaling.
        scalers_ (dict): Stored mapping of column names to fitted scikit-learn scaling instances.
        feature_names_in_ (list of str): Input feature names seen in fit.
        feature_names_out_ (list of str): Output feature names (matches input).
    """
    
    def __init__(self,
                 default_strategy: str = "standard",
                 column_strategies: Optional[Dict[str, str]] = None,
                 exclude_cols: Optional[List[str]] = None):
        """Initializes the FeatureScaler.

        Args:
            default_strategy (str): Default scaling strategy. Defaults to "standard".
            column_strategies (dict, optional): Custom column strategies overrides.
            exclude_cols (list of str, optional): List of columns to exclude.
        """
        super().__init__()
        self.default_strategy = default_strategy
        self.column_strategies = column_strategies or {}
        self.exclude_cols = exclude_cols or []

        # Fitted parameters
        self.scalers_: Dict[str, Any] = {}
        self.feature_names_in_: List[str] = []
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fits standard, robust, minmax, or quantile scalers to numeric columns.

        Args:
            X (pd.DataFrame): Training DataFrame.
            y (pd.Series, optional): Target variable. Defaults to None.

        Returns:
            FeatureScaler: The fitted instance of the transformer.

        Raises:
            TypeError: If X is not a pandas DataFrame.
            ValueError: If an unknown scaling strategy is specified.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        self.feature_names_in_ = list(X.columns)
        self.scalers_ = {}

        for col in X.columns:
            if col in self.exclude_cols or col == "TARGET" or col == "y":
                continue

            # Scale only numerical columns that are not boolean indicators (0 or 1)
            is_numeric = pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col])
            
            # Why: Skip binary columns (0/1 values only). Scaling binary columns destroys 
            # their direct interpretability and doesn't add value.
            is_binary = X[col].nunique(dropna=True) <= 2 and set(X[col].dropna().unique()).issubset({0, 1, 0.0, 1.0})
            
            if not is_numeric or is_binary:
                continue

            strategy = self.column_strategies.get(col, self.default_strategy)
            
            if strategy == "standard":
                scaler = StandardScaler()
            elif strategy == "minmax":
                scaler = MinMaxScaler()
            elif strategy == "robust":
                # Why: RobustScaler uses median and IQR, making it resilient to outliers.
                scaler = RobustScaler()
            elif strategy == "quantile":
                # Why: QuantileTransformer maps values to a standard normal distribution. 
                # Very effective for Neural Networks to flatten complex, skewed shapes.
                scaler = QuantileTransformer(random_state=42, n_quantiles=min(1000, len(X)))
            elif strategy == "none":
                continue
            else:
                raise ValueError(f"Unknown scaling strategy: {strategy}")

            # Fit scaler on non-null values only to prevent propagation of NaN statistics
            non_nulls = X[[col]].dropna()
            if not non_nulls.empty:
                scaler.fit(non_nulls)
                self.scalers_[col] = scaler

        self.feature_names_out_ = self.feature_names_in_
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scales numerical features using fitted scaling estimators.

        Args:
            X (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame.

        Raises:
            TypeError: If X is not a pandas DataFrame.
        """
        super().transform(X)
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        df = X.copy()

        # Scale only non-null values to preserve NaN properties for downstream operations
        for col, scaler in self.scalers_.items():
            if col in df.columns:
                non_null_mask = df[col].notnull()
                if non_null_mask.any():
                    vals = df.loc[non_null_mask, [col]]
                    scaled_vals = scaler.transform(vals)
                    df.loc[non_null_mask, col] = scaled_vals.ravel()

        return df

    def get_feature_names_out(self, input_features=None):
        """Returns feature output names list.

        Args:
            input_features (list of str, optional): Input names. Defaults to None.

        Returns:
            list of str: Feature names.
        """
        return self.feature_names_out_
