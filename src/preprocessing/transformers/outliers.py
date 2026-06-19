"""Outlier detection and clipping transformer.

This module provides the OutlierTransformer, which identifies and treats numerical 
outliers using standard methods such as Interquartile Range (IQR), Z-Score, or Winsorization.

Outliers can heavily bias downstream estimators, specifically linear models (Logistic 
Regression) and Neural Networks. The OutlierTransformer calculates the clipping bounds 
during the `fit` phase on the training split, and clips values to these boundaries 
during `transform` on both train and validation splits to prevent target/data leakage.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from src.preprocessing.transformers.base import BaseTransformer

class OutlierTransformer(BaseTransformer):
    """Clips numerical values outside statistical boundaries.

    Attributes:
        strategy (str): Treatment strategy ('clip' or 'none'). Default is 'clip'.
        method (str): Detection method ('iqr', 'z_score', 'winsorize'). Defaults to 'iqr'.
        threshold (float): Threshold multiplier for IQR (usually 1.5) or Z-score (usually 3.0).
        lower_quantile (float): Bottom quantile for Winsorization (default 0.01).
        upper_quantile (float): Top quantile for Winsorization (default 0.99).
        columns (list of str, optional): Specific columns to treat. Defaults to all numericals.
        limits_ (dict): Stored mapping of column names to computed boundaries (lower, upper).
        feature_names_in_ (list of str): Input feature names seen in fit.
        feature_names_out_ (list of str): Output feature names (matches input).
        outlier_counts_ (dict): Tracked number of outliers per column for reporting.
    """
    
    def __init__(self,
                 strategy: str = "clip",
                 method: str = "iqr",
                 threshold: float = 1.5,
                 lower_quantile: float = 0.01,
                 upper_quantile: float = 0.99,
                 columns: Optional[List[str]] = None):
        """Initializes the OutlierTransformer.

        Args:
            strategy (str): Treatment strategy ('clip' or 'none'). Defaults to "clip".
            method (str): Boundary calculation method ('iqr', 'z_score', 'winsorize'). Defaults to "iqr".
            threshold (float): Standard multiplier threshold. Defaults to 1.5.
            lower_quantile (float): Lower percentile for winsorization. Defaults to 0.01.
            upper_quantile (float): Upper percentile for winsorization. Defaults to 0.99.
            columns (list of str, optional): Columns to process. Defaults to None.
        """
        super().__init__()
        self.strategy = strategy  # Row removal is avoided here because it changes output shapes in test pipelines
        self.method = method
        self.threshold = threshold
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.columns = columns

        # Fitted parameters
        self.limits_: Dict[str, Tuple[float, float]] = {}
        self.feature_names_in_: List[str] = []
        self.feature_names_out_: List[str] = []
        self.outlier_counts_: Dict[str, int] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Calculates clipping limits for numerical features.

        Args:
            X (pd.DataFrame): Training DataFrame.
            y (pd.Series, optional): Target series. Defaults to None.

        Returns:
            OutlierTransformer: The fitted instance of the transformer.

        Raises:
            TypeError: If X is not a pandas DataFrame.
            ValueError: If an unknown method is specified.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        self.feature_names_in_ = list(X.columns)
        self.limits_ = {}
        self.outlier_counts_ = {}

        # 1. Determine target columns to treat
        target_cols = self.columns
        if target_cols is None:
            target_cols = [
                col for col in X.columns 
                if pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col])
            ]

        # 2. Compute outlier limits
        # Why: Fitting outlier limits on training data and applying it to test/validation prevents 
        # validation set statistics from leaking back into model evaluation.
        for col in X.columns:
            if col not in target_cols or col == "TARGET" or col == "y":
                continue

            series = X[col].dropna()
            if series.empty:
                continue

            if self.method == "iqr":
                q25 = float(series.quantile(0.25))
                q75 = float(series.quantile(0.75))
                iqr = q75 - q25
                lower_limit = q25 - self.threshold * iqr
                upper_limit = q75 + self.threshold * iqr
            elif self.method == "z_score":
                mean = float(series.mean())
                std = float(series.std())
                if std == 0:
                    lower_limit, upper_limit = mean, mean
                else:
                    lower_limit = mean - self.threshold * std
                    upper_limit = mean + self.threshold * std
            elif self.method == "winsorize":
                lower_limit = float(series.quantile(self.lower_quantile))
                upper_limit = float(series.quantile(self.upper_quantile))
            else:
                raise ValueError(f"Unknown outlier detection method: {self.method}")

            self.limits_[col] = (lower_limit, upper_limit)
            
            # 3. Track number of outliers for preprocessing validation audits
            outliers = X[col].notnull() & ((X[col] < lower_limit) | (X[col] > upper_limit))
            self.outlier_counts_[col] = int(outliers.sum())

        self.feature_names_out_ = self.feature_names_in_
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clips values to computed limits for configured features.

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

        if self.strategy == "clip":
            for col, (lower_limit, upper_limit) in self.limits_.items():
                if col in df.columns:
                    df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)

        return df

    def get_feature_names_out(self, input_features=None):
        """Returns feature output names list.

        Args:
            input_features (list of str, optional): Input names. Defaults to None.

        Returns:
            list of str: Feature names.
        """
        return self.feature_names_out_
