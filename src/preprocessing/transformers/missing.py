"""Missing value imputation transformer.

This module provides the MissingValueTransformer, which imputes missing values 
for both numerical and categorical features. It preserves pandas structures, 
supports custom column-level strategies, and can generate indicator columns 
that record where missing values were originally located.

Imputation is crucial to prevent down-stream estimators (like scikit-learn models or Neural 
Networks) from breaking due to NaN/null values, while preserving patterns of missingness.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Any, Optional
from sklearn.impute import SimpleImputer, KNNImputer
# Import IterativeImputer (requires enabling experimental feature in scikit-learn)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from src.preprocessing.transformers.base import BaseTransformer

class MissingValueTransformer(BaseTransformer):
    """Imputes missing values for numerical and categorical columns.

    Attributes:
        num_strategy (str): Default strategy for numerical columns ('median', 'mean', 'constant', 'knn', 'iterative').
        num_fill_value (Any, optional): Value to use for constant imputation in numerical columns.
        cat_strategy (str): Default strategy for categorical columns ('constant', 'most_frequent').
        cat_fill_value (str): Value to use for constant imputation in categorical columns.
        add_indicators (bool): If True, appends boolean flags marking missing occurrences.
        column_strategies (dict): Specific mapping of column names to strategy dicts, 
            e.g., {'AMT_ANNUITY': {'strategy': 'mean'}}.
        knn_neighbors (int): Number of neighbors to use for KNN imputation.
        imputers_ (dict): Stored mapping of column names to fitted SimpleImputer/KNNImputer objects.
        missing_indicator_cols_ (list of str): List of indicator columns generated during fit.
        feature_names_in_ (list of str): Input feature names seen in fit.
        feature_names_out_ (list of str): Total output feature names (including indicators).
        imputation_values_report_ (dict): Diagnostic dictionary detailing fill values for reporting.
    """
    
    def __init__(self,
                 num_strategy: str = "median",
                 num_fill_value: Optional[Any] = None,
                 cat_strategy: str = "constant",
                 cat_fill_value: str = "Unknown",
                 add_indicators: bool = True,
                 column_strategies: Optional[Dict[str, Dict[str, Any]]] = None,
                 knn_neighbors: int = 5):
        """Initializes the MissingValueTransformer.

        Args:
            num_strategy (str): Default strategy for numerical features. Defaults to "median".
            num_fill_value (Any, optional): Default fill value if num_strategy is "constant".
            cat_strategy (str): Default strategy for categorical features. Defaults to "constant".
            cat_fill_value (str): Default fill value if cat_strategy is "constant". Defaults to "Unknown".
            add_indicators (bool): Whether to create binary missingness indicator columns. Defaults to True.
            column_strategies (dict, optional): Custom column-specific configuration overrides.
            knn_neighbors (int): K neighbors for KNN imputer if used. Defaults to 5.
        """
        super().__init__()
        self.num_strategy = num_strategy
        self.num_fill_value = num_fill_value
        self.cat_strategy = cat_strategy
        self.cat_fill_value = cat_fill_value
        self.add_indicators = add_indicators
        self.column_strategies = column_strategies or {}
        self.knn_neighbors = knn_neighbors

        # Fitted parameters to store transformer state
        self.imputers_: Dict[str, Any] = {}
        self.missing_indicator_cols_: List[str] = []
        self.feature_names_in_: List[str] = []
        self.feature_names_out_: List[str] = []
        self.imputation_values_report_: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fits imputation estimators and saves missingness patterns.

        Args:
            X (pd.DataFrame): Input DataFrame to learn median/mean/mode stats from.
            y (pd.Series, optional): The target variable. Defaults to None.

        Returns:
            MissingValueTransformer: The fitted instance of the transformer.

        Raises:
            TypeError: If X is not a pandas DataFrame.
            ValueError: If an unknown imputation strategy is provided.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        self.feature_names_in_ = list(X.columns)
        self.imputers_ = {}
        self.missing_indicator_cols_ = []
        self.imputation_values_report_ = {}

        # 1. Identify columns with missing values to build indicators
        # Why: Retaining indicators like "EXT_SOURCE_1_isnull" is highly predictive, as 
        # the absence of a credit score is often statistically linked to defaults.
        if self.add_indicators:
            for col in X.columns:
                if X[col].isnull().any():
                    self.missing_indicator_cols_.append(f"{col}_isnull")

        # 2. Fit individual imputers for each column
        # Why: Creating a separate imputer per column ensures we preserve column names 
        # and prevent errors when columns are mixed types.
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                continue  # Skip datetime columns from numeric/categorical imputation

            # Determine column specific strategy or fallback to default
            col_strat_info = self.column_strategies.get(col, {})
            is_numeric = pd.api.types.is_numeric_dtype(X[col])

            if is_numeric:
                strategy = col_strat_info.get("strategy", self.num_strategy)
                fill_val = col_strat_info.get("fill_value", self.num_fill_value)
                
                # Cast to float to avoid casting warnings or exceptions when fitting float stats 
                # (like mean/median) into integer series.
                col_data = X[[col]].astype(float)
                
                if strategy == "median":
                    val = float(col_data[col].median(numeric_only=True))
                    if np.isnan(val):
                        val = 0.0
                    imputer = SimpleImputer(strategy="constant", fill_value=val)
                    imputer.fit(col_data)
                    self.imputers_[col] = imputer
                    self.imputation_values_report_[col] = {"strategy": "median", "value": val}
                elif strategy == "mean":
                    val = float(col_data[col].mean(numeric_only=True))
                    if np.isnan(val):
                        val = 0.0
                    imputer = SimpleImputer(strategy="constant", fill_value=val)
                    imputer.fit(col_data)
                    self.imputers_[col] = imputer
                    self.imputation_values_report_[col] = {"strategy": "mean", "value": val}
                elif strategy == "constant":
                    val = fill_val if fill_val is not None else 0.0
                    imputer = SimpleImputer(strategy="constant", fill_value=val)
                    imputer.fit(col_data)
                    self.imputers_[col] = imputer
                    self.imputation_values_report_[col] = {"strategy": "constant", "value": val}
                elif strategy == "knn":
                    imputer = KNNImputer(n_neighbors=self.knn_neighbors)
                    imputer.fit(col_data)
                    self.imputers_[col] = imputer
                    self.imputation_values_report_[col] = {"strategy": "knn"}
                elif strategy == "iterative":
                    imputer = IterativeImputer(random_state=42)
                    imputer.fit(col_data)
                    self.imputers_[col] = imputer
                    self.imputation_values_report_[col] = {"strategy": "iterative"}
                else:
                    raise ValueError(f"Unknown numerical imputation strategy: {strategy}")
            else:
                # Categorical Imputation
                strategy = col_strat_info.get("strategy", self.cat_strategy)
                fill_val = col_strat_info.get("fill_value", self.cat_fill_value)

                if strategy == "most_frequent" or strategy == "mode":
                    mode_series = X[col].mode()
                    val = str(mode_series.iloc[0]) if not mode_series.empty else str(fill_val)
                    imputer = SimpleImputer(strategy="constant", fill_value=val)
                    imputer.fit(X[[col]].astype(str))
                    self.imputers_[col] = imputer
                    self.imputation_values_report_[col] = {"strategy": "most_frequent", "value": val}
                elif strategy == "constant":
                    val = str(fill_val)
                    imputer = SimpleImputer(strategy="constant", fill_value=val)
                    imputer.fit(X[[col]].astype(str))
                    self.imputers_[col] = imputer
                    self.imputation_values_report_[col] = {"strategy": "constant", "value": val}
                else:
                    raise ValueError(f"Unknown categorical imputation strategy: {strategy}")

        # Construct final output list of feature names
        self.feature_names_out_ = [
            c for c in self.feature_names_in_ 
            if c in self.imputers_ or pd.api.types.is_datetime64_any_dtype(X[c])
        ] + self.missing_indicator_cols_
        
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Imputes missing values and appends indicators.

        Args:
            X (pd.DataFrame): Input DataFrame to impute.

        Returns:
            pd.DataFrame: Imputed DataFrame with extra indicator columns.

        Raises:
            TypeError: If X is not a pandas DataFrame.
        """
        super().transform(X)
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        df = X.copy()

        # 1. Create missingness indicator columns
        # Why: Test set missingness must match training indicators. If a feature had missing values in 
        # train but not in test, we output 0s for that column to align schemas.
        if self.add_indicators:
            for ind_col in self.missing_indicator_cols_:
                orig_col = ind_col.replace("_isnull", "")
                if orig_col in df.columns:
                    df[ind_col] = df[orig_col].isnull().astype(int)
                else:
                    df[ind_col] = 0

        # 2. Impute columns using fitted estimators
        for col in df.columns:
            if col in self.imputers_:
                imputer = self.imputers_[col]
                is_numeric = pd.api.types.is_numeric_dtype(df[col])
                
                if is_numeric:
                    # Cast slice to float to prevent SimpleImputer casting errors
                    df[[col]] = imputer.transform(df[[col]].astype(float))
                else:
                    # Cast categorical object values safely to prevent SimpleImputer breaking on unseen values
                    temp_col = df[col].astype(str)
                    temp_col = temp_col.replace("nan", np.nan)
                    imputed = imputer.transform(temp_col.to_frame())
                    
                    df[col] = imputed.ravel()
                    if isinstance(X[col].dtype, pd.CategoricalDtype):
                        # Ensure category type constraints remain consistent after string casting
                        unique_vals = df[col].unique()
                        df[col] = df[col].astype(pd.CategoricalDtype(categories=unique_vals))

        return df[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        """Returns the list of output feature names (including indicators).

        Args:
            input_features (list of str, optional): The input feature names. Defaults to None.

        Returns:
            list of str: Final list of feature names.
        """
        return self.feature_names_out_
