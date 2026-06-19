"""Data cleaning and column standardization transformer.

This module provides the DataCleaner transformer which handles standard initial tabular 
cleaning. Its main use is to coerce column data types, drop duplicate columns, 
and handle duplicate rows during the initial ingestion phase.

Data cleaning is performed as the very first step of the pipeline to guarantee that all 
subsequent transformers (imputers, encoders, scalers) receive columns with predictable 
schemas and standard pandas data types.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from src.preprocessing.transformers.base import BaseTransformer

class DataCleaner(BaseTransformer):
    """Standardizes types, drops column-level duplicates, and handles row-level duplicates.

    Attributes:
        drop_row_duplicates (bool): If True, drops duplicate rows during fit_transform.
        drop_col_duplicates (bool): If True, checks for and drops duplicate feature columns.
        datetime_cols (list of str): List of column names that should be coerced to datetimes.
        coerce_objects_to_category (bool): If True, converts object columns to pandas categories.
        duplicate_features_ (list of str): Stored list of duplicate columns found during fitting.
        column_types_ (dict): Stored mapping of column names to their inferred/standardized string types.
        feature_names_in_ (list of str): Input feature names seen in fit.
        feature_names_out_ (list of str): Output feature names remaining after duplicates are dropped.
    """
    
    def __init__(self, 
                 drop_row_duplicates: bool = True,
                 drop_col_duplicates: bool = True,
                 datetime_cols: Optional[List[str]] = None,
                 coerce_objects_to_category: bool = True):
        """Initializes the DataCleaner transformer.

        Args:
            drop_row_duplicates (bool): Whether to drop row-level duplicates on train data.
            drop_col_duplicates (bool): Whether to detect and drop identical columns.
            datetime_cols (list of str, optional): Column names to interpret as datetimes.
            coerce_objects_to_category (bool): Whether to convert string columns to category type.
        """
        super().__init__()
        self.drop_row_duplicates = drop_row_duplicates
        self.drop_col_duplicates = drop_col_duplicates
        self.datetime_cols = datetime_cols or []
        self.coerce_objects_to_category = coerce_objects_to_category
        
        # Fitted parameters to store transformer state
        self.duplicate_features_: List[str] = []
        self.column_types_: Dict[str, str] = {}
        self.feature_names_in_: List[str] = []
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Identifies identical columns and learns standard data types for each feature.

        Args:
            X (pd.DataFrame): The input DataFrame to fit on.
            y (pd.Series, optional): The target variable. Defaults to None.

        Returns:
            DataCleaner: The fitted instance of the transformer.

        Raises:
            TypeError: If X is not a pandas DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        self.feature_names_in_ = list(X.columns)
        
        # 1. Detect Duplicate Features
        # Why: Duplicate features introduce multicollinearity and inflate memory usage.
        # Check pairs sequentially. If X[col_i] == X[col_j] for all elements, flag col_j for dropping.
        self.duplicate_features_ = []
        if self.drop_col_duplicates:
            cols = X.columns
            for i in range(len(cols)):
                col_i = cols[i]
                if col_i in self.duplicate_features_:
                    continue
                for j in range(i + 1, len(cols)):
                    col_j = cols[j]
                    if col_j in self.duplicate_features_:
                        continue
                    if X[col_i].equals(X[col_j]):
                        self.duplicate_features_.append(col_j)

        # 2. Learn column types
        # Why: Coercing columns to categories or specific numeric types speeds up downstream processing 
        # and standardizes treatment of string columns.
        remaining_cols = [c for c in X.columns if c not in self.duplicate_features_]
        for col in remaining_cols:
            dtype = X[col].dtype
            if col in self.datetime_cols:
                self.column_types_[col] = "datetime"
            elif self.coerce_objects_to_category and (isinstance(dtype, pd.CategoricalDtype) or dtype == "object"):
                self.column_types_[col] = "category"
            elif pd.api.types.is_integer_dtype(dtype):
                self.column_types_[col] = "int64"
            elif pd.api.types.is_float_dtype(dtype):
                self.column_types_[col] = "float64"
            else:
                self.column_types_[col] = str(dtype)

        self.feature_names_out_ = remaining_cols
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies duplicate column dropping and column type standardization.

        Args:
            X (pd.DataFrame): The input DataFrame to transform.

        Returns:
            pd.DataFrame: Cleaned DataFrame with correct column types and no identical columns.

        Raises:
            TypeError: If X is not a pandas DataFrame.
        """
        super().transform(X)
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Copy to avoid side-effects on original data (SettingWithCopyWarning)
        df = X.copy()

        # 1. Drop duplicate features learned during fit
        if self.duplicate_features_:
            cols_to_drop = [c for c in self.duplicate_features_ if c in df.columns]
            df = df.drop(columns=cols_to_drop)

        # 2. Coerce types based on fitted types
        # Why: Coercing integers with NaNs requires pandas nullable type 'Int64', 
        # otherwise pandas automatically converts integer series with NaNs to floats.
        for col in df.columns:
            if col in self.column_types_:
                expected_type = self.column_types_[col]
                try:
                    if expected_type == "datetime":
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                    elif expected_type == "category":
                        df[col] = df[col].astype("category")
                    elif expected_type == "int64":
                        if df[col].isnull().any():
                            df[col] = df[col].astype("Int64")
                        else:
                            df[col] = df[col].astype("int64")
                    elif expected_type == "float64":
                        df[col] = df[col].astype("float64")
                except Exception:
                    pass  # Fallback to keeping the original data type if casting fails

        return df

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fits the cleaner, optionally removes duplicate rows, and returns the cleaned DataFrame.

        Why override: Scikit-learn's standard fit_transform cannot alter row sizes. By providing 
        a custom fit_transform, we can drop duplicate rows on training data before fitting.

        Args:
            X (pd.DataFrame): The input DataFrame to fit and transform.
            y (pd.Series, optional): The target variable. Defaults to None.

        Returns:
            pd.DataFrame: Cleaned DataFrame (with duplicates optionally removed).
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Drop duplicate rows during training to avoid overfitting
        if self.drop_row_duplicates:
            non_duplicate_mask = ~X.duplicated()
            X = X[non_duplicate_mask]
        
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        """Returns the list of remaining column names after duplicate columns are dropped.

        Args:
            input_features (list of str, optional): The input feature names. Defaults to None.

        Returns:
            list of str: Remaining column names.
        """
        return self.feature_names_out_
