"""Mathematical transformations for numerical features.

This module provides the FeatureTransformer, which applies transformations such as 
Log, Square Root, Box-Cox, and Yeo-Johnson to numerical columns.

Heavily skewed columns (like income or credit amount) violate normality assumptions 
required by linear models and neural networks. Feature transformations redistribute 
these values, promoting stability and faster gradient updates. Fitted stats 
(specifically the optimal transformation lambda value for Box-Cox/Yeo-Johnson) 
are computed during fit and re-applied during transform to guarantee consistency.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from scipy.stats import boxcox, yeojohnson
from src.preprocessing.transformers.base import BaseTransformer
from src.preprocessing.strategies import TRANSFORMATION_REGISTRY

class FeatureTransformer(BaseTransformer):
    """Applies mathematical transformations to numerical features.

    Maintains lambda properties for Box-Cox and Yeo-Johnson transforms.

    Attributes:
        transformations (dict): A mapping of column names to transformation types 
            (e.g., {'AMT_INCOME_TOTAL': 'log1p'}).
        fitted_params_ (dict): Stored dictionary of fitted lambdas and transformation types.
        feature_names_in_ (list of str): Input feature names seen in fit.
        feature_names_out_ (list of str): Output feature names (matches input).
    """
    
    def __init__(self, transformations: Optional[Dict[str, str]] = None):
        """Initializes the FeatureTransformer.

        Args:
            transformations (dict, optional): Column-to-transform mapping.
        """
        super().__init__()
        self.transformations = transformations or {}

        # Fitted parameters
        self.fitted_params_: Dict[str, Any] = {}
        self.transformation_report_: Dict[str, Any] = {}
        self.feature_names_in_: List[str] = []
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Learns transformation properties (optimal lambdas) for configured features.

        Note: Falls back to Yeo-Johnson if Box-Cox is requested but negative 
        or zero values exist in the column, as Box-Cox requires strictly positive inputs.

        Args:
            X (pd.DataFrame): Training DataFrame.
            y (pd.Series, optional): Target series. Defaults to None.

        Returns:
            FeatureTransformer: The fitted instance of the transformer.

        Raises:
            TypeError: If X is not a pandas DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        self.feature_names_in_ = list(X.columns)
        self.fitted_params_ = {}
        self.transformation_report_ = {}

        for col, transform_type in self.transformations.items():
            if col not in X.columns:
                continue
            if transform_type not in TRANSFORMATION_REGISTRY:
                raise ValueError(f"Unknown transformation strategy: {transform_type}")

            series = X[col].dropna()
            if series.empty:
                continue

            if transform_type == "box_cox":
                # Why: Box-Cox fails on non-positive inputs. Fallback to Yeo-Johnson.
                if (series <= 0).any():
                    _, lmbda = yeojohnson(series)
                    self.fitted_params_[col] = {"type": "yeo_johnson", "lambda": lmbda}
                    self.transformation_report_[col] = {
                        "operation": "yeo_johnson",
                        "parameters": {"lambda": float(lmbda)},
                    }
                else:
                    _, lmbda = boxcox(series)
                    self.fitted_params_[col] = {"type": "box_cox", "lambda": lmbda}
                    self.transformation_report_[col] = {
                        "operation": "box_cox",
                        "parameters": {"lambda": float(lmbda)},
                    }
            elif transform_type == "yeo_johnson":
                # Why: Yeo-Johnson can handle negative/zero inputs natively.
                _, lmbda = yeojohnson(series)
                self.fitted_params_[col] = {"type": "yeo_johnson", "lambda": lmbda}
                self.transformation_report_[col] = {
                    "operation": "yeo_johnson",
                    "parameters": {"lambda": float(lmbda)},
                }
            elif transform_type in ["log", "log1p", "sqrt"]:
                # Static transformations (no parameter fitting required)
                self.fitted_params_[col] = {"type": transform_type}
                self.transformation_report_[col] = {
                    "operation": transform_type,
                    "parameters": {},
                }

        self.feature_names_out_ = self.feature_names_in_
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies mathematical transformations to configured columns.

        Uses the exact optimal lambdas fitted during the training step.

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

        for col, params in self.fitted_params_.items():
            if col not in df.columns:
                continue

            transform_type = params["type"]
            non_null_mask = df[col].notnull()
            
            if not non_null_mask.any():
                continue

            # Perform transformations only on non-null values to preserve missingness indicators
            vals = df.loc[non_null_mask, col].values

            if transform_type == "log":
                df.loc[non_null_mask, col] = np.log(np.maximum(vals, 1e-9))
            elif transform_type == "log1p":
                df.loc[non_null_mask, col] = np.log1p(np.maximum(vals, -1.0 + 1e-9))
            elif transform_type == "sqrt":
                df.loc[non_null_mask, col] = np.sqrt(np.maximum(vals, 0.0))
            elif transform_type == "box_cox":
                lmbda = params["lambda"]
                positive_vals = np.maximum(vals, 1e-9)
                df.loc[non_null_mask, col] = boxcox(positive_vals, lmbda=lmbda)
            elif transform_type == "yeo_johnson":
                lmbda = params["lambda"]
                df.loc[non_null_mask, col] = yeojohnson(vals, lmbda=lmbda)

        return df

    def get_feature_names_out(self, input_features=None):
        """Returns feature output names list.

        Args:
            input_features (list of str, optional): Input names. Defaults to None.

        Returns:
            list of str: Feature names.
        """
        return self.feature_names_out_

    def get_reports(self) -> Dict[str, Any]:
        """Returns transformation metadata for pipeline-level reporting."""

        return {"transformations": self.transformation_report_}
