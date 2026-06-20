"""Categorical variable encoding transformer.

This module provides the EncodingTransformer, which orchestrates categorical level 
encoding strategies. Supported encodings:
- One-Hot encoding (standard for linear models and neural networks).
- Ordinal encoding (standard for tree models).
- Frequency encoding (replaces labels with their frequencies).
- Target encoding (replaces category with average target probabilities).
- CatBoost-style ordered encoding (reduces leakage using running target statistics).

Critical Requirement: Target encoding on the training set introduces massive target 
leakage if calculated naively. To prevent this, our fit_transform implementation uses 
out-of-fold (OOF) target value computation via K-Fold validation on the training data, 
while preserving global target statistics for test set transformations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Any, Optional
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from src.preprocessing.transformers.base import BaseTransformer
from src.preprocessing.strategies import ENCODER_REGISTRY

class EncodingTransformer(BaseTransformer):
    """Encodes categorical levels into model-ready numerical columns.

    Supports a global default strategy or column-specific configuration overrides.

    Attributes:
        default_strategy (str): Default strategy name ('one_hot', 'ordinal', 'target', 'frequency', 'catboost').
        column_strategies (dict): Specific overrides per column.
        target_cv_folds (int): Number of folds for out-of-fold target encoding calculations.
        target_smoothing (float): Smoothing parameter (prior weight) for target encoding.
        handle_unknown (str): Strategy to handle unknown categories during transform ('ignore').
        encoding_info_ (dict): Stored mapping details, priors, models, or dicts learned in fit.
        feature_names_in_ (list of str): Input feature names seen in fit.
        feature_names_out_ (list of str): Output feature names remaining after encoding.
    """
    
    def __init__(self,
                 default_strategy: str = "one_hot",
                 column_strategies: Optional[Dict[str, str]] = None,
                 target_cv_folds: int = 5,
                 target_smoothing: float = 10.0,
                 catboost_prior: float = 10.0,
                 random_state: int = 42,
                 handle_unknown: str = "ignore"):
        """Initializes the EncodingTransformer.

        Args:
            default_strategy (str): Default encoding strategy to apply. Defaults to "one_hot".
            column_strategies (dict, optional): Custom column-specific strategies overrides.
            target_cv_folds (int): Number of folds for out-of-fold target encoding. Defaults to 5.
            target_smoothing (float): Smoothing factor for target encoding. Defaults to 10.0.
            handle_unknown (str): Standard scikit-learn handle_unknown configuration. Defaults to "ignore".
        """
        super().__init__()
        self.default_strategy = default_strategy
        self.column_strategies = column_strategies or {}
        self.target_cv_folds = target_cv_folds
        self.target_smoothing = target_smoothing
        self.catboost_prior = catboost_prior
        self.random_state = random_state
        self.handle_unknown = handle_unknown

        # Fitted parameters
        self.encoding_info_: Dict[str, Dict[str, Any]] = {}
        self.feature_names_in_: List[str] = []
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Learns categorical mappings and target ratios for each column.

        Note: When y is not provided during fitting, target-derived encoders default 
        to standard ordinal encoding to prevent failures.

        Args:
            X (pd.DataFrame): Input DataFrame to learn encodings from.
            y (pd.Series, optional): Target series required for target encoding. Defaults to None.

        Returns:
            EncodingTransformer: The fitted instance of the transformer.

        Raises:
            TypeError: If X is not a pandas DataFrame.
            ValueError: If an unknown strategy is encountered.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        self.feature_names_in_ = list(X.columns)
        self.encoding_info_ = {}
        self.feature_names_out_ = []

        # Find categorical columns (objects or categories)
        cat_cols = [col for col in X.columns if isinstance(X[col].dtype, pd.CategoricalDtype) or X[col].dtype == "object"]

        for col in X.columns:
            if col not in cat_cols:
                # Numerical columns pass through unmodified
                self.feature_names_out_.append(col)
                continue

            strategy = self.column_strategies.get(col, self.default_strategy)
            if strategy not in ENCODER_REGISTRY:
                raise ValueError(f"Unknown encoding strategy: {strategy}")
            
            if strategy == "one_hot":
                # Why: Standard OneHotEncoder creates binary indicators per category. 
                # We cast to string to handle mixed types gracefully.
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
                encoder.fit(X[[col]].astype(str))
                
                categories = encoder.categories_[0]
                out_cols = [f"{col}_{cat}" for cat in categories]
                
                self.encoding_info_[col] = {
                    "strategy": "one_hot",
                    "encoder": encoder,
                    "output_columns": out_cols
                }
                self.feature_names_out_.extend(out_cols)

            elif strategy == "ordinal":
                # Why: Maps categories to simple integers. Ideal for decision tree methods.
                categories = list(X[col].dropna().unique())
                mapping = {cat: idx for idx, cat in enumerate(categories)}
                
                self.encoding_info_[col] = {
                    "strategy": "ordinal",
                    "mapping": mapping,
                    "output_columns": [col]
                }
                self.feature_names_out_.append(col)

            elif strategy == "frequency":
                # Why: Replaces label with occurrence frequency, capturing prevalence patterns.
                counts = X[col].value_counts(dropna=True)
                total = len(X[col].dropna())
                mapping = (counts / total).to_dict() if total > 0 else {}
                
                self.encoding_info_[col] = {
                    "strategy": "frequency",
                    "mapping": mapping,
                    "default_value": 0.0,
                    "output_columns": [col]
                }
                self.feature_names_out_.append(col)

            elif strategy == "target":
                # Fallback to ordinal encoding if target variable y is absent
                if y is None:
                    categories = list(X[col].dropna().unique())
                    mapping = {cat: idx for idx, cat in enumerate(categories)}
                    self.encoding_info_[col] = {
                        "strategy": "target_fallback",
                        "mapping": mapping,
                        "output_columns": [col]
                    }
                    self.feature_names_out_.append(col)
                    continue

                # Why: Target encoding computes smoothed category target probability.
                # Smoothed formula: S = (count * mean + smoothing * prior) / (count + smoothing)
                # It handles rare levels by bringing them closer to the global target prior.
                global_prior = float(y.mean())
                col_y_df = pd.DataFrame({col: X[col], "y": y})
                stats = col_y_df.groupby(col, observed=True)["y"].agg(["count", "mean"])
                
                smoothed_mapping = {}
                for category, row in stats.iterrows():
                    count = row["count"]
                    mean = row["mean"]
                    smoothed = (count * mean + self.target_smoothing * global_prior) / (count + self.target_smoothing)
                    smoothed_mapping[str(category)] = float(smoothed)

                self.encoding_info_[col] = {
                    "strategy": "target",
                    "mapping": smoothed_mapping,
                    "global_prior": global_prior,
                    "output_columns": [col]
                }
                self.feature_names_out_.append(col)

            elif strategy == "catboost":
                if y is None:
                    categories = list(X[col].dropna().unique())
                    mapping = {cat: idx for idx, cat in enumerate(categories)}
                    self.encoding_info_[col] = {
                        "strategy": "catboost_fallback",
                        "mapping": mapping,
                        "output_columns": [col]
                    }
                    self.feature_names_out_.append(col)
                    continue

                global_prior = float(y.mean())
                col_y_df = pd.DataFrame({col: X[col], "y": y})
                stats = col_y_df.groupby(col, observed=True)["y"].agg(["count", "mean"])

                smoothed_mapping = {}
                for category, row in stats.iterrows():
                    count = row["count"]
                    mean = row["mean"]
                    smoothed = (count * mean + self.catboost_prior * global_prior) / (count + self.catboost_prior)
                    smoothed_mapping[str(category)] = float(smoothed)

                self.encoding_info_[col] = {
                    "strategy": "catboost",
                    "mapping": smoothed_mapping,
                    "global_prior": global_prior,
                    "output_columns": [col]
                }
                self.feature_names_out_.append(col)

            else:
                raise ValueError(f"Unknown encoding strategy: {strategy}")

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms categorical columns using learned encoding mappings.

        Note: When applying target encoding, this method uses the global mapped 
        priors learned during fitting.

        Args:
            X (pd.DataFrame): Input DataFrame to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame with all categorical features encoded.

        Raises:
            TypeError: If X is not a pandas DataFrame.
        """
        super().transform(X)
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        df = X.copy()
        output_dfs = []

        # Maintain exact column ordering during concatenation
        for col in X.columns:
            if col not in self.encoding_info_:
                output_dfs.append(df[[col]])
                continue

            info = self.encoding_info_[col]
            strategy = info["strategy"]

            if strategy == "one_hot":
                encoder = info["encoder"]
                out_cols = info["output_columns"]
                encoded_arr = encoder.transform(df[[col]].astype(str))
                encoded_df = pd.DataFrame(encoded_arr, columns=out_cols, index=df.index)
                output_dfs.append(encoded_df)

            elif strategy == "ordinal":
                mapping = info["mapping"]
                encoded_series = df[col].astype(str).map(mapping).fillna(-1.0)
                output_dfs.append(encoded_series.to_frame(name=col))

            elif strategy == "frequency":
                mapping = info["mapping"]
                default_val = info["default_value"]
                encoded_series = df[col].astype(str).map(mapping).fillna(default_val)
                output_dfs.append(encoded_series.to_frame(name=col))

            elif strategy in {"target", "target_fallback", "catboost", "catboost_fallback"}:
                mapping = info["mapping"]
                prior = info.get("global_prior", 0.0)
                encoded_series = df[col].astype(str).map(mapping).fillna(prior)
                output_dfs.append(encoded_series.to_frame(name=col))

        return pd.concat(output_dfs, axis=1)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Executes fit and transform. Applies CV-safe target encoding if target is specified.

        Why override: Naive target encoding causes massive target leakage because training features 
        are mapped using target information from their own rows. To avoid this, we compute 
        Out-of-Fold (OOF) target encodings using KFold partitioning for the training partition.

        Args:
            X (pd.DataFrame): Training DataFrame.
            y (pd.Series, optional): Training target variable. Defaults to None.

        Returns:
            pd.DataFrame: Training DataFrame with out-of-fold target values.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # 1. Fits the mapping parameters globally (so they can be used for transform on val/test sets)
        self.fit(X, y)

        if y is None:
            return self.transform(X)

        # 2. Check if target encoding is active
        oof_strategies = {"target", "catboost"}
        encoded_with_oof = {
            col: info["strategy"]
            for col, info in self.encoding_info_.items()
            if info["strategy"] in oof_strategies
        }

        if not encoded_with_oof:
            return self.transform(X)

        # 3. Perform Out-of-Fold calculation
        df_oof = X.copy()
        kf = KFold(n_splits=self.target_cv_folds, shuffle=True, random_state=42)
        oof_values = {
            col: np.zeros(len(X)) 
            for col, info in self.encoding_info_.items() 
            if info["strategy"] in oof_strategies
        }

        for train_idx, val_idx in kf.split(X, y):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            global_prior_fold = float(y_train_fold.mean())

            for col, oof_arr in oof_values.items():
                strategy = self.encoding_info_[col]["strategy"]
                if strategy == "target":
                    col_y_df = pd.DataFrame({col: X_train_fold[col], "y": y_train_fold})
                    stats = col_y_df.groupby(col, observed=True)["y"].agg(["count", "mean"])

                    fold_mapping = {}
                    for category, row in stats.iterrows():
                        count = row["count"]
                        mean = row["mean"]
                        smoothed = (count * mean + self.target_smoothing * global_prior_fold) / (count + self.target_smoothing)
                        fold_mapping[str(category)] = float(smoothed)

                    oof_arr[val_idx] = X_val_fold[col].astype(str).map(fold_mapping).fillna(global_prior_fold).values
                elif strategy == "catboost":
                    ordered_series = self._catboost_encode_train(
                        X_train_fold[col],
                        y_train_fold,
                        global_prior_fold,
                    )
                    fold_mapping = pd.Series(ordered_series, index=X_train_fold.index)
                    category_means = (
                        pd.DataFrame({"feature": X_train_fold[col].astype(str), "encoded": fold_mapping})
                        .groupby("feature", observed=True)["encoded"]
                        .mean()
                        .to_dict()
                    )
                    oof_arr[val_idx] = (
                        X_val_fold[col].astype(str).map(category_means).fillna(global_prior_fold).values
                    )

        # Build output dataframe, inserting OOF arrays for target columns
        output_dfs = []
        for col in X.columns:
            if col not in self.encoding_info_:
                output_dfs.append(df_oof[[col]])
                continue

            info = self.encoding_info_[col]
            strategy = info["strategy"]

            if strategy in {"target", "catboost"}:
                oof_series = pd.Series(oof_values[col], index=X.index, name=col)
                output_dfs.append(oof_series.to_frame())
            else:
                col_transformed = self.transform(X[[col]])
                output_dfs.append(col_transformed)

        return pd.concat(output_dfs, axis=1)

    def get_feature_names_out(self, input_features=None):
        """Returns the list of final encoded feature names.

        Args:
            input_features (list of str, optional): The input feature names. Defaults to None.

        Returns:
            list of str: Final list of feature names.
        """
        return self.feature_names_out_

    def get_reports(self) -> Dict[str, Any]:
        """Returns encoding metadata for pipeline-level reporting."""

        return {
            "encodings": {
                col: {
                    "strategy": info["strategy"],
                    "output_columns": info["output_columns"],
                }
                for col, info in self.encoding_info_.items()
            }
        }

    def _catboost_encode_train(
        self,
        feature: pd.Series,
        target: pd.Series,
        global_prior: float,
    ) -> np.ndarray:
        """Builds ordered target statistics for CatBoost-style training encoding."""

        ordered_values = np.zeros(len(feature), dtype=float)
        cat_sums: Dict[str, float] = {}
        cat_counts: Dict[str, int] = {}
        permutation = np.random.RandomState(self.random_state).permutation(len(feature))
        feature_values = feature.astype(str).values
        target_values = target.values

        for idx in permutation:
            category = feature_values[idx]
            running_sum = cat_sums.get(category, 0.0)
            running_count = cat_counts.get(category, 0)
            ordered_values[idx] = (
                running_sum + self.catboost_prior * global_prior
            ) / (running_count + self.catboost_prior)
            cat_sums[category] = running_sum + float(target_values[idx])
            cat_counts[category] = running_count + 1

        return ordered_values
