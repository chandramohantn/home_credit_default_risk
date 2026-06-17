"""Module for feature importance exploration in EDA."""

import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

class ImportanceAnalyzer:
    """Analyzes feature importance using model-agnostic methods."""

    @staticmethod
    def calculate_mutual_info(df: pd.DataFrame, target_column: str, top_n: int = 30) -> pd.Series:
        """Calculates Mutual Information scores for all features.

        Args:
            df (pd.DataFrame): The dataset.
            target_column (str): The target column.
            top_n (int): Number of top features to return.

        Returns:
            pd.Series: Mutual Information scores.
        """
        data = df.copy()
        y = data[target_column]
        X = data.drop(columns=[target_column])
        
        # Handle missing and categorical values for MI
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            X[col] = X[col].fillna(X[col].median() if X[col].dtype != 'object' else -1)

        # Use a sample for speed on large datasets
        if len(X) > 50000:
            sample_idx = np.random.choice(len(X), 50000, replace=False)
            X_sample = X.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]
        else:
            X_sample = X
            y_sample = y

        mi_scores = mutual_info_classif(X_sample, y_sample)
        return pd.Series(mi_scores, index=X.columns).sort_values(ascending=False).head(top_n)
