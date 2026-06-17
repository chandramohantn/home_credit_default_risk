"""Module for dimensionality reduction analysis in EDA."""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple

class DimensionalityAnalyzer:
    """Performs PCA and other dimensionality reduction techniques."""

    @staticmethod
    def run_pca(df: pd.DataFrame, n_components: int = 2) -> Tuple[np.ndarray, PCA]:
        """Runs PCA on numerical features.

        Args:
            df (pd.DataFrame): The dataset (numeric only or preprocessed).
            n_components (int): Number of components.

        Returns:
            Tuple[np.ndarray, PCA]: Projected data and the PCA object.
        """
        # Select numeric and handle missing
        X = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
        X = X.fillna(X.median())
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        return X_pca, pca

    @staticmethod
    def run_umap(df: pd.DataFrame, n_components: int = 2):
        """Runs UMAP on numerical features (requires umap-learn)."""
        try:
            import umap
            X = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
            X = X.fillna(X.median())
            X_scaled = StandardScaler().fit_transform(X)
            
            reducer = umap.UMAP(n_components=n_components, n_jobs=-1)
            return reducer.fit_transform(X_scaled)
        except ImportError:
            print("UMAP not installed. Skipping UMAP.")
            return None
