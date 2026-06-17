"""Module for dimensionality reduction visualizations."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class DimensionalityVisualizer:
    """Visualizes PCA and UMAP projections."""

    @staticmethod
    def plot_projection(X_proj: np.ndarray, target: np.ndarray, title: str = "Projection"):
        """Plots a 2D projection colored by target.

        Args:
            X_proj (np.ndarray): 2D projected data (N, 2).
            target (np.ndarray): Target labels for coloring.
            title (str): Plot title.

        Returns:
            matplotlib.figure.Figure: Generated figure.
        """
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=X_proj[:, 0], y=X_proj[:, 1], hue=target, alpha=0.5, palette='viridis')
        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.tight_layout()
        return plt.gcf()
