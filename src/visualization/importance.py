"""Module for feature importance visualizations."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class ImportanceVisualizer:
    """Visualizes feature importance scores."""

    @staticmethod
    def plot_importance(importance_scores: pd.Series, title: str = "Feature Importance"):
        """Plots a bar chart of importance scores.

        Args:
            importance_scores (pd.Series): Scores indexed by feature name.
            title (str): Plot title.

        Returns:
            matplotlib.figure.Figure: Generated figure.
        """
        plt.figure(figsize=(12, 8))
        sns.barplot(x=importance_scores.values, y=importance_scores.index)
        plt.title(title)
        plt.xlabel("Score")
        plt.ylabel("Features")
        plt.tight_layout()
        return plt.gcf()
