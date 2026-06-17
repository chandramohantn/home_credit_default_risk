"""Module for outlier visualizations."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class OutlierVisualizer:
    """Visualizes outliers using scatter and box plots."""

    @staticmethod
    def plot_outliers(df: pd.DataFrame, feature: str, target_column: str = None):
        """Plots scatter plot of feature values to highlight outliers.

        Args:
            df (pd.DataFrame): The dataset.
            feature (str): Feature to plot.
            target_column (str, optional): Target for coloring.

        Returns:
            matplotlib.figure.Figure: Generated figure.
        """
        plt.figure(figsize=(10, 6))
        if target_column:
            sns.scatterplot(data=df, x=df.index, y=feature, hue=target_column, alpha=0.6)
        else:
            sns.scatterplot(x=df.index, y=df[feature], alpha=0.6)
            
        plt.title(f"Outlier Detection for {feature}")
        plt.xlabel("Index")
        plt.ylabel(feature)
        plt.tight_layout()
        return plt.gcf()
