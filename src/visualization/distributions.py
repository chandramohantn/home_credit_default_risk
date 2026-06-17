"""Module for univariate distribution visualizations."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List

class DistributionVisualizer:
    """Plots histograms, KDEs, and boxplots for features."""

    @staticmethod
    def plot_numerical(df: pd.DataFrame, feature: str):
        """Plots distribution (Hist/KDE) and Boxplot for a numerical feature.

        Args:
            df (pd.DataFrame): The dataset.
            feature (str): Feature name.

        Returns:
            matplotlib.figure.Figure: Generated figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Hist + KDE
        sns.histplot(df[feature].dropna(), kde=True, ax=axes[0])
        axes[0].set_title(f"Distribution of {feature}")
        
        # Boxplot
        sns.boxplot(y=df[feature].dropna(), ax=axes[1])
        axes[1].set_title(f"Boxplot of {feature}")
        
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_categorical(df: pd.DataFrame, feature: str, top_n: int = 20):
        """Plots countplot for a categorical feature.

        Args:
            df (pd.DataFrame): The dataset.
            feature (str): Feature name.
            top_n (int): Max categories to show.

        Returns:
            matplotlib.figure.Figure: Generated figure.
        """
        plt.figure(figsize=(12, 6))
        order = df[feature].value_counts().head(top_n).index
        sns.countplot(data=df, y=feature, order=order)
        plt.title(f"Top {top_n} Categories in {feature}")
        plt.tight_layout()
        return plt.gcf()
