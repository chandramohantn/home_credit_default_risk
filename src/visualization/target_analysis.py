"""Module for visualizing target distribution."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class TargetVisualizer:
    """Visualizes the distribution of the target variable."""

    @staticmethod
    def plot_distribution(df: pd.DataFrame, target_column: str):
        """Plots a count plot of the target variable with percentage labels.

        Args:
            df (pd.DataFrame): The DataFrame containing the target column.
            target_column (str): The name of the target column.

        Returns:
            matplotlib.figure.Figure: The generated figure object.
        """
        plt.figure(figsize=(8, 6))
        sns.countplot(x=target_column, data=df)
        plt.title(f"Distribution of {target_column}")
        
        # Add percentage labels
        total = len(df)
        for p in plt.gca().patches:
            height = p.get_height()
            percentage = '{:.1f}%'.format(100 * height/total)
            x = p.get_x() + p.get_width() / 2 - 0.05
            y = p.get_y() + height
            plt.gca().annotate(percentage, (x, y), size=12)
            
        plt.tight_layout()
        return plt.gcf()

    @staticmethod
    def plot_numeric_vs_target(df: pd.DataFrame, feature: str, target_column: str):
        """Plots distribution overlay and boxplot vs target.

        Args:
            df (pd.DataFrame): The dataset.
            feature (str): Numerical feature.
            target_column (str): Target column.

        Returns:
            matplotlib.figure.Figure: Generated figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # KDE overlay
        sns.kdeplot(data=df, x=feature, hue=target_column, common_norm=False, ax=axes[0])
        axes[0].set_title(f"{feature} Distribution by {target_column}")
        
        # Boxplot vs target
        sns.boxplot(data=df, x=target_column, y=feature, ax=axes[1])
        axes[1].set_title(f"{feature} Boxplot by {target_column}")
        
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_categorical_vs_target(df: pd.DataFrame, feature: str, target_column: str, top_n: int = 15):
        """Plots stacked bar chart and target rate per category.

        Args:
            df (pd.DataFrame): The dataset.
            feature (str): Categorical feature.
            target_column (str): Target column.

        Returns:
            matplotlib.figure.Figure: Generated figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        order = df[feature].value_counts().head(top_n).index
        
        # Stacked bar (Count)
        sns.countplot(data=df, y=feature, hue=target_column, order=order, ax=axes[0])
        axes[0].set_title(f"{feature} Counts by {target_column}")
        
        # Target Rate bar
        target_rates = df.groupby(feature)[target_column].mean().loc[order]
        sns.barplot(x=target_rates.values, y=target_rates.index, ax=axes[1])
        axes[1].set_title(f"{feature} Target Rate")
        axes[1].axvline(df[target_column].mean(), color='red', linestyle='--', label='Global Average')
        
        plt.tight_layout()
        return fig
