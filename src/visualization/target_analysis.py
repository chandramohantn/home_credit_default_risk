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
