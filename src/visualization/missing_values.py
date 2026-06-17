"""Module for visualizing missing values."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict

class MissingValueVisualizer:
    """Visualizes missing value statistics."""

    @staticmethod
    def plot_missing_percentages(percentages: Dict[str, float], top_n: int = 20):
        """Plots a bar chart of features with the highest missing percentages.

        Args:
            percentages (Dict[str, float]): Dictionary mapping features to 
                their missing percentage.
            top_n (int): The number of top missing features to display. 
                Defaults to 20.

        Returns:
            matplotlib.figure.Figure: The generated figure object.
        """
        missing_df = pd.Series(percentages).sort_values(ascending=False).head(top_n)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=missing_df.values, y=missing_df.index)
        plt.title(f"Top {top_n} Features by Missing Percentage")
        plt.xlabel("Percentage Missing (%)")
        plt.ylabel("Features")
        plt.tight_layout()
        return plt.gcf()
