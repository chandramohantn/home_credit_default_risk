"""Module for visualizing feature correlations."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class CorrelationVisualizer:
    """Visualizes correlations between features."""

    @staticmethod
    def plot_heatmap(df: pd.DataFrame, columns: list = None, top_n: int = 20):
        """Plots a correlation heatmap for numeric features.

        Args:
            df (pd.DataFrame): The DataFrame to analyze.
            columns (list, optional): List of specific columns to include. 
                If None, all numeric columns are used. Defaults to None.
            top_n (int): If more than top_n columns are present, only the 
                top_n columns with the highest average absolute correlation 
                are shown. Defaults to 20.

        Returns:
            matplotlib.figure.Figure: The generated figure object.
        """
        if columns is None:
            # Only numeric columns for correlation
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
        corr = df[columns].corr()
        
        # If too many columns, take top N columns with highest average correlation
        if len(columns) > top_n:
            avg_corr = corr.abs().mean().sort_values(ascending=False)
            top_cols = avg_corr.head(top_n).index.tolist()
            corr = df[top_cols].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        return plt.gcf()
