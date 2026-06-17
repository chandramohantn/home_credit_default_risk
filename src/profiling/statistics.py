"""Module for calculating general dataset statistics."""

from .schema import DatasetStats

class DatasetStatistics:
    """Calculates high-level statistics for a pandas DataFrame."""

    @staticmethod
    def summarize(df) -> DatasetStats:
        """Summarizes the dimensions and memory usage of a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to summarize.

        Returns:
            DatasetStats: A dataclass containing row count, column count, 
                and memory usage in MB.
        """
        return DatasetStats(
            rows=len(df),
            columns=len(df.columns),
            memory_mb=df.memory_usage(deep=True).sum() / 1024**2
        )