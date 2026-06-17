"""Module for data quality analysis including missing values, duplicates, and leakage."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .schema import (
    MissingValueReport,
    DuplicateReport,
    ConstantFeatureReport,
    CardinalityReport,
    LeakageReport
)

class FeatureInspector:
    """Utility class for inspecting feature types in a DataFrame."""

    @staticmethod
    def get_numeric_columns(df: pd.DataFrame) -> List[str]:
        """Identifies numeric columns in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to inspect.

        Returns:
            List[str]: List of numeric column names.
        """
        return df.select_dtypes(include=[np.number]).columns.tolist()

    @staticmethod
    def get_categorical_columns(df: pd.DataFrame) -> List[str]:
        """Identifies categorical (object/category) columns.

        Args:
            df (pd.DataFrame): The DataFrame to inspect.

        Returns:
            List[str]: List of categorical column names.
        """
        return df.select_dtypes(include=["object", "category"]).columns.tolist()

    @staticmethod
    def get_datetime_columns(df: pd.DataFrame) -> List[str]:
        """Identifies datetime columns.

        Args:
            df (pd.DataFrame): The DataFrame to inspect.

        Returns:
            List[str]: List of datetime column names.
        """
        return df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    @staticmethod
    def get_boolean_columns(df: pd.DataFrame) -> List[str]:
        """Identifies boolean columns.

        Args:
            df (pd.DataFrame): The DataFrame to inspect.

        Returns:
            List[str]: List of boolean column names.
        """
        return df.select_dtypes(include=["bool"]).columns.tolist()

class MissingValueAnalyzer:
    """Analyzes missing values in a DataFrame."""

    @staticmethod
    def analyze(df: pd.DataFrame, threshold: float = 50.0) -> MissingValueReport:
        """Calculates missing counts, percentages, and identifies high-missing features.

        Args:
            df (pd.DataFrame): The DataFrame to analyze.
            threshold (float): Percentage threshold for flagging features with 
                high missing values. Defaults to 50.0.

        Returns:
            MissingValueReport: Dataclass containing missing value metrics.
        """
        counts = df.isnull().sum().to_dict()
        percentages = (df.isnull().sum() / len(df) * 100).to_dict()
        features_above_threshold = [
            f for f, p in percentages.items() if p > threshold
        ]
        return MissingValueReport(
            counts=counts,
            percentages=percentages,
            features_above_threshold=features_above_threshold
        )

class DuplicateAnalyzer:
    """Analyzes duplicate rows in a DataFrame."""

    @staticmethod
    def analyze(df: pd.DataFrame) -> DuplicateReport:
        """Counts duplicate rows in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to analyze.

        Returns:
            DuplicateReport: Dataclass containing duplicate count.
        """
        return DuplicateReport(duplicate_count=int(df.duplicated().sum()))

class ConstantFeatureAnalyzer:
    """Detects constant and near-constant features."""

    @staticmethod
    def analyze(df: pd.DataFrame, near_constant_threshold: float = 99.0) -> ConstantFeatureReport:
        """Identifies features with zero or very low variance.

        Args:
            df (pd.DataFrame): The DataFrame to analyze.
            near_constant_threshold (float): Percentage threshold for the 
                most frequent value to consider a feature near-constant. 
                Defaults to 99.0.

        Returns:
            ConstantFeatureReport: Dataclass containing lists of constant 
                and near-constant features.
        """
        constant_features = []
        near_constant_features = []

        for col in df.columns:
            nunique = df[col].nunique()
            if nunique <= 1:
                constant_features.append(col)
            else:
                most_freq_pct = (df[col].value_counts(normalize=True).iloc[0] * 100)
                if most_freq_pct >= near_constant_threshold:
                    near_constant_features.append(col)

        return ConstantFeatureReport(
            constant_features=constant_features,
            near_constant_features=near_constant_features
        )

class CardinalityAnalyzer:
    """Analyzes the cardinality of features."""

    @staticmethod
    def analyze(df: pd.DataFrame, categorical_only: bool = True) -> CardinalityReport:
        """Counts unique values for features.

        Args:
            df (pd.DataFrame): The DataFrame to analyze.
            categorical_only (bool): If True, only analyzes categorical features. 
                Defaults to True.

        Returns:
            CardinalityReport: Dataclass containing cardinality for each feature.
        """
        if categorical_only:
            cols = FeatureInspector.get_categorical_columns(df)
        else:
            cols = df.columns.tolist()
        
        cardinality = df[cols].nunique().to_dict()
        return CardinalityReport(cardinality=cardinality)

class LeakageAnalyzer:
    """Detects potential data leakage columns."""

    @staticmethod
    def analyze(
        df: pd.DataFrame, 
        target_column: str = None, 
        corr_threshold: float = 0.95,
        unique_ratio_threshold: float = 0.95
    ) -> LeakageReport:
        """Identifies columns that might lead to data leakage.

        Logic:
        1. ID-like columns: Features with a very high ratio of unique values 
           to row count (e.g., primary keys like SK_ID_CURR).
        2. High Target Correlation: Numeric features with extremely high 
           absolute correlation with the target variable.

        Args:
            df (pd.DataFrame): The DataFrame to analyze.
            target_column (str, optional): The name of the target column. 
                Required for correlation analysis.
            corr_threshold (float): Absolute correlation threshold above 
                which a feature is flagged. Defaults to 0.95.
            unique_ratio_threshold (float): Unique value ratio threshold 
                above which a feature is flagged as ID-like. Defaults to 0.95.

        Returns:
            LeakageReport: Dataclass containing potential leakage columns.
        """
        potential_leakage_columns = []

        # ID-like columns (high uniqueness)
        for col in df.columns:
            if col == target_column:
                continue
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > unique_ratio_threshold:
                potential_leakage_columns.append(col)

        # High correlation with target
        if target_column and target_column in df.columns:
            numeric_cols = FeatureInspector.get_numeric_columns(df)
            if target_column in numeric_cols:
                correlations = df[numeric_cols].corr()[target_column].abs()
                leaky_corr = correlations[correlations > corr_threshold].index.tolist()
                potential_leakage_columns.extend([c for c in leaky_corr if c != target_column])

        return LeakageReport(potential_leakage_columns=list(set(potential_leakage_columns)))
