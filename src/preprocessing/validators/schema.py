"""Validation schema check module.

This module provides the SchemaValidator, which verifies that datasets (train, test, 
and validation) match expected column configurations, schemas, and data types.

In real-world credit risk systems, schemas can drift, columns can get renamed, or target 
variables can get dropped during feature extraction. The SchemaValidator acts as a 
fail-fast gateway at the beginning of the pipeline, raising explicit errors early 
rather than allowing silent failures further down the line.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

class SchemaValidator:
    """Validates that a DataFrame conforms to the expected tabular schema.

    Attributes:
        target_column (str, optional): Target column name to verify.
        key_column (str, optional): Primary key column (like 'SK_ID_CURR') to verify uniqueness.
        expected_schema_ (dict): Stored mapping of column names to string types learned in fit.
        columns_ (list of str): Stored list of expected columns.
    """
    
    def __init__(
        self,
        target_column: str = None,
        key_column: str = None,
        allow_extra_columns: bool = True,
    ):
        """Initializes the SchemaValidator.

        Args:
            target_column (str, optional): Name of the target label column.
            key_column (str, optional): Name of the ID/key column.
        """
        self.target_column = target_column
        self.key_column = key_column
        self.allow_extra_columns = allow_extra_columns
        self.expected_schema_: Dict[str, str] = {}
        self.columns_: List[str] = []

    def fit(self, df: pd.DataFrame):
        """Learns reference schema names and column datatypes.

        Args:
            df (pd.DataFrame): Reference DataFrame (typically the training dataset).

        Returns:
            SchemaValidator: The fitted instance of the validator.
        """
        self.columns_ = list(df.columns)
        self.expected_schema_ = {col: str(dtype) for col, dtype in df.dtypes.items()}
        return self

    def validate(self, df: pd.DataFrame, is_train: bool = True) -> Tuple[bool, Dict[str, Any]]:
        """Verifies if the given DataFrame matches the learned reference schema.

        Specifically, validates:
        - Presence of all required columns.
        - Matching of basic numerical vs categorical column types.
        - Absence of duplicate column names.
        - Uniqueness of primary key column values.
        - Target variable existence and non-null status (during training).

        Args:
            df (pd.DataFrame): DataFrame to validate.
            is_train (bool): If True, checks for target variable presence. Defaults to True.

        Returns:
            Tuple[bool, Dict[str, Any]]:
                - bool: True if the DataFrame conforms to the schema, False otherwise.
                - dict: Detailed validation report mapping failure checks.
        """
        report = {
            "missing_columns": [],
            "extra_columns": [],
            "type_mismatches": {},
            "duplicate_columns": [],
            "key_violations": [],
            "target_check": "Passed"
        }
        
        # 1. Check duplicate columns
        # Why: Duplicate column names in pandas can cause errors on selection or indexing.
        cols = list(df.columns)
        if len(cols) != len(set(cols)):
            duplicates = [c for c in set(cols) if cols.count(c) > 1]
            report["duplicate_columns"] = duplicates

        # 2. Check if Schema has been fitted
        if not self.expected_schema_:
            is_valid = len(report["duplicate_columns"]) == 0
            if self.target_column and is_train and self.target_column not in df.columns:
                report["target_check"] = f"Missing target column '{self.target_column}'"
                is_valid = False
            return is_valid, report

        # 3. Check expected columns (excluding target if this is test/inference data)
        expected_cols = list(self.expected_schema_.keys())
        if not is_train and self.target_column in expected_cols:
            expected_cols.remove(self.target_column)

        for col in expected_cols:
            if col not in df.columns:
                report["missing_columns"].append(col)

        # 4. Check for extra columns not defined in the training set
        for col in df.columns:
            if col not in self.expected_schema_ and col != self.target_column:
                report["extra_columns"].append(col)

        # 5. Check data types
        # Why: We group numerical types together (ints and floats) to avoid failing on simple 
        # int64 vs float64 mismatches, but strictly raise errors if numeric turns to text or vice-versa.
        for col in df.columns:
            if col in self.expected_schema_:
                expected_type = self.expected_schema_[col]
                actual_type = str(df[col].dtype)
                is_expected_numeric = "int" in expected_type or "float" in expected_type
                is_actual_numeric = "int" in actual_type or "float" in actual_type
                
                if is_expected_numeric != is_actual_numeric:
                    report["type_mismatches"][col] = {
                        "expected": expected_type,
                        "actual": actual_type
                    }

        # 6. Check target variables constraints (during training)
        if is_train and self.target_column:
            if self.target_column not in df.columns:
                report["target_check"] = f"Missing target column '{self.target_column}'"
            else:
                null_count = df[self.target_column].isnull().sum()
                if null_count > 0:
                    report["target_check"] = f"Target contains {null_count} missing values"

        # 7. Check key column uniqueness constraints
        if self.key_column and self.key_column in df.columns:
            if df[self.key_column].duplicated().any():
                dup_count = df[self.key_column].duplicated().sum()
                report["key_violations"].append(
                    f"Key column '{self.key_column}' contains {dup_count} duplicate values"
                )

        # Evaluate final conforms check
        is_valid = (
            len(report["missing_columns"]) == 0 and
            (self.allow_extra_columns or len(report["extra_columns"]) == 0) and
            len(report["type_mismatches"]) == 0 and
            len(report["duplicate_columns"]) == 0 and
            len(report["key_violations"]) == 0 and
            (report["target_check"] == "Passed" or not is_train)
        )
        
        return is_valid, report
