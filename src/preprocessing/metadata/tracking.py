"""Metadata tracking and reporting module.

This module provides the MetadataTracker, which logs runtime details, shapes, 
step execution times, and fitted parameters (such as imputation stats, category 
mappings, and outlier bounds) during preprocessing.

In ML engineering, audit trails are required to verify pipeline consistency and debug 
inference errors. By storing the fitted state of all steps in structured JSON files, 
this module ensures full system lineage and reproducibility.
"""

import json
from pathlib import Path
import time
from typing import Dict, Any, List, Optional

class MetadataTracker:
    """Tracks preprocessing statistics and writes runtime JSON reports.

    Attributes:
        output_dir (Path): The directory path to write the JSON reports to.
        run_metadata_ (dict): Master runtime dict detailing step times and shapes.
        missing_report_ (dict): Imputed values and features map.
        outlier_report_ (dict): Outlier limits and clipped record counts.
        encoding_report_ (dict): Mapped categorical levels and output columns.
        scaling_report_ (dict): Fit scaling stats per column.
        feature_metadata_ (dict): Master list of column names, validation outputs, and dtypes.
    """
    
    def __init__(self, output_dir: Path, version: str = "1.0"):
        """Initializes the MetadataTracker and creates target output folders.

        Args:
            output_dir (Path or str): Output directory path for diagnostic reports.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.version = version
        
        # Instantiate master runtime logging dict
        self.run_metadata_: Dict[str, Any] = {
            "version": version,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "steps": [],
            "input_shape": None,
            "output_shape": None,
        }
        self.missing_report_: Dict[str, Any] = {}
        self.outlier_report_: Dict[str, Any] = {}
        self.encoding_report_: Dict[str, Any] = {}
        self.scaling_report_: Dict[str, Any] = {}
        self.transformation_report_: Dict[str, Any] = {}
        self.feature_metadata_: Dict[str, Any] = {}

    def log_step(self, step_name: str, input_cols: int, output_cols: int, elapsed_time: float):
        """Logs execution stats for an individual pipeline step.

        Args:
            step_name (str): The name of the preprocessing step (e.g. 'cleaner').
            input_cols (int): Number of columns before executing the step.
            output_cols (int): Number of columns after executing the step.
            elapsed_time (float): Step execution time in seconds.
        """
        self.run_metadata_["steps"].append({
            "step": step_name,
            "input_columns": input_cols,
            "output_columns": output_cols,
            "elapsed_seconds": elapsed_time
        })

    def log_shapes(self, input_shape: tuple, output_shape: tuple):
        """Logs overall input and output shapes of the preprocessed dataset.

        Args:
            input_shape (tuple): Initial shape of the input DataFrame (rows, columns).
            output_shape (tuple): Final shape of the transformed DataFrame (rows, columns).
        """
        self.run_metadata_["input_shape"] = list(input_shape)
        self.run_metadata_["output_shape"] = list(output_shape)

    def record_missing_values(self, report: Dict[str, Any]):
        """Records imputation statistics (e.g., medians, modes, strategies).

        Args:
            report (dict): Mapped statistics from the MissingValueTransformer.
        """
        self.missing_report_ = report

    def record_outliers(self, report: Dict[str, Any]):
        """Records outlier limits and clipped record counts.

        Args:
            report (dict): Mapped limits from the OutlierTransformer.
        """
        self.outlier_report_ = report

    def record_encodings(self, report: Dict[str, Any]):
        """Records categorical level encoding mapping details.

        Args:
            report (dict): Mapped columns from the EncodingTransformer.
        """
        self.encoding_report_ = report

    def record_scaling(self, report: Dict[str, Any]):
        """Records feature scaling class details.

        Args:
            report (dict): Mapped scaler classes from the FeatureScaler.
        """
        self.scaling_report_ = report

    def record_feature_metadata(self, metadata: Dict[str, Any]):
        """Records schema verification features and validation logs.

        Args:
            metadata (dict): Inferred datatypes and validation warnings.
        """
        self.feature_metadata_ = metadata

    def record_transformations(self, report: Dict[str, Any]):
        """Records feature transformation details and fitted parameters."""

        self.transformation_report_ = report

    def record_report(self, name: str, payload: Dict[str, Any]):
        """Records a report payload under a known report name."""

        if name == "missing_values":
            self.record_missing_values(payload)
        elif name == "outliers":
            self.record_outliers(payload)
        elif name == "encodings":
            self.record_encodings(payload)
        elif name == "scaling":
            self.record_scaling(payload)
        elif name == "transformations":
            self.record_transformations(payload)
        elif name == "feature_metadata":
            self.record_feature_metadata(payload)
        else:
            self.run_metadata_.setdefault("extra_reports", {})[name] = payload

    def save_all_reports(self):
        """Writes all recorded preprocessing reports to designated JSON files.

        Why: Ensures that training runs output separate diagnostic reports 
        that can be archived, versioned, or loaded to audit inference tasks.
        """
        def add_version(data: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(data, dict):
                return {"version": self.version, "data": data}
            return {"version": self.version, **data} if "version" not in data else data

        def save_json(data: Dict[str, Any], filename: str):
            path = self.output_dir / filename
            with open(path, "w") as f:
                json.dump(add_version(data), f, indent=2)

        save_json(self.run_metadata_, "preprocessing_report.json")
        save_json(self.missing_report_, "missing_value_report.json")
        save_json(self.outlier_report_, "outlier_report.json")
        save_json(self.encoding_report_, "encoding_report.json")
        save_json(self.scaling_report_, "scaling_report.json")
        save_json(self.transformation_report_, "transformation_report.json")
        save_json(self.feature_metadata_, "feature_metadata.json")
