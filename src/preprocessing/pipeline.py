"""Preprocessing pipeline orchestration framework.

This module provides the main PreprocessingPipeline class. It integrates 
individual data transformers (DataCleaner, MissingValueTransformer, 
RareCategoryTransformer, EncodingTransformer, OutlierTransformer, FeatureTransformer, 
and FeatureScaler) with validation schemas (SchemaValidator) and leakage trackers 
(LeakageValidator) into a single unified sklearn-like API.

By orchestrating all cleaning and transformation processes through this single pipeline, 
we ensure that transformations are applied in a strict, predictable order without 
introducing target leakage, and that we track execution metrics via a MetadataTracker.
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from src.preprocessing.validators.schema import SchemaValidator
from src.preprocessing.validators.leakage import LeakageValidator
from src.preprocessing.metadata.tracking import MetadataTracker
from src.preprocessing.registry import TRANSFORMER_REGISTRY

class PreprocessingPipeline:
    """Orchestrator for fitting and executing preprocessing configurations.

    Translates static dictionary/YAML configurations into active scikit-learn 
    transformer objects and coordinates dataset-level validation.

    Attributes:
        config (dict): Pipeline execution parameters dictionary.
        reports_dir (Path): Output directory for JSON logs and reports.
        target_column (str): Target column name.
        schema_validator (SchemaValidator): Schema integrity checker.
        leakage_validator (LeakageValidator): Data leakage verification module.
        tracker (MetadataTracker): Auditing and reporting module.
        steps_ (list of Tuple[str, Any]): Preprocessing steps as name/transformer pairs.
        fitted_ (bool): Fits state flag.
    """
    
    def __init__(self, config: Dict[str, Any], reports_dir: str = "reports", target_column: str = "TARGET"):
        """Initializes the PreprocessingPipeline orchestrator.

        Args:
            config (dict): Pipeline parameters dictionary (typically loaded from YAML).
            reports_dir (str): Folder path to save execution JSONs. Defaults to "reports".
            target_column (str): Label column name. Defaults to "TARGET".
        """
        self.config = config
        self.reports_dir = Path(reports_dir)
        self.target_column = target_column
        
        # Instantiate validators and trackers
        self.schema_validator = SchemaValidator(target_column=self.target_column)
        self.leakage_validator = LeakageValidator(target_column=self.target_column)
        self.tracker = MetadataTracker(self.reports_dir)
        
        self.steps_: List[Tuple[str, Any]] = []
        self.fitted_ = False
        
        # Dynamic build process
        self._build_pipeline()

    def _build_pipeline(self):
        """Parses the configuration parameters to instantiate transformers in sequence.

        Why: Dynamic assembly based on config definitions allows trees, linear, and neural 
        networks pipelines to share implementation files but run different combinations 
        (e.g., skip scaling for trees, apply one-hot vs target encoding).
        """
        pipeline_cfg = self.config.get("pipeline", {})
        
        # 1. Schema Validation Configuration (checked before transforming)
        self.run_schema_validation = pipeline_cfg.get("schema_validation", True)
        
        # 2. Data Cleaner Setup
        if pipeline_cfg.get("type_standardization", True) or pipeline_cfg.get("drop_duplicates", True):
            cleaner_cfg = {
                "drop_row_duplicates": pipeline_cfg.get("drop_duplicates", True),
                "drop_col_duplicates": pipeline_cfg.get("drop_duplicates", True),
                "coerce_objects_to_category": pipeline_cfg.get("type_standardization", True),
                "datetime_cols": pipeline_cfg.get("datetime_cols", [])
            }
            cleaner_class = TRANSFORMER_REGISTRY["cleaner"]
            self.steps_.append(("cleaner", cleaner_class(**cleaner_cfg)))
            
        # 3. Missing Value Imputer Setup
        if "imputation" in pipeline_cfg:
            imp_cfg = pipeline_cfg["imputation"]
            missing_cfg = {
                "num_strategy": imp_cfg.get("numerical", {}).get("strategy", "median"),
                "num_fill_value": imp_cfg.get("numerical", {}).get("fill_value", None),
                "cat_strategy": imp_cfg.get("categorical", {}).get("strategy", "constant"),
                "cat_fill_value": imp_cfg.get("categorical", {}).get("fill_value", "Unknown"),
                "add_indicators": imp_cfg.get("numerical", {}).get("add_indicators", True),
                "column_strategies": imp_cfg.get("column_strategies", {})
            }
            missing_class = TRANSFORMER_REGISTRY["missing"]
            self.steps_.append(("missing", missing_class(**missing_cfg)))
            
        # 4. Rare Category Setup
        if "rare_categories" in pipeline_cfg:
            rare_cfg = pipeline_cfg["rare_categories"]
            rare_class = TRANSFORMER_REGISTRY["rare_categories"]
            self.steps_.append(("rare_categories", rare_class(
                threshold_pct=rare_cfg.get("threshold_pct", 0.01),
                threshold_count=rare_cfg.get("threshold_count", None),
                top_k=rare_cfg.get("top_k", None),
                fill_value=rare_cfg.get("fill_value", "OTHER"),
                exclude_cols=rare_cfg.get("exclude_cols", [])
            )))

        # 5. Encoding Setup
        if "encoding" in pipeline_cfg:
            enc_cfg = pipeline_cfg["encoding"]
            encoding_class = TRANSFORMER_REGISTRY["encoder"]
            self.steps_.append(("encoder", encoding_class(
                default_strategy=enc_cfg.get("strategy", "one_hot"),
                column_strategies=enc_cfg.get("column_strategies", {}),
                target_cv_folds=enc_cfg.get("target_cv_folds", 5),
                target_smoothing=enc_cfg.get("target_smoothing", 10.0)
            )))

        # 6. Outlier Setup
        if "outliers" in pipeline_cfg:
            out_cfg = pipeline_cfg["outliers"]
            if out_cfg.get("strategy", "none") != "none":
                outlier_class = TRANSFORMER_REGISTRY["outliers"]
                self.steps_.append(("outliers", outlier_class(
                    strategy=out_cfg.get("strategy", "clip"),
                    method=out_cfg.get("method", "iqr"),
                    threshold=out_cfg.get("threshold", 1.5),
                    lower_quantile=out_cfg.get("lower_quantile", 0.01),
                    upper_quantile=out_cfg.get("upper_quantile", 0.99),
                    columns=out_cfg.get("columns", None)
                )))

        # 7. Transformations Setup
        if "transformations" in pipeline_cfg:
            trans_cfg = pipeline_cfg["transformations"]
            if trans_cfg:
                trans_class = TRANSFORMER_REGISTRY["transformations"]
                self.steps_.append(("transformations", trans_class(transformations=trans_cfg)))

        # 8. Feature Scaler Setup
        if "scaling" in pipeline_cfg:
            scale_cfg = pipeline_cfg["scaling"]
            if scale_cfg.get("strategy", "none") != "none":
                scaler_class = TRANSFORMER_REGISTRY["scaler"]
                self.steps_.append(("scaler", scaler_class(
                    default_strategy=scale_cfg.get("strategy", "standard"),
                    column_strategies=scale_cfg.get("column_strategies", {}),
                    exclude_cols=scale_cfg.get("exclude_cols", [])
                )))

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fits all transformers sequentially on the training split.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series, optional): Target variable labels. Defaults to None.

        Returns:
            PreprocessingPipeline: The fitted orchestrator instance.
        """
        self.fit_transform(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms validation or testing features using fitted estimators.

        Does not perform leakage detection, validation schema fitting, or row 
        dropping (to prevent changing sample sizes during predictions).

        Args:
            X (pd.DataFrame): Features dataset to transform.

        Returns:
            pd.DataFrame: Transformed features DataFrame.

        Raises:
            ValueError: If the pipeline has not been fitted prior to transform.
        """
        if not self.fitted_:
            raise ValueError("PreprocessingPipeline is not fitted yet. Call 'fit' or 'fit_transform' first.")
            
        df = X.copy()
        
        # Apply validation check against training schema representation
        if self.run_schema_validation:
            is_valid, report = self.schema_validator.validate(df, is_train=False)
            if not is_valid:
                # Logs warnings but allows processing to attempt execution. Can raise error if strictness is modified.
                pass
                
        # Transform sequentially through each step
        for name, transformer in self.steps_:
            df = transformer.transform(df)
            
        return df

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fits all steps and transforms training features.

        Performs:
        - Schema check fitting.
        - Data leakage checks prior to fit.
        - Sequentially fits and transforms steps.
        - Updates index alignment if cleaner drops duplicates.
        - Extracts diagnostic logs for tracker logging.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series, optional): Target variable. Defaults to None.

        Returns:
            pd.DataFrame: Fully preprocessed training DataFrame.

        Raises:
            ValueError: If schema checks fail on critical target variable requirements.
        """
        df = X.copy()
        df_y = y.copy() if y is not None else None
        
        self.tracker.log_shapes(df.shape, df.shape) # Initial shape log

        # 1. Schema Validation (Fit & Validate on concatenated X & y)
        # Why: Schema validator requires target presence on fit check during train.
        if self.run_schema_validation:
            df_to_validate = pd.concat([df, y], axis=1) if y is not None else df
            self.schema_validator.fit(df_to_validate)
            is_valid, val_report = self.schema_validator.validate(df_to_validate, is_train=True)
            self.tracker.record_feature_metadata({
                "expected_schema": self.schema_validator.expected_schema_,
                "validation_report": val_report
            })
            if not is_valid:
                if val_report.get("target_check") != "Passed" and y is not None:
                    raise ValueError(f"Schema validation failed: {val_report['target_check']}")
        
        # 2. Leakage Validation Checks
        if y is not None:
            leakage_report = self.leakage_validator.analyze(pd.concat([df, y], axis=1))
            self.tracker.run_metadata_["leakage_analysis"] = leakage_report

        # 3. Fit and Transform sequentially
        for name, transformer in self.steps_:
            start_time = time.time()
            input_cols = df.shape[1]
            
            # Why: The Cleaner handles duplicate row deletion on training files. 
            # We explicitly subset df_y index here to maintain coordinate alignment with df features.
            if name == "cleaner" and transformer.drop_row_duplicates and df_y is not None:
                non_duplicate_mask = ~df.duplicated()
                df = df[non_duplicate_mask]
                df_y = df_y.loc[df.index]
                
            # Perform fit_transform or fit + transform
            if hasattr(transformer, "fit_transform"):
                df = transformer.fit_transform(df, df_y)
            else:
                transformer.fit(df, df_y)
                df = transformer.transform(df)
                
            elapsed = time.time() - start_time
            output_cols = df.shape[1]
            
            # Log step metrics to tracker
            self.tracker.log_step(name, input_cols, output_cols, elapsed)

            # Extract diagnostic values for logging reports
            if name == "missing":
                self.tracker.record_missing_values(transformer.imputation_values_report_)
            elif name == "outliers":
                self.tracker.record_outliers({
                    "limits": {col: list(lim) for col, lim in transformer.limits_.items()},
                    "outlier_counts": transformer.outlier_counts_
                })
            elif name == "encoder":
                self.tracker.record_encodings({
                    col: {
                        "strategy": info["strategy"],
                        "output_columns": info["output_columns"]
                    } for col, info in transformer.encoding_info_.items()
                })
            elif name == "scaler":
                self.tracker.record_scaling({
                    col: str(type(scaler)) for col, scaler in transformer.scalers_.items()
                })

        self.fitted_ = True
        self.tracker.log_shapes(X.shape, df.shape)
        
        # Save all reports to JSON files
        self.tracker.save_all_reports()
        
        return df

    def get_feature_names_out(self):
        """Retraces the final list of columns outputted by the pipeline.

        Returns:
            list of str: Final list of feature column names.

        Raises:
            ValueError: If the pipeline is not fitted.
        """
        if not self.fitted_:
            raise ValueError("Pipeline is not fitted yet.")
        
        # Retrace output columns from the final active step in steps_ list
        if self.steps_:
            return self.steps_[-1][1].get_feature_names_out()
        return self.schema_validator.columns_
