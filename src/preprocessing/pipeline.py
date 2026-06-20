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
import time
from copy import deepcopy
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from src.preprocessing.validators.schema import SchemaValidator
from src.preprocessing.validators.leakage import LeakageValidator
from src.preprocessing.metadata.tracking import MetadataTracker
from src.preprocessing.registry import LEGACY_STEP_ALIASES, TRANSFORMER_REGISTRY

class PreprocessingPipeline:
    """Orchestrator for fitting and executing preprocessing configurations.

    Translates static dictionary/YAML configurations into active scikit-learn
    transformer objects, supports config-driven step ordering/component selection,
    and coordinates dataset-level validation.

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
    
    def __init__(
        self,
        config: Dict[str, Any],
        reports_dir: str = "reports",
        target_column: str = "TARGET",
        component_registry: Optional[Dict[str, Any]] = None,
    ):
        """Initializes the PreprocessingPipeline orchestrator.

        Args:
            config (dict): Pipeline parameters dictionary (typically loaded from YAML).
            reports_dir (str): Folder path to save execution JSONs. Defaults to "reports".
            target_column (str): Label column name. Defaults to "TARGET".
        """
        self.config = config
        self.reports_dir = Path(reports_dir)
        self.target_column = target_column
        self.component_registry = {**TRANSFORMER_REGISTRY, **(component_registry or {})}
        self.pipeline_cfg = self.config.get("pipeline", {})
        metadata_cfg = self.pipeline_cfg.get("metadata", {})
        self.pipeline_version = metadata_cfg.get("version", "1.0")
         
        # Instantiate validators and trackers
        schema_cfg = self._parse_schema_validation_config(self.pipeline_cfg.get("schema_validation", True))
        self.run_schema_validation = schema_cfg["enabled"]
        self.strict_schema_validation = schema_cfg["strict"]
        self.schema_validator = SchemaValidator(
            target_column=self.target_column,
            key_column=schema_cfg.get("key_column"),
            allow_extra_columns=schema_cfg.get("allow_extra_columns", True),
        )

        leakage_cfg = self._parse_leakage_validation_config(self.pipeline_cfg.get("leakage_validation", True))
        self.run_leakage_validation = leakage_cfg["enabled"]
        self.fail_on_leakage = leakage_cfg["fail_on_detection"]
        self.leakage_validator = LeakageValidator(
            target_column=self.target_column,
            cardinality_threshold=leakage_cfg["cardinality_threshold"],
            correlation_threshold=leakage_cfg["correlation_threshold"],
        )
        self.tracker = MetadataTracker(self.reports_dir, version=self.pipeline_version)
         
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
        pipeline_cfg = self.pipeline_cfg
        available_components = {
            LEGACY_STEP_ALIASES.get(name, name)
            for name in pipeline_cfg.get("available_components", self.component_registry.keys())
        }
        step_configs = pipeline_cfg.get("steps") or self._build_legacy_step_configs(pipeline_cfg)

        for step_cfg in step_configs:
            normalized_step = self._normalize_step_config(step_cfg)
            if not normalized_step["enabled"]:
                continue

            component_name = normalized_step["component"]
            if component_name not in available_components:
                raise ValueError(
                    f"Component '{component_name}' is not available. "
                    f"Available components: {sorted(available_components)}"
                )

            transformer_class = self.component_registry.get(component_name)
            if transformer_class is None:
                raise ValueError(f"Unknown pipeline component: {component_name}")

            self.steps_.append(
                (normalized_step["name"], transformer_class(**normalized_step["params"]))
            )

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
            self._handle_schema_validation(is_valid, report, stage="transform")
                
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
        
        self.tracker.log_shapes(df.shape, df.shape)
        self.tracker.run_metadata_["pipeline_steps"] = [name for name, _ in self.steps_]
        self.tracker.run_metadata_["config_snapshot"] = deepcopy(self.config)

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
            self._handle_schema_validation(is_valid, val_report, stage="fit")
        
        # 2. Leakage Validation Checks
        if y is not None and self.run_leakage_validation:
            leakage_report = self.leakage_validator.analyze(pd.concat([df, y], axis=1))
            self.tracker.run_metadata_["leakage_analysis"] = leakage_report
            self._handle_leakage_report(leakage_report)

        # 3. Fit and Transform sequentially
        for name, transformer in self.steps_:
            start_time = time.time()
            input_cols = df.shape[1]

            if hasattr(transformer, "fit_transform"):
                df = transformer.fit_transform(df, df_y)
            else:
                transformer.fit(df, df_y)
                df = transformer.transform(df)

            if df_y is not None and not df.index.equals(df_y.index):
                df_y = df_y.loc[df.index]
                
            elapsed = time.time() - start_time
            output_cols = df.shape[1]
            
            # Log step metrics to tracker
            self.tracker.log_step(name, input_cols, output_cols, elapsed)

            for report_name, payload in transformer.get_reports().items():
                self.tracker.record_report(report_name, payload)

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

    def _build_legacy_step_configs(self, pipeline_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Builds ordered step configs from the legacy preprocessing configuration format."""

        step_configs: List[Dict[str, Any]] = []
        if pipeline_cfg.get("type_standardization", True) or pipeline_cfg.get("drop_duplicates", True):
            step_configs.append({
                "component": "cleaner",
                "params": {
                    "drop_row_duplicates": pipeline_cfg.get("drop_duplicates", True),
                    "drop_col_duplicates": pipeline_cfg.get("drop_duplicates", True),
                    "coerce_objects_to_category": pipeline_cfg.get("type_standardization", True),
                    "datetime_cols": pipeline_cfg.get("datetime_cols", []),
                },
            })

        if "imputation" in pipeline_cfg:
            imp_cfg = pipeline_cfg["imputation"]
            step_configs.append({
                "component": "missing",
                "params": {
                    "num_strategy": imp_cfg.get("numerical", {}).get("strategy", "median"),
                    "num_fill_value": imp_cfg.get("numerical", {}).get("fill_value"),
                    "cat_strategy": imp_cfg.get("categorical", {}).get("strategy", "constant"),
                    "cat_fill_value": imp_cfg.get("categorical", {}).get("fill_value", "Unknown"),
                    "add_indicators": imp_cfg.get("numerical", {}).get("add_indicators", True),
                    "column_strategies": imp_cfg.get("column_strategies", {}),
                    "knn_neighbors": imp_cfg.get("knn_neighbors", 5),
                },
            })

        if "rare_categories" in pipeline_cfg:
            rare_cfg = pipeline_cfg["rare_categories"]
            step_configs.append({
                "component": "rare_categories",
                "params": {
                    "threshold_pct": rare_cfg.get("threshold_pct", 0.01),
                    "threshold_count": rare_cfg.get("threshold_count"),
                    "top_k": rare_cfg.get("top_k"),
                    "fill_value": rare_cfg.get("fill_value", "OTHER"),
                    "exclude_cols": rare_cfg.get("exclude_cols", []),
                },
            })

        if "encoding" in pipeline_cfg:
            enc_cfg = pipeline_cfg["encoding"]
            step_configs.append({
                "component": "encoder",
                "params": {
                    "default_strategy": enc_cfg.get("strategy", "one_hot"),
                    "column_strategies": enc_cfg.get("column_strategies", {}),
                    "target_cv_folds": enc_cfg.get("target_cv_folds", 5),
                    "target_smoothing": enc_cfg.get("target_smoothing", 10.0),
                    "catboost_prior": enc_cfg.get("catboost_prior", 10.0),
                    "random_state": enc_cfg.get("random_state", 42),
                },
            })

        if "outliers" in pipeline_cfg and pipeline_cfg["outliers"].get("strategy", "none") != "none":
            out_cfg = pipeline_cfg["outliers"]
            step_configs.append({
                "component": "outliers",
                "params": {
                    "strategy": out_cfg.get("strategy", "clip"),
                    "method": out_cfg.get("method", "iqr"),
                    "threshold": out_cfg.get("threshold", 1.5),
                    "lower_quantile": out_cfg.get("lower_quantile", 0.01),
                    "upper_quantile": out_cfg.get("upper_quantile", 0.99),
                    "contamination": out_cfg.get("contamination", 0.05),
                    "random_state": out_cfg.get("random_state", 42),
                    "columns": out_cfg.get("columns"),
                },
            })

        if pipeline_cfg.get("transformations"):
            step_configs.append({
                "component": "transformations",
                "params": {"transformations": pipeline_cfg["transformations"]},
            })

        if "scaling" in pipeline_cfg and pipeline_cfg["scaling"].get("strategy", "none") != "none":
            scale_cfg = pipeline_cfg["scaling"]
            step_configs.append({
                "component": "scaler",
                "params": {
                    "default_strategy": scale_cfg.get("strategy", "standard"),
                    "column_strategies": scale_cfg.get("column_strategies", {}),
                    "exclude_cols": scale_cfg.get("exclude_cols", []),
                },
            })

        return step_configs

    def _normalize_step_config(self, step_cfg: Any) -> Dict[str, Any]:
        """Normalizes user-provided step configs into a consistent internal structure."""

        if isinstance(step_cfg, str):
            component = LEGACY_STEP_ALIASES.get(step_cfg, step_cfg)
            return {"name": component, "component": component, "params": {}, "enabled": True}

        if not isinstance(step_cfg, dict):
            raise TypeError("Each pipeline step must be a string or dictionary.")

        component = step_cfg.get("component") or step_cfg.get("type") or step_cfg.get("name")
        if component is None:
            raise ValueError("Each pipeline step dictionary must define 'component'.")

        component = LEGACY_STEP_ALIASES.get(component, component)
        return {
            "name": step_cfg.get("name", component),
            "component": component,
            "params": step_cfg.get("params", {}),
            "enabled": step_cfg.get("enabled", True),
        }

    def _parse_schema_validation_config(self, config_value: Any) -> Dict[str, Any]:
        """Parses schema validation settings from a bool or dict."""

        if isinstance(config_value, dict):
            return {
                "enabled": config_value.get("enabled", True),
                "strict": config_value.get("strict", True),
                "key_column": config_value.get("key_column"),
                "allow_extra_columns": config_value.get("allow_extra_columns", True),
            }
        return {
            "enabled": bool(config_value),
            "strict": bool(config_value),
            "key_column": None,
            "allow_extra_columns": True,
        }

    def _parse_leakage_validation_config(self, config_value: Any) -> Dict[str, Any]:
        """Parses leakage validation settings from a bool or dict."""

        if isinstance(config_value, dict):
            return {
                "enabled": config_value.get("enabled", True),
                "fail_on_detection": config_value.get("fail_on_detection", False),
                "cardinality_threshold": config_value.get("cardinality_threshold", 0.95),
                "correlation_threshold": config_value.get("correlation_threshold", 0.95),
            }
        return {
            "enabled": bool(config_value),
            "fail_on_detection": False,
            "cardinality_threshold": 0.95,
            "correlation_threshold": 0.95,
        }

    def _handle_schema_validation(self, is_valid: bool, report: Dict[str, Any], stage: str) -> None:
        """Raises a fail-fast schema error when validation is configured as strict."""

        if is_valid or not self.strict_schema_validation:
            return
        raise ValueError(f"Schema validation failed during {stage}: {report}")

    def _handle_leakage_report(self, leakage_report: Dict[str, Any]) -> None:
        """Raises when configured leakage checks detect suspicious columns."""

        has_findings = any(bool(value) for value in leakage_report.values())
        if self.fail_on_leakage and has_findings:
            raise ValueError(f"Leakage validation failed: {leakage_report}")
