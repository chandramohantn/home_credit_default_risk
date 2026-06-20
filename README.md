# Home Credit Default Risk

This project is building a reusable credit-risk experimentation framework on top of the Home Credit Default Risk dataset. So far, the implementation covers three major phases: **data profiling and quality analysis**, **exploratory data analysis (EDA)**, and **data cleaning/preprocessing**.

At a high level, we have already implemented:

1. **Profiling and quality analysis** to summarize datasets, measure missingness, detect duplicates and near-constant features, analyze categorical cardinality, inspect target imbalance, and flag potential data leakage.
2. **EDA orchestration** to generate target analysis, univariate and bivariate insights, correlation studies, outlier checks, mutual-information feature importance, dimensionality reduction outputs, figures, CSV deliverables, and markdown summaries.
3. **A reusable preprocessing framework** with strict/configurable validation, leakage checks, metadata tracking, configurable transformers, strategy registries, and model-specific pipelines for tree-based, linear, and neural-network workflows.
4. **Execution scripts and reports** that load raw data, run the implemented phases, and persist processed datasets plus JSON/CSV/figure artifacts under `reports/` and `data/processed/`.

## Implemented modules, classes, and functions

The sections below keep the same information as the original inventory, but use a cleaner hierarchy:

- **Domain** -> top-level package or project area
- **Module** -> Python file / logical unit
- **Class / object** -> implemented class or exported object
- **Method** -> one-line responsibility

### Data

#### `loader`

**`DatasetLoader`** — loads datasets from a configured data directory.

- `load_csv` — loads a CSV file from the configured data directory into a DataFrame.

### Profiling

#### `profiler`

**`DataProfiler`** — orchestrates the full profiling workflow for a dataset.

- `run` — executes all profiling analysis steps on the given DataFrame.

#### `statistics`

**`DatasetStatistics`** — computes high-level dataset statistics.

- `summarize` — summarizes row count, column count, and memory usage for a DataFrame.

#### `quality`

**`FeatureInspector`** — provides helper methods for inspecting feature types in a DataFrame.

- `get_numeric_columns` — identifies numeric columns in the DataFrame.
- `get_categorical_columns` — identifies categorical columns in the DataFrame.
- `get_datetime_columns` — identifies datetime columns in the DataFrame.
- `get_boolean_columns` — identifies boolean columns in the DataFrame.

**`MissingValueAnalyzer`** — analyzes missing-value patterns in a DataFrame.

- `analyze` — calculates missing counts, missing percentages, and high-missing features.

**`DuplicateAnalyzer`** — analyzes duplicate rows in a DataFrame.

- `analyze` — counts duplicate rows in the DataFrame.

**`ConstantFeatureAnalyzer`** — detects constant and near-constant features.

- `analyze` — identifies zero-variance and low-variance features using a frequency threshold.

**`CardinalityAnalyzer`** — analyzes feature cardinality.

- `analyze` — counts unique values for categorical features or all features.

**`LeakageAnalyzer`** — flags columns that may introduce data leakage.

- `analyze` — detects ID-like columns and features with suspiciously high target correlation.

#### `target`

**`TargetAnalyzer`** — analyzes target distribution and class imbalance for profiling.

- `analyze` — calculates positive/negative counts, imbalance ratio, and baseline accuracy.

#### `schema`

**`DatasetStats`** — stores dataset shape and memory-usage summary values.

**`MissingValueReport`** — stores missing-value counts, percentages, and flagged features.

**`DuplicateReport`** — stores the duplicate-row count.

**`ConstantFeatureReport`** — stores constant and near-constant feature lists.

**`CardinalityReport`** — stores per-feature unique-value counts.

**`TargetAnalysisReport`** — stores target-distribution and imbalance metrics.

**`LeakageReport`** — stores columns flagged as potential leakage sources.

**`ProfilingResult`** — aggregates all profiling outputs into a single result object.

#### `report`

**`ReportGenerator`** — saves profiling outputs to structured report files.

- `save_results` — writes profiling results to JSON and CSV files.
- `save_figure` — saves a matplotlib figure to the profiling figures directory.

### Analyzers

#### `schema`

**`TargetEDAResult`** — stores target-distribution and baseline-metric outputs for EDA.

**`NumericalUnivariateStats`** — stores descriptive statistics for one numerical feature.

**`CategoricalUnivariateStats`** — stores category-frequency and rarity statistics for one categorical feature.

**`NumericBivariateStats`** — stores target-separation statistics for one numerical feature.

**`CategoricalBivariateStats`** — stores target-rate and association metrics for one categorical feature.

**`OutlierStats`** — stores IQR-based and Z-score-based outlier counts and indices for a feature.

#### `target`

**`TargetAnalyzer`** — analyzes the target variable for EDA.

- `analyze` — calculates target counts, percentages, and majority-class baseline metrics.

#### `univariate`

**`UnivariateAnalyzer`** — analyzes features independently.

- `analyze_numeric` — computes descriptive statistics for numerical columns.
- `analyze_categorical` — computes cardinality, top-category, and rare-category statistics for categorical columns.

#### `bivariate`

**`BivariateAnalyzer`** — analyzes relationships between features and the target.

- `analyze_numeric` — computes mean/median differences, Cohen's d, and mutual information for numerical features.
- `analyze_categorical` — computes target rates, chi-square scores, and mutual information for categorical features.

#### `correlation`

**`CorrelationAnalyzer`** — analyzes feature correlation and multicollinearity.

- `analyze_correlations` — calculates a numeric correlation matrix.
- `calculate_vif` — calculates variance inflation factor scores for selected columns.

#### `outlier`

**`OutlierAnalyzer`** — detects outliers in numerical features.

- `analyze` — identifies outliers using IQR and Z-score methods.

#### `importance`

**`ImportanceAnalyzer`** — estimates feature importance using model-agnostic statistics.

- `calculate_mutual_info` — calculates top mutual-information scores for features against the target.

#### `dimensionality`

**`DimensionalityAnalyzer`** — runs dimensionality-reduction analyses.

- `run_pca` — performs PCA on numerical features and returns the projection plus fitted PCA object.
- `run_umap` — performs UMAP on numerical features when `umap-learn` is available.

#### `missing_patterns`

**`MissingPatternAnalyzer`** — analyzes structured missingness patterns.

- `calculate_missing_correlations` — calculates correlations between feature-level missingness indicators.
- `analyze_missing_vs_target` — measures whether missingness is associated with target behavior.

### EDA

#### `eda_orchestrator`

**`EDAOrchestrator`** — coordinates the full Phase 3 exploratory data analysis workflow.

- `run` — executes target analysis, feature analysis, plotting, and report generation for a dataset.
- `_save_csv_deliverables` — saves structured CSV outputs for target relationships and univariate insights.
- `_generate_summary_report` — writes the markdown EDA summary report.

### Visualization

#### `target_analysis`

**`TargetVisualizer`** — creates target-related visualizations.

- `plot_distribution` — plots the target distribution with percentage labels.
- `plot_numeric_vs_target` — plots a numerical feature against the target using distribution and box plots.
- `plot_categorical_vs_target` — plots categorical counts and target rates by category.

#### `distributions`

**`DistributionVisualizer`** — creates univariate distribution plots.

- `plot_numerical` — plots histogram/KDE and boxplot views for a numerical feature.
- `plot_categorical` — plots category counts for a categorical feature.

#### `correlations`

**`CorrelationVisualizer`** — visualizes feature-correlation structure.

- `plot_heatmap` — plots a correlation heatmap for selected numeric features.

#### `outliers`

**`OutlierVisualizer`** — visualizes outlier behavior in a feature.

- `plot_outliers` — plots feature values, optionally colored by target, to highlight outliers.

#### `importance`

**`ImportanceVisualizer`** — visualizes feature-importance scores.

- `plot_importance` — plots a bar chart of importance scores.

#### `dimensionality`

**`DimensionalityVisualizer`** — visualizes low-dimensional projections.

- `plot_projection` — plots a 2D projection colored by target values.

#### `missing_values`

**`MissingValueVisualizer`** — visualizes missing-value statistics.

- `plot_missing_percentages` — plots the features with the highest missing percentages.

### Preprocessing

#### `pipeline`

**`PreprocessingPipeline`** — orchestrates schema checks, leakage checks, transformers, and reporting in a sklearn-like preprocessing workflow.

- `fit` — fits the preprocessing pipeline on training data.
- `transform` — applies previously fitted preprocessing steps to new data.
- `fit_transform` — fits all preprocessing steps and transforms the training data in sequence.
- `get_feature_names_out` — returns the final output feature names after preprocessing.

This pipeline now supports **config-driven step order** and **component whitelisting** through `pipeline.steps` and `pipeline.available_components`.

#### `registry`

**`TRANSFORMER_REGISTRY`** — maps configuration keys to transformer classes for dynamic pipeline construction.

**`LEGACY_STEP_ALIASES`** — maps legacy config section names to pipeline component names.

#### `strategies`

**`ENCODER_REGISTRY`** — lists the supported categorical encoding strategies.

**`SCALER_REGISTRY`** — maps scaling strategy names to scaler implementations.

**`TRANSFORMATION_REGISTRY`** — lists the supported feature-transformation strategies and their metadata.

**`OUTLIER_METHOD_REGISTRY`** — lists the supported outlier-detection methods.

#### `validators`

**`SchemaValidator`** — validates that incoming datasets match the expected schema.

- `fit` — learns the reference schema from a DataFrame.
- `validate` — checks required columns, type compatibility, duplicate columns, key uniqueness, and target validity.

**`LeakageValidator`** — validates whether features show classic leakage patterns before training.

- `analyze` — flags ID-like columns, near-constant columns, and highly target-correlated numeric columns.

#### `metadata`

**`MetadataTracker`** — records preprocessing metadata and writes JSON diagnostics.

- `log_step` — records execution metrics for one preprocessing step.
- `log_shapes` — records the overall input and output dataset shapes.
- `record_missing_values` — records fitted missing-value statistics.
- `record_outliers` — records fitted outlier limits and counts.
- `record_encodings` — records categorical-encoding details.
- `record_scaling` — records the scaler used for each feature.
- `record_feature_metadata` — records schema-validation and feature-level metadata.
- `save_all_reports` — writes all tracked preprocessing reports to JSON files.

#### `transformers`

**`BaseTransformer`** — provides a common sklearn-compatible interface for custom transformers.

- `fit` — marks the transformer as fitted and provides the standard fit contract.
- `transform` — validates fitted state before a subclass applies transformations.
- `get_feature_names_out` — returns output feature names for downstream tracking.

**`DataCleaner`** — standardizes types and removes duplicate rows or duplicate columns.

- `fit` — learns duplicate columns and standardized column types from training data.
- `transform` — drops learned duplicate columns and coerces columns to fitted types.
- `fit_transform` — removes duplicate rows during training and returns the cleaned DataFrame.
- `get_feature_names_out` — returns the remaining feature names after cleaning.

**`MissingValueTransformer`** — imputes missing values and optionally creates missingness indicators.

- `fit` — learns per-column imputation strategies and fill values.
- `transform` — imputes missing values and appends missing-indicator columns.
- `get_feature_names_out` — returns feature names including added indicator columns.

**`RareCategoryTransformer`** — groups low-frequency categorical levels into a fallback label.

- `fit` — learns the frequent categories to retain for each categorical column.
- `transform` — replaces rare categories with the configured fallback value.
- `get_feature_names_out` — returns the output feature names after rare-category handling.

**`EncodingTransformer`** — converts categorical variables into model-ready numerical representations.

- `fit` — learns one-hot, ordinal, frequency, or target-encoding mappings from training data.
- `transform` — applies the learned encodings to categorical columns.
- `fit_transform` — applies cross-validation-safe out-of-fold target encoding when target encoding is enabled.
- `get_feature_names_out` — returns the final encoded feature names.

**`OutlierTransformer`** — detects and treats numerical outliers using fitted statistical limits.

- `fit` — calculates per-column clipping or winsorization limits from training data.
- `transform` — clips values to the fitted outlier limits.
- `get_feature_names_out` — returns output feature names after outlier handling.

**`FeatureTransformer`** — applies mathematical transformations to skewed numerical features.

- `fit` — learns transformation parameters such as Box-Cox or Yeo-Johnson lambdas.
- `transform` — applies the configured transformations using fitted parameters.
- `get_feature_names_out` — returns output feature names after feature transformation.

**`FeatureScaler`** — scales numerical features for models sensitive to feature magnitude.

- `fit` — fits the configured scaler for each eligible numerical feature.
- `transform` — applies the fitted scaling transformations to new data.
- `get_feature_names_out` — returns output feature names after scaling.

### Scripts

#### `run_preprocessing.py`

- `run_pipeline_for_config` — loads a YAML config, runs a preprocessing pipeline, and saves processed train/validation datasets plus reports.
- `main` — loads raw data, creates a stratified train/validation split, and runs all configured preprocessing pipelines.

#### `verify_eda.py`

- `main` — loads a sample of the raw dataset and runs the EDA orchestrator for verification.
