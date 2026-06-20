# Data Profiling & Quality Analysis Checklist

## Implemented class diagram

**How to read this diagram**

- `+` denotes a public attribute or method.
- `<<service>>` marks stateless analysis/reporting classes; `<<utility>>` marks helper classes.
- Method signatures show the main inputs and, where relevant, default argument values.
- `..>` indicates a dependency or usage relationship.
- `*--` indicates strong ownership/composition inside `ProfilingResult`; `o--` indicates optional contained results.
- Read the flow from `DataProfiler` outward: it coordinates analyzers, assembles a `ProfilingResult`, and that result is later consumed by reporting and visualization helpers.

```mermaid
classDiagram
    class DataProfiler {
        +target_column
        +DataProfiler(target_column=None)
        +run(df) ProfilingResult
    }

    class DatasetStatistics {
        <<service>>
        +summarize(df) DatasetStats
    }

    class MissingValueAnalyzer {
        <<service>>
        +analyze(df, threshold=50.0) MissingValueReport
    }

    class DuplicateAnalyzer {
        <<service>>
        +analyze(df) DuplicateReport
    }

    class ConstantFeatureAnalyzer {
        <<service>>
        +analyze(df, near_constant_threshold=99.0) ConstantFeatureReport
    }

    DataProfiler ..> DatasetStatistics : summarize()
    DataProfiler ..> MissingValueAnalyzer : analyze()
    DataProfiler ..> DuplicateAnalyzer : analyze()
    DataProfiler ..> ConstantFeatureAnalyzer : analyze()
```


```mermaid
classDiagram
    class DataProfiler {
        +target_column
        +DataProfiler(target_column=None)
        +run(df) ProfilingResult
    }

    class FeatureInspector {
        <<utility>>
        +get_numeric_columns(df) columns
        +get_categorical_columns(df) columns
        +get_datetime_columns(df) columns
        +get_boolean_columns(df) columns
    }

    class CardinalityAnalyzer {
        <<service>>
        +analyze(df, categorical_only=True) CardinalityReport
    }

    class TargetAnalyzer {
        <<service>>
        +analyze(df, target_column) TargetAnalysisReport
    } 

    class LeakageAnalyzer {
        <<service>>
        +analyze(df, target_column=None, corr_threshold=0.95, unique_ratio_threshold=0.95) LeakageReport
    }

	  CardinalityAnalyzer ..> FeatureInspector : categorical columns
    LeakageAnalyzer ..> FeatureInspector : numeric columns
    DataProfiler ..> CardinalityAnalyzer : analyze()
    DataProfiler ..> LeakageAnalyzer : analyze()
    DataProfiler ..> TargetAnalyzer : analyze()
```


```mermaid
classDiagram
    class DataProfiler {
        +target_column
        +DataProfiler(target_column=None)
        +run(df) ProfilingResult
    }

    class MissingValueVisualizer {
        +plot_missing_percentages(percentages, top_n=20) Figure
    }

    class MissingValueReport {
        +counts: per-feature missing count
        +percentages: per-feature missing percent
        +features_above_threshold: flagged features
    }

    class ReportGenerator {
        +ReportGenerator(output_dir)
        +save_results(results) void
        +save_figure(fig, filename) void
    }

    class ProfilingResult {
        +dataset_stats: DatasetStats
        +missing_values: MissingValueReport
        +duplicates: DuplicateReport
        +constant_features: ConstantFeatureReport
        +cardinality: CardinalityReport
        +target_analysis: TargetAnalysisReport
        +leakage: LeakageReport
    }

    class DatasetStats {
        +rows: int
        +columns: int
        +memory_mb: float
    }

    class DuplicateReport {
        +duplicate_count: int
    }

    class ConstantFeatureReport {
        +constant_features: columns
        +near_constant_features: columns
    }

    class CardinalityReport {
        +cardinality: per-feature unique count
    }

    class LeakageReport {
        +potential_leakage_columns: columns
    }

    class TargetAnalysisReport {
        +positive_count: int
        +negative_count: int
        +imbalance_ratio: float
        +baseline_accuracy: float
    }

    DataProfiler --> ProfilingResult : returns
    ProfilingResult *-- DatasetStats
    ProfilingResult *-- MissingValueReport
    ProfilingResult *-- DuplicateReport
    ProfilingResult *-- ConstantFeatureReport
    ProfilingResult *-- CardinalityReport
    ProfilingResult o-- TargetAnalysisReport
    ProfilingResult o-- LeakageReport
    ReportGenerator ..> ProfilingResult : persists
    MissingValueVisualizer ..> MissingValueReport : plots
```

## Dataset Understanding
- Load dataset
- Dataset dimensions
- Feature types
- Target variable distribution
- Memory consumption

## Data Quality
- Missing value analysis
- Duplicate row analysis
- Constant feature detection
- Near-constant feature detection
- Unique value analysis
- High cardinality detection

## Statistical Analysis
- Numerical feature summary
- Categorical feature summary
- Target-wise summary

## Class Imbalance Analysis
- Class distribution
- Imbalance ratio
- Baseline majority class accuracy

## Data Leakage Detection
Data leakage occurs when information from outside the training dataset is used to create the model. This can lead to overly optimistic performance during training and poor performance in production.

Our profiling framework implements a `LeakageAnalyzer` that flags potential leakage based on two criteria:

1.  **ID-like Column Detection**: Features that have an extremely high ratio of unique values to total rows (e.g., > 95%). These are often primary keys (like `SK_ID_CURR`) or unique identifiers that shouldn't be used as predictive features.
2.  **High Target Correlation**: Numeric features that exhibit an exceptionally high absolute correlation with the target variable (e.g., > 0.95). While high correlation is generally good, extreme values often indicate that the feature is a proxy for the target or is recorded after the target event has occurred.

Features flagged by this analyzer should be manually reviewed before being included in the training pipeline.

- Suspicious columns
- ID columns
- Features highly correlated with target

## Visualization
- Missing value heatmap
- Missing value percentages
- Target distribution
- Numerical feature distributions
- Correlation matrix

## Reporting
- Generate profiling report
- Save plots
- Save quality metrics
- Save summary report