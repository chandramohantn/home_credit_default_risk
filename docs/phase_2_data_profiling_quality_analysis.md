# Data Profiling & Quality Analysis Checklist

## Implemented class diagram

```mermaid
classDiagram
    class DataProfiler {
        +target_column: optional target
        +run(df) ProfilingResult
    }

    class DatasetStatistics {
        +summarize(df) DatasetStats
    }

    class MissingValueAnalyzer {
        +analyze(df, threshold) MissingValueReport
    }

    class DuplicateAnalyzer {
        +analyze(df) DuplicateReport
    }

    class ConstantFeatureAnalyzer {
        +analyze(df, near_constant_threshold) ConstantFeatureReport
    }

    class CardinalityAnalyzer {
        +analyze(df, categorical_only) CardinalityReport
    }

    class TargetAnalyzer {
        +analyze(df, target_column) TargetAnalysisReport
    }

    class LeakageAnalyzer {
        +analyze(df, target_column, corr_threshold, unique_ratio_threshold) LeakageReport
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

    class ReportGenerator {
        +save_results(results) void
        +save_figure(fig, filename) void
    }

    class MissingValueVisualizer {
        +plot_missing_percentages(percentages, top_n) Figure
    }

    class DatasetStats
    class MissingValueReport
    class DuplicateReport
    class ConstantFeatureReport
    class CardinalityReport
    class TargetAnalysisReport
    class LeakageReport

    DataProfiler ..> DatasetStatistics : summarize()
    DataProfiler ..> MissingValueAnalyzer : analyze()
    DataProfiler ..> DuplicateAnalyzer : analyze()
    DataProfiler ..> ConstantFeatureAnalyzer : analyze()
    DataProfiler ..> CardinalityAnalyzer : analyze()
    DataProfiler ..> TargetAnalyzer : analyze()
    DataProfiler ..> LeakageAnalyzer : analyze()
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