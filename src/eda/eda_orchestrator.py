"""Orchestration module for EDA Phase 3."""

import pandas as pd
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt

from analyzers.target import TargetAnalyzer
from analyzers.univariate import UnivariateAnalyzer
from analyzers.bivariate import BivariateAnalyzer
from analyzers.correlation import CorrelationAnalyzer
from analyzers.outlier import OutlierAnalyzer
from analyzers.importance import ImportanceAnalyzer
from analyzers.dimensionality import DimensionalityAnalyzer
from analyzers.missing_patterns import MissingPatternAnalyzer

from visualization.target_analysis import TargetVisualizer
from visualization.distributions import DistributionVisualizer
from visualization.correlations import CorrelationVisualizer
from visualization.outliers import OutlierVisualizer
from visualization.importance import ImportanceVisualizer
from visualization.dimensionality import DimensionalityVisualizer

class EDAOrchestrator:
    """Orchestrates the execution of Phase 3 EDA."""

    def __init__(self, output_dir: Path, target_column: str):
        """Initializes the orchestrator.

        Args:
            output_dir (Path): Root directory for reports.
            target_column (str): Target column name.
        """
        self.output_dir = output_dir
        self.target_column = target_column
        self.figures_dir = self.output_dir / "figures"
        
        # Create subdirectories
        for sub in ["distributions", "target_relationships", "correlations", "outliers", "dimensionality_reduction", "feature_importance"]:
            (self.figures_dir / sub).mkdir(parents=True, exist_ok=True)

    def run(self, df: pd.DataFrame, num_features: List[str] = None, cat_features: List[str] = None):
        """Runs the complete EDA pipeline.

        Args:
            df (pd.DataFrame): The dataset.
            num_features (List[str], optional): Numerical features to analyze.
            cat_features (List[str], optional): Categorical features to analyze.
        """
        if num_features is None:
            num_features = df.select_dtypes(include=['number']).columns.tolist()
            if self.target_column in num_features:
                num_features.remove(self.target_column)
        
        if cat_features is None:
            cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

        print("--- Section 1: Target Analysis ---")
        target_results = TargetAnalyzer.analyze(df, self.target_column)
        target_fig = TargetVisualizer.plot_distribution(df, self.target_column)
        target_fig.savefig(self.figures_dir / "target_relationships" / "target_distribution.png")
        plt.close(target_fig)

        print("--- Section 2: Univariate Analysis ---")
        # Just top 10 for speed in this run, or all
        for col in num_features[:15]:
            fig = DistributionVisualizer.plot_numerical(df, col)
            fig.savefig(self.figures_dir / "distributions" / f"{col}_dist.png")
            plt.close(fig)
            
        for col in cat_features[:10]:
            fig = DistributionVisualizer.plot_categorical(df, col)
            fig.savefig(self.figures_dir / "distributions" / f"{col}_count.png")
            plt.close(fig)

        print("--- Section 3: Target Relationship Analysis ---")
        numeric_biv = BivariateAnalyzer.analyze_numeric(df, self.target_column, num_features[:15])
        cat_biv = BivariateAnalyzer.analyze_categorical(df, self.target_column, cat_features[:10])
        
        for col in num_features[:15]:
            fig = TargetVisualizer.plot_numeric_vs_target(df, col, self.target_column)
            fig.savefig(self.figures_dir / "target_relationships" / f"{col}_vs_target.png")
            plt.close(fig)
            
        for col in cat_features[:10]:
            fig = TargetVisualizer.plot_categorical_vs_target(df, col, self.target_column)
            fig.savefig(self.figures_dir / "target_relationships" / f"{col}_vs_target.png")
            plt.close(fig)

        print("--- Section 4: Correlation Analysis ---")
        corr_matrix = CorrelationAnalyzer.analyze_correlations(df)
        fig = CorrelationVisualizer.plot_heatmap(df, columns=num_features[:30])
        fig.savefig(self.figures_dir / "correlations" / "correlation_heatmap.png")
        plt.close(fig)

        print("--- Section 8: Feature Importance (MI) ---")
        mi_scores = ImportanceAnalyzer.calculate_mutual_info(df, self.target_column)
        fig = ImportanceVisualizer.plot_importance(mi_scores, "Mutual Information Scores")
        fig.savefig(self.figures_dir / "feature_importance" / "mi_scores.png")
        plt.close(fig)

        print("--- Section 9: Dimensionality Reduction ---")
        X_pca, _ = DimensionalityAnalyzer.run_pca(df[num_features].dropna())
        pca_fig = DimensionalityVisualizer.plot_projection(X_pca, df.loc[df[num_features].dropna().index, self.target_column], "PCA Projection")
        pca_fig.savefig(self.figures_dir / "dimensionality_reduction" / "pca_2d.png")
        plt.close(pca_fig)

        self._generate_summary_report(target_results, mi_scores)
        self._save_csv_deliverables(numeric_biv, cat_biv, num_features, cat_features, df)

    def _save_csv_deliverables(self, numeric_biv, cat_biv, num_features, cat_features, df):
        """Saves structured insights to CSV files."""
        # target_relationships.csv
        target_rel_data = []
        for col, stats in numeric_biv.items():
            target_rel_data.append({
                "feature": col,
                "type": "numeric",
                "mean_diff": stats.mean_diff,
                "cohen_d": stats.cohen_d,
                "mutual_info": stats.mutual_info
            })
        for col, stats in cat_biv.items():
            target_rel_data.append({
                "feature": col,
                "type": "categorical",
                "chi2_score": stats.chi2_score,
                "mutual_info": stats.mutual_info
            })
        pd.DataFrame(target_rel_data).to_csv(self.output_dir / "target_relationships.csv", index=False)

        # feature_insights.csv (Univariate)
        univariate_analyzer = UnivariateAnalyzer()
        num_stats = univariate_analyzer.analyze_numeric(df, num_features[:15])
        
        feature_insights = []
        for col, stats in num_stats.items():
            feature_insights.append({
                "feature": col,
                "mean": stats.mean,
                "skew": stats.skew,
                "kurtosis": stats.kurtosis
            })
        pd.DataFrame(feature_insights).to_csv(self.output_dir / "feature_insights.csv", index=False)

    def _generate_summary_report(self, target_results, mi_scores):
        """Generates the markdown summary report."""
        report_path = self.output_dir / "eda_summary.md"
        with open(report_path, "w") as f:
            f.write("# Phase 3: Exploratory Data Analysis Summary\n\n")
            f.write("## Target Analysis\n")
            f.write(f"- Baseline Accuracy: {target_results.baseline_accuracy:.2f}%\n")
            f.write(f"- Counts: {target_results.counts}\n\n")
            f.write("## Top Predictive Features (Mutual Information)\n")
            f.write(mi_scores.to_markdown())
            f.write("\n\n## Key Observations\n")
            f.write("- [Add observations here based on generated plots]\n")
