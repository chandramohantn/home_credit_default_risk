"""Module for generating and saving profiling reports."""

import json
import pandas as pd
from pathlib import Path
from dataclasses import asdict
from .schema import ProfilingResult

class ReportGenerator:
    """Handles saving profiling results to various file formats."""

    def __init__(self, output_dir: Path):
        """Initializes the report generator.

        Args:
            output_dir (Path): The directory where reports and figures 
                will be saved.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)

    def save_results(self, results: ProfilingResult):
        """Saves analysis results to JSON and CSV files.

        Args:
            results (ProfilingResult): The profiling results to save.
        """
        # Save dataset stats
        with open(self.output_dir / "dataset_summary.json", "w") as f:
            json.dump(asdict(results.dataset_stats), f, indent=4)

        # Save missing values
        missing_df = pd.DataFrame({
            "feature": list(results.missing_values.counts.keys()),
            "missing_count": list(results.missing_values.counts.values()),
            "missing_pct": list(results.missing_values.percentages.values())
        }).sort_values("missing_pct", ascending=False)
        missing_df.to_csv(self.output_dir / "missing_values.csv", index=False)

        # Save duplicates
        with open(self.output_dir / "duplicates.json", "w") as f:
            json.dump(asdict(results.duplicates), f, indent=4)

        # Save cardinality
        cardinality_df = pd.DataFrame({
            "feature": list(results.cardinality.cardinality.keys()),
            "unique_values": list(results.cardinality.cardinality.values())
        }).sort_values("unique_values", ascending=False)
        cardinality_df.to_csv(self.output_dir / "cardinality.csv", index=False)

        # Save target analysis if available
        if results.target_analysis:
            with open(self.output_dir / "target_analysis.json", "w") as f:
                json.dump(asdict(results.target_analysis), f, indent=4)

        # Save leakage
        if results.leakage:
            with open(self.output_dir / "leakage.json", "w") as f:
                json.dump(asdict(results.leakage), f, indent=4)

    def save_figure(self, fig, filename: str):
        """Saves a matplotlib figure to the figures subdirectory.

        Args:
            fig (matplotlib.figure.Figure): The figure object to save.
            filename (str): The name of the file (e.g., 'plot.png').
        """
        fig.savefig(self.output_dir / "figures" / filename)
