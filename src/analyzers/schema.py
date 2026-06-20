"""Shared data schemas for EDA analyzer outputs."""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class TargetEDAResult:
    """Dataclass for target analysis results."""

    counts: Dict[Any, int]
    percentages: Dict[Any, float]
    baseline_accuracy: float
    baseline_precision: float
    baseline_recall: float
    baseline_f1: float


@dataclass
class NumericalUnivariateStats:
    """Statistics for a single numerical feature."""

    mean: float
    median: float
    std: float
    min: float
    max: float
    skew: float
    kurtosis: float


@dataclass
class CategoricalUnivariateStats:
    """Statistics for a single categorical feature."""

    cardinality: int
    top_categories: Dict[Any, int]
    rare_categories: List[Any]
    rare_threshold: float


@dataclass
class NumericBivariateStats:
    """Statistics for a numerical feature vs target."""

    mean_diff: float
    median_diff: float
    cohen_d: float
    mutual_info: float


@dataclass
class CategoricalBivariateStats:
    """Statistics for a categorical feature vs target."""

    target_rate_per_category: Dict[Any, float]
    chi2_score: float
    mutual_info: float


@dataclass
class OutlierStats:
    """Statistics for outliers in a feature."""

    iqr_outliers_count: int
    zscore_outliers_count: int
    outlier_indices: List[int]
