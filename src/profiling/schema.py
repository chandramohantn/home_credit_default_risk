from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class DatasetStats:
    rows: int
    columns: int
    memory_mb: float

@dataclass
class MissingValueReport:
    counts: Dict[str, int]
    percentages: Dict[str, float]
    features_above_threshold: List[str]

@dataclass
class DuplicateReport:
    duplicate_count: int

@dataclass
class ConstantFeatureReport:
    constant_features: List[str]
    near_constant_features: List[str]

@dataclass
class CardinalityReport:
    cardinality: Dict[str, int]

@dataclass
class TargetAnalysisReport:
    positive_count: int
    negative_count: int
    imbalance_ratio: float
    baseline_accuracy: float

@dataclass
class LeakageReport:
    potential_leakage_columns: List[str]

@dataclass
class ProfilingResult:
    dataset_stats: DatasetStats
    missing_values: MissingValueReport
    duplicates: DuplicateReport
    constant_features: ConstantFeatureReport
    cardinality: CardinalityReport
    target_analysis: Optional[TargetAnalysisReport] = None
    leakage: Optional[LeakageReport] = None
