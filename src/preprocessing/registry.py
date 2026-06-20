"""Registries for configurable preprocessing components and supported strategies."""

from src.preprocessing.transformers.cleaning import DataCleaner
from src.preprocessing.transformers.missing import MissingValueTransformer
from src.preprocessing.transformers.rare_categories import RareCategoryTransformer
from src.preprocessing.transformers.encoding import EncodingTransformer
from src.preprocessing.transformers.outliers import OutlierTransformer
from src.preprocessing.transformers.transformations import FeatureTransformer
from src.preprocessing.transformers.scaling import FeatureScaler
from src.preprocessing.strategies import (
    ENCODER_REGISTRY,
    OUTLIER_METHOD_REGISTRY,
    SCALER_REGISTRY,
    TRANSFORMATION_REGISTRY,
)

# Global registry mapping configuration strings to custom transformer classes.
# Why: This architecture facilitates decoupling configuration from implementation.
TRANSFORMER_REGISTRY = {
    "cleaner": DataCleaner,
    "missing": MissingValueTransformer,
    "rare_categories": RareCategoryTransformer,
    "encoder": EncodingTransformer,
    "outliers": OutlierTransformer,
    "transformations": FeatureTransformer,
    "scaler": FeatureScaler,
}


LEGACY_STEP_ALIASES = {
    "cleaner": "cleaner",
    "imputation": "missing",
    "missing": "missing",
    "rare_categories": "rare_categories",
    "encoding": "encoder",
    "encoder": "encoder",
    "outliers": "outliers",
    "transformations": "transformations",
    "scaling": "scaler",
    "scaler": "scaler",
}
