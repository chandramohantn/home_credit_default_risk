"""Transformer registration directory.

This module exposes the TRANSFORMER_REGISTRY dictionary which maps text-based 
configuration keys (used in preprocessing YAML configurations) to actual custom 
transformer classes.

A global registry makes it easy to instantiate pipelines dynamically from static 
configuration files without writing conditional initialization logic in python code.
"""

from src.preprocessing.transformers.cleaning import DataCleaner
from src.preprocessing.transformers.missing import MissingValueTransformer
from src.preprocessing.transformers.rare_categories import RareCategoryTransformer
from src.preprocessing.transformers.encoding import EncodingTransformer
from src.preprocessing.transformers.outliers import OutlierTransformer
from src.preprocessing.transformers.transformations import FeatureTransformer
from src.preprocessing.transformers.scaling import FeatureScaler

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
