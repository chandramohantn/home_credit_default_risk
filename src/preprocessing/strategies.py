"""Shared strategy registries for configurable preprocessing components."""

from sklearn.preprocessing import (
    MinMaxScaler,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


ENCODER_REGISTRY = {
    "one_hot": "One-hot encoding for linear and sparse-friendly models.",
    "ordinal": "Ordinal integer mapping for tree-based models.",
    "frequency": "Frequency encoding based on category prevalence.",
    "target": "Smoothed target encoding with out-of-fold training support.",
    "catboost": "Ordered target encoding inspired by CatBoost statistics.",
}


SCALER_REGISTRY = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "quantile": QuantileTransformer,
    "none": None,
}


TRANSFORMATION_REGISTRY = {
    "log": {"requires_positive": True, "learns_lambda": False},
    "log1p": {"requires_positive": False, "learns_lambda": False},
    "sqrt": {"requires_positive": False, "learns_lambda": False},
    "box_cox": {"requires_positive": True, "learns_lambda": True},
    "yeo_johnson": {"requires_positive": False, "learns_lambda": True},
}


OUTLIER_METHOD_REGISTRY = {
    "iqr": "Interquartile range bounds.",
    "z_score": "Standard deviation thresholding.",
    "winsorize": "Quantile-based clipping bounds.",
    "isolation_forest": "IsolationForest-based row scoring.",
}
