"""
Analyzers module for core data analysis tasks.

This package provides specialized analyzers for target distribution, 
univariate/bivariate statistics, correlations, outliers, and feature importance.

The __init__.py file simplifies the package's public API by exposing core 
analyzers directly at the package level.
"""

from .target import TargetAnalyzer
from .univariate import UnivariateAnalyzer
from .bivariate import BivariateAnalyzer
from .correlation import CorrelationAnalyzer
from .outlier import OutlierAnalyzer
from .importance import ImportanceAnalyzer
from .dimensionality import DimensionalityAnalyzer
from .missing_patterns import MissingPatternAnalyzer
