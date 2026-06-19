"""Base custom transformer for the preprocessing pipeline.

This module defines the BaseTransformer, which serves as the base class for all 
custom estimators/transformers in the preprocessing framework. It inherits from 
scikit-learn's BaseEstimator and TransformerMixin to ensure seamless integration 
with scikit-learn Pipelines, Grid Searches, and other tools.

By inheriting from this base class, custom transformers get automatic fit/transform 
state checking and the default implementation of standard interfaces, promoting 
code reuse and reducing boilerplate.
"""

from sklearn.base import BaseEstimator, TransformerMixin

class BaseTransformer(BaseEstimator, TransformerMixin):
    """Abstract base class for all preprocessing transformers.

    Ensures that subclassed estimators conform to the scikit-learn API 
    (providing fit, transform, and fit_transform) while managing internal 
    state verification to prevent calling transform before fit.
    
    Attributes:
        fitted_ (bool): State variable indicating whether the transformer 
            has been fitted on a dataset.
    """
    
    def __init__(self):
        """Initializes the base transformer, setting the fitted state to False."""
        self.fitted_ = False

    def fit(self, X, y=None):
        """Fits the transformer on the input dataset.

        This method marks the transformer as fitted. Custom subclass estimators 
        should override this to learn parameters (like mean, median, modes, 
        encodings) from the training set.

        Args:
            X (pd.DataFrame or np.ndarray): The input features to fit on.
            y (pd.Series or np.ndarray, optional): The target variable. Defaults to None.

        Returns:
            BaseTransformer: The fitted instance of the transformer.
        """
        self.fitted_ = True
        return self

    def transform(self, X):
        """Applies the learned transformations to the input features.

        Custom subclass estimators must override this method. The base class 
        implements state verification to ensure the transformer has been fitted 
        prior to transforming.

        Args:
            X (pd.DataFrame or np.ndarray): The input features to transform.

        Returns:
            pd.DataFrame or np.ndarray: The transformed features.

        Raises:
            ValueError: If the transformer has not been fitted prior to transformation.
        """
        if not self.fitted_:
            raise ValueError(
                "This transformer instance is not fitted yet. "
                "You must call 'fit' before calling 'transform' to avoid target/data leakage."
            )
        return X

    def get_feature_names_out(self, input_features=None):
        """Returns the list of output feature names after transformation.

        Ensures compatibility with scikit-learn's feature name tracking API, 
        making it easy to trace how columns are added, removed, or renamed.

        Args:
            input_features (list of str, optional): The list of input feature names. 
                Defaults to None.

        Returns:
            list of str: The output feature names. By default, returns the input names unchanged.
        """
        return input_features
