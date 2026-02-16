import numpy as np
from .exceptions import ValidationError, InconsistentSamplesError

def check_array(X, ensure_2d=True, allow_nd=False):
    if not isinstance(X, (np.ndarray, list, tuple)):
        raise ValidationError(f"Expected numpy array or array-like, got {type(X)}")
    
    X = np.array(X)
    
    if ensure_2d and X.ndim == 1:
        X = X.reshape(-1, 1)
    
    if not allow_nd and X.ndim > 2:
        raise ValidationError(f"Found array with dim {X.ndim}. Expected <= 2.")
        
    if np.any(np.isnan(X)):
        raise ValidationError("Input contains NaN values.")
        
    if np.any(np.isinf(X)):
        raise ValidationError("Input contains infinity or a value too large for dtype.")
        
    return X

def check_X_y(X, y):
    X = check_array(X)
    y = np.array(y)
    
    if X.shape[0] != y.shape[0]:
        raise InconsistentSamplesError(
            f"Found input variables with inconsistent numbers of samples: [{X.shape[0]}, {y.shape[0]}]"
        )
        
    return X, y

def check_is_fitted(estimator, attributes):
    if not all(hasattr(estimator, attr) for attr in attributes):
        raise NotFittedError(
            f"This {type(estimator).__name__} instance is not fitted yet. Call 'fit' first."
        )