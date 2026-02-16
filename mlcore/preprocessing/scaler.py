import numpy as np
from ..utils.validation import check_array
from ..utils.exceptions import NotFittedError

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = check_array(X)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise NotFittedError("Scaler must be fitted before transforming.")
        X = check_array(X)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)