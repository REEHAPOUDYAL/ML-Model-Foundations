import numpy as np
from ..utils.validation import check_X_y, check_is_fitted

class KNeighborsClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train, self.y_train = check_X_y(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, ['X_train', 'y_train'])
        X = np.array(X)
        preds = [self._predict_single(x) for x in X]
        return np.array(preds)

    def _predict_single(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        counts = np.bincount(k_nearest_labels.astype(int))
        return np.argmax(counts)