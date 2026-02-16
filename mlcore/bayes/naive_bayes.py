import numpy as np
from ..utils.validation import check_X_y

class GaussianNB:
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes = np.unique(y)
        self.stats = []

        for c in self.classes:
            X_c = X[y == c]
            self.stats.append({
                "mean": np.mean(X_c, axis=0),
                "var": np.var(X_c, axis=0) + 1e-9,
                "prior": X_c.shape[0] / X.shape[0]
            })
        return self

    def _get_log_pdf(self, class_idx, x):
        mean = self.stats[class_idx]["mean"]
        var = self.stats[class_idx]["var"]
        exponent = -((x - mean) ** 2) / (2 * var)
        log_coeff = -0.5 * np.log(2 * np.pi * var)
        return log_coeff + exponent

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        posteriors = []
        for i, c in enumerate(self.classes):
            log_prior = np.log(self.stats[i]["prior"])
            log_likelihood = np.sum(self._get_log_pdf(i, x))
            posteriors.append(log_prior + log_likelihood)
        return self.classes[np.argmax(posteriors)]