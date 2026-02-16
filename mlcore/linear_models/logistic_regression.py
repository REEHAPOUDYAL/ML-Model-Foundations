import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000, alpha=0.0):
        self.lr = lr
        self.n_iters = n_iters
        self.alpha = alpha
        self.w = None
        self.b = None

    def _sigmoid(self, z):
        z = np.asanyarray(z)
        mask_pos = z >= 0
        mask_neg = z < -709
        
        res = np.empty_like(z, dtype=np.float64)
        
        # Standard stable sigmoid
        res[mask_pos] = 1 / (1 + np.exp(-z[mask_pos]))
        exp_z = np.exp(z[~mask_pos])
        res[~mask_pos] = exp_z / (1 + exp_z)
        
        # Force saturation for test stability
        res[z > 709] = 1.0
        res[mask_neg] = 0.0
        
        return res

    def fit(self, X, y):
        X = np.asanyarray(X)
        y = np.asanyarray(y).flatten()
        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.n_iters):
            y_pred = self._sigmoid(np.dot(X, self.w) + self.b)
            error = y_pred - y
            
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            if self.alpha > 0:
                dw += self.alpha * self.w

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        return self._sigmoid(np.dot(X, self.w) + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)