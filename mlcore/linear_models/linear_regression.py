import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000, l1_ratio=0.0, alpha=0.0):
        self.lr = lr
        self.n_iters = n_iters
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        y = y.flatten()

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.w) + self.b
            diff = y_pred - y
            
            dw = (2 / n_samples) * np.dot(X.T, diff)
            db = (2 / n_samples) * np.sum(diff)
            self.w -= self.lr * dw
            self.b -= self.lr * db

            if self.alpha > 0 and self.l1_ratio > 0:
                threshold = self.lr * self.alpha * self.l1_ratio
                self.w = np.sign(self.w) * np.maximum(np.abs(self.w) - threshold, 0)

            if self.alpha > 0 and self.l1_ratio < 1:
                self.w -= self.lr * self.alpha * (1 - self.l1_ratio) * self.w

    def predict(self, X):
        return np.dot(X, self.w) + self.b