import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        X = np.asanyarray(X)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        cov = np.cov(X_centered, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        total_var = np.sum(vals)
        self.explained_variance_ratio_ = (vals / total_var)[:self.n_components]        
        self.components = vecs[:, :self.n_components].T
        return self

    def transform(self, X):
        X = np.asanyarray(X)
        return np.dot(X - self.mean, self.components.T)

    def fit_transform(self, X):
        return self.fit(X).transform(X)