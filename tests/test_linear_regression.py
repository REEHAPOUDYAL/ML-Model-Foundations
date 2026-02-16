import numpy as np
import pytest
from mlcore.linear_models.linear_regression import LinearRegression

def test_convergence():
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 2 * X.flatten() + 1
    
    model = LinearRegression(lr=0.01, n_iters=500)
    model.fit(X, y)    
    assert np.allclose(model.w, [2.0], atol=1e-1)
    assert np.allclose(model.b, 1.0, atol=1e-1)

def test_regularization_shrinkage():
    X = np.random.randn(50, 5)
    y = np.dot(X, [10, -5, 0, 0, 0]) + np.random.normal(0, 0.1, 50)
    
    m_no_reg = LinearRegression(lr=0.01, n_iters=1000, alpha=0)
    m_no_reg.fit(X, y)
    m_reg = LinearRegression(lr=0.01, n_iters=1000, alpha=10.0, l1_ratio=0)
    m_reg.fit(X, y)
    
    assert np.linalg.norm(m_reg.w) < np.linalg.norm(m_no_reg.w)

def test_lasso_sparsity():
    X = np.random.randn(50, 10)
    y = 5 * X[:, 0] + 3 * X[:, 1] + np.random.normal(0, 0.1, 50)
    model = LinearRegression(lr=0.01, n_iters=2000, alpha=1.0, l1_ratio=1.0)
    model.fit(X, y)

    zero_weights = np.sum(np.abs(model.w) < 1e-3)
    assert zero_weights > 0

def test_prediction_shape():
    X = np.random.randn(10, 3)
    y = np.random.randn(10)
    model = LinearRegression()
    model.fit(X, y)
    
    preds = model.predict(X)
    assert preds.shape == y.shape